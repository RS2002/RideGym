"""Verify the standalone simulator's wheel and source-distribution boundaries.

    python .github/scripts/verify_distribution.py dist/*

Only standard-library modules are required. This script is repository tooling;
it is deliberately excluded from both published distributions.
"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

SOURCE_FILES = {
    "LICENSE",
    "MANIFEST.in",
    "PKG-INFO",
    "README_PYPI.md",
    "pyproject.toml",
    "setup.cfg",
}
EGG_INFO_FILES = {
    "PKG-INFO",
    "SOURCES.txt",
    "dependency_links.txt",
    "requires.txt",
    "top_level.txt",
}
WHEEL_INFO_FILES = {"METADATA", "RECORD", "WHEEL", "top_level.txt"}
REQUIRED_MODULES = {"ride_gym/__init__.py", "ride_gym/env.py"}


def checked_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def is_module(path: PurePosixPath) -> bool:
    return (
        len(path.parts) >= 2
        and path.parts[0] == "ride_gym"
        and path.suffix == ".py"
        and "__pycache__" not in path.parts
    )


def check_metadata(data: bytes, archive: Path) -> tuple[str, str]:
    metadata = BytesParser().parsebytes(data)
    if metadata["Name"] != "ride-gym" or not metadata["Version"]:
        raise ValueError(f"Unexpected distribution metadata: {archive.name}")
    if metadata["License-Expression"] != "MIT":
        raise ValueError(f"Expected MIT license metadata: {archive.name}")
    return metadata["Name"], metadata["Version"]


def check_wheel(path: Path) -> tuple[set[str], tuple[str, str]]:
    modules = set()
    info_roots = set()
    metadata = None
    has_license = False
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = checked_path(member.filename)
            if member.is_dir():
                continue
            if is_module(relative):
                modules.add(relative.as_posix())
            elif (
                len(relative.parts) == 2
                and relative.parts[0].startswith("ride_gym-")
                and relative.parts[0].endswith(".dist-info")
                and relative.parts[1] in WHEEL_INFO_FILES
            ):
                info_roots.add(relative.parts[0])
                if relative.name == "METADATA":
                    metadata = check_metadata(archive.read(member), path)
            elif (
                len(relative.parts) == 3
                and relative.parts[0].startswith("ride_gym-")
                and relative.parts[0].endswith(".dist-info")
                and relative.parts[1:] == ("licenses", "LICENSE")
            ):
                info_roots.add(relative.parts[0])
                has_license = True
            else:
                raise ValueError(f"Unexpected wheel member: {member.filename}")
    if len(info_roots) != 1 or metadata is None:
        raise ValueError(f"Missing or ambiguous wheel metadata: {path.name}")
    if info_roots != {f"ride_gym-{metadata[1]}.dist-info"}:
        raise ValueError(f"Wheel metadata directory does not match version: {path.name}")
    if not has_license:
        raise ValueError(f"Missing wheel license: {path.name}")
    return modules, metadata


def check_sdist(path: Path) -> tuple[set[str], tuple[str, str]]:
    modules = set()
    roots = set()
    metadata = None
    has_license = False
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            full = checked_path(member.name)
            roots.add(full.parts[0])
            if member.isdir():
                continue
            if not member.isfile() or len(full.parts) < 2:
                raise ValueError(f"Unexpected source member: {member.name}")
            relative = PurePosixPath(*full.parts[1:])
            if is_module(relative):
                modules.add(relative.as_posix())
            elif len(relative.parts) == 1 and relative.name in SOURCE_FILES:
                if relative.name == "LICENSE":
                    has_license = True
                if relative.name == "PKG-INFO":
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("Could not read source metadata")
                    with stream:
                        metadata = check_metadata(stream.read(), path)
            elif (
                len(relative.parts) == 2
                and relative.parts[0] == "ride_gym.egg-info"
                and relative.parts[1] in EGG_INFO_FILES
            ):
                pass
            else:
                raise ValueError(f"Unexpected source member: {member.name}")
    if len(roots) != 1 or metadata is None:
        raise ValueError(f"Missing or ambiguous source metadata: {path.name}")
    if roots != {f"ride_gym-{metadata[1]}"}:
        raise ValueError(f"Source directory does not match version: {path.name}")
    if not has_license:
        raise ValueError(f"Missing source license: {path.name}")
    return modules, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    results = []
    for path in args.archives:
        if path.name.endswith(".whl"):
            result = check_wheel(path)
        elif path.name.endswith(".tar.gz"):
            result = check_sdist(path)
        else:
            parser.error(f"Unsupported distribution: {path}")
        modules, metadata = result
        if not REQUIRED_MODULES <= modules:
            raise ValueError(f"Missing simulator modules in {path.name}")
        results.append(result)
        print(f"Verified {path.name}: {len(modules)} simulator modules, metadata and docs only")
    if any(result != results[0] for result in results[1:]):
        raise ValueError("Distributions disagree on simulator modules or package identity")


if __name__ == "__main__":
    main()
