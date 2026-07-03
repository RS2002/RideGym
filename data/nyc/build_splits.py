"""Batch-generate multiple time-window order files for train/val/test splits.

The single-window :mod:`data.nyc.preprocess_orders` produces ONE order file for
ONE time window, which makes every training episode replay identical demand.
This script slices the raw FHVHV parquet into MANY fixed-length windows grouped
into disjoint train / val / test pools, so a multi-window order generator can:

* sample a RANDOM training window each episode (demand diversity / regularisation);
* hold out separate windows for validation and test (no temporal leakage).

Each window becomes one order parquet under ``data/nyc/splits/<split>/`` and a
single ``manifest.json`` records, per split, the list of (file, start, end)
windows so the generator can load them without rescanning the raw data.

Design
------
The split boundaries are by DAY ranges (train days / val days / test days are
disjoint), and within each chosen day we cut one or more fixed-length windows
(e.g. every morning 07:00-10:00 in 1-hour windows). This keeps the demand
distribution comparable across splits (same hour-of-day) while guaranteeing the
DAYS never overlap -- the cleanest hold-out for a benchmark.

Run::

    python -m data.nyc.build_splits                 # use the defaults below
    python -m data.nyc.build_splits --window-min 60 # 60-min windows

Adjust ``TRAIN_DAYS`` / ``VAL_DAYS`` / ``TEST_DAYS`` and ``DAILY_WINDOWS`` to
your dataset month and desired coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import pandas as pd

from data.nyc.preprocess_orders import preprocess_orders, DEFAULT_PARQUET
from data.nyc.build_nyc_network import REGION_A_BBOX

_HERE = os.path.dirname(__file__)
DEFAULT_SPLITS_DIR = os.path.join(_HERE, "splits")

# ---------------------------------------------------------------------------
# Split definition (EDIT THESE to match your data month / desired coverage).
#
# Disjoint day ranges guarantee no temporal leakage across splits. The dataset
# bundled here is 2026-04 (April), so all dates are 2026-04-DD.
# ---------------------------------------------------------------------------
TRAIN_DAYS: List[str] = [f"2026-04-{d:02d}" for d in range(6, 9)]   # 1..20
VAL_DAYS:   List[str] = [f"2026-04-{d:02d}" for d in range(9, 10)]  # 21..25
TEST_DAYS:  List[str] = [f"2026-04-{d:02d}" for d in range(10, 11)]  # 26..30

# Within each day, cut fixed-length windows starting at these "HH:MM" times.
# Default: the morning peak split into hourly windows. Add more (e.g. "17:00",
# "18:00") to cover the evening peak as well.
DAILY_WINDOW_STARTS: List[str] = ["08:00", "09:00"]


def _windows_for_days(
    days: List[str], starts: List[str], window_min: int
) -> List[Tuple[str, str]]:
    """Expand (days x daily-start-times) into concrete (start, end) timestamps."""
    out: List[Tuple[str, str]] = []
    for day in days:
        for hhmm in starts:
            start = pd.Timestamp(f"{day} {hhmm}")
            end = start + pd.Timedelta(minutes=window_min)
            out.append((str(start), str(end)))
    return out


def _build_split(
    split: str,
    windows: List[Tuple[str, str]],
    splits_dir: str,
    parquet_path: str,
    bbox: tuple,
    sample_rate: float,
    seed: int,
) -> List[dict]:
    """Generate one order file per window for a split; return its manifest list."""
    split_dir = os.path.join(splits_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    entries: List[dict] = []
    for i, (start, end) in enumerate(windows):
        out_path = os.path.join(split_dir, f"window_{i:04d}.parquet")
        try:
            preprocess_orders(
                parquet_path=parquet_path,
                out_path=out_path,
                start=start,
                end=end,
                bbox=bbox,
                sample_rate=sample_rate,
                seed=seed,
            )
        except RuntimeError as exc:
            # A window with no trips (e.g. a date outside the file) is skipped
            # rather than aborting the whole batch.
            print(f"  [skip] {split} {start}..{end}: {exc}")
            continue
        entries.append(
            {
                "file": os.path.relpath(out_path, splits_dir),
                "start": start,
                "end": end,
                "horizon_min": (
                    (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
                    / 60.0
                ),
            }
        )
    return entries


def build_splits(
    splits_dir: str = DEFAULT_SPLITS_DIR,
    parquet_path: str = DEFAULT_PARQUET,
    bbox: tuple = REGION_A_BBOX,
    window_min: int = 60,
    sample_rate: float = 1.0,
    seed: int = 0,
) -> str:
    """Generate all train/val/test window order files + a manifest. Returns the
    manifest path."""
    os.makedirs(splits_dir, exist_ok=True)
    manifest = {
        "window_min": window_min,
        "sample_rate": sample_rate,
        "bbox": list(bbox),
        "splits": {},
    }
    for split, days in (
        ("train", TRAIN_DAYS),
        ("val", VAL_DAYS),
        ("test", TEST_DAYS),
    ):
        windows = _windows_for_days(days, DAILY_WINDOW_STARTS, window_min)
        print(f"\n=== building split '{split}': {len(windows)} windows ===")
        entries = _build_split(
            split, windows, splits_dir, parquet_path, bbox, sample_rate, seed
        )
        manifest["splits"][split] = entries
        print(f"  -> {len(entries)} non-empty windows kept for '{split}'")

    manifest_path = os.path.join(splits_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nsaved manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch-generate train/val/test window order files."
    )
    p.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--window-min", type=int, default=60,
                   help="length of each window in minutes (default 60).")
    p.add_argument("--sample-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build_splits(
        splits_dir=args.splits_dir,
        parquet_path=args.parquet,
        window_min=args.window_min,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()