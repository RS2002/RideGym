"""Fixed-window test runner: compare model-based + RL dispatch on NYC windows.

This is a thin, opinionated wrapper around :mod:`benchmark.evaluate` for the
common "final test" workflow: pick a seed, a set of comparison methods
(model-based baselines and one trained RL checkpoint), and a small set of
*specific* NYC test windows, then print one KPI table per window.

Why pin specific windows?
-------------------------
In ``nyc_split="test"`` mode the ``MultiWindowNYCOrderGenerator`` traverses the
split's windows cyclically (window_0000, window_0001, ...), so you cannot ask
for "just window_0002 and window_0010". To evaluate on an EXACT window we drop
to single-file replay mode instead: set ``nyc_splits_dir=None`` and point
``nyc_order_path`` at that window's parquet file. ``make_benchmark_env`` then
builds a plain :class:`NYCOrderGenerator`, giving a deterministic replay of that
one window -- identical order stream every run for a given seed.

Fair-comparison guarantee
--------------------------
Every method runs through :func:`benchmark.runner.run_episode`, which builds a
fresh env and calls ``env.reset(seed=cfg.seed)``. That seed fixes ALL
randomness (driver initial positions, party sizes, OD perturbation), so all
methods see a byte-identical scenario on each window. Any KPI difference
reflects only the dispatch policy.

Usage
-----
Programmatic::

    from benchmark.run_test import run_test
    run_test(
        seed=42,
        baselines=["nearest_distance", "hungarian", "gale_shapley"],
        rl_ckpt="iddqn/runs/<run>/checkpoints/iddqn_ep500.pt",
        splits_dir="data/nyc/splits/test",
        windows=[2, 10],
    )

Multiple RL methods can be compared in one table by passing several
checkpoints (repeat ``--rl-ckpt`` on the CLI, or pass a list to ``rl=``);
each is auto-named from its checkpoint filename (collisions are de-duplicated),
or named explicitly via ``--rl NAME CKPT`` / ``rl=[(name, ckpt), ...]``.

CLI::

    python -m benchmark.run_test \\
        --seed 42 \\
        --rl-ckpt iddqn/runs/<run>/checkpoints/iddqn_ep500.pt \\
        --rl-ckpt bmgq/runs/<run>/checkpoints/bmgq_ep500.pt \\
        --rl mfddqn_v2 mfddqn/runs/<run>/checkpoints/mfddqn_ep500.pt \\
        --windows 2 10
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Union

from benchmark.config import BenchmarkConfig
from benchmark.evaluate import (
    TestSet,
    MethodFactory,
    BASELINE_FACTORIES,
    RL_METHODS,
    load_rl_dispatch,
    evaluate,
)

# Defaults for the standard "final test" run.
DEFAULT_SEED = 42
DEFAULT_WINDOWS = (2, 10)
DEFAULT_SPLITS_DIR = os.path.join("data", "nyc", "splits", "test")
# All model-based baselines are included by default so the RL method is always
# compared against the full model-based field.
DEFAULT_BASELINES = tuple(BASELINE_FACTORIES.keys())


def _window_path(splits_dir: str, window: int) -> str:
    """Return the parquet path for a window index, validating it exists.

    Windows are named ``window_XXXX.parquet`` (4-digit, zero-padded) inside the
    split directory. We check existence up front so a typo'd index fails with a
    clear message instead of deep inside env construction.
    """
    fname = f"window_{int(window):04d}.parquet"
    path = os.path.join(splits_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"test window not found: {path!r}. Expected a file named {fname!r} "
            f"in {splits_dir!r}."
        )
    return path


def _infer_rl_method(ckpt_path: str) -> Optional[str]:
    """Guess the RL method from a checkpoint path via its filename prefix.

    Trainers save checkpoints as ``<method>_ep<N>.pt`` (e.g. ``iddqn_ep500.pt``,
    ``bmgq_ep500.pt``, ``mfddqn_ep500.pt``), so the leading token before the
    first underscore identifies the method. Returns ``None`` if no known method
    name is found, leaving the caller to fall back to its default / explicit
    choice.
    """
    base = os.path.basename(ckpt_path).lower()
    for name in RL_METHODS:
        if base.startswith(name):
            return name
    return None


def _infer_run_dir(ckpt_path: str) -> str:
    """Locate the training run dir (holding ``config.json``) for a checkpoint.

    Checkpoints typically live at ``<run_dir>/checkpoints/<name>.pt`` while the
    run's ``config.json`` sits at ``<run_dir>/config.json``. We walk up from the
    checkpoint's directory until we find a ``config.json``, so the caller only
    has to supply the ``.pt`` path.
    """
    d = os.path.dirname(os.path.abspath(ckpt_path))
    # Walk up a bounded number of levels; config.json is normally 0-2 levels up.
    for _ in range(6):
        if os.path.exists(os.path.join(d, "config.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root
            break
        d = parent
    raise FileNotFoundError(
        f"could not locate config.json above checkpoint {ckpt_path!r}; pass "
        f"run_dir explicitly."
    )


# A single RL method to evaluate. Accepted spec shapes (normalised by
# :func:`_normalise_rl_spec`):
#   * ``"path/to/ckpt.pt"``                       -> name & method inferred
#   * ``("display_name", "path/to/ckpt.pt")``     -> explicit name
#   * ``("display_name", "ckpt.pt", "bmgq")``     -> explicit name + method
#   * ``{"ckpt": ..., "name": ..., "method": ..., "run_dir": ...}``
RLSpec = Union[str, Sequence, Dict[str, Optional[str]]]


def _normalise_rl_spec(spec: RLSpec) -> Dict[str, Optional[str]]:
    """Coerce any accepted RL spec shape into a canonical dict.

    Returns a dict with keys ``ckpt`` (required), ``name``, ``method`` and
    ``run_dir`` (the latter three optional / ``None`` when unspecified).
    """
    if isinstance(spec, str):
        return {"ckpt": spec, "name": None, "method": None, "run_dir": None}
    if isinstance(spec, dict):
        if "ckpt" not in spec:
            raise ValueError(f"RL spec dict missing required 'ckpt': {spec!r}")
        return {
            "ckpt": spec["ckpt"],
            "name": spec.get("name"),
            "method": spec.get("method"),
            "run_dir": spec.get("run_dir"),
        }
    # Sequence: (name, ckpt) or (name, ckpt, method).
    seq = list(spec)
    if len(seq) == 2:
        name, ckpt = seq
        return {"ckpt": ckpt, "name": name, "method": None, "run_dir": None}
    if len(seq) == 3:
        name, ckpt, method = seq
        return {"ckpt": ckpt, "name": name, "method": method, "run_dir": None}
    raise ValueError(
        f"RL spec sequence must be (name, ckpt) or (name, ckpt, method); "
        f"got {spec!r}"
    )


def _unique_name(name: str, taken: set) -> str:
    """Return ``name`` (or ``name_2``, ``name_3``, ...) not already in ``taken``.

    Two checkpoints of the same family infer the same display name; suffix the
    later ones so every method is a distinct key in the results table.
    """
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def _make_window_test_sets(
    splits_dir: str, windows: Sequence[int], seed: int
) -> List[TestSet]:
    """Build one :class:`TestSet` per window, pinned via single-file replay.

    Each test set overrides the config to single-file mode (``nyc_splits_dir``
    = None) pointed at that window's parquet, so the window is replayed exactly
    rather than drawn cyclically from the split pool.
    """
    test_sets: List[TestSet] = []
    for w in windows:
        path = _window_path(splits_dir, w)
        test_sets.append(
            TestSet(
                name=f"window_{int(w):04d}",
                seed=seed,
                # split is irrelevant in single-file mode; leave base untouched.
                split=None,
                overrides={
                    "network_kind": "nyc",
                    "nyc_splits_dir": None,      # -> single-file NYCOrderGenerator
                    "nyc_order_path": path,      # replay THIS window exactly
                },
            )
        )
    return test_sets


def run_test(
    seed: int = DEFAULT_SEED,
    baselines: Optional[Sequence[str]] = None,
    rl: Optional[Sequence[RLSpec]] = None,
    rl_ckpt: Optional[str] = None,
    rl_run_dir: Optional[str] = None,
    rl_method: Optional[str] = None,
    rl_name: Optional[str] = None,
    splits_dir: str = DEFAULT_SPLITS_DIR,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    device: Optional[str] = None,
    out_dir: Optional[str] = None,
    base_cfg: Optional[BenchmarkConfig] = None,
    verbose: bool = False,
):
    """Run model-based baselines and an RL checkpoint on fixed NYC windows.

    Parameters
    ----------
    seed:
        Master seed fixing all randomness for every window (default 42).
    baselines:
        Model-based baseline names to include (keys of
        :data:`benchmark.evaluate.BASELINE_FACTORIES`). ``None`` -> all of them.
    rl:
        Zero or more RL methods to evaluate together in the same table. Each
        entry is a spec accepted by :func:`_normalise_rl_spec`: a bare
        checkpoint path (name & method inferred), a ``(name, ckpt)`` /
        ``(name, ckpt, method)`` tuple, or a dict with keys ``ckpt``/``name``/
        ``method``/``run_dir``. Names that collide are de-duplicated with a
        numeric suffix.
    rl_ckpt:
        Convenience shortcut for a SINGLE RL checkpoint (equivalent to adding
        one entry to ``rl``); combined with ``rl_run_dir`` / ``rl_method`` /
        ``rl_name`` below. ``None`` and empty ``rl`` -> run baselines only.
    rl_run_dir:
        Training run dir (``config.json``) for ``rl_ckpt``. ``None`` -> inferred
        by walking up from the checkpoint.
    rl_method:
        RL family for ``rl_ckpt`` (``"iddqn"`` | ``"bmgq"`` | ``"mfddqn"``).
        ``None`` -> inferred from the checkpoint filename prefix, defaulting to
        ``"iddqn"`` if that fails.
    rl_name:
        Display name for ``rl_ckpt`` in the table. ``None`` -> the resolved
        method name.
    splits_dir:
        Directory holding the ``window_XXXX.parquet`` test windows.
    windows:
        Window indices to evaluate on (default (2, 10)).
    device:
        Torch device override for the RL method (e.g. ``"cpu"``).
    out_dir:
        If set, write per-method detailed records under this directory.
    base_cfg:
        Base :class:`BenchmarkConfig`; each window overrides the order source on
        top of it. Defaults to the standard benchmark scenario.
    verbose:
        Forward per-step progress to ``run_episode`` (the KPI table always
        prints regardless).

    Returns
    -------
    results:
        ``{window_name: {method_name: summary_dict}}``.
    """
    baseline_names = list(baselines) if baselines is not None else list(DEFAULT_BASELINES)

    # Assemble the method dict: every requested model-based baseline first, then
    # the RL method (if a checkpoint was supplied) so it appears last in tables.
    methods: dict[str, MethodFactory] = {}
    for name in baseline_names:
        if name not in BASELINE_FACTORIES:
            raise ValueError(
                f"unknown baseline {name!r}; choose from {list(BASELINE_FACTORIES)}"
            )
        methods[name] = BASELINE_FACTORIES[name]

    # Collect all RL specs: the list form first, then the single-checkpoint
    # convenience shortcut (if given) as one more spec.
    rl_specs: List[Dict[str, Optional[str]]] = [
        _normalise_rl_spec(s) for s in (rl or [])
    ]
    if rl_ckpt is not None:
        rl_specs.append(
            {
                "ckpt": rl_ckpt,
                "name": rl_name,
                "method": rl_method,
                "run_dir": rl_run_dir,
            }
        )

    for spec in rl_specs:
        ckpt = spec["ckpt"]
        method = spec["method"] or _infer_rl_method(ckpt) or "iddqn"
        if method not in RL_METHODS:
            raise ValueError(
                f"unknown rl method {method!r} for {ckpt!r}; "
                f"choose from {list(RL_METHODS)}"
            )
        run_dir = spec["run_dir"] or _infer_run_dir(ckpt)
        # Resolve a unique display name so multiple checkpoints of the same
        # family (e.g. two iddqn runs) don't overwrite each other in the table.
        name = _unique_name(spec["name"] or method, set(methods))
        methods[name] = load_rl_dispatch(
            run_dir, ckpt, device=device, method=method
        )

    if not methods:
        raise ValueError("no methods selected: pass baselines and/or rl.")

    test_sets = _make_window_test_sets(splits_dir, windows, seed)
    base_cfg = base_cfg or BenchmarkConfig()
    return evaluate(
        methods, test_sets, base_cfg=base_cfg, out_dir=out_dir, verbose=verbose
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"master seed fixing all randomness (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--baselines", type=str, nargs="*", default=list(DEFAULT_BASELINES),
        help="model-based baselines to run (default: all).",
    )
    parser.add_argument(
        "--rl-ckpt", type=str, action="append", default=None,
        metavar="CKPT",
        help="trained RL .pt checkpoint; repeat to compare several. Name and "
             "method are inferred from each filename.",
    )
    parser.add_argument(
        "--rl", action="append", nargs=2, default=None,
        metavar=("NAME", "CKPT"),
        help="an explicitly-named RL checkpoint (method inferred); repeatable.",
    )
    parser.add_argument(
        "--rl-run-dir", type=str, default=None,
        help="run dir with config.json for a SINGLE --rl-ckpt "
             "(default: inferred from the checkpoint path).",
    )
    parser.add_argument(
        "--rl-method", type=str, default=None, choices=list(RL_METHODS),
        help="RL family for a SINGLE --rl-ckpt (default: inferred from name).",
    )
    parser.add_argument(
        "--rl-name", type=str, default=None,
        help="display name for a SINGLE --rl-ckpt (default: the method name).",
    )
    parser.add_argument(
        "--splits-dir", type=str, default=DEFAULT_SPLITS_DIR,
        help=f"test windows directory (default {DEFAULT_SPLITS_DIR!r}).",
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS),
        help="window indices to test on (default: 2 10).",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device.")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="if set, write per-method detailed records under this dir.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ckpts = list(args.rl_ckpt or [])
    named = list(args.rl or [])

    # --rl-run-dir / --rl-method / --rl-name customise a SINGLE checkpoint,
    # where disambiguation is unnecessary. They are only valid when exactly one
    # --rl-ckpt (and no --rl) was given; reject otherwise so they can't silently
    # apply to the wrong method.
    single = len(ckpts) == 1 and not named
    if (args.rl_run_dir or args.rl_method or args.rl_name) and not single:
        raise SystemExit(
            "--rl-run-dir/--rl-method/--rl-name apply to a single --rl-ckpt "
            "only; for multiple checkpoints use --rl NAME CKPT per method."
        )

    if single:
        # Route through the single-checkpoint shortcut so the run-dir / method /
        # name overrides actually take effect (they are ignored on ``rl=`` list
        # entries).
        run_test(
            seed=args.seed,
            baselines=args.baselines,
            rl_ckpt=ckpts[0],
            rl_run_dir=args.rl_run_dir,
            rl_method=args.rl_method,
            rl_name=args.rl_name,
            splits_dir=args.splits_dir,
            windows=args.windows,
            device=args.device,
            out_dir=args.out_dir,
            verbose=args.verbose,
        )
        return

    # Multiple (or zero) RL methods: build the spec list from both CLI forms.
    #   --rl-ckpt CKPT   (repeatable; name/method inferred from filename)
    #   --rl NAME CKPT   (repeatable; explicit name, method inferred)
    rl_specs: List[RLSpec] = list(ckpts)
    rl_specs.extend((name, ckpt) for name, ckpt in named)

    run_test(
        seed=args.seed,
        baselines=args.baselines,
        rl=rl_specs,
        splits_dir=args.splits_dir,
        windows=args.windows,
        device=args.device,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()