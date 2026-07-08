"""Test-time evaluation harness: compare dispatch methods on fixed test sets.

Goals
-----
This module runs one or more dispatch methods (model-based baselines and/or
trained RL checkpoints) on one or more *test sets*, and prints, per test set:

* the standard KPI table (reward / service / complete / wait / ride / detour /
  empty% / util), plus
* the number of orders in that test set (``orders``), and
* each method's runtime in seconds (``time_s``) -- measured strictly from the
  first ``env.reset()`` to the last ``env.step()`` (excluding environment
  construction / graph loading), as ``run_episode`` reports in
  ``summary["wall_time_seconds"]``.

Fair-comparison guarantee (fully fixed randomness)
--------------------------------------------------
Every method is evaluated on the SAME scenario by running each through
``run_episode``, which builds a fresh env and calls ``env.reset(seed=cfg.seed)``.
``env.reset(seed)`` reseeds BOTH the env RNG (driver initial positions) AND the
order generator's RNG (party sizes, OD spatial perturbation, and -- for the
multi-window NYC source -- the window selection). Because a fresh env + fixed
seed is used for each method, all these random components are byte-for-byte
identical across methods, so any KPI difference reflects only the dispatch
policy, never the scenario. A "test set" here is simply a ``(seed, split)``
pair: change the seed and/or the NYC split to evaluate a different fixed
scenario, and every method still sees that exact same scenario.

Usage
-----
Programmatic::

    from benchmark.evaluate import evaluate, TestSet
    from benchmark.config import BenchmarkConfig

    base = BenchmarkConfig(nyc_split="test")
    test_sets = [TestSet(name="test-s0", seed=0), TestSet(name="test-s1", seed=1)]
    methods = {
        "nearest": lambda cfg: NearestDistanceDispatch.from_config(cfg),
        "iddqn":   load_rl_dispatch("iddqn/runs/<run>", "checkpoints/iddqn_epXXX.pt"),
    }
    evaluate(methods, test_sets, base_cfg=base)

CLI::

    python -m benchmark.evaluate --seeds 0 1 2 --split test \\
        --rl iddqn iddqn/runs/<run> checkpoints/iddqn_ep500.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from benchmark.config import BenchmarkConfig
from benchmark.runner import run_episode
from benchmark.baselines import (
    NearestDistanceDispatch,
    HungarianDispatch,
    RandomRadiusDispatch,
    GaleShapleyDispatch,
)


# A dispatch factory takes a (per-test-set) BenchmarkConfig and returns an
# object exposing ``act(observations)`` (and optionally
# ``last_assignment_distances``). Using a factory (not a prebuilt instance) lets
# each test set build the method against its own config, e.g. so a baseline's
# gate metric / network matches that test set exactly.
MethodFactory = Callable[[BenchmarkConfig], object]


# Registry of built-in model-based baselines, keyed by CLI/lookup name.
BASELINE_FACTORIES: Dict[str, MethodFactory] = {
    "random_radius": lambda cfg: RandomRadiusDispatch.from_config(cfg, k_nearest=20),
    "gale_shapley": lambda cfg: GaleShapleyDispatch.from_config(cfg, k_nearest=20),
    "nearest_distance": lambda cfg: NearestDistanceDispatch.from_config(cfg, k_nearest=20),
    "hungarian": lambda cfg: HungarianDispatch.from_config(cfg, k_nearest=20),
}


# KPI columns shown in the summary table, as (summary_key, header_label, width).
_KPI_COLS = [
    ("total_reward", "reward", 12),
    ("service_rate", "service", 10),
    ("complete_rate", "complete", 10),
    ("avg_wait_time", "wait", 10),
    ("avg_ride_time", "ride", 10),
    ("avg_detour_time", "detour", 10),
    ("empty_distance_ratio", "empty%", 10),
    ("avg_driver_utilisation", "util", 10),
    # Requested extras: order count of this test set + model runtime (s).
    ("total_orders", "orders", 10),
    ("wall_time_seconds", "time_s", 10),
]


@dataclass
class TestSet:
    """One fixed test scenario.

    A test set fully determines the scenario via ``seed`` (which fixes every
    random component -- driver positions, party sizes, OD perturbation, NYC
    window selection) and any config overrides (e.g. ``split="test"``). Every
    method is run on this exact scenario, so comparisons are apples-to-apples.

    Parameters
    ----------
    name:
        Human-readable label for this test set (used as the table title).
    seed:
        Master seed. ``env.reset(seed=...)`` threads it into the env RNG and the
        order generator RNG, fixing ALL randomness.
    split:
        NYC split to draw the window(s) from (``"train"``/``"val"``/``"test"``).
        ``None`` leaves the base config's split unchanged. Ignored for the
        abstract (non-NYC) scenarios.
    overrides:
        Extra ``BenchmarkConfig`` field overrides applied on top of the base
        config for this test set (e.g. ``{"num_orders": 5000}``).
    """

    name: str
    seed: int = 0
    split: Optional[str] = None
    overrides: Dict[str, object] = field(default_factory=dict)

    def make_config(self, base_cfg: BenchmarkConfig) -> BenchmarkConfig:
        """Build this test set's concrete config from a base config."""
        changes = dict(self.overrides)
        changes["seed"] = self.seed
        if self.split is not None:
            changes["nyc_split"] = self.split
        return dataclasses.replace(base_cfg, **changes)


# Registry of trainable RL methods: name -> (module, TrainConfig class,
# Trainer class). Every trainer shares the same shape -- a config dataclass with
# a ``benchmark`` field, a ``_GreedyActorDispatch(actor, network)`` adapter, and
# an agent whose ``online`` net loads from the checkpoint's ``"online"`` key --
# so one loader handles them all; only the import targets differ.
RL_METHODS: Dict[str, tuple] = {
    "iddqn": ("iddqn.train_iddqn", "TrainConfig", "IDDQNTrainer"),
    "bmgq": ("bmgq.train_bmgq", "BMGTrainConfig", "BMGQTrainer"),
    "mfddqn": ("mfddqn.train_mfddqn", "MFTrainConfig", "MFDDQNTrainer"),
}


def load_rl_dispatch(
    run_dir: str,
    checkpoint: str,
    device: Optional[str] = None,
    method: str = "iddqn",
) -> MethodFactory:
    """Return a method factory that loads a trained RL checkpoint.

    The returned factory rebuilds the trainer's actor (network + feature
    encoder) from the run's ``config.json`` and the given checkpoint, then wraps
    it as a greedy dispatcher exposing ``act(observations)`` -- exactly the
    object the trainer uses for evaluation. Rebuilding via the trainer avoids
    duplicating the (arch-dependent) network-construction logic.

    Supports every method in :data:`RL_METHODS` (``iddqn``, ``bmgq``,
    ``mfddqn``), which all share the same trainer shape; ``method`` selects
    which trainer/config classes to import.

    Parameters
    ----------
    run_dir:
        Training run directory containing ``config.json`` (and typically a
        ``checkpoints/`` subdir).
    checkpoint:
        Path to the ``.pt`` checkpoint. If not absolute and not found as given,
        it is resolved relative to ``run_dir``.
    device:
        Torch device override (e.g. ``"cpu"``). ``None`` keeps the run's device
        if available, else falls back to CPU.
    method:
        Which RL method the checkpoint belongs to (key of :data:`RL_METHODS`).
    """
    # Imported lazily so the baselines-only path does not require torch.
    import importlib
    import torch

    if method not in RL_METHODS:
        raise ValueError(
            f"unknown RL method {method!r}; choose from {list(RL_METHODS)}"
        )
    mod_name, cfg_cls_name, trainer_cls_name = RL_METHODS[method]
    mod = importlib.import_module(mod_name)
    TrainConfig = getattr(mod, cfg_cls_name)
    Trainer = getattr(mod, trainer_cls_name)
    GreedyActorDispatch = getattr(mod, "_GreedyActorDispatch")

    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in run dir: {run_dir!r}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Resolve the checkpoint path (allow a path relative to the run dir).
    ckpt_path = checkpoint
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        cand = os.path.join(run_dir, checkpoint)
        if os.path.exists(cand):
            ckpt_path = cand
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint!r}")

    # Rebuild the TrainConfig from the stored JSON. The benchmark sub-config is
    # rebuilt as a BenchmarkConfig; unknown keys are ignored defensively so a
    # newer/older config file still loads.
    bm_fields = {f.name for f in dataclasses.fields(BenchmarkConfig)}
    bm_kwargs = {k: v for k, v in (raw.get("benchmark") or {}).items() if k in bm_fields}
    # Tuple-typed fields deserialize from JSON as lists; coerce back.
    for k in ("area", "relocation_grid"):
        if k in bm_kwargs and isinstance(bm_kwargs[k], list):
            bm_kwargs[k] = tuple(bm_kwargs[k])
    base_bm = BenchmarkConfig(**bm_kwargs)

    tc_field_map = {f.name: f for f in dataclasses.fields(TrainConfig)}
    tc_kwargs = {
        k: v
        for k, v in raw.items()
        if k in tc_field_map and k != "benchmark"
    }
    # Tuple-typed fields (e.g. hidden, cv_resolutions) deserialize from JSON as
    # lists; coerce any list back to a tuple when the field's default is a
    # tuple, so this works for every method's config without hardcoding names.
    for k, v in list(tc_kwargs.items()):
        default = tc_field_map[k].default
        if isinstance(v, list) and isinstance(default, tuple):
            tc_kwargs[k] = tuple(v)
    tc_kwargs["benchmark"] = base_bm
    if device is not None:
        tc_kwargs["device"] = device
    train_cfg = TrainConfig(**tc_kwargs)

    def factory(cfg: BenchmarkConfig) -> object:
        # Rebuild the trainer with THIS test set's benchmark config so the actor
        # (feature encoder area / gate metric) matches the scenario, then load
        # the trained weights and return the greedy dispatcher.
        this_cfg = dataclasses.replace(train_cfg, benchmark=cfg, verbose=False)
        trainer = Trainer(this_cfg)
        # weights_only=True is safe here (the checkpoint stores only tensors /
        # plain dicts) and silences the torch.load pickle FutureWarning; fall
        # back to a full load on older torch versions lacking the kwarg.
        try:
            ckpt = torch.load(
                ckpt_path, map_location=trainer.cfg.device, weights_only=True
            )
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=trainer.cfg.device)
        trainer.agent.online.load_state_dict(ckpt["online"])
        trainer.agent.online.eval()
        dispatch = GreedyActorDispatch(trainer.actor, trainer.network)
        # mfddqn's adapter carries a mean field across steps; reset it so the
        # first step starts clean. Harmless no-op for methods without it.
        if hasattr(dispatch, "reset_episode"):
            dispatch.reset_episode()
        return dispatch

    return factory


def evaluate(
    methods: Dict[str, MethodFactory],
    test_sets: List[TestSet],
    base_cfg: Optional[BenchmarkConfig] = None,
    out_dir: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Dict]]:
    """Run every method on every test set and print a per-test-set KPI table.

    Parameters
    ----------
    methods:
        ``{display_name: factory(cfg) -> dispatcher}``. A factory builds the
        method against the test set's config (so gate metric / network / feature
        encoder match that exact scenario).
    test_sets:
        The fixed test scenarios to evaluate on.
    base_cfg:
        Base :class:`BenchmarkConfig`; each test set overrides ``seed`` (and
        optionally ``split`` / extra fields) on top of it. Defaults to the
        standard benchmark scenario.
    out_dir:
        If given, per-method detailed records are written under
        ``out_dir/<test_set_name>/<method>/`` (via the recorder).
    verbose:
        Forwarded to ``run_episode`` (per-step progress). The KPI table is
        always printed regardless.

    Returns
    -------
    results:
        ``{test_set_name: {method_name: summary_dict}}`` for programmatic use.
    """
    base_cfg = base_cfg or BenchmarkConfig()
    results: Dict[str, Dict[str, Dict]] = {}

    for ts in test_sets:
        cfg = ts.make_config(base_cfg)
        ts_out = os.path.join(out_dir, ts.name) if out_dir else None
        summaries: Dict[str, Dict] = {}

        for name, factory in methods.items():
            # Each method builds fresh against this test set's config; run_episode
            # constructs a fresh env and reset(seed=cfg.seed), so the scenario
            # (all randomness) is identical across methods.
            dispatcher = factory(cfg)
            summary, _rec = run_episode(
                dispatcher,
                algorithm_name=name,
                cfg=cfg,
                out_dir=ts_out,
                verbose=verbose,
            )
            summaries[name] = summary

        results[ts.name] = summaries
        _print_table(ts, summaries)

    return results


def _print_table(ts: TestSet, summaries: Dict[str, Dict]) -> None:
    """Print the KPI table for one test set (incl. order count + runtime)."""
    # Order count is a property of the test set (identical across methods); show
    # it in the title as well as the per-row column for convenience.
    n_orders = 0
    for s in summaries.values():
        n_orders = int(s.get("total_orders", 0))
        break

    title = (
        f"=== test set: {ts.name}  (seed={ts.seed}"
        + (f", split={ts.split}" if ts.split is not None else "")
        + f", orders={n_orders}) ==="
    )
    print("\n" + title)

    header = f"{'algo':16s}" + "".join(
        f"{lbl:>{w}s}" for _k, lbl, w in _KPI_COLS
    )
    print(header)
    print("-" * len(header))

    for name, s in summaries.items():
        row = f"{name:16s}"
        for key, _lbl, w in _KPI_COLS:
            v = s.get(key, 0.0)
            if key == "total_orders":
                row += f"{int(v):>{w}d}"
            elif key == "wall_time_seconds":
                row += f"{float(v):>{w}.2f}"
            else:
                row += f"{float(v):>{w}.4f}"
        print(row)


def _build_cli_methods(args) -> Dict[str, MethodFactory]:
    """Assemble the method dict from CLI args (baselines + RL checkpoints)."""
    methods: Dict[str, MethodFactory] = {}
    for name in args.baselines:
        if name not in BASELINE_FACTORIES:
            raise SystemExit(
                f"unknown baseline {name!r}; choose from {list(BASELINE_FACTORIES)}"
            )
        methods[name] = BASELINE_FACTORIES[name]
    # --rl NAME RUN_DIR CHECKPOINT (repeatable)
    for spec in args.rl or []:
        if len(spec) != 3:
            raise SystemExit(
                "--rl expects 3 values: NAME RUN_DIR CHECKPOINT "
                f"(got {spec!r})"
            )
        name, run_dir, ckpt = spec
        methods[name] = load_rl_dispatch(run_dir, ckpt, device=args.device)
    if not methods:
        raise SystemExit("no methods selected: pass --baselines and/or --rl.")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="one test set per seed (all randomness fixed by the seed).",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="NYC split for every test set (train/val/test); None keeps base.",
    )
    parser.add_argument(
        "--baselines", type=str, nargs="*", default=list(BASELINE_FACTORIES),
        help="which model-based baselines to run.",
    )
    parser.add_argument(
        "--rl", action="append", nargs=3, metavar=("NAME", "RUN_DIR", "CHECKPOINT"),
        help="a trained RL method to evaluate; repeatable.",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device.")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="if set, write per-method detailed records under this dir.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_cfg = BenchmarkConfig()
    if args.split is not None:
        base_cfg = dataclasses.replace(base_cfg, nyc_split=args.split)

    test_sets = [
        TestSet(name=f"seed{seed}", seed=seed, split=args.split)
        for seed in args.seeds
    ]
    methods = _build_cli_methods(args)
    evaluate(methods, test_sets, base_cfg=base_cfg, out_dir=args.out_dir,
             verbose=args.verbose)


if __name__ == "__main__":
    main()
