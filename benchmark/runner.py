"""Episode runner: ties a dispatch algorithm to the benchmark env + recorder.

Runs one full episode of the standard benchmark scenario with a given dispatch
algorithm, recording detailed per-step / per-order / per-driver metrics, and
optionally persisting them for cross-algorithm comparison.
"""

from __future__ import annotations

import time
from typing import Optional

from benchmark.config import BenchmarkConfig, make_benchmark_env
from benchmark.recorder import EpisodeRecorder


def run_episode(
    algorithm,
    algorithm_name: str,
    cfg: Optional[BenchmarkConfig] = None,
    out_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Run one benchmark episode and return ``(summary, recorder)``.

    Parameters
    ----------
    algorithm:
        A dispatch object exposing ``act(observations) -> {driver_id: action}``
        and, optionally, ``last_assignment_distances`` (logged each step).
    algorithm_name:
        Name used for output directory and result labelling.
    cfg:
        Benchmark configuration. Defaults to the standard scenario.
    out_dir:
        If given, results are written under ``out_dir/<algorithm_name>/``.
    verbose:
        Print progress and the final summary.
    """
    cfg = cfg or BenchmarkConfig()
    env = make_benchmark_env(cfg)
    recorder = EpisodeRecorder(algorithm=algorithm_name, config=cfg.to_dict())

    obs, _ = env.reset(seed=cfg.seed)
    t0 = time.time()
    step = 0
    while True:
        actions = algorithm.act(obs)
        obs, rewards, dones, info = env.step(actions)
        assign_log = getattr(algorithm, "last_assignment_distances", None)
        recorder.record_step(
            env, info, assign_log=assign_log, rewards=rewards
        )
        step += 1
        if verbose and step % 10 == 0:
            served = recorder.step_rows[-1]["completed"]
            pending = recorder.step_rows[-1]["pending"]
            print(
                f"  [step {step:3d}/{int(cfg.horizon / cfg.dt)}] "
                f"t={info['time']:.0f} pending={pending} "
                f"completed_this_step={served}"
            )
        if dones["__all__"]:
            break

    summary = recorder.finalize(env)
    elapsed = time.time() - t0
    summary["wall_time_seconds"] = elapsed

    if verbose:
        print(f"\n=== {algorithm_name} summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.4f}")
            else:
                print(f"  {k:24s}: {v}")

    if out_dir:
        paths = recorder.save(out_dir)
        if verbose:
            print(f"\nResults written under: {out_dir}/{algorithm_name}/")
            for name, p in paths.items():
                print(f"  {name:8s}: {p}")

    return summary, recorder