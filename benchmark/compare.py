"""Run several dispatch baselines on the standard benchmark and compare them.

Usage:

    python -m benchmark.compare

Runs each registered baseline on the *same* scenario (same seed, same orders /
drivers), persists each one's detailed records under ``results/<name>/``, and
prints a side-by-side summary table of key KPIs.
"""

from __future__ import annotations

from benchmark.config import BenchmarkConfig
from benchmark.baselines import (
    NearestDistanceDispatch,
    HungarianDispatch,
    RandomRadiusDispatch,
    GaleShapleyDispatch,
)
from benchmark.runner import run_episode

# Registry of (display name, factory taking a BenchmarkConfig) baselines.
BASELINES = {
    "random_radius": lambda cfg: RandomRadiusDispatch.from_config(cfg, k_nearest=20),
    "gale_shapley": lambda cfg: GaleShapleyDispatch.from_config(cfg, k_nearest=20),
    "nearest_distance": lambda cfg: NearestDistanceDispatch.from_config(cfg, k_nearest=20),
    "hungarian": lambda cfg: HungarianDispatch.from_config(cfg, k_nearest=20),
}

COMPARE_KEYS = [
    "completed",
    "confirmed",
    # service_rate = confirmed/total (dispatch coverage);
    # complete_rate = completed/total (delivered within horizon).
    "service_rate",
    "complete_rate",
    "avg_wait_time",
    "avg_ride_time",
    "avg_detour_time",
    "total_distance_km",
    "empty_distance_ratio",
    "avg_driver_utilisation",
    "wall_time_seconds",
]


def main(cfg: BenchmarkConfig = None, out_dir: str = "results") -> None:
    cfg = cfg or BenchmarkConfig()
    summaries = {}
    for name, factory in BASELINES.items():
        print(f"running {name} ...")
        summary, _ = run_episode(
            algorithm=factory(cfg),
            algorithm_name=name,
            cfg=cfg,
            out_dir=out_dir,
            verbose=False,
        )
        summaries[name] = summary

    names = list(summaries.keys())
    header = f"{'metric':26s}" + "".join(f"{n:>18s}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for key in COMPARE_KEYS:
        row = f"{key:26s}"
        for n in names:
            row += f"{summaries[n][key]:18.4f}"
        print(row)


if __name__ == "__main__":
    main()
