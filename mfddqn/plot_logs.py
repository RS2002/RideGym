"""Plot training / evaluation metric curves from an IDDQN run.

Reads ``train_log.csv`` and ``eval_log.csv`` produced by
:mod:`iddqn.train_iddqn` and draws every numeric metric as a curve against the
episode axis. Two figures (one per log) are written as PNGs next to the logs
and, unless ``--no-show`` is given, displayed interactively.

Usage::

    conda run -n zzj python -m iddqn.plot_logs iddqn/runs/<run_name>
    conda run -n zzj python -m iddqn.plot_logs iddqn/runs/<run_name> --no-show

If no run directory is given, the most recently modified sub-directory of
``iddqn/runs`` is used.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional

import pandas as pd
import matplotlib

# Use a non-interactive backend automatically when no display is available so
# the script still saves figures on headless machines.
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)


# Columns that index the rows rather than being plottable metrics.
_NON_METRIC_COLS = {"episode", "global_step"}


def _latest_run(runs_dir: str) -> str:
    """Return the most recently modified sub-directory of ``runs_dir``."""
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")
    subdirs = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"no run sub-directories under: {runs_dir}")
    return max(subdirs, key=os.path.getmtime)


def _numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    """Numeric columns worth plotting (excludes index/identifier columns)."""
    cols = []
    for c in df.columns:
        if c in _NON_METRIC_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _plot_log(csv_path: str, title: str, save_path: str) -> Optional[str]:
    """Plot every numeric metric in ``csv_path`` as a small-multiples grid.

    Returns the saved figure path, or ``None`` if the CSV is missing/empty.
    """
    if not os.path.isfile(csv_path):
        print(f"[skip] not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[skip] empty log: {csv_path}")
        return None

    # X axis: prefer the episode column, else fall back to the row index.
    x_col = "episode" if "episode" in df.columns else None
    x = df[x_col] if x_col else df.index
    x_label = x_col or "row"

    metrics = _numeric_metric_columns(df)
    if not metrics:
        print(f"[skip] no numeric metrics in: {csv_path}")
        return None

    # Lay the metrics out on a roughly square grid of sub-plots.
    n = len(metrics)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False
    )

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        # Drop NaNs (e.g. mean_loss before warmup) so the line stays clean.
        sub = df[[x_col, metric]].dropna() if x_col else df[[metric]].dropna()
        xs = sub[x_col] if x_col else sub.index
        ax.plot(xs, sub[metric], marker=".", markersize=3, linewidth=1.2)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel(x_label, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any unused sub-plot cells in the final row.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=120)
    print(f"[saved] {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot IDDQN train/eval metric curves from a run directory."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="run directory containing train_log.csv / eval_log.csv "
        "(default: latest under iddqn/runs).",
    )
    parser.add_argument(
        "--runs-dir",
        default=os.path.join("iddqn", "runs"),
        help="root runs directory used to resolve the latest run "
        "(default: iddqn/runs).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save figures without opening an interactive window.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run(args.runs_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    print(f"run dir: {run_dir}")

    saved = []
    p1 = _plot_log(
        os.path.join(run_dir, "train_log.csv"),
        title="Training curves",
        save_path=os.path.join(run_dir, "train_curves.png"),
    )
    p2 = _plot_log(
        os.path.join(run_dir, "eval_log.csv"),
        title="Evaluation curves",
        save_path=os.path.join(run_dir, "eval_curves.png"),
    )
    saved = [p for p in (p1, p2) if p]

    if saved and not args.no_show:
        # Only call show() if an interactive backend is active.
        if matplotlib.get_backend().lower() != "agg":
            plt.show()


if __name__ == "__main__":
    main()