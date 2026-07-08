"""Visualization level 3: aggregate demand / supply analysis plots.

Unlike the spatial snapshot (level 1) and replay animation (level 2), these
functions summarise the *statistics* of an episode over space and time:

* A. demand heatmap        -- spatial density of order origins
* B. supply-demand gap     -- per-cell (pending orders - idle vehicles) over time
* C. service-rate heatmap  -- per-cell fraction of orders completed
* D. time series           -- pending / onboard / completed / cancelled vs time
* E. wait-time histogram   -- pickup-wait distribution of completed orders

A, C, E read the finished ``env`` directly (all orders carry their final status
and timestamps). B, D consume an :class:`AnalysisRecorder` populated per step.
All plots use a plain grid (2-D histogram) over the service area, so they work
for every scenario (NYC / abstract) without needing region polygons; an optional
region-based aggregation is provided for B.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ride_gym.enums import DriverStatus, OrderStatus


# --------------------------------------------------------------------------- #
#  Per-step recorder for the time-dependent plots (B, D).
# --------------------------------------------------------------------------- #
class AnalysisRecorder:
    """Records per-step aggregate counts for the time-series / gap plots.

    Call :meth:`snapshot` once per step. It stores, per step: the clock, the
    number of pending / onboard / (cumulative) completed / cancelled orders, and
    -- for the supply-demand gap -- the per-cell pending-order origins and idle-
    vehicle locations. Lightweight: only counts and coordinate lists, no drawing.
    """

    def __init__(self):
        self.area = None
        self.horizon = None
        self.times: List[float] = []
        self.pending: List[int] = []
        self.onboard: List[int] = []
        self.completed: List[int] = []
        self.cancelled: List[int] = []
        # Per step: coordinate arrays for spatial supply/demand gap (B).
        self.pending_xy: List[np.ndarray] = []
        self.idle_xy: List[np.ndarray] = []

    def reset(self) -> None:
        self.__init__()

    def snapshot(self, env) -> None:
        if self.area is None:
            self.area = env.area
            self.horizon = env.horizon

        self.times.append(env.time)

        onboard = sum(
            o.status == OrderStatus.ONBOARD for o in env.orders.values()
        )
        completed = sum(
            o.status == OrderStatus.COMPLETED for o in env.orders.values()
        )
        cancelled = sum(
            o.status == OrderStatus.CANCELLED for o in env.orders.values()
        )
        self.pending.append(len(env._pending_ids))
        self.onboard.append(onboard)
        self.completed.append(completed)
        self.cancelled.append(cancelled)

        pend_xy = np.array(
            [env.orders[oid].origin for oid in env._pending_ids], dtype=float
        ).reshape(-1, 2)
        idle_xy = np.array(
            [
                d.location
                for d in env.drivers.values()
                if d.status == DriverStatus.IDLE
            ],
            dtype=float,
        ).reshape(-1, 2)
        self.pending_xy.append(pend_xy)
        self.idle_xy.append(idle_xy)


# --------------------------------------------------------------------------- #
#  Small helpers.
# --------------------------------------------------------------------------- #
def _new_ax(ax, figsize, dpi):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        return fig, ax, True
    return ax.figure, ax, False


def _finish(fig, ax, save_path, created):
    ax.set_aspect("equal", adjustable="box")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _hist2d_counts(xy: np.ndarray, area, bins: int):
    """2-D histogram counts over the service area; returns (H, xedges, yedges)."""
    xmin, ymin, xmax, ymax = area
    if xy.size == 0:
        H = np.zeros((bins, bins))
        xe = np.linspace(xmin, xmax, bins + 1)
        ye = np.linspace(ymin, ymax, bins + 1)
        return H, xe, ye
    H, xe, ye = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=bins, range=[[xmin, xmax], [ymin, ymax]]
    )
    return H.T, xe, ye  # transpose so H[row=y, col=x] matches imshow orientation


# --------------------------------------------------------------------------- #
#  A. Demand heatmap.
# --------------------------------------------------------------------------- #
def plot_demand_heatmap(
    env,
    bins: int = 40,
    ax=None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    dpi: int = 110,
    cmap: str = "magma",
):
    """Spatial density of all order origins over the whole episode (A)."""
    fig, ax, created = _new_ax(ax, figsize, dpi)
    origins = np.array(
        [o.origin for o in env.orders.values()], dtype=float
    ).reshape(-1, 2)
    H, xe, ye = _hist2d_counts(origins, env.area, bins)
    im = ax.imshow(
        H, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap=cmap, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="order count")
    ax.set_title(f"Demand heatmap (origins, {len(origins)} orders)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return _finish(fig, ax, save_path, created)


# --------------------------------------------------------------------------- #
#  B. Supply-demand gap (pending orders - idle vehicles), time-averaged.
# --------------------------------------------------------------------------- #
def plot_supply_demand_gap(
    rec: AnalysisRecorder,
    bins: int = 24,
    ax=None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    dpi: int = 110,
    cmap: str = "coolwarm",
):
    """Per-cell (pending demand - idle supply), averaged over all steps (B).

    Positive (warm) cells are under-served (more waiting orders than idle cars);
    negative (cool) cells have idle supply to spare. Aggregated over the episode
    from the recorder's per-step pending / idle coordinate arrays.
    """
    if not rec.times:
        raise ValueError("AnalysisRecorder is empty; call snapshot() per step.")
    fig, ax, created = _new_ax(ax, figsize, dpi)

    dem = np.zeros((bins, bins))
    sup = np.zeros((bins, bins))
    for pxy, ixy in zip(rec.pending_xy, rec.idle_xy):
        Hd, xe, ye = _hist2d_counts(pxy, rec.area, bins)
        Hs, _, _ = _hist2d_counts(ixy, rec.area, bins)
        dem += Hd
        sup += Hs
    n = max(len(rec.times), 1)
    gap = (dem - sup) / n  # time-averaged per-cell gap

    vmax = np.abs(gap).max() or 1.0
    im = ax.imshow(
        gap, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="avg (pending - idle) per cell")
    ax.set_title("Supply-demand gap (warm = under-served)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return _finish(fig, ax, save_path, created)


# --------------------------------------------------------------------------- #
#  C. Service-rate heatmap.
# --------------------------------------------------------------------------- #
def plot_service_rate_heatmap(
    env,
    bins: int = 24,
    ax=None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 6.0),
    dpi: int = 110,
    cmap: str = "RdYlGn",
    min_orders: int = 1,
):
    """Per-cell fraction of orders (by origin) that were completed (C).

    A cell's value is completed / (completed + cancelled) among orders whose
    origin falls in it. Cells with fewer than ``min_orders`` are left blank
    (NaN) so sparse cells do not dominate the picture.
    """
    fig, ax, created = _new_ax(ax, figsize, dpi)
    orders = list(env.orders.values())
    origins = np.array([o.origin for o in orders], dtype=float).reshape(-1, 2)
    completed = np.array(
        [o.status == OrderStatus.COMPLETED for o in orders], dtype=float
    )
    total_mask = np.array(
        [
            o.status in (OrderStatus.COMPLETED, OrderStatus.CANCELLED)
            for o in orders
        ],
        dtype=float,
    )

    comp_H, xe, ye = _hist2d_counts(origins[completed > 0], env.area, bins)
    tot_H, _, _ = _hist2d_counts(origins[total_mask > 0], env.area, bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(tot_H >= min_orders, comp_H / tot_H, np.nan)

    im = ax.imshow(
        rate, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="completion rate")
    ax.set_title("Service-rate heatmap (by order origin)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return _finish(fig, ax, save_path, created)


# --------------------------------------------------------------------------- #
#  D. Time series of system load.
# --------------------------------------------------------------------------- #
def plot_time_series(
    rec: AnalysisRecorder,
    ax=None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),
    dpi: int = 110,
):
    """Pending / onboard / completed / cancelled counts over time (D)."""
    if not rec.times:
        raise ValueError("AnalysisRecorder is empty; call snapshot() per step.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created = True
    else:
        fig, created = ax.figure, False

    t = rec.times
    ax.plot(t, rec.pending, label="pending", color="#d62728")
    ax.plot(t, rec.onboard, label="onboard", color="#1f77b4")
    ax.plot(t, rec.completed, label="completed (cum.)", color="#2ca02c")
    ax.plot(t, rec.cancelled, label="cancelled (cum.)", color="#7f7f7f")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("order count")
    ax.set_title("System load over time", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  E. Pickup-wait-time histogram.
# --------------------------------------------------------------------------- #
def plot_wait_time_hist(
    env,
    bins: int = 30,
    ax=None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7.0, 4.5),
    dpi: int = 110,
    color: str = "#4c72b0",
):
    """Distribution of pickup wait (pickup_time - request_time) for served orders (E)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created = True
    else:
        fig, created = ax.figure, False

    waits = [
        o.pickup_time - o.request_time
        for o in env.orders.values()
        if o.pickup_time is not None
    ]
    if waits:
        ax.hist(waits, bins=bins, color=color, edgecolor="white")
        mean_w = float(np.mean(waits))
        ax.axvline(mean_w, color="#d62728", linestyle="--",
                   label=f"mean = {mean_w:.1f} min")
        ax.legend(fontsize=9)
    ax.set_xlabel("pickup wait (min)")
    ax.set_ylabel("number of orders")
    ax.set_title(f"Pickup-wait distribution ({len(waits)} served)", fontsize=11)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
#  Convenience: render all five as one dashboard figure.
# --------------------------------------------------------------------------- #
def plot_analysis_dashboard(
    env,
    rec: AnalysisRecorder,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (16.0, 10.0),
    dpi: int = 110,
):
    """Lay A-E out on a single 2x3 dashboard figure for a quick overview."""
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    plot_demand_heatmap(env, ax=axes[0, 0])
    plot_supply_demand_gap(rec, ax=axes[0, 1])
    plot_service_rate_heatmap(env, ax=axes[0, 2])
    plot_time_series(rec, ax=axes[1, 0])
    plot_wait_time_hist(env, ax=axes[1, 1])
    axes[1, 2].axis("off")  # spare cell
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig