"""Static frame rendering for the ride-pooling environment (visualization L1).

A single, self-contained ``render_frame`` that draws one snapshot of the
simulation onto a matplotlib axis: the road network (or service-area bounds),
every vehicle coloured by status, the pending-order origins, and the planned
route of each busy vehicle. It reads only the public runtime state already held
by :class:`~ride_gym.env.RidePoolEnv`, so it never mutates the simulation.

Two output modes, mirroring the Gym convention:

* ``mode="human"``     -> draw (and optionally save) a figure; returns the Figure.
* ``mode="rgb_array"`` -> render to an offscreen canvas; returns an ``(H, W, 3)``
                          uint8 numpy array (the frame), suitable for stitching
                          into an animation (visualization level 2).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import itertools 
import matplotlib
# Pick a non-interactive backend automatically on headless machines, exactly as
# iddqn/plot_logs.py does, so frames still render where there is no display.
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib import animation
from ride_gym.enums import DriverStatus, OrderStatus

# Status -> colour. Kept module-level so level-2 animation reuses the same map.
STATUS_COLORS = {
    DriverStatus.IDLE: "#9e9e9e",        # grey
    DriverStatus.TO_PICKUP: "#1f77b4",   # blue
    DriverStatus.TO_DROPOFF: "#2ca02c",  # green
    DriverStatus.RELOCATING: "#ff7f0e",  # orange
}
STATUS_LABEL = {
    DriverStatus.IDLE: "idle",
    DriverStatus.TO_PICKUP: "to pickup",
    DriverStatus.TO_DROPOFF: "to dropoff",
    DriverStatus.RELOCATING: "relocating",
}


def _is_graph_network(net) -> bool:
    """Duck-typed graph-network check (mirrors RidePoolEnv._is_graph_network)."""
    return all(
        hasattr(net, a) for a in ("snap", "node_path", "node_distance", "node_coord")
    )


def _draw_network(ax, env, max_edges: int = 20000) -> None:
    """Draw the street graph (graph mode) or the service-area box (abstract)."""
    net = env.network
    if _is_graph_network(net) and hasattr(net, "graph"):
        g = net.graph
        segs = []
        for u, v in g.edges():
            xu, yu = g.nodes[u]["x"], g.nodes[u]["y"]
            xv, yv = g.nodes[v]["x"], g.nodes[v]["y"]
            segs.append([(xu, yu), (xv, yv)])
            if len(segs) >= max_edges:  # guard huge graphs
                break
        ax.add_collection(
            LineCollection(segs, colors="#dddddd", linewidths=0.4, zorder=0)
        )
        lon0, lat0, lon1, lat1 = net.bounds
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(lat0, lat1)
    else:
        xmin, ymin, xmax, ymax = env.area
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color="#cccccc", linewidth=1.0, zorder=0,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


def _route_polyline(env, driver):
    """Ordered coordinate list a driver will follow: location -> each task stop.

    Graph mode reconstructs the real street path between consecutive stops via
    ``node_path``; abstract networks use ``shortest_path().nodes`` (which yields
    the L-shaped Manhattan path or the straight Euclidean segment).
    """
    if not driver.task_points:
        return []
    net = env.network
    pts = [driver.location] + [tp.location for tp in driver.task_points]
    poly = [pts[0]]
    graph = _is_graph_network(net)
    for a, b in zip(pts[:-1], pts[1:]):
        if graph:
            path = net.node_path(net.snap(a), net.snap(b))
            poly.extend(net.node_coord(i) for i in path[1:])
        else:
            nodes = net.shortest_path(a, b).nodes
            poly.extend(nodes[1:])
    return poly

def _select_drivers(env, driver_ids, max_drivers, only_busy, rng):
    """Pick which drivers to draw, applying the busy filter and a subsample cap.

    Selection order:
      1. If ``driver_ids`` is given, start from exactly those (ignores the cap
         unless it is also exceeded); else start from all drivers.
      2. If ``only_busy``, drop IDLE drivers (keep those with tasks/relocating).
      3. If more than ``max_drivers`` remain, take a random subsample of that
         size (reproducible via ``rng``), so the picture is uncluttered but
         representative.
    """
    if driver_ids is not None:
        chosen = [env.drivers[i] for i in driver_ids if i in env.drivers]
    else:
        chosen = list(env.drivers.values())

    if only_busy:
        chosen = [d for d in chosen if d.status != DriverStatus.IDLE]

    if max_drivers is not None and len(chosen) > max_drivers:
        idx = rng.choice(len(chosen), size=max_drivers, replace=False)
        chosen = [chosen[i] for i in sorted(idx)]
    return chosen


def render_frame(
    env,
    ax=None,
    mode: str = "human",
    show_routes: bool = True,
    show_pending: bool = True,
    max_drivers: Optional[int] = None,
    driver_ids: Optional[list] = None,
    only_busy: bool = False,
    max_pending: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    dpi: int = 110,
    seed: int = 0,
):
    """Render one snapshot of ``env``.

    Decluttering controls
    ---------------------
    max_drivers:
        Draw at most this many vehicles (a reproducible random subsample when
        the fleet is larger). ``None`` (default) draws all.
    driver_ids:
        Draw exactly these driver ids (still subject to ``max_drivers`` /
        ``only_busy``). ``None`` considers the whole fleet.
    only_busy:
        Drop idle vehicles, keeping only those serving or relocating -- usually
        the most informative view on a large fleet.
    show_routes / show_pending:
        Toggle the planned-route polylines and the pending-order markers.
    max_pending:
        Cap how many pending-order origins are drawn (random subsample).
    seed:
        Seed for the subsampling RNG, so the same frame is reproducible.

    Modes
    -----
    ``"human"`` returns the Figure (and saves it when ``save_path`` is set);
    ``"rgb_array"`` returns an ``(H, W, 3)`` uint8 frame for animation.
    """
    rng = np.random.default_rng(seed)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        ax.clear()

    _draw_network(ax, env)

    # Which vehicles to draw (subsampled / filtered for legibility).
    drivers = _select_drivers(env, driver_ids, max_drivers, only_busy, rng)

    # --- planned routes (only for the selected, busy vehicles) -------------
    if show_routes:
        for d in drivers:
            poly = _route_polyline(env, d)
            if len(poly) >= 2:
                xs, ys = zip(*poly)
                ax.plot(xs, ys, color=STATUS_COLORS.get(d.status, "#888888"),
                        linewidth=0.9, alpha=0.6, zorder=1)

    # --- pending order origins (optionally capped) -------------------------
    n_pending_total = len(env._pending_ids)
    if show_pending and n_pending_total:
        pend_ids = env._pending_ids
        if max_pending is not None and n_pending_total > max_pending:
            pick = rng.choice(n_pending_total, size=max_pending, replace=False)
            pend_ids = [pend_ids[i] for i in pick]
        pend = [env.orders[oid] for oid in pend_ids]
        ox, oy = zip(*[o.origin for o in pend])
        ax.scatter(ox, oy, s=18, facecolors="none", edgecolors="#d62728",
                   linewidths=1.0, label="pending order", zorder=2)

    # --- selected vehicles, grouped by status ------------------------------
    by_status = {}
    for d in drivers:
        by_status.setdefault(d.status, []).append(d)
    for status, ds in by_status.items():
        xs = [d.location[0] for d in ds]
        ys = [d.location[1] for d in ds]
        sizes = [16 + 12 * d.onboard_passengers for d in ds]
        ax.scatter(xs, ys, s=sizes, c=STATUS_COLORS.get(status, "#333333"),
                   edgecolors="white", linewidths=0.3, zorder=3,
                   label=STATUS_LABEL.get(status, str(status)))

    # --- counts + cosmetics -------------------------------------------------
    served = sum(o.status == OrderStatus.COMPLETED for o in env.orders.values())
    cancelled = sum(o.status == OrderStatus.CANCELLED for o in env.orders.values())
    shown = f"showing {len(drivers)}/{env.num_drivers} veh"
    ax.set_title(
        f"t = {env.time:.0f} / {env.horizon:.0f} min   |   "
        f"served {served}   cancelled {cancelled}   "
        f"pending {n_pending_total}   |   {shown}",
        fontsize=10,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=1.2)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if mode == "rgb_array":
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[..., :3].copy()
        if created:
            plt.close(fig)
        return frame
    return fig


_FOCUS_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _driver_route_segments(env, driver):
    """Return the driver's street polyline plus its ordered (coord, kind) stops."""
    stops = [(tp.location, tp.kind, tp.order_id) for tp in driver.task_points]
    poly = _route_polyline(env, driver)   # same-module helper, no self-import
    return poly, stops

def render_focus(
    env,
    driver_ids,
    ax=None,
    mode: str = "human",
    show_od_link: bool = True,
    annotate: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (9.0, 9.0),
    dpi: int = 150,
    pad_frac: float = 0.08,
    # --- publication-quality typography / sizing knobs ---
    title_fontsize: int = 18,
    legend_fontsize: int = 15,
    label_fontsize: int = 15,
    veh_marker: float = 320.0,
    od_marker: float = 190.0,
    route_lw: float = 3.0,
    od_lw: float = 2.2,
):
    """Render ONLY a few vehicles, each in its own colour, with full semantics.

    Sizing / typography (all enlarged for figures in a paper):
      title_fontsize / legend_fontsize / label_fontsize:
          font sizes for the title, the shape legend, and the ``veh <id>``
          annotations respectively.
      veh_marker / od_marker:
          scatter marker areas for the vehicle dot and the OD triangles.
      route_lw / od_lw:
          line widths of the planned-route polyline and the OD link.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        ax.clear()

    _draw_network(ax, env)

    drivers = [env.drivers[i] for i in driver_ids if i in env.drivers]
    colors = itertools.cycle(_FOCUS_PALETTE)

    all_x, all_y = [], []

    for d in drivers:
        c = next(colors)
        poly, stops = _driver_route_segments(env, d)

        # 1) planned route along the streets
        if len(poly) >= 2:
            xs, ys = zip(*poly)
            ax.plot(xs, ys, color=c, linewidth=route_lw, alpha=0.85, zorder=2)
            all_x += list(xs); all_y += list(ys)

        # 2) each order's OD: origin (^) + destination (v) + dashed link
        for od in _order_details_for(env, d):
            (oxo, oyo), (oxd, oyd) = od["origin"], od["destination"]
            ax.scatter([oxo], [oyo], marker="^", s=od_marker, color=c,
                       edgecolors="k", linewidths=0.8, zorder=4)
            ax.scatter([oxd], [oyd], marker="v", s=od_marker, color=c,
                       edgecolors="k", linewidths=0.8, zorder=4)
            if show_od_link:
                ax.plot([oxo, oxd], [oyo, oyd], color=c, linestyle=":",
                        linewidth=od_lw, alpha=0.75, zorder=3)
            all_x += [oxo, oxd]; all_y += [oyo, oyd]

        # 3) the vehicle itself (big dot + id label)
        vx, vy = d.location
        ax.scatter([vx], [vy], marker="o", s=veh_marker, color=c,
                   edgecolors="white", linewidths=1.8, zorder=5)
        if annotate:
            ax.annotate(f"veh {d.driver_id}", (vx, vy),
                        textcoords="offset points", xytext=(9, 9),
                        fontsize=label_fontsize, color=c, weight="bold")
        all_x.append(vx); all_y.append(vy)

    # Auto-zoom to the focused content.
    if all_x and all_y:
        x0, x1 = min(all_x), max(all_x)
        y0, y1 = min(all_y), max(all_y)
        px = (x1 - x0) * pad_frac + 1e-9
        py = (y1 - y0) * pad_frac + 1e-9
        ax.set_xlim(x0 - px, x1 + px)
        ax.set_ylim(y0 - py, y1 + py)

    # Shape legend (colour = which vehicle) with enlarged markers/text.
    from matplotlib.lines import Line2D
    shape_legend = [
        Line2D([0], [0], marker="o", color="k", linestyle="none",
               markersize=14, label="vehicle"),
        Line2D([0], [0], marker="^", color="k", linestyle="none",
               markersize=14, label="order origin"),
        Line2D([0], [0], marker="v", color="k", linestyle="none",
               markersize=14, label="order destination"),
        Line2D([0], [0], color="k", linestyle="-", linewidth=route_lw,
               label="planned route"),
        Line2D([0], [0], color="k", linestyle=":", linewidth=od_lw,
               label="order OD link"),
    ]
    ax.legend(handles=shape_legend, loc="upper right",
              fontsize=legend_fontsize, framealpha=0.95)

    ax.set_title(
        f"t = {env.time:.0f} min   |   focus on {len(drivers)} vehicle(s): "
        f"{[d.driver_id for d in drivers]}",
        fontsize=title_fontsize,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if mode == "rgb_array":
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[..., :3].copy()
        if created:
            plt.close(fig)
        return frame
    return fig

def _order_details_for(env, driver):
    """Origin/destination of every order this driver carries or heads to.

    Uses the driver's assigned_orders (onboard + assigned-not-yet-picked-up),
    reading each order's true origin/destination from the env's order table.
    """
    out = []
    for oid in driver.assigned_orders:
        o = env.orders[oid]
        out.append({"origin": o.origin, "destination": o.destination,
                    "order_id": oid})
    return out


class TrajectoryRecorder:
    """Records a lightweight per-step snapshot of the simulation for replay.

    ``snapshot(env)`` is called once per step (or once every ``stride`` steps)
    and stores only the plain data needed to redraw a frame later -- it never
    draws, so recording adds negligible cost to the simulation loop and can be
    left off entirely during training.

    Each snapshot captures BOTH the fleet-wide overview fields and the per-order
    OD / route fields, so the same recording can be rendered either as a
    fleet-wide animation (:func:`render_frame`) or a focused one
    (:func:`render_focus`) without re-running the episode.
    """

    def __init__(self, stride: int = 1):
        """``stride``: keep one snapshot every ``stride`` steps (1 = every step)."""
        self.stride = max(1, int(stride))
        self._i = 0
        self.frames: list = []
        # Static, episode-invariant context captured once at the first snapshot.
        self.area = None
        self.horizon = None
        self.num_drivers = None
        self.network = None  # kept by reference so frames can draw the streets

    def reset(self) -> None:
        """Clear all recorded frames (call at the start of an episode)."""
        self._i = 0
        self.frames = []

    def snapshot(self, env) -> None:
        """Capture the current env state (respecting ``stride``)."""
        if self._i % self.stride == 0:
            self._capture(env)
        self._i += 1

    def _capture(self, env) -> None:
        if self.area is None:  # one-off static context
            self.area = env.area
            self.horizon = env.horizon
            self.num_drivers = env.num_drivers
            self.network = env.network

        drivers = {}
        for did, d in env.drivers.items():
            drivers[did] = {
                "location": tuple(d.location),
                "status": d.status,                 # DriverStatus enum
                "onboard_passengers": d.onboard_passengers,
                "assigned_orders": list(d.assigned_orders),
                # Copy the task-point stops (coord + kind) so routes redraw
                # exactly; kept as plain tuples to decouple from live entities.
                "task_points": [
                    (tp.kind, tp.order_id, tuple(tp.location))
                    for tp in d.task_points
                ],
            }

        pending = [tuple(env.orders[oid].origin) for oid in env._pending_ids]

        # OD of every currently-assigned order, so focused frames can draw them.
        order_od = {}
        for d in env.drivers.values():
            for oid in d.assigned_orders:
                o = env.orders[oid]
                order_od[oid] = (tuple(o.origin), tuple(o.destination))

        # Running completion / cancellation counts for the frame title.
        served = sum(
            o.status == OrderStatus.COMPLETED for o in env.orders.values()
        )
        cancelled = sum(
            o.status == OrderStatus.CANCELLED for o in env.orders.values()
        )

        self.frames.append({
            "time": env.time,
            "drivers": drivers,
            "pending": pending,
            "order_od": order_od,
            "served": served,
            "cancelled": cancelled,
        })


class _SnapshotView:
    """Adapts one recorded frame into an env-like object for the renderers.

    Exposes exactly the attributes ``render_frame`` reads (``drivers`` with
    ``location``/``status``/``onboard_passengers``/``task_points``,
    ``_pending_ids`` + ``orders``, ``network``, ``area``, ``time``, ``horizon``,
    ``num_drivers``), reconstructing lightweight stand-in driver/order objects
    from the plain snapshot data. This lets the animation reuse the level-1
    drawing code verbatim -- no separate frame-drawing logic to maintain.
    """

    class _Order:
        __slots__ = ("origin", "destination", "status")

        def __init__(self, origin, destination, status):
            self.origin = origin
            self.destination = destination
            self.status = status

    class _Task:
        __slots__ = ("kind", "order_id", "location")

        def __init__(self, kind, order_id, location):
            self.kind = kind
            self.order_id = order_id
            self.location = location

    class _Driver:
        __slots__ = (
            "driver_id", "location", "status", "onboard_passengers",
            "assigned_orders", "task_points",
        )

        def __init__(self, did, rec):
            self.driver_id = did
            self.location = rec["location"]
            self.status = rec["status"]
            self.onboard_passengers = rec["onboard_passengers"]
            self.assigned_orders = rec["assigned_orders"]
            self.task_points = [
                _SnapshotView._Task(k, oid, loc)
                for (k, oid, loc) in rec["task_points"]
            ]

    def __init__(self, frame, recorder):
        self.network = recorder.network
        self.area = recorder.area
        self.horizon = recorder.horizon
        self.num_drivers = recorder.num_drivers
        self.time = frame["time"]

        self.drivers = {
            did: _SnapshotView._Driver(did, rec)
            for did, rec in frame["drivers"].items()
        }
        # Reconstruct an ``orders`` table + pending id list. Pending origins are
        # stored without ids, so we mint synthetic negative ids for them (only
        # their origin is drawn). Assigned-order OD get their real ids.
        self.orders = {}
        self._pending_ids = []
        for k, origin in enumerate(frame["pending"]):
            pid = -(k + 1)
            self.orders[pid] = _SnapshotView._Order(
                origin, origin, OrderStatus.PENDING
            )
            self._pending_ids.append(pid)
        for oid, (o, dst) in frame["order_od"].items():
            self.orders[oid] = _SnapshotView._Order(
                o, dst, OrderStatus.ASSIGNED
            )

        # Pre-computed running counts for the title (render_frame recomputes
        # served/cancelled from ``orders``; our reconstructed table has only the
        # live orders, so we stash the true totals for the title override).
        self._served = frame["served"]
        self._cancelled = frame["cancelled"]


def render_animation(
    recorder: TrajectoryRecorder,
    out_path: str = "trajectory.gif",
    fps: int = 4,
    focus_ids: Optional[list] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    dpi: int = 100,
    **render_kwargs,
):
    """Render a recorded trajectory to a GIF (default) or MP4.

    Parameters
    ----------
    recorder:
        A :class:`TrajectoryRecorder` populated over one episode.
    out_path:
        Output file. ``.gif`` uses Pillow (no extra system dep); ``.mp4`` uses
        FFMpeg (requires ``ffmpeg`` on PATH).
    fps:
        Frames per second of the output.
    focus_ids:
        When given, each frame is drawn with :func:`render_focus` on these
        vehicle ids (the focused animation); otherwise :func:`render_frame`
        draws the fleet-wide overview.
    render_kwargs:
        Forwarded to the per-frame renderer (e.g. ``max_drivers``, ``only_busy``,
        ``show_routes`` for the overview; ``show_od_link`` for focus).
    """
    if not recorder.frames:
        raise ValueError("recorder has no frames; call snapshot() during the run.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _draw(frame_idx: int):
        view = _SnapshotView(recorder.frames[frame_idx], recorder)
        if focus_ids is not None:
            render_focus(view, driver_ids=focus_ids, ax=ax, mode="human",
                         **render_kwargs)
        else:
            render_frame(view, ax=ax, mode="human", **render_kwargs)
        # Override the title with the true running totals from the snapshot.
        f = recorder.frames[frame_idx]
        ax.set_title(
            f"t = {f['time']:.0f} / {recorder.horizon:.0f} min   |   "
            f"served {f['served']}   cancelled {f['cancelled']}   "
            f"pending {len(f['pending'])}",
            fontsize=10,
        )

    anim = animation.FuncAnimation(
        fig, _draw, frames=len(recorder.frames), interval=1000 / max(fps, 1)
    )

    if out_path.lower().endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps)
    else:
        writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path