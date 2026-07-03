"""Decentralised multi-agent ride-pooling environment.

Gym-like (not Gym-dependent) interface:

* ``reset(seed=None)`` -> (observations, info)
* ``step(actions)``    -> (observations, rewards, dones, info)

where ``actions`` is ``{driver_id: action}`` and outputs are keyed by driver id.

Action schema (a plain dict per driver, the two keys are mutually exclusive):

    {"orders": Iterable[int]}     # bid on a set of pending order ids (may be [])
    {"relocate": int | (x, y)}    # relocate to a preset index or a coordinate

Providing both keys (with a non-empty order set AND a relocation target) raises
``InvalidActionError``. An empty/absent action keeps the driver's current state.

Strict event flow per step (order is intentional and MUST NOT be reordered):

    1. Cancel timed-out pending orders (BEFORE any action handling).
    2. (Observations were handed out at the previous step / reset.)
    3. Validate actions, run conflict detection, then assign orders & set
       relocation targets.
    4. Physically move every driver one time step along its planned route.
    5. Update order/driver state on pickup/drop-off arrival.
    6. Advance the clock and inject newly-arrived orders.
    7. Compute per-driver rewards from the step event log.
    8. Evaluate termination.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ridepool_sim.entities import Driver, Order, TaskPoint
from ridepool_sim.enums import DriverStatus, OrderStatus
from ridepool_sim.exceptions import ConflictError, InvalidActionError
from ridepool_sim.order_generator import OrderGenerator, RandomOrderGenerator
from ridepool_sim.rewards import DefaultRewardFunction, RewardFunction
from ridepool_sim.road_network import EuclideanNetwork, RoadNetwork
from ridepool_sim.routing import GreedyInsertionPlanner, RoutesPlanner

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]
RelocateTarget = Union[int, Coord]


class RidePoolEnv:
    """Multi-agent ride-pooling and relocation simulation environment."""

    def __init__(
        self,
        area: Area = (0.0, 0.0, 100.0, 100.0),
        num_drivers: int = 10,
        driver_capacity: int = 4,
        driver_capacities: Optional[List[int]] = None,
        driver_speeds: Optional[List[float]] = None,
        dt: float = 1.0,
        horizon: float = 240.0,
        order_timeout: Optional[float] = 10.0,
        order_generator: Optional[OrderGenerator] = None,
        road_network: Optional[RoadNetwork] = None,
        routes_planner: Optional[RoutesPlanner] = None,
        reward_function: Optional[RewardFunction] = None,
        relocation_points: Optional[List[Coord]] = None,
        relocation_grid: Tuple[int, int] = (5, 5),
        relocation_centroids: Optional[List[Coord]] = None,
        relocation_adjacency: int = 8,
        relocation_neighbours_k: int = 8,
        seed: Optional[int] = None,
    ):
        """Configure the environment.

        Parameters
        ----------
        area:
            Service-area bounds ``(xmin, ymin, xmax, ymax)``.
        num_drivers:
            Number of driver agents.
        driver_capacity:
            Default per-driver passenger capacity, applied to every vehicle when
            ``driver_capacities`` is not given (homogeneous fleet).
        driver_capacities:
            Optional explicit per-vehicle capacities, length ``num_drivers``
            (vehicle ``i`` gets ``driver_capacities[i]``). When given it
            overrides ``driver_capacity`` and enables a heterogeneous fleet.
        driver_speeds:
            Optional explicit per-vehicle travel speeds (distance units /
            minute), length ``num_drivers``. ``None`` (default) makes every
            vehicle inherit the road network's default speed (homogeneous
            fleet); a per-vehicle value is used in place of ``network.speed``
            in the movement model.
        dt:
            Time-step length in minutes.
        horizon:
            Total simulation duration in minutes.
        order_timeout:
            Minutes a pending order may wait before auto-cancellation. ``None``
            disables cancellation.
        order_generator:
            Source of orders. Defaults to a uniform :class:`RandomOrderGenerator`.
        road_network:
            Distance/path backend. Defaults to :class:`EuclideanNetwork`.
        routes_planner:
            Task-point sequencer. Defaults to :class:`GreedyInsertionPlanner`.
        reward_function:
            Per-driver reward. Defaults to :class:`DefaultRewardFunction`.
        relocation_points:
            Preset relocation coordinates. If ``None`` they are derived from
            ``relocation_centroids`` (if given) or an evenly spaced grid of
            ``relocation_grid`` cell centres. These are the *region centres*;
            each is a selectable relocation target and defines one region.
        relocation_grid:
            ``(rows, cols)`` for the auto-generated uniform-grid region model.
            Used only when neither ``relocation_points`` nor
            ``relocation_centroids`` is supplied. The grid model supports
            geometric N-adjacency (see ``relocation_adjacency``).
        relocation_centroids:
            User-supplied region centres ``[(x, y), ...]`` (e.g. NYC taxi-zone
            centroids). When given, the region model is the (generally
            irregular) set of these centres and region adjacency degenerates to
            the K spatially-nearest centres (see ``relocation_neighbours_k``),
            since N-gon adjacency has no meaning for an irregular point set.
        relocation_adjacency:
            Number of geometric neighbours per region in the *grid* model: 4
            (von Neumann: up/down/left/right) or 8 (Moore: incl. diagonals).
            Ignored for the centroid model.
        relocation_neighbours_k:
            Number of spatially-nearest neighbour regions per region in the
            *centroid* model. Ignored for the grid model.
        seed:
            Base RNG seed for reproducibility.

        Region model / single source of truth
        --------------------------------------
        The environment owns the region partition AND the region-adjacency
        graph, exposing both via observations (``self.current_region`` and the
        shared ``region_neighbours``) so that any policy reuses *exactly* the
        same regions/adjacency the env will enforce -- there is no second,
        possibly-inconsistent definition on the policy side.
        """
        self.area = area
        self.num_drivers = int(num_drivers)
        self.driver_capacity = int(driver_capacity)
        # Resolve per-vehicle capacities / speeds into length-num_drivers lists
        # once at construction. Capacities default to the homogeneous
        # ``driver_capacity``; speeds default to ``None`` per vehicle (inherit
        # the network speed). Explicit lists must match ``num_drivers``.
        self.driver_capacities: List[int] = self._resolve_per_driver(
            driver_capacities,
            default=self.driver_capacity,
            name="driver_capacities",
            cast=int,
        )
        self.driver_speeds: List[Optional[float]] = self._resolve_per_driver(
            driver_speeds,
            default=None,
            name="driver_speeds",
            cast=lambda v: None if v is None else float(v),
        )
        self.dt = float(dt)
        self.horizon = float(horizon)
        self.order_timeout = order_timeout

        self.network = road_network or EuclideanNetwork(speed=1.0)
        self.planner = routes_planner or GreedyInsertionPlanner()
        self.reward_function = reward_function or DefaultRewardFunction()
        self.order_generator = order_generator or RandomOrderGenerator(
            area=area, horizon=self.horizon, num_orders=200, rng=seed
        )

        # --- Region model (single source of truth) -------------------------
        # Region centres come from, in priority order: explicit
        # ``relocation_points``, then user ``relocation_centroids``, then the
        # auto-generated uniform grid. Only the grid model carries (rows, cols)
        # metadata enabling geometric N-adjacency; the other two are treated as
        # an irregular point set whose adjacency is K-nearest.
        self.relocation_adjacency = int(relocation_adjacency)
        self.relocation_neighbours_k = int(relocation_neighbours_k)
        self._grid_shape: Optional[Tuple[int, int]] = None
        if relocation_points is not None:
            centres = list(relocation_points)
        elif relocation_centroids is not None:
            centres = list(relocation_centroids)
        else:
            centres = self._build_relocation_grid(relocation_grid)
            # rows, cols recorded so adjacency can use the regular lattice.
            self._grid_shape = (int(relocation_grid[0]), int(relocation_grid[1]))

        # Stored as an immutable tuple so that observations can share it by
        # reference with zero copy while making it impossible for a policy to
        # corrupt env state by mutating the relocation grid in place.
        self.relocation_points: Tuple[Coord, ...] = tuple(
            (float(x), float(y)) for (x, y) in centres
        )
        # Per-region neighbour region indices (the region-adjacency graph),
        # precomputed once: geometric N-adjacency for the grid model, else
        # K-nearest for an irregular centroid set. Exposed via observations so
        # the policy reuses exactly this adjacency.
        self.region_neighbours: Tuple[Tuple[int, ...], ...] = (
            self._build_region_neighbours()
        )

        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Runtime state (populated in reset()).
        self.time: float = 0.0
        self.drivers: Dict[int, Driver] = {}
        self.orders: Dict[int, Order] = {}
        self._pending_ids: List[int] = []
        self._all_orders: List[Order] = []
        self._next_inject_idx: int = 0

    # ------------------------------------------------------------------ setup
    def _resolve_per_driver(self, values, default, name: str, cast):
        """Normalise a per-driver spec into a length-``num_drivers`` list.

        ``values`` may be ``None`` (every vehicle gets ``default``) or a
        sequence of exactly ``num_drivers`` entries (vehicle ``i`` gets
        ``values[i]``). Each entry is passed through ``cast`` for type/None
        normalisation. A wrong-length sequence is a configuration error and
        raises ``ValueError`` so mistakes surface immediately rather than
        silently truncating/recycling the fleet.
        """
        if values is None:
            return [cast(default) for _ in range(self.num_drivers)]
        seq = list(values)
        if len(seq) != self.num_drivers:
            raise ValueError(
                f"{name} must have length num_drivers ({self.num_drivers}), "
                f"got {len(seq)}."
            )
        return [cast(v) for v in seq]

    def _build_relocation_grid(self, grid: Tuple[int, int]) -> List[Coord]:
        """Evenly spaced cell-centre coordinates over the service area."""
        rows, cols = grid
        xmin, ymin, xmax, ymax = self.area
        xs = (np.arange(cols) + 0.5) / cols * (xmax - xmin) + xmin
        ys = (np.arange(rows) + 0.5) / rows * (ymax - ymin) + ymin
        return [(float(x), float(y)) for y in ys for x in xs]

    def _build_region_neighbours(self) -> Tuple[Tuple[int, ...], ...]:
        """Precompute each region's neighbour-region indices (adjacency graph).

        Two adjacency models:

        * Grid model (``self._grid_shape`` is set): geometric N-adjacency on the
          regular lattice -- 4 (von Neumann) or 8 (Moore, incl. diagonals)
          neighbours per cell, clipped at the lattice border. The region index
          is row-major (``idx = row * cols + col``), matching
          :meth:`_build_relocation_grid`'s emission order.
        * Centroid model (irregular point set): the K spatially-nearest other
          region centres by Euclidean distance, since N-gon adjacency is
          undefined for an arbitrary point set. K = ``relocation_neighbours_k``.

        Returns a tuple (immutable, shareable by reference) of per-region
        neighbour-index tuples, self excluded.
        """
        pts = self.relocation_points
        n = len(pts)
        if n == 0:
            return tuple()

        if self._grid_shape is not None:
            rows, cols = self._grid_shape
            if self.relocation_adjacency == 4:
                offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            else:  # 8-adjacency (Moore) is the default for any non-4 value.
                offsets = [
                    (dr, dc)
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                ]
            neigh: List[Tuple[int, ...]] = []
            for idx in range(n):
                r, c = divmod(idx, cols)
                here = []
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        here.append(nr * cols + nc)
                neigh.append(tuple(here))
            return tuple(neigh)

        # Centroid model: K-nearest other centres (brute force; the region count
        # is small -- e.g. a few hundred NYC zones -- so O(n^2) is negligible and
        # paid once at construction).
        k = max(0, min(self.relocation_neighbours_k, n - 1))
        arr = np.asarray(pts, dtype=float)  # [n, 2]
        neigh = []
        for i in range(n):
            d2 = ((arr - arr[i]) ** 2).sum(axis=1)
            d2[i] = np.inf  # exclude self
            nearest = np.argsort(d2)[:k]
            neigh.append(tuple(int(j) for j in nearest))
        return tuple(neigh)

    def _current_region(self, coord: Coord) -> int:
        """Index of the region whose centre is nearest to ``coord``.

        The driver's 'current region' is the nearest region centre (a
        well-defined, unambiguous partition for both the grid and the irregular
        centroid models). With no regions defined, returns -1.
        """
        if not self.relocation_points:
            return -1
        arr = np.asarray(self.relocation_points, dtype=float)
        d2 = ((arr - np.asarray(coord, dtype=float)) ** 2).sum(axis=1)
        return int(np.argmin(d2))

    def _in_area(self, coord: Coord) -> bool:
        xmin, ymin, xmax, ymax = self.area
        x, y = coord
        return xmin <= x <= xmax and ymin <= y <= ymax

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None):
        """Reset the environment and return ``(observations, info)``."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            if hasattr(self.order_generator, "reseed"):
                self.order_generator.reseed(seed)

        self.time = 0.0
        xmin, ymin, xmax, ymax = self.area
        graph_mode = self._is_graph_network()

        # Spawn drivers. On a real graph they start ON a random network node (so
        # they are always on a drivable road and their node position is
        # well-defined); on abstract networks they start at a random point in
        # the service-area bounding box.
        self.drivers = {}
        for did in range(self.num_drivers):
            cap = self.driver_capacities[did]
            spd = self.driver_speeds[did]
            if graph_mode:
                n_nodes = len(self.network.node_coords)
                node_idx = int(self._rng.integers(0, n_nodes))
                loc = self.network.node_coord(node_idx)
                self.drivers[did] = Driver(
                    driver_id=did,
                    location=loc,
                    capacity=cap,
                    speed=spd,
                    node_idx=node_idx,
                )
            else:
                loc = (
                    float(self._rng.uniform(xmin, xmax)),
                    float(self._rng.uniform(ymin, ymax)),
                )
                self.drivers[did] = Driver(
                    driver_id=did, location=loc, capacity=cap, speed=spd
                )

        # Generate the full order set; inject those due at t == 0.
        self._all_orders = sorted(
            self.order_generator.generate(), key=lambda o: o.request_time
        )
        self.orders = {o.order_id: o for o in self._all_orders}
        self._pending_ids = []
        self._next_inject_idx = 0
        self._inject_due_orders()

        return self._build_observations(), {"time": self.time}

    def _inject_due_orders(self) -> None:
        """Move orders whose request_time <= current time into the pending pool."""
        while (
            self._next_inject_idx < len(self._all_orders)
            and self._all_orders[self._next_inject_idx].request_time <= self.time
        ):
            order = self._all_orders[self._next_inject_idx]
            if order.status == OrderStatus.PENDING:
                self._pending_ids.append(order.order_id)
            self._next_inject_idx += 1

    # ------------------------------------------------------------- step (1) ----
    def _cancel_timed_out(self) -> None:
        """Step 1: cancel pending orders past the timeout. Runs FIRST."""
        if self.order_timeout is None:
            return
        survivors: List[int] = []
        for oid in self._pending_ids:
            order = self.orders[oid]
            if order.waiting_time(self.time) > self.order_timeout:
                order.status = OrderStatus.CANCELLED
                order.cancel_time = self.time
            else:
                survivors.append(oid)
        self._pending_ids = survivors

    # ------------------------------------------------------------- step (3) ----
    def _parse_action(self, driver_id: int, action) -> Tuple[List[int], Optional[Coord]]:
        """Validate a single driver's action; return (order_ids, reloc_coord).

        Enforces the mutual-exclusion rule and all single-driver legality
        checks except the cross-driver conflict check (done globally) and the
        greedy capacity drop (applied during assignment).
        """
        if action is None:
            return [], None

        order_ids = list(action.get("orders", []) or [])
        relocate = action.get("relocate", None)

        wants_orders = len(order_ids) > 0
        wants_relocate = relocate is not None

        # Mutual exclusion: cannot bid AND relocate in the same step.
        if wants_orders and wants_relocate:
            raise InvalidActionError(
                f"Driver {driver_id}: action is mutually exclusive but contains "
                f"both an order bid set and a relocation target."
            )

        # Validate bid targets. Unknown ids are a genuine policy bug -> raise.
        # Orders that became non-pending *between* observation and this step
        # (e.g. auto-cancelled by timeout, or already grabbed) are a fact of the
        # world, not a policy error: silently drop them from the bid set so the
        # policy is not punished for an unavoidable race.
        pending_set = set(self._pending_ids)
        validated_ids = []
        for oid in order_ids:
            if oid not in self.orders:
                raise InvalidActionError(
                    f"Driver {driver_id}: bids on unknown order {oid!r}."
                )
            if oid in pending_set:
                validated_ids.append(oid)
        order_ids = validated_ids
        # Re-evaluate whether the driver still effectively bids after filtering.
        wants_orders = len(order_ids) > 0

        reloc_coord: Optional[Coord] = None
        if wants_relocate:
            driver = self.drivers[driver_id]
            # Relocation precondition: must be idle and not bidding this step.
            if driver.status != DriverStatus.IDLE:
                raise InvalidActionError(
                    f"Driver {driver_id}: relocation requires IDLE status "
                    f"(current={driver.status})."
                )
            reloc_coord = self._resolve_relocation(driver_id, relocate)

        return order_ids, reloc_coord

    def _resolve_relocation(self, driver_id: int, relocate: RelocateTarget) -> Coord:
        """Resolve a relocation target (preset index or coordinate) to a coord."""
        if isinstance(relocate, (int, np.integer)):
            idx = int(relocate)
            if not (0 <= idx < len(self.relocation_points)):
                raise InvalidActionError(
                    f"Driver {driver_id}: relocation index {idx} out of range "
                    f"[0, {len(self.relocation_points)})."
                )
            return self.relocation_points[idx]

        # Otherwise treat as a coordinate.
        try:
            x, y = relocate
            coord = (float(x), float(y))
        except (TypeError, ValueError):
            raise InvalidActionError(
                f"Driver {driver_id}: relocation target {relocate!r} is neither "
                f"a valid preset index nor an (x, y) coordinate."
            )
        if not self._in_area(coord):
            raise InvalidActionError(
                f"Driver {driver_id}: relocation coordinate {coord} is outside "
                f"the service area {self.area}."
            )
        return coord

    def _detect_conflicts(self, bids: Dict[int, List[int]]) -> None:
        """Raise ConflictError if any pending order is bid on by >1 driver."""
        order_to_drivers: Dict[int, List[int]] = {}
        for did, oids in bids.items():
            for oid in oids:
                order_to_drivers.setdefault(oid, []).append(did)
        for oid, dids in order_to_drivers.items():
            if len(dids) > 1:
                raise ConflictError(order_id=oid, driver_ids=dids)

    def _assign_orders(
        self, driver_id: int, order_ids: List[int], event: Dict
    ) -> None:
        """Assign a (conflict-free) bid set to a driver under the capacity rule.

        Per the spec correction: if the requested set would exceed capacity, do
        NOT raise; greedily drop the largest-party orders until it fits, so as
        many orders as possible are still assigned. Ties broken by larger id
        first for determinism. Dropped orders stay pending.
        """
        if not order_ids:
            return

        driver = self.drivers[driver_id]
        baseline = driver.committed_passengers(self.orders)
        budget = driver.capacity - baseline

        # Greedy drop: sort so that we keep smallest parties first.
        candidates = sorted(
            order_ids,
            key=lambda oid: (self.orders[oid].num_passengers, -oid),
        )
        accepted: List[int] = []
        used = 0
        for oid in candidates:
            party = self.orders[oid].num_passengers
            if used + party <= budget:
                accepted.append(oid)
                used += party
            # else: dropped (remains pending, available next step)

        # Counterfactual baseline for already-committed orders: their estimated
        # remaining time-to-delivery (``order.eta``) as maintained up to this
        # step. This is the drop-off time under the route that did NOT yet
        # include the new orders, captured incrementally (decremented each step,
        # reset on every re-plan) so no extra route traversal is needed here.
        # Snapshotted now, before insertion / re-planning overwrites it.
        before_eta = {
            oid: self.orders[oid].eta
            for oid in driver.assigned_orders
            if self.orders[oid].eta is not None
        }

        for oid in accepted:
            order = self.orders[oid]
            order.status = OrderStatus.ASSIGNED
            order.assigned_driver = driver_id
            driver.assigned_orders.append(oid)
            driver.task_points.append(
                TaskPoint("pickup", oid, order.origin)
            )
            driver.task_points.append(
                TaskPoint("dropoff", oid, order.destination)
            )
            self._pending_ids.remove(oid)

            event["assigned_orders"].append(oid)
            event["assigned_party_sizes"][oid] = order.num_passengers
            solo = self.network.shortest_path(order.origin, order.destination)
            event["assigned_solo_times"][oid] = solo.travel_time

        if accepted:
            # Re-optimise the whole plan and refresh status/route. The planner
            # reports each stop's arrival time as a by-product of the sequencing
            # it already does, so the realised-route arrival times come for free
            # (no second pass over the finished route).
            driver.task_points, after_times = self.planner.plan_with_times(
                driver.location, driver.task_points, self.network
            )

            # ----------------------------------------------------------------
            # (1) Service-time of each NEWLY assigned order: its predicted
            #     END-TO-END time from the user's request to the planned
            #     drop-off, measured on the re-optimised route. Both segments
            #     are included:
            #       * (now - request_time): time already spent on the platform
            #         from the moment the user sent the request to this dispatch
            #         (request -> pending -> assignment);
            #       * after_times[oid]["dropoff"]: planned minutes from NOW until
            #         the order is dropped off (covers the remaining pickup wait
            #         AND the in-vehicle ride on the pooled route).
            #     after_times is measured from the driver's current position
            #     (relative to the current clock T == self.time), so the
            #     absolute predicted drop-off clock is T + after_times[...] and
            #     the end-to-end time is that minus request_time.
            for oid in accepted:
                at = after_times.get(oid, {})
                if "dropoff" in at:
                    req = self.orders[oid].request_time
                    end_to_end = (self.time + at["dropoff"]) - req
                    event["assigned_service_times"][oid] = max(0.0, end_to_end)

            # ----------------------------------------------------------------
            # (2) Re-routing impact on already-committed EN-ROUTE orders: how
            #     much each one's predicted drop-off time changes because the
            #     new orders were inserted and the whole route re-optimised.
            #     SIGNED (not clamped): a later drop-off is a positive penalty,
            #     an earlier one is negative (a reward). Each en-route order is
            #     measured separately and summed. The drop-off time already
            #     reflects any change to that order's pickup time too (its pickup
            #     precedes its drop-off on the same route), per design.
            #
            #     before_eta[oid] and after_times[oid]["dropoff"] share the same
            #     basis (minutes-to-drop-off from the current position/clock),
            #     so their difference is the pure re-routing delta.
            extra_detour = 0.0
            for oid, old_eta in before_eta.items():
                at = after_times.get(oid, {})
                if "dropoff" in at:
                    extra_detour += at["dropoff"] - old_eta

            event["extra_detour_time"] += extra_detour

            # Refresh the maintained time-to-delivery for every order now on the
            # driver's plan from the freshly computed arrival times (free, from
            # the planner). These become the baseline for the next assignment.
            for oid, at in after_times.items():
                if "dropoff" in at:
                    self.orders[oid].eta = at["dropoff"]

            if driver.status in (DriverStatus.IDLE, DriverStatus.RELOCATING):
                driver.relocation_target = None
            self._refresh_driver_status(driver)

    def _set_relocation(self, driver_id: int, coord: Coord) -> None:
        driver = self.drivers[driver_id]
        driver.relocation_target = coord
        driver.status = DriverStatus.RELOCATING

    def _refresh_driver_status(self, driver: Driver) -> None:
        """Derive status from the next task point."""
        if driver.task_points:
            nxt = driver.task_points[0]
            driver.status = (
                DriverStatus.TO_PICKUP
                if nxt.kind == "pickup"
                else DriverStatus.TO_DROPOFF
            )
        elif driver.relocation_target is not None:
            driver.status = DriverStatus.RELOCATING
        else:
            driver.status = DriverStatus.IDLE

    # ------------------------------------------------------------- step (4-5) --
    def _move_driver(self, driver: Driver, event: Dict) -> None:
        """Move one driver up to ``speed * dt`` along its plan, updating orders.

        Movement model depends on the road network:

        * If the network exposes node-level routing (``node_path`` /
          ``node_distance`` / ``snap`` / ``node_coord``), as
          :class:`OSMnxNetwork` does, the driver advances **along the real
          street graph with in-edge positioning**: within one step it walks the
          shortest node path toward each target, consuming a constant-speed
          distance budget ``speed * dt`` edge by edge, and may stop part-way
          along an edge (its continuous ``location`` is interpolated between the
          edge endpoints). See :meth:`_move_driver_graph`.
        * Otherwise (Euclidean / Manhattan abstract networks) it falls back to
          the original straight-line interpolation toward each target.

        Deadlock guarantee (graph mode): a long edge is consumed across multiple
        steps via in-edge progress (``edge_pos_m`` accumulates), so no single
        edge can stall a driver forever; the only precondition is a strictly
        positive budget (``speed * dt > 0``), which is asserted.
        """
        if self._is_graph_network():
            self._move_driver_graph(driver, event)
            return

        budget_dist = self._driver_speed(driver) * self.dt
        moved = 0.0
        started_onboard = driver.onboard_passengers

        target = self._current_target(driver)
        while target is not None and budget_dist > 1e-12:
            seg = self.network.shortest_path(driver.location, target)
            if seg.distance <= budget_dist + 1e-12:
                # Reach this target this step.
                driver.location = target
                budget_dist -= seg.distance
                moved += seg.distance
                self._arrive_at_target(driver, event)
                target = self._current_target(driver)
            else:
                # Partial progress toward target along straight interpolation.
                frac = budget_dist / seg.distance
                ox, oy = driver.location
                tx, ty = target
                driver.location = (ox + (tx - ox) * frac, oy + (ty - oy) * frac)
                moved += budget_dist
                budget_dist = 0.0

        event["distance_moved"] = moved
        # Time spent moving uses the driver's own speed when set so a
        # heterogeneous fleet's per-vehicle travel time is reported correctly.
        event["time_moved"] = moved / max(self._driver_speed(driver), 1e-12)
        event["is_empty_move"] = moved > 0 and started_onboard == 0
        event["is_idle_wait"] = (
            moved == 0
            and not driver.task_points
            and driver.relocation_target is None
        )

        # Decrement the maintained time-to-delivery for every order still on the
        # driver's plan. Any order NOT delivered this step necessarily leaves
        # the driver still travelling toward a remaining stop, which only
        # happens when the whole movement budget (== dt minutes at network
        # speed) was consumed; hence exactly dt of remaining time has elapsed.
        # Orders dropped off this step had their eta cleared on arrival, so they
        # are skipped. Re-planning later overwrites eta, so error cannot
        # accumulate.
        for oid in driver.assigned_orders:
            order = self.orders[oid]
            if order.eta is not None:
                order.eta -= self.dt

    # ---------------------------------------------------- graph-mode movement
    def _driver_speed(self, driver: Driver) -> float:
        """Effective travel speed for a driver this step.

        Returns the driver's own ``speed`` when set (heterogeneous fleet), else
        the road network's default ``speed`` (homogeneous fleet). Centralising
        this keeps both the abstract and graph movement paths consistent.
        """
        return self.network.speed if driver.speed is None else driver.speed

    def _is_graph_network(self) -> bool:
        """True if the road network supports node-level routing (OSMnx-style).

        Detected structurally (duck typing) by the presence of the node-routing
        API, so any network providing it gets graph movement without a hard
        import dependency on :class:`OSMnxNetwork`.
        """
        net = self.network
        return (
            hasattr(net, "snap")
            and hasattr(net, "node_path")
            and hasattr(net, "node_distance")
            and hasattr(net, "node_coord")
        )

    def _edge_interp_location(self, driver: Driver) -> Coord:
        """Continuous (lon, lat) for a driver positioned part-way along an edge.

        Linearly interpolates between the edge's endpoint node coordinates by
        the fraction ``edge_pos_m / edge_length``. Used only in graph mode to
        keep ``driver.location`` a real coordinate for observations/features.
        """
        net = self.network
        ax, ay = net.node_coord(driver.edge_from)
        bx, by = net.node_coord(driver.edge_to)
        edge_len = net.node_distance(driver.edge_from, driver.edge_to)
        frac = 0.0 if edge_len <= 0 else min(1.0, driver.edge_pos_m / edge_len)
        return (ax + (bx - ax) * frac, ay + (by - ay) * frac)

    def _move_driver_graph(self, driver: Driver, event: Dict) -> None:
        """Advance a driver along the real street graph with in-edge position.

        Invariant on driver position: the driver is EITHER exactly on a node
        (``node_idx`` set, ``edge_to`` None) OR part-way along a directed edge
        (``edge_from``/``edge_to``/``edge_pos_m`` set, ``node_idx`` None). Within
        one step it spends a constant-speed distance budget ``speed * dt`` metres
        walking the shortest node path toward each successive task target,
        finishing partial edges across steps.

        Deadlock-free: each loop iteration either consumes strictly positive
        budget (advancing along an edge) or arrives at a target (shrinking the
        task list); a single edge longer than one step's budget is consumed over
        several steps via ``edge_pos_m``. Requires ``budget > 0`` (asserted).
        """
        net = self.network
        speed = self._driver_speed(driver)
        budget = speed * self.dt
        assert budget > 0, (
            "graph movement requires a strictly positive distance budget "
            "(driver/network speed * dt); got speed="
            f"{speed} dt={self.dt}. A zero budget would deadlock drivers."
        )
        moved = 0.0
        started_onboard = driver.onboard_passengers
        eps = 1e-9

        # If the driver is currently mid-edge, finish (or progress along) that
        # edge before any node-path routing.
        if driver.edge_to is not None:
            edge_len = net.node_distance(driver.edge_from, driver.edge_to)
            remain = edge_len - driver.edge_pos_m
            if budget + eps >= remain:
                # Reach the edge's end node this step.
                budget -= remain
                moved += remain
                driver.node_idx = driver.edge_to
                driver.edge_from = driver.edge_to = None
                driver.edge_pos_m = 0.0
                driver.location = net.node_coord(driver.node_idx)
            else:
                driver.edge_pos_m += budget
                moved += budget
                budget = 0.0
                driver.location = self._edge_interp_location(driver)

        # Walk node paths toward each target while budget remains.
        while budget > eps:
            target = self._current_target(driver)
            if target is None:
                break
            src = driver.node_idx
            dst = net.snap(target)
            if src == dst:
                # Already at the target node -> handle pickup/dropoff/reloc and
                # move on to the next target.
                self._arrive_at_target(driver, event)
                continue
            path = net.node_path(src, dst)
            progressed = False
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                edge_len = net.node_distance(u, v)
                if budget + eps >= edge_len:
                    budget -= edge_len
                    moved += edge_len
                    driver.node_idx = v
                    driver.location = net.node_coord(v)
                    progressed = True
                    if v == dst:
                        self._arrive_at_target(driver, event)
                        break
                else:
                    # Enter edge u->v and stop part-way (budget is strictly
                    # positive here, so this always makes progress).
                    driver.edge_from = u
                    driver.edge_to = v
                    driver.edge_pos_m = budget
                    driver.node_idx = None
                    moved += budget
                    budget = 0.0
                    driver.location = self._edge_interp_location(driver)
                    progressed = True
                    break
            if not progressed:
                # Defensive: no edge consumed (e.g. degenerate path). Break to
                # avoid any chance of an infinite loop.
                break

        event["distance_moved"] = moved
        event["time_moved"] = moved / max(speed, 1e-12)
        event["is_empty_move"] = moved > 0 and started_onboard == 0
        event["is_idle_wait"] = (
            moved == 0
            and not driver.task_points
            and driver.relocation_target is None
        )

        # Same eta maintenance as the abstract-network path: any order still
        # assigned (undelivered) had a full dt of travel time elapse this step.
        for oid in driver.assigned_orders:
            order = self.orders[oid]
            if order.eta is not None:
                order.eta -= self.dt

    def _current_target(self, driver: Driver) -> Optional[Coord]:
        """Where the driver is heading right now: next task point or reloc."""
        if driver.task_points:
            return driver.task_points[0].location
        if driver.relocation_target is not None:
            return driver.relocation_target
        return None

    def _arrive_at_target(self, driver: Driver, event: Dict) -> None:
        """Handle arrival at the current target (task point or relocation)."""
        if driver.task_points:
            tp = driver.task_points.pop(0)
            order = self.orders[tp.order_id]
            if tp.kind == "pickup":
                order.status = OrderStatus.ONBOARD
                order.pickup_time = self.time
                driver.onboard_passengers += order.num_passengers
                event["picked_up_orders"].append(order.order_id)
            else:  # dropoff
                order.status = OrderStatus.COMPLETED
                order.dropoff_time = self.time
                order.eta = None  # delivered: no remaining time-to-delivery
                driver.onboard_passengers -= order.num_passengers
                if order.order_id in driver.assigned_orders:
                    driver.assigned_orders.remove(order.order_id)
                event["completed_orders"].append(order.order_id)
            self._refresh_driver_status(driver)
        elif driver.relocation_target is not None:
            # Reached relocation goal -> become idle.
            driver.relocation_target = None
            driver.status = DriverStatus.IDLE

    # ------------------------------------------------------------- observations
    def _build_observations(self) -> Dict[int, Dict]:
        """Build per-driver observations.

        Shared, per-step-invariant structures (the pending-order list, the
        public driver state of *all* drivers, the relocation grid, the clock)
        are constructed once and shared by reference across every driver's
        observation, avoiding the previous O(num_drivers**2) cost of building a
        per-driver 'others-excluding-self' dict.

        Each driver's observation exposes:

        * ``self``           : that driver's full private state.
        * ``all_drivers``    : public state of every driver, keyed by id,
                               *including this driver itself*. A policy that
                               wants only the others filters by ``driver_id``.
        * ``pending_orders`` : shared pending-order snapshot.
        * ``relocation_points`` / ``time`` : shared.

        All shared structures are treated as read-only by convention; callers
        must not mutate them.
        """
        pending = [
            {
                "order_id": o.order_id,
                "origin": o.origin,
                "destination": o.destination,
                "num_passengers": o.num_passengers,
                "waiting_time": o.waiting_time(self.time),
            }
            for oid in self._pending_ids
            for o in (self.orders[oid],)
        ]

        # Public state of all drivers, built once and shared by reference.
        all_drivers = {
            did: {
                "location": d.location,
                "status": d.status.value,
                "onboard_passengers": d.onboard_passengers,
            }
            for did, d in self.drivers.items()
        }

        obs: Dict[int, Dict] = {}
        for did, d in self.drivers.items():
            obs[did] = {
                "self": {
                    "driver_id": did,
                    "location": d.location,
                    "status": d.status.value,
                    "capacity": d.capacity,
                    # Effective per-vehicle travel speed (own speed if set, else
                    # the network default), exposed so a policy can reason about
                    # heterogeneous fleets.
                    "speed": self._driver_speed(d),
                    "onboard_passengers": d.onboard_passengers,
                    "assigned_orders": list(d.assigned_orders),
                    # En-route order details (origin, destination, party,
                    # onboard flag, remaining eta) for sequence-based feature
                    # encoders. The flat MLP encoder ignores this field.
                    "assigned_order_details": [
                        {
                            "order_id": oid,
                            "origin": self.orders[oid].origin,
                            "destination": self.orders[oid].destination,
                            "num_passengers": self.orders[oid].num_passengers,
                            "onboard": self.orders[oid].status
                            == OrderStatus.ONBOARD,
                            "eta": self.orders[oid].eta,
                        }
                        for oid in d.assigned_orders
                    ],
                    # Onboard + assigned-but-not-yet-picked-up passengers: the
                    # authoritative capacity baseline against which new bids are
                    # checked. Exposed so policies / learners see true remaining
                    # capacity and future committed load, not just onboard.
                    "committed_passengers": d.committed_passengers(self.orders),
                    # Index of the region the driver is currently in (nearest
                    # region centre). Lets a relocation policy restrict targets
                    # to this region's neighbours -- the env's own adjacency.
                    "current_region": self._current_region(d.location),
                },
                "all_drivers": all_drivers,
                "pending_orders": pending,
                "time": self.time,
                "relocation_points": self.relocation_points,
                # Region-adjacency graph (per-region neighbour indices) and the
                # region count, shared by reference. Single source of truth for
                # which regions a driver may relocate to from its current one.
                "region_neighbours": self.region_neighbours,
            }
        return obs

    def _new_event(self) -> Dict:
        return {
            "assigned_orders": [],
            # Per newly-assigned order: its passenger (party) count. Used by the
            # reward's revenue bonus, which scales fare by the number of riders.
            "assigned_party_sizes": {},
            "assigned_solo_times": {},
            # Per newly-assigned order: predicted END-TO-END service time (min)
            # from the user's request to the planned drop-off, i.e.
            #   (now - order.request_time)            # platform + dispatch wait
            # + planned time-to-dropoff on the re-optimised route   # pickup +
            #                                                        # in-vehicle
            # Used by the reward's service-time penalty.
            "assigned_service_times": {},
            "completed_orders": [],
            "picked_up_orders": [],
            "distance_moved": 0.0,
            "time_moved": 0.0,
            "is_empty_move": False,
            "is_idle_wait": False,
            "extra_detour_time": 0.0,
        }

    # -------------------------------------------------------------------- step
    def step(self, actions: Dict[int, Dict]):
        """Advance the simulation by one time step.

        Returns ``(observations, rewards, dones, info)`` keyed by driver id
        (``info`` is a single shared dict containing the per-driver event log).
        """
        actions = actions or {}
        events = {did: self._new_event() for did in self.drivers}

        # --- Step 1: cancel timed-out pending orders BEFORE anything else. ---
        self._cancel_timed_out()

        # --- Step 3: validate, detect conflicts, then assign / relocate. ---
        parsed_bids: Dict[int, List[int]] = {}
        parsed_reloc: Dict[int, Coord] = {}
        for did in self.drivers:
            order_ids, reloc = self._parse_action(did, actions.get(did))
            if order_ids:
                parsed_bids[did] = order_ids
            if reloc is not None:
                parsed_reloc[did] = reloc

        self._detect_conflicts(parsed_bids)

        for did, order_ids in parsed_bids.items():
            self._assign_orders(did, order_ids, events[did])
        for did, coord in parsed_reloc.items():
            self._set_relocation(did, coord)

        # --- Step 4 & 5: physical movement + order state updates. ---
        for did, driver in self.drivers.items():
            self._move_driver(driver, events[did])

        # --- Step 6: advance clock, inject new orders. ---
        self.time += self.dt
        self._inject_due_orders()

        # --- Step 7: rewards. ---
        rewards = {
            did: self.reward_function(did, events[did]) for did in self.drivers
        }

        # --- Step 8: termination. ---
        done = self.time >= self.horizon
        dones = {did: done for did in self.drivers}
        dones["__all__"] = done

        observations = self._build_observations()
        info = {"time": self.time, "events": events, "done": done}
        return observations, rewards, dones, info

    # -------------------------------------------------------------- utilities
    def render(self) -> None:
        """Placeholder for trajectory / heat-map visualisation (future work)."""
        served = sum(
            1 for o in self.orders.values() if o.status == OrderStatus.COMPLETED
        )
        cancelled = sum(
            1 for o in self.orders.values() if o.status == OrderStatus.CANCELLED
        )
        pending = len(self._pending_ids)
        print(
            f"[t={self.time:6.1f}] drivers={self.num_drivers} "
            f"served={served} cancelled={cancelled} pending={pending}"
        )