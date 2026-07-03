"""Baseline dispatch algorithms for the benchmark.

All baselines are *centralised* one-shot matchers run each decision step. They
consume the per-driver observation dict from :class:`RidePoolEnv` and return a
conflict-free ``{driver_id: {"orders": [...]}}`` action mapping (each pending
order is bid on by at most one driver, satisfying the env's strict conflict
rule).

Pickup-distance gate (units / metric)
-------------------------------------
The matching COST (and greedy ordering) is always the true road-network pickup
distance -- that is the definition of the nearest / Hungarian baselines and is
never changed. The *gate* that decides whether a (driver, order) pair is even
eligible, however, follows ``cfg.pickup_distance_metric`` so it is identical to
the learning methods' gate, keeping the comparison fair:

* ``"euclidean"`` (default): a pair is eligible iff the straight-line distance
  (origin -> driver location), in KILOMETRES, is within the threshold. On the
  abstract km scenarios coordinates are already km; on graph scenarios (osmnx /
  nyc) coordinates are ``(lon, lat)`` degrees and are scaled to km with a
  lat-linear correction ``(111*cos(lat0), 111)`` at the area-centre latitude.
* ``"network"``: a pair is eligible iff the road-network distance is within the
  threshold. ``network.distance`` returns metres on graph scenarios, so the km
  threshold is scaled to metres there.

The threshold itself (``cfg.pickup_distance_threshold``) is always in km.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ridepool_sim.road_network import RoadNetwork
from benchmark.spatial import GridIndex

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]


def _gate_params_from_cfg(cfg, area: Area):
    """Derive the pickup-distance gate parameters from a BenchmarkConfig.

    Returns ``(metric, coord_to_km, thr_network)`` where:

    * ``metric`` is ``"euclidean"`` or ``"network"``;
    * ``coord_to_km = (kx, ky)`` scales a coordinate delta to kilometres for the
      Euclidean gate ((1, 1) on abstract scenarios; lat-linear on graph ones);
    * ``thr_network`` is the threshold expressed in the ROAD-NETWORK metric's
      units (metres on graph scenarios, km on abstract ones), or ``None`` when
      the gate is disabled. Used only by the ``"network"`` metric.
    """
    metric = getattr(cfg, "pickup_distance_metric", "euclidean")
    is_graph = cfg.network_kind in ("osmnx", "nyc")
    if is_graph:
        lat0 = 0.5 * (area[1] + area[3])
        coord_to_km = (111.0 * float(np.cos(np.radians(lat0))), 111.0)
    else:
        coord_to_km = (1.0, 1.0)
    thr_km = cfg.pickup_distance_threshold
    thr_network = (
        None if thr_km is None else thr_km * (1000.0 if is_graph else 1.0)
    )
    return metric, coord_to_km, thr_network


class NearestDistanceDispatch:
    """Greedy nearest-(pickup)-distance assignment, accelerated by a grid index.

    Each step:

    1. Build a :class:`GridIndex` over current driver locations (O(drivers)).
    2. For each pending order, query the ``k`` nearest *free-capacity* drivers
       to form a small candidate set, avoiding the O(orders * drivers) full
       pairing.
    3. Greedily commit (order, driver) pairs in ascending pickup distance,
       consuming each order once and decrementing driver free capacity. A
       driver may take several nearby orders while capacity allows (pooling).
    4. **Fallback pass**: any order left unassigned while free-capacity drivers
       still remain is matched against those remaining drivers exactly.

    Bids are conflict-free by construction (each order committed at most once),
    so the environment never raises ConflictError.

    The eligibility gate follows ``pickup_distance_metric`` (see module docstring)
    while the greedy ordering / committed distance is always the road-network
    pickup distance.
    """

    def __init__(
        self,
        network: RoadNetwork,
        area: Area,
        cell_size: float = None,
        k_nearest: int = 20,
        use_knn: bool = True,
        max_orders_per_driver: int = 1,
        pickup_distance_threshold: float = None,
        pickup_distance_metric: str = "euclidean",
        coord_to_km: Tuple[float, float] = (1.0, 1.0),
        thr_network: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        pickup_distance_threshold:
            Gate threshold in KILOMETRES (``None`` disables the gate).
        pickup_distance_metric:
            ``"euclidean"`` (default) or ``"network"`` -- which distance the gate
            compares against (see module docstring). The matching cost is always
            the road-network distance regardless of this.
        coord_to_km:
            ``(kx, ky)`` scaling a coordinate delta to km for the Euclidean gate.
        thr_network:
            Threshold pre-scaled to the road-network metric's units (metres on
            graph scenarios), used only by the ``"network"`` gate. ``None``
            disables the gate.
        """
        if max_orders_per_driver < 1:
            raise ValueError(
                f"max_orders_per_driver must be >= 1, got {max_orders_per_driver}"
            )
        self.network = network
        self.area = area
        self.cell_size = cell_size if cell_size else max(network.speed, 1e-6)
        self.k_nearest = int(k_nearest)
        self.use_knn = bool(use_knn)
        self.max_orders_per_driver = int(max_orders_per_driver)
        self.pickup_distance_threshold = pickup_distance_threshold
        self.pickup_distance_metric = pickup_distance_metric
        self.coord_to_km = coord_to_km
        self.thr_network = thr_network
        self._index = GridIndex(area, self.cell_size)
        # {order_id: committed pickup distance} for the recorder.
        self.last_assignment_distances: Dict[int, float] = {}

    @classmethod
    def from_config(cls, cfg, k_nearest: int = 20, use_knn: bool = True, max_orders_per_driver: int = 1):
        """Build the dispatcher directly from a :class:`BenchmarkConfig`.

        Uses the same road network the benchmark env is built with, so pickup
        distances are measured under the scenario's metric. The gate metric and
        threshold are read from ``cfg`` (km threshold; metric-aware gate).
        """
        from benchmark.config import _make_network

        network = _make_network(cfg)
        area = network.bounds if cfg.network_kind in ("osmnx", "nyc") else cfg.area
        metric, coord_to_km, thr_network = _gate_params_from_cfg(cfg, area)
        return cls(
            network=network,
            area=area,
            k_nearest=k_nearest,
            use_knn=use_knn,
            max_orders_per_driver=max_orders_per_driver,
            pickup_distance_threshold=cfg.pickup_distance_threshold,
            pickup_distance_metric=metric,
            coord_to_km=coord_to_km,
            thr_network=thr_network,
        )

    def _gate_ok(self, origin: Coord, driver_loc: Coord, d_network: float) -> bool:
        """Whether a (order origin, driver location) pair passes the gate.

        ``d_network`` is the already-computed road-network distance for the pair
        (reused for the ``"network"`` metric so no extra graph query is paid).
        """
        thr_km = self.pickup_distance_threshold
        if thr_km is None:
            return True
        if self.pickup_distance_metric == "network":
            thr = self.thr_network if self.thr_network is not None else thr_km
            return d_network <= thr
        # Euclidean (straight-line) km distance with the coord->km scaling.
        kx, ky = self.coord_to_km
        dx = (origin[0] - driver_loc[0]) * kx
        dy = (origin[1] - driver_loc[1]) * ky
        return (dx * dx + dy * dy) <= (thr_km * thr_km)

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        self.last_assignment_distances = {}
        if not observations:
            return {}

        any_obs = next(iter(observations.values()))
        pending = any_obs["pending_orders"]
        if not pending:
            return {did: {"orders": []} for did in observations}

        # Remaining free capacity and location per driver for this step.
        free_cap: Dict[int, int] = {}
        driver_loc: Dict[int, Coord] = {}
        for did, obs in observations.items():
            s = obs["self"]
            free_cap[did] = s["capacity"] - s["onboard_passengers"]
            driver_loc[did] = s["location"]

        # Build the spatial index over all drivers (cheap, O(drivers)).
        self._index.build(driver_loc)
        dist_fn = self.network.distance
        eff_k = self.k_nearest if self.use_knn else len(observations)

        party_of: Dict[int, int] = {o["order_id"]: o["num_passengers"] for o in pending}
        origin_of: Dict[int, Coord] = {o["order_id"]: o["origin"] for o in pending}
        bids: Dict[int, List[int]] = {did: [] for did in observations}
        assigned_orders = set()
        max_n = self.max_orders_per_driver

        # --- Fast pass: k-NN candidate pairs, greedy by ascending distance. ---
        pairs: List[Tuple[float, int, int]] = []
        for oid, origin in origin_of.items():
            party = party_of[oid]
            nearest = self._index.nearest(
                origin,
                eff_k,
                distance_fn=dist_fn,
                candidate_filter=lambda d, p=party: free_cap[d] >= p,
            )
            for d, did in nearest:
                pairs.append((d, oid, did))

        pairs.sort(key=lambda t: t[0])
        for d, oid, did in pairs:
            if oid in assigned_orders:
                continue
            if not self._gate_ok(origin_of[oid], driver_loc[did], d):
                continue  # pickup distance gate: too far to serve
            if len(bids[did]) >= max_n:
                continue  # driver already hit its per-step order cap
            party = party_of[oid]
            if free_cap[did] < party:
                continue
            bids[did].append(oid)
            free_cap[did] -= party
            assigned_orders.add(oid)
            self.last_assignment_distances[oid] = d

        # --- Fallback pass: exact match for residual orders vs free drivers. ---
        residual_orders = [oid for oid in origin_of if oid not in assigned_orders]
        free_drivers = [
            did
            for did, c in free_cap.items()
            if c > 0 and len(bids[did]) < max_n
        ]
        if residual_orders and free_drivers:
            fb_pairs: List[Tuple[float, int, int]] = []
            for oid in residual_orders:
                origin = origin_of[oid]
                party = party_of[oid]
                for did in free_drivers:
                    if free_cap[did] >= party:
                        fb_pairs.append((dist_fn(origin, driver_loc[did]), oid, did))
            fb_pairs.sort(key=lambda t: t[0])
            for d, oid, did in fb_pairs:
                if oid in assigned_orders:
                    continue
                if not self._gate_ok(origin_of[oid], driver_loc[did], d):
                    continue  # pickup distance gate: too far to serve
                if len(bids[did]) >= max_n:
                    continue  # respect the per-step order cap in fallback too
                party = party_of[oid]
                if free_cap[did] < party:
                    continue
                bids[did].append(oid)
                free_cap[did] -= party
                assigned_orders.add(oid)
                self.last_assignment_distances[oid] = d

        return {did: {"orders": oids} for did, oids in bids.items()}


class HungarianDispatch:
    """Globally optimal (minimum total pickup distance) one-to-one matching.

    Solves the assignment problem exactly each step via
    ``scipy.optimize.linear_sum_assignment`` (Jonker-Volgenant), minimising the
    **total** road-network pickup distance over all matched pairs. The natural
    optimal-matching counterpart of the greedy nearest baseline.

    * **One-to-one per step**: each free-capacity driver is matched to at most
       one order this step; further pooling happens on subsequent steps.
    * **Candidate pruning**: the same :class:`GridIndex` k-NN candidate set is
       used. Non-candidate / gated-out entries are set to a large sentinel cost
       (INF) and any match landing on a sentinel is discarded, so an order with
       no free eligible driver simply stays pending.
    * **Pickup-distance gate**: follows ``pickup_distance_metric`` exactly like
       :class:`NearestDistanceDispatch`; gated-out pairs are left at INF.

    Resulting bids are conflict-free (each order matched at most once).
    """

    _INF = 1e9

    def __init__(
        self,
        network: RoadNetwork,
        area: Area,
        cell_size: float = None,
        k_nearest: int = 20,
        use_knn: bool = True,
        pickup_distance_threshold: float = None,
        pickup_distance_metric: str = "euclidean",
        coord_to_km: Tuple[float, float] = (1.0, 1.0),
        thr_network: Optional[float] = None,
    ):
        """See :class:`NearestDistanceDispatch` for parameter meanings; the
        candidate set and gate are built identically for a fair comparison."""
        self.network = network
        self.area = area
        self.cell_size = cell_size if cell_size else max(network.speed, 1e-6)
        self.k_nearest = int(k_nearest)
        self.use_knn = bool(use_knn)
        self.pickup_distance_threshold = pickup_distance_threshold
        self.pickup_distance_metric = pickup_distance_metric
        self.coord_to_km = coord_to_km
        self.thr_network = thr_network
        self._index = GridIndex(area, self.cell_size)
        self.last_assignment_distances: Dict[int, float] = {}

    @classmethod
    def from_config(cls, cfg, k_nearest: int = 20, use_knn: bool = True):
        """Build directly from a :class:`BenchmarkConfig` (same metric as env).
        Gate metric and km threshold are read from ``cfg`` (metric-aware gate)."""
        from benchmark.config import _make_network

        network = _make_network(cfg)
        area = network.bounds if cfg.network_kind in ("osmnx", "nyc") else cfg.area
        metric, coord_to_km, thr_network = _gate_params_from_cfg(cfg, area)
        return cls(
            network=network,
            area=area,
            k_nearest=k_nearest,
            use_knn=use_knn,
            pickup_distance_threshold=cfg.pickup_distance_threshold,
            pickup_distance_metric=metric,
            coord_to_km=coord_to_km,
            thr_network=thr_network,
        )

    def _gate_ok(self, origin: Coord, driver_loc: Coord, d_network: float) -> bool:
        """Whether a pair passes the gate (see NearestDistanceDispatch._gate_ok)."""
        thr_km = self.pickup_distance_threshold
        if thr_km is None:
            return True
        if self.pickup_distance_metric == "network":
            thr = self.thr_network if self.thr_network is not None else thr_km
            return d_network <= thr
        kx, ky = self.coord_to_km
        dx = (origin[0] - driver_loc[0]) * kx
        dy = (origin[1] - driver_loc[1]) * ky
        return (dx * dx + dy * dy) <= (thr_km * thr_km)

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        from scipy.optimize import linear_sum_assignment

        self.last_assignment_distances = {}
        if not observations:
            return {}

        any_obs = next(iter(observations.values()))
        pending = any_obs["pending_orders"]
        if not pending:
            return {did: {"orders": []} for did in observations}

        free_cap: Dict[int, int] = {}
        driver_loc: Dict[int, Coord] = {}
        for did, obs in observations.items():
            s = obs["self"]
            free_cap[did] = s["capacity"] - s["onboard_passengers"]
            driver_loc[did] = s["location"]

        self._index.build(driver_loc)
        dist_fn = self.network.distance
        eff_k = self.k_nearest if self.use_knn else len(observations)

        # Gather k-NN candidate drivers per order; collect the involved drivers.
        order_ids = [o["order_id"] for o in pending]
        party_of = {o["order_id"]: o["num_passengers"] for o in pending}
        origin_of = {o["order_id"]: o["origin"] for o in pending}

        # candidate distances: {order_id: {driver_id: distance}}
        cand: Dict[int, Dict[int, float]] = {}
        involved_drivers = set()
        for oid in order_ids:
            party = party_of[oid]
            nearest = self._index.nearest(
                origin_of[oid],
                eff_k,
                distance_fn=dist_fn,
                candidate_filter=lambda d, p=party: free_cap[d] >= p,
            )
            cand[oid] = {did: d for d, did in nearest}
            involved_drivers.update(cand[oid].keys())

        bids: Dict[int, List[int]] = {did: [] for did in observations}
        if not involved_drivers:
            return {did: {"orders": oids} for did, oids in bids.items()}

        driver_index = list(involved_drivers)
        col_of = {did: j for j, did in enumerate(driver_index)}
        n_rows = len(order_ids)
        n_cols = len(driver_index)

        # Build the sparse cost matrix with INF sentinels for non-candidates.
        # Pairs failing the (metric-aware) pickup gate are also left at INF so
        # the solver never matches them (the order then stays pending).
        cost = np.full((n_rows, n_cols), self._INF, dtype=float)
        for i, oid in enumerate(order_ids):
            origin = origin_of[oid]
            for did, d in cand[oid].items():
                if not self._gate_ok(origin, driver_loc[did], d):
                    continue
                cost[i, col_of[did]] = d

        # Optimal one-to-one assignment minimising total pickup distance.
        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            d = cost[i, j]
            if d >= self._INF / 2:
                continue  # sentinel: no real candidate -> order stays pending
            oid = order_ids[i]
            did = driver_index[j]
            bids[did].append(oid)
            self.last_assignment_distances[oid] = d

        return {did: {"orders": oids} for did, oids in bids.items()}