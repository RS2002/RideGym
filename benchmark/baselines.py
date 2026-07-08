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

from ride_gym.road_network import RoadNetwork
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


class RandomRadiusDispatch:
    """Random (order, driver) matching within a fixed pickup radius.

    The simplest model-based baseline: instead of preferring the *nearest* free
    driver like :class:`NearestDistanceDispatch`, each pending order is matched
    to a RANDOM eligible driver among those within the pickup radius. It shares
    every fairness-relevant mechanism with the nearest baseline -- the same grid
    index, the same metric-aware pickup gate, the same capacity handling, the
    same conflict-free-by-construction guarantee, and the same
    ``max_orders_per_driver`` control over whether a driver may take several
    orders in one step (pooling) -- so the ONLY difference is *how* an eligible
    (order, driver) pair is chosen: uniformly at random rather than by ascending
    distance.

    Radius (units)
    --------------
    The matching radius is exactly the pickup-distance gate: a pair is eligible
    iff its pickup distance (order origin -> driver location) is within the
    threshold, measured under ``pickup_distance_metric`` (see module docstring).
    Unlike the nearest / Hungarian baselines (whose gate defaults to *disabled*),
    this baseline's radius defaults to **1 km**, because "random within a radius"
    is only meaningful with a bounded neighbourhood. Pass an explicit
    ``pickup_distance_threshold`` (km) to widen / tighten it, or ``None`` to
    disable the radius entirely (random over ALL free drivers).

    Determinism
    -----------
    A ``seed`` makes the random matching reproducible across runs; the same seed
    + same scenario yields the same assignments.

    Bids are conflict-free by construction (each order committed at most once),
    so the environment never raises ConflictError.
    """

    # Default matching radius (km) when the config leaves the gate unset. Random
    # matching needs a bounded neighbourhood to be meaningful.
    DEFAULT_RADIUS_KM = 1.0

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
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        pickup_distance_threshold:
            Matching radius in KILOMETRES. ``None`` disables the radius (random
            over all free drivers). See :class:`NearestDistanceDispatch` for the
            gate-metric parameters, which are identical here.
        max_orders_per_driver:
            Max orders a single driver may take in one step (pooling). ``1``
            (default) is one-order-per-step, matching the nearest baseline's
            default; a larger value lets a driver be randomly matched to several
            in-radius orders while capacity allows.
        seed:
            RNG seed for the random matching (reproducibility).
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
        self._rng = np.random.default_rng(seed)
        # {order_id: committed pickup distance} for the recorder.
        self.last_assignment_distances: Dict[int, float] = {}

    @classmethod
    def from_config(
        cls,
        cfg,
        k_nearest: int = 20,
        use_knn: bool = True,
        max_orders_per_driver: int = 1,
        seed: Optional[int] = None,
    ):
        """Build the dispatcher directly from a :class:`BenchmarkConfig`.

        Uses the same road network / gate metric as the benchmark env. The
        matching radius is ``cfg.pickup_distance_threshold`` when set, else the
        class default (:attr:`DEFAULT_RADIUS_KM`, 1 km) -- so this baseline
        always matches within a bounded neighbourhood unless the config
        explicitly requests an unbounded random match (use ``None`` in code for
        unbounded). The ``seed`` defaults to the scenario seed for
        reproducibility.
        """
        from benchmark.config import _make_network

        network = _make_network(cfg)
        area = network.bounds if cfg.network_kind in ("osmnx", "nyc") else cfg.area
        metric, coord_to_km, _thr_network = _gate_params_from_cfg(cfg, area)
        # Radius: config threshold if given, else the 1 km default. Re-derive
        # thr_network for THIS radius (the config helper only scales the config
        # threshold, which may be None here).
        thr_km = cfg.pickup_distance_threshold
        if thr_km is None:
            thr_km = cls.DEFAULT_RADIUS_KM
        is_graph = cfg.network_kind in ("osmnx", "nyc")
        thr_network = thr_km * (1000.0 if is_graph else 1.0)
        return cls(
            network=network,
            area=area,
            k_nearest=k_nearest,
            use_knn=use_knn,
            max_orders_per_driver=max_orders_per_driver,
            pickup_distance_threshold=thr_km,
            pickup_distance_metric=metric,
            coord_to_km=coord_to_km,
            thr_network=thr_network,
            seed=cfg.seed if seed is None else seed,
        )

    def _gate_ok(self, origin: Coord, driver_loc: Coord, d_network: float) -> bool:
        """Whether a pair is within the matching radius (see NearestDistanceDispatch)."""
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

        party_of: Dict[int, int] = {o["order_id"]: o["num_passengers"] for o in pending}
        origin_of: Dict[int, Coord] = {o["order_id"]: o["origin"] for o in pending}
        bids: Dict[int, List[int]] = {did: [] for did in observations}
        assigned_orders = set()
        max_n = self.max_orders_per_driver

        # Collect every in-radius, capacity-feasible (order, driver) pair. The
        # grid index yields each order's nearest free drivers as a candidate
        # set (cheap); the gate then keeps only those within the radius. The
        # committed distance is kept for the recorder, but -- unlike the nearest
        # baseline -- it is NOT used to order the commits.
        pairs: List[Tuple[int, int, float]] = []  # (order_id, driver_id, distance)
        for oid, origin in origin_of.items():
            party = party_of[oid]
            nearest = self._index.nearest(
                origin,
                eff_k,
                distance_fn=dist_fn,
                candidate_filter=lambda d, p=party: free_cap[d] >= p,
            )
            for d, did in nearest:
                if self._gate_ok(origin, driver_loc[did], d):
                    pairs.append((oid, did, d))

        # Random matching: shuffle all eligible pairs, then commit greedily in
        # that random order under the conflict / capacity / per-driver-cap rules.
        # Shuffling the pair list (rather than picking a random driver per order
        # independently) keeps every commit conflict-free while giving each
        # eligible pair an unbiased chance.
        self._rng.shuffle(pairs)
        for oid, did, d in pairs:
            if oid in assigned_orders:
                continue  # order already taken
            if len(bids[did]) >= max_n:
                continue  # driver hit its per-step order cap
            party = party_of[oid]
            if free_cap[did] < party:
                continue  # not enough remaining capacity
            bids[did].append(oid)
            free_cap[did] -= party
            assigned_orders.add(oid)
            self.last_assignment_distances[oid] = d

        return {did: {"orders": oids} for did, oids in bids.items()}


class GaleShapleyDispatch:
    """Gale-Shapley (deferred-acceptance) stable matching dispatcher.

    An online stable-matching baseline in the spirit of the assignment engine
    used at Didi (see Gale & Shapley 1962; Yue et al. 2024). Each decision step
    runs the deferred-acceptance algorithm between the pending orders and the
    free-capacity drivers, using the two-sided preferences below, and commits
    the resulting stable matching.

    Two-sided preferences
    ---------------------
    * **Orders propose to drivers** (order-optimal stable matching), so the
      result is the stable matching most preferred by the riders -- consistent
      with a platform that prioritises rider waiting time.
    * **An order prefers the CLOSEST driver** (ascending pickup distance): the
      nearer the driver, the shorter the rider's expected waiting time, so the
      order's preference list over drivers is sorted by ascending road-network
      pickup distance.
    * **A driver prefers the HIGHER-PRICED order** (descending price): a
      driver's earnings scale with the order's fare, so it prefers the order
      that pays most. The price is taken proportional to the trip distance
      (origin -> destination, road-network) times the passenger count::

          price(order) = trip_distance_km * num_passengers

      matching the paper's "price proportional to distance and passenger
      count". Ties are broken by ascending pickup distance (a closer rider is
      preferred at equal price), then by order id for determinism.

    Shared mechanisms (fair comparison)
    -----------------------------------
    Like the other baselines it uses the same :class:`GridIndex` k-NN candidate
    set, the same metric-aware pickup gate, the same capacity handling, and the
    same ``max_orders_per_driver`` control (a driver holds up to that many
    orders, keeping its most-preferred proposals and rejecting the rest while
    capacity allows). Bids are conflict-free by construction (deferred
    acceptance never double-books an order), so the env never raises
    ConflictError.
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
        """See :class:`NearestDistanceDispatch` for the shared parameters.

        Parameters
        ----------
        max_orders_per_driver:
            Number of orders a driver may hold simultaneously in the stable
            matching (its "quota"). ``1`` (default) is classic one-to-one
            deferred acceptance; a larger quota lets a driver hold several of
            its top-preferred in-radius orders while capacity allows (pooling).
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
            max_orders_per_driver=max_orders_per_driver,
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

        party_of: Dict[int, int] = {o["order_id"]: o["num_passengers"] for o in pending}
        origin_of: Dict[int, Coord] = {o["order_id"]: o["origin"] for o in pending}
        dest_of: Dict[int, Coord] = {o["order_id"]: o["destination"] for o in pending}

        # Order fare / price = trip_distance * party (proportional to distance
        # and passenger count). Higher price is more preferred by drivers.
        price_of: Dict[int, float] = {}
        for oid in origin_of:
            trip = dist_fn(origin_of[oid], dest_of[oid])
            price_of[oid] = trip * party_of[oid]

        # --- Build each order's preference list over drivers ---------------
        # Candidate drivers come from the shared k-NN + gate; the order prefers
        # the CLOSEST driver first (ascending pickup distance -> shortest wait).
        # ``pref[oid]`` is a list of (driver_id, pickup_distance) sorted by
        # ascending distance; ``pickup[(oid, did)]`` caches the distance.
        pref: Dict[int, List[Tuple[int, float]]] = {}
        pickup: Dict[Tuple[int, int], float] = {}
        for oid, origin in origin_of.items():
            party = party_of[oid]
            nearest = self._index.nearest(
                origin,
                eff_k,
                distance_fn=dist_fn,
                candidate_filter=lambda d, p=party: free_cap[d] >= p,
            )
            cand: List[Tuple[int, float]] = []
            for d, did in nearest:
                if not self._gate_ok(origin, driver_loc[did], d):
                    continue  # outside the pickup radius -> ineligible
                cand.append((did, d))
                pickup[(oid, did)] = d
            # Ascending pickup distance: nearest driver is most preferred.
            cand.sort(key=lambda t: t[1])
            pref[oid] = cand

        # --- Deferred acceptance (orders propose) -------------------------
        # Each order walks down its preference list proposing to drivers. A
        # driver tentatively holds up to (quota, remaining capacity) proposals,
        # keeping the highest-priced ones and rejecting the rest; rejected
        # orders propose to their next-preferred driver. Iterates until every
        # order is held or has exhausted its list.
        max_n = self.max_orders_per_driver
        # next pointer into each order's preference list.
        next_idx: Dict[int, int] = {oid: 0 for oid in pref}
        # tentative holds: {driver_id: set of held order ids}.
        held: Dict[int, set] = {did: set() for did in observations}
        # remaining free capacity as we tentatively fill each driver.
        rem_cap: Dict[int, int] = dict(free_cap)

        # Orders that still want to propose (have a next choice, not held).
        free_orders = [oid for oid in pref if pref[oid]]

        def _driver_can_take(did: int, party: int) -> bool:
            """Driver has quota slots left AND enough remaining capacity."""
            return len(held[did]) < max_n and rem_cap[did] >= party

        while free_orders:
            oid = free_orders.pop()
            plist = pref[oid]
            party = party_of[oid]
            placed = False
            # Walk down the order's remaining preferences until placed / exhausted.
            while next_idx[oid] < len(plist):
                did, _d = plist[next_idx[oid]]
                next_idx[oid] += 1
                if _driver_can_take(did, party):
                    # Free slot: driver tentatively accepts.
                    held[did].add(oid)
                    rem_cap[did] -= party
                    placed = True
                    break
                # Driver full: does this order out-rank (higher price, tie ->
                # closer) its currently-held least-preferred order that this
                # order could DISPLACE while respecting capacity?
                # Find the held order the driver likes LEAST.
                worst = min(
                    held[did],
                    key=lambda h: (price_of[h], -pickup.get((h, did), 0.0), -h),
                )
                # The proposing order is preferred iff it pays more (tie: closer
                # pickup, then smaller id). Only displace if freeing the worst
                # order yields enough capacity for this one.
                better = (
                    price_of[oid],
                    -pickup.get((oid, did), 0.0),
                    -oid,
                ) > (
                    price_of[worst],
                    -pickup.get((worst, did), 0.0),
                    -worst,
                )
                if better and rem_cap[did] + party_of[worst] >= party:
                    # Evict the worst held order, admit this one.
                    held[did].discard(worst)
                    rem_cap[did] += party_of[worst]
                    held[did].add(oid)
                    rem_cap[did] -= party
                    placed = True
                    # The evicted order becomes free again to re-propose.
                    if next_idx[worst] < len(pref[worst]):
                        free_orders.append(worst)
                    break
                # else: driver rejects; try this order's next preference.
            # If not placed and list exhausted, the order stays pending.
            del placed  # (documentation-only local)

        # --- Emit the committed stable matching ---------------------------
        bids: Dict[int, List[int]] = {did: [] for did in observations}
        for did, oids in held.items():
            for oid in oids:
                bids[did].append(oid)
                self.last_assignment_distances[oid] = pickup[(oid, did)]

        return {did: {"orders": oids} for did, oids in bids.items()}