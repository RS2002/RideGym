"""Baseline dispatch algorithms for the benchmark.

All baselines are *centralised* one-shot matchers run each decision step. They
consume the per-driver observation dict from :class:`RidePoolEnv` and return a
conflict-free ``{driver_id: {"orders": [...]}}`` action mapping (each pending
order is bid on by at most one driver, satisfying the env's strict conflict
rule).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ridepool_sim.road_network import RoadNetwork
from benchmark.spatial import GridIndex

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]


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
       still remain is matched against those remaining drivers exactly. This
       removes the artificial under-assignment that pure k-NN truncation could
       otherwise cause; the residual set is typically small so this is cheap.

    Bids are conflict-free by construction (each order committed at most once),
    so the environment never raises ConflictError.
    """

    def __init__(
        self,
        network: RoadNetwork,
        area: Area,
                cell_size: float = None,
        k_nearest: int = 20,
        use_knn: bool = True,
        max_orders_per_driver: int = 1,
    ):
        """
        Parameters
        ----------
        network:
            Road network used to measure pickup distance (order origin ->
            driver location).
        area:
            Service-area bounds, needed to build the spatial index.
        cell_size:
            Grid cell side length in coordinate units. Defaults to one per-step
            travel distance (``network.speed``), keeping the nearest free driver
            within a few rings.
        k_nearest:
            Number of nearest free-capacity candidate drivers retrieved per
            order in the fast pass. Larger values approach exact matching at
            higher cost; the fallback pass guarantees no order is dropped while
            capacity remains regardless of this value.
        max_orders_per_driver:
            Maximum number of orders a single driver may be assigned in ONE
            step. Defaults to 1, making the greedy baseline one-to-one per step
            (directly comparable to :class:`HungarianDispatch`); further pooling
            then happens over subsequent steps as the env re-exposes the driver
            with remaining capacity. Set higher (e.g. the driver capacity) to
            allow multi-order pooling within a single step, still bounded by the
            driver's free capacity. Must be >= 1.
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
        self._index = GridIndex(area, self.cell_size)
        # {order_id: committed pickup distance (km)} for the recorder.
        self.last_assignment_distances: Dict[int, float] = {}

    @classmethod
    def from_config(cls, cfg, k_nearest: int = 20, use_knn: bool = True, max_orders_per_driver: int = 1):
        """Build the dispatcher directly from a :class:`BenchmarkConfig`.

        Uses the same road network the benchmark env is built with, so pickup
                distances are measured under the scenario's metric.
        """
        from benchmark.config import _make_network

        network = _make_network(cfg)
        area = network.bounds if cfg.network_kind in ("osmnx", "nyc") else cfg.area
        return cls(
            network=network,
            area=area,
            k_nearest=k_nearest,
            use_knn=use_knn,
            max_orders_per_driver=max_orders_per_driver,
        )

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
        # Per-step cap on how many orders one driver may be assigned. This is an
        # additional constraint ON TOP of capacity: even if a driver could fit
        # more passengers, it accepts at most ``max_n`` orders this step (the
        # remaining demand is served on later steps). ``len(bids[did])`` is the
        # number already committed to that driver this step.
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
        # Eligible fallback drivers still have capacity AND room under the
        # per-step order cap.
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

    Unlike :class:`NearestDistanceDispatch` (which commits greedily, closest
    pair first), this solves the assignment problem exactly each step via
    ``scipy.optimize.linear_sum_assignment`` (Jonker-Volgenant), minimising the
    **total** pickup distance over all matched (order, driver) pairs. It is the
    natural optimal-matching counterpart of the greedy nearest baseline.

    Design choices (kept directly comparable to the nearest baseline):

    * **One-to-one per step**: each free-capacity driver is matched to at most
       one order this step; further pooling happens on subsequent steps. This
       is the clean 'optimal bipartite matching' semantics.
    * **Candidate pruning**: the same :class:`GridIndex` k-NN candidate set is
       used, so only nearby (order, driver) pairs enter the cost matrix. Non
       candidate entries are set to a large sentinel cost (INF) and any match
       landing on a sentinel is discarded, so an order with no free nearby
       driver simply stays pending.

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
    ):
        """See :class:`NearestDistanceDispatch` for parameter meanings; the
        candidate set is built identically for a fair comparison."""
        self.network = network
        self.area = area
        self.cell_size = cell_size if cell_size else max(network.speed, 1e-6)
        self.k_nearest = int(k_nearest)
        self.use_knn = bool(use_knn)
        self._index = GridIndex(area, self.cell_size)
        self.last_assignment_distances: Dict[int, float] = {}

    @classmethod
    def from_config(cls, cfg, k_nearest: int = 20, use_knn: bool = True):
        """Build directly from a :class:`BenchmarkConfig` (same metric as env)."""
        from benchmark.config import _make_network

        network = _make_network(cfg)
        area = network.bounds if cfg.network_kind in ("osmnx", "nyc") else cfg.area
        return cls(
            network=network,
            area=area,
            k_nearest=k_nearest,
            use_knn=use_knn,
        )

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
        cost = np.full((n_rows, n_cols), self._INF, dtype=float)
        for i, oid in enumerate(order_ids):
            for did, d in cand[oid].items():
                cost[i, col_of[did]] = d

        # Optimal one-to-one assignment minimising total pickup distance.
        row_ind, col_ind = linear_sum_assignment(cost)

        for i, j in zip(row_ind, col_ind):
            d = cost[i, j]
            if d >= self._INF / 2:
                continue  # sentinel: no real candidate -> order stays pending
            oid = order_ids[i]
            did = driver_index[j]
            # Capacity is guaranteed by candidate_filter (party <= free_cap).
            bids[did].append(oid)
            self.last_assignment_distances[oid] = d

        return {did: {"orders": oids} for did, oids in bids.items()}