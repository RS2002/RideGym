"""Standard benchmark scenario configuration.

The canonical benchmark used for comparing dispatch baselines:

* Simulation horizon : 1 hour (60 minutes)
* Drivers            : 1000
* Orders             : 10000
* Decision interval  : 1 minute  -> 60 steps
* Driver speed       : 60 km/h   -> 1.0 km / minute
* Driver capacity    : 3
* Passengers / order : exactly 1
* Relocation         : disabled for now (no repositioning)

Coordinate units are kilometres. The service area defaults to a 10 km x 10 km
urban region; ``speed`` is therefore ``60 / 60 = 1.0`` km per minute so a driver
covers 1 km per decision step. All quantities are centralised here so a single
edit re-parameterises every baseline run identically and reproducibly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from ridepool_sim.env import RidePoolEnv
from ridepool_sim.order_generator import RandomOrderGenerator
from ridepool_sim.rewards import DefaultRewardFunction
from ridepool_sim.road_network import RoadNetwork, ManhattanNetwork

Area = Tuple[float, float, float, float]
Coord = Tuple[float, float]


@dataclass
class BenchmarkConfig:
    """Immutable-ish description of the benchmark scenario.

    Attributes
    ----------
    area:
        Service-area bounds in km, ``(xmin, ymin, xmax, ymax)``.
    num_drivers:
        Number of drivers (agents).
    num_orders:
        Total orders over the horizon.
    horizon:
        Total simulation duration in minutes.
    dt:
        Decision interval in minutes.
    speed_kmh:
        Driver speed in km/h (converted to km/min internally).
    driver_capacity:
        Per-driver passenger capacity.
    order_timeout:
        Minutes a pending order may wait before auto-cancellation.
    arrival:
        Temporal demand distribution ("uniform" | "poisson" | "peak").
    network_kind:
        Road-network metric for the benchmark ("manhattan" | "euclidean").
    seed:
        Base RNG seed for reproducibility.
    """

    area: Area = (0.0, 0.0, 10.0, 10.0)
    num_drivers: int = 800
    num_orders: int = 15000
    horizon: float = 60.0
    dt: float = 1.0
    speed_kmh: float = 40.0
    driver_capacity: int = 3
    order_timeout: Optional[float] = 3.0
    passengers_per_order: int = 1
    arrival: str = "uniform"
    network_kind: str = "nyc"
    # Path to a cached OSM graph (used only when network_kind in {"osmnx",
    # "nyc"}). None -> the bundled default region (data/guomao.gpickle).
    osmnx_graph_path: Optional[str] = "data/guomao.gpickle"
    # Path to the cached Manhattan graph (used only when network_kind == "nyc");
    # None -> data/nyc/manhattan.gpickle (built by data/nyc/build_nyc_network).
    nyc_graph_path: Optional[str] = "data/nyc/manhattan.gpickle"
    # Path to the preprocessed NYC order file (used only when
    # network_kind == "nyc"); produced by data/nyc/preprocess_orders.py.
    nyc_order_path: Optional[str] = "data/nyc/orders.parquet"
    # Optional cap on the number of NYC orders loaded (None = all).
    nyc_order_limit: Optional[int] = None

    # --- Multi-window train/val/test split (NYC only) -----------------------
    # When ``nyc_splits_dir`` is set (and network_kind == "nyc"), the env uses
    # the MultiWindowNYCOrderGenerator instead of the single-file
    # NYCOrderGenerator: it draws orders from the pool of time windows under
    # ``<nyc_splits_dir>/<nyc_split>/`` recorded in the split manifest. This is
    # how a single config switches between training (random window each episode)
    # and held-out validation / test windows.
    #   nyc_splits_dir : directory produced by data/nyc/build_splits.py
    #                    (contains manifest.json + train/val/test/ subdirs).
    #                    None (default) -> use the single nyc_order_path file.
    #   nyc_split      : which pool to draw from -- "train" (random window per
    #                    episode) | "val" | "test" (deterministic cyclic).
    nyc_splits_dir: Optional[str] = "data/nyc/splits"
    nyc_split: str = "train"

    # --- Random party size (NYC scenarios) ----------------------------
    # FHVHV records carry no real passenger count, so by default every NYC
    # order is a single passenger. When random_party_size is True (DEFAULT)
    # each order draws a passenger count uniformly in [1, max_party_size]
    # (default 1..3), making pooling non-trivial. Reproducible per episode
    # via the env seed and re-randomised across episodes.
    random_party_size: bool = False
    max_party_size: int = 3

    # --- OD spatial perturbation (NYC scenarios) --------------------------
    # NYC orders are located by taxi-zone CENTROID, so every order in a zone
    # shares the exact same origin/destination and snaps to the same road
    # node -- an unrealistic centroid-collapse artefact. When this radius
    # (kilometres) is > 0, each endpoint is jittered uniformly within a disc
    # of this radius BEFORE snapping, scattering same-zone orders across
    # nearby real nodes. 0.0 (default) keeps the exact-centroid behaviour.
    # Reproducible per episode (env seed) and re-randomised across episodes.
    nyc_perturb_od_radius_km: float = 0.5

    # --- Relocation / region model ------------------------------------------
    # Whether the env builds a relocation region model. The dispatch baselines
    # never emit relocate actions, so this only matters for a relocation-aware
    # policy; building the region model is cheap and harmless otherwise.
    # The region CENTRES are NOT stored on this config -- they are passed
    # explicitly to ``make_benchmark_env(cfg, relocation_centroids=...)`` so the
    # user fully owns how regions are defined (a plain coordinate list, or the
    # ``nyc_zone_centroids`` helper for NYC's real zones). Only the scalar knobs
    # of the region model live here.
    #
    # Whether the env builds a relocation region model at all. Building it is
    # cheap and harmless even for the relocation-free baselines.
    relocation_enabled: bool = False
    # Uniform-grid region model resolution (rows, cols), used when the user
    # passes no explicit centres -> the env falls back to this regular lattice.
    relocation_grid: Tuple[int, int] = (10, 10)
    # Grid-model geometric adjacency: 4 (von Neumann) or 8 (Moore).
    relocation_adjacency: int = 8
    # Centroid-model adjacency: number of spatially-nearest neighbour regions.
    relocation_neighbours_k: int = 8

    # --- Pickup-distance gating ---------------------------------------------
    # When set (coordinate units, i.e. km for the abstract scenarios; the same
    # units as the road network's ``distance``), a driver may only be matched
    # to an order whose straight-line / network pickup distance (order origin ->
    # driver location) is <= this threshold. This caps how long any rider waits
    # to be picked up. ``None`` (default) disables the gate entirely, leaving
    # matching unconstrained by distance. Honoured uniformly by every method
    # (iddqn, iddqn_reposition, bmg-q, mfddqn, nearest, hungarian).
    pickup_distance_threshold: Optional[float] = None
    # Metric used to measure that pickup distance:
    #   "euclidean" (default) -> straight-line distance, computed vectorised in
    #       numpy. Fast: O(N*M) flops with no graph queries, suitable at scale.
    #   "network"             -> the true road-network distance (origin ->
    #       driver location). Exact but SLOW on graph scenarios (osmnx / nyc):
    #       each (driver, order) pair is a shortest-path lookup, so the gate
    #       becomes the dominant per-step cost. Use only when the straight-line
    #       approximation is unacceptable.
    # Ignored when ``pickup_distance_threshold`` is None.
    pickup_distance_metric: str = "euclidean"

    # --- Reward shaping -----------------------------------------------------
    # All DefaultRewardFunction coefficients are exposed here so a single config
    # controls the reward for every method (iddqn, bmg-q, mfddqn, baselines)
    # uniformly and reproducibly. They are threaded into the env's
    # DefaultRewardFunction in make_benchmark_env. Defaults match
    # DefaultRewardFunction's own defaults, so omitting them changes nothing.
    #
    #   assignment_bonus  : fixed positive reward per newly assigned order.
    #   revenue_coef      : fare per order, proportional to its solo service
    #                       time x party size (longer/larger trips earn more).
    #   service_time_coef : penalty per order's predicted end-to-end service
    #                       time (request -> planned drop-off).
    #   detour_coef       : penalty on the SIGNED re-routing impact on already
    #                       committed en-route orders.
    #   empty_move_penalty: penalty for moving while empty (no onboard rider).
    #   idle_penalty      : penalty for staying put with no tasks.
    assignment_bonus: float = 1.0
    revenue_coef: float = 0.01
    service_time_coef: float = 0.04
    detour_coef: float = 0.08
    empty_move_penalty: float = 0
    idle_penalty: float = 0

    # --- Idle (take-no-order) policy ----------------------------------------
    # Whether a driver may ACTIVELY choose to take no order when a legal order
    # is available (RL methods only -- the nearest/hungarian baselines always
    # assign greedily).
    #   True  -> the bipartite matching lets the dummy (no-order) action
    #            compete on its own Q-value, so the agent can decide idling is
    #            best. This can inflate the empty/idle rate if the agent learns
    #            to refuse demand.
    #   False -> idling is only a PASSIVE fallback: a driver is assigned a legal
    #            order whenever one is available, and lands on the dummy only
    #            when it has no assignable legal order (all taken by others, or
    #            none within capacity / candidate set). Mirrored identically in
    #            the Q-target computation. Threaded into every matching-based
    #            method (iddqn, bmg-q, mfddqn).
    allow_idle: bool = False

    seed: int = 0

    @property
    def speed_km_per_min(self) -> float:
        """Driver speed in km per minute (matches coordinate units)."""
        return self.speed_kmh / 60.0

    def to_dict(self) -> dict:
        """Serialisable view of the configuration (for result provenance)."""
        d = asdict(self)
        d["speed_km_per_min"] = self.speed_km_per_min
        return d


def _make_network(cfg: BenchmarkConfig) -> RoadNetwork:
    speed = cfg.speed_km_per_min
    if cfg.network_kind == "manhattan":
        return ManhattanNetwork(speed=speed)
    if cfg.network_kind == "euclidean":
        from ridepool_sim.road_network import EuclideanNetwork

        return EuclideanNetwork(speed=speed)
    if cfg.network_kind == "osmnx":
        # Real road network backed by a cached OSM graph. Distances are in
        # metres and travel times honour per-segment speeds (the scalar
        # ``speed_kmh`` is ignored). Building precomputes the all-pairs matrices
        # once. The graph path can be overridden via ``osmnx_graph_path``.
        from ridepool_sim.osmnx_network import OSMnxNetwork, DEFAULT_GRAPH

        return OSMnxNetwork(
            graph_path=cfg.osmnx_graph_path or DEFAULT_GRAPH,
            speed_kmh=cfg.speed_kmh,
        )
    if cfg.network_kind == "nyc":
        # Real Manhattan road network for the NYC FHVHV scenario. Same backend
        # as "osmnx" but pointed at the cached Manhattan graph; demand comes
        # from real historical trips (see make_benchmark_env).
        from ridepool_sim.osmnx_network import OSMnxNetwork
        from data.nyc.build_nyc_network import DEFAULT_OUT as NYC_GRAPH

        return OSMnxNetwork(
            graph_path=cfg.nyc_graph_path or NYC_GRAPH,
            speed_kmh=cfg.speed_kmh,
        )
    raise ValueError(f"Unknown network_kind: {cfg.network_kind!r}")


def nyc_zone_centroids(area: Optional[Area] = None) -> List[Coord]:
    """Return NYC taxi-zone centroids as ``[(lon, lat), ...]`` region centres.

    A convenience source of *real administrative* region centres for the NYC
    scenario, so the user does not have to hand-list hundreds of points. When
    ``area`` is given, only centres inside that bounding box are returned (drop
    zones outside the modelled service area). Pass the result straight into
    :func:`make_benchmark_env`::

        env = make_benchmark_env(cfg, relocation_centroids=nyc_zone_centroids(cfg.area))
    """
    from data.nyc.zone_centroids import load_zone_centroids

    centroids = load_zone_centroids()
    pts = list(centroids.values())
    if area is not None:
        xmin, ymin, xmax, ymax = area
        pts = [
            (x, y) for (x, y) in pts if xmin <= x <= xmax and ymin <= y <= ymax
        ]
    return [(float(x), float(y)) for (x, y) in pts]


def load_split_window_paths(splits_dir: str, split: str) -> List[str]:
    """Return the window order-file paths for a split from its manifest.

    Reads <splits_dir>/manifest.json (written by data/nyc/build_splits.py)
    and returns the absolute paths of every window order file for ``split``
    ("train"|"val"|"test"). Paths are stored relative to splits_dir.
    """
    import json

    manifest_path = os.path.join(splits_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"split manifest not found: {manifest_path!r}. Run "
            f"`python -m data.nyc.build_splits` first to generate windows."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("splits", {}).get(split, [])
    if not entries:
        raise ValueError(
            f"split {split!r} has no windows in {manifest_path!r}. "
            f"Available: {list(manifest.get(chr(39)+chr(115)+chr(112)+chr(108)+chr(105)+chr(116)+chr(115)+chr(39), {}).keys())}."
        )
    return [os.path.join(splits_dir, e["file"]) for e in entries]


def make_benchmark_env(
    cfg: Optional[BenchmarkConfig] = None,
    relocation_centroids: Optional[List[Coord]] = None,
) -> RidePoolEnv:
    """Construct the standard benchmark :class:`RidePoolEnv` from a config.

    Relocation is effectively disabled by the baselines (they never emit a
    relocate action); the relocation grid is left at the env default but unused.

    Parameters
    ----------
    relocation_centroids:
        User-defined region centres ``[(x, y), ...]``. This is the primary way
        to customise the region partition: the user supplies whatever centres
        they want (or uses :func:`nyc_zone_centroids` for NYC's real zones).
        When ``None``, the env falls back to the uniform-grid region model
        (``cfg.relocation_grid``). Region adjacency for an explicit centre set
        is K-nearest (``cfg.relocation_neighbours_k``).
    """
    cfg = cfg or BenchmarkConfig()

    network = _make_network(cfg)

    if cfg.network_kind == "osmnx":
        # Real network: the service area is the graph's geographic extent and
        # order endpoints are sampled on real nodes (guaranteed reachable). The
        # env is given the SAME network instance so its precomputed matrices /
        # snap cache are shared, not rebuilt.
        from ridepool_sim.order_generator import OSMnxOrderGenerator

        area = network.bounds
        order_gen = OSMnxOrderGenerator(
            network=network,
            horizon=cfg.horizon,
            num_orders=cfg.num_orders,
            arrival=cfg.arrival,
            max_party_size=cfg.passengers_per_order,
            fixed_party_size=cfg.passengers_per_order,
            rng=cfg.seed,
        )
    elif cfg.network_kind == "nyc":
        # Real Manhattan network + real historical FHVHV demand. The service







        # area is the graph's geographic extent; orders are snapped onto the
        # SAME network instance (shared matrices / snap cache).
        area = network.bounds






        if cfg.nyc_splits_dir is not None:
            # Multi-window train/val/test mode: draw orders from the pool of
            # time windows recorded in the split manifest. Training draws a
            # random window per episode; val/test traverse held-out windows
            # deterministically.
            from ridepool_sim.order_generator import MultiWindowNYCOrderGenerator

            window_paths = load_split_window_paths(
                cfg.nyc_splits_dir, cfg.nyc_split
            )
            order_gen = MultiWindowNYCOrderGenerator(
                network=network,
                order_paths=window_paths,
                horizon=cfg.horizon,
                mode=cfg.nyc_split,
                limit=cfg.nyc_order_limit,
                random_party_size=cfg.random_party_size,
                max_party_size=cfg.max_party_size,
                perturb_od_radius_km=cfg.nyc_perturb_od_radius_km,
                rng=cfg.seed,
            )
        else:
            # Single-file mode: deterministic replay of one preprocessed window.
            from ridepool_sim.order_generator import NYCOrderGenerator
            from data.nyc.preprocess_orders import DEFAULT_OUT as NYC_ORDERS

            order_gen = NYCOrderGenerator(
                network=network,
                order_path=cfg.nyc_order_path or NYC_ORDERS,
                horizon=cfg.horizon,
                limit=cfg.nyc_order_limit,
                random_party_size=cfg.random_party_size,
                max_party_size=cfg.max_party_size,
                perturb_od_radius_km=cfg.nyc_perturb_od_radius_km,
                rng=cfg.seed,
            )
    else:
        # Abstract-coordinate scenario (manhattan / euclidean).
        # The core RandomOrderGenerator supports random party sizes in general;
        # this benchmark scenario fixes it via passengers_per_order (default 1).
        area = cfg.area
        order_gen = RandomOrderGenerator(
            area=cfg.area,
            horizon=cfg.horizon,
            num_orders=cfg.num_orders,
            arrival=cfg.arrival,
            max_party_size=cfg.passengers_per_order,
            rng=cfg.seed,
        )

    env = RidePoolEnv(
        area=area,
        num_drivers=cfg.num_drivers,
        driver_capacity=cfg.driver_capacity,
        dt=cfg.dt,
        horizon=cfg.horizon,
        order_timeout=cfg.order_timeout,
        order_generator=order_gen,
        road_network=network,
        reward_function=DefaultRewardFunction(
            assignment_bonus=cfg.assignment_bonus,
            revenue_coef=cfg.revenue_coef,
            service_time_coef=cfg.service_time_coef,
            detour_coef=cfg.detour_coef,
            empty_move_penalty=cfg.empty_move_penalty,
            idle_penalty=cfg.idle_penalty,
        ),
        relocation_centroids=relocation_centroids,
        relocation_grid=cfg.relocation_grid,
        relocation_adjacency=cfg.relocation_adjacency,
        relocation_neighbours_k=cfg.relocation_neighbours_k,
        seed=cfg.seed,
    )
    return env