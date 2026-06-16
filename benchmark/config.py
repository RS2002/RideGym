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

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

from ridepool_sim.env import RidePoolEnv
from ridepool_sim.order_generator import RandomOrderGenerator
from ridepool_sim.road_network import RoadNetwork, ManhattanNetwork

Area = Tuple[float, float, float, float]


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
    num_drivers: int = 1000
    num_orders: int = 15000
    horizon: float = 60.0
    dt: float = 1.0
    speed_kmh: float = 60.0
    driver_capacity: int = 3
    order_timeout: Optional[float] = 5.0
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


def make_benchmark_env(cfg: Optional[BenchmarkConfig] = None) -> RidePoolEnv:
    """Construct the standard benchmark :class:`RidePoolEnv` from a config.

    Relocation is effectively disabled by the baselines (they never emit a
    relocate action); the relocation grid is left at the env default but unused.
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
        # area is the graph's geographic extent; orders are loaded from the
        # preprocessed file and snapped onto the SAME network instance (shared
        # matrices / snap cache). Demand is deterministic (no num_orders /
        # arrival knobs -- the file fixes which trips occur and when).
        from ridepool_sim.order_generator import NYCOrderGenerator
        from data.nyc.preprocess_orders import DEFAULT_OUT as NYC_ORDERS

        area = network.bounds
        order_gen = NYCOrderGenerator(
            network=network,
            order_path=cfg.nyc_order_path or NYC_ORDERS,
            horizon=cfg.horizon,
            limit=cfg.nyc_order_limit,
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
        seed=cfg.seed,
    )
    return env