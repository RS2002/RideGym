"""Order generation strategies.

Orders can be supplied from historical data or generated procedurally. All
generators yield a pre-sorted (by ``request_time``) list of :class:`Order`
objects at reset time; the environment injects each order into the pending pool
when the simulation clock reaches its ``request_time``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ridepool_sim.entities import Order

Area = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


class OrderGenerator(ABC):
    """Abstract order source."""

    @abstractmethod
    def generate(self) -> List[Order]:
        """Return all orders for one episode, sorted by ``request_time``."""
        raise NotImplementedError


class RandomOrderGenerator(OrderGenerator):
    """Procedurally generate orders over the service area and horizon.

    Parameters
    ----------
    area:
        Service-area bounds ``(xmin, ymin, xmax, ymax)``.
    horizon:
        Total simulation duration in minutes.
    num_orders:
        Total number of orders to generate.
    arrival:
        Temporal distribution of request times: ``"uniform"`` spreads orders
        uniformly over the horizon; ``"poisson"`` draws inter-arrival times
        from an exponential distribution (a Poisson process); ``"peak"`` biases
        arrivals toward configurable peak centres.
        max_party_size:
        Maximum passengers per order (party size is uniform in 1..this) when
        ``fixed_party_size`` is ``None``.
    fixed_party_size:
        If set, every order has exactly this many passengers (overrides the
        random 1..``max_party_size`` draw). ``None`` (default) preserves the
        random behaviour.
    peak_centers:
        For ``arrival="peak"``: minute offsets of demand peaks.
    peak_std:
        For ``arrival="peak"``: standard deviation (minutes) of each peak.
    rng:
        Optional ``numpy`` random generator / seed for reproducibility.
    """

    def __init__(
        self,
        area: Area,
        horizon: float,
        num_orders: int,
                arrival: str = "uniform",
        max_party_size: int = 1,
        fixed_party_size: Optional[int] = None,
        peak_centers: Optional[Sequence[float]] = None,
        peak_std: float = 30.0,
        rng=None,
    ):
        self.area = area
        self.horizon = float(horizon)
        self.num_orders = int(num_orders)
        self.arrival = arrival
        self.max_party_size = int(max_party_size)
        self.fixed_party_size = (
            int(fixed_party_size) if fixed_party_size is not None else None
        )
        self.peak_centers = list(peak_centers) if peak_centers else [horizon / 2.0]
        self.peak_std = float(peak_std)
        self._rng = np.random.default_rng(rng)

    def reseed(self, rng) -> None:
        """Reset the internal RNG (used by ``env.reset(seed=...)``)."""
        self._rng = np.random.default_rng(rng)

    def _request_times(self) -> np.ndarray:
        n, h = self.num_orders, self.horizon
        if self.arrival == "uniform":
            times = self._rng.uniform(0.0, h, size=n)
        elif self.arrival == "poisson":
            # Inter-arrival ~ Exponential(mean = h / n); cumulative, clipped.
            gaps = self._rng.exponential(scale=h / max(n, 1), size=n)
            times = np.cumsum(gaps)
            times = np.clip(times, 0.0, h)
        elif self.arrival == "peak":
            centers = self._rng.choice(self.peak_centers, size=n)
            times = self._rng.normal(loc=centers, scale=self.peak_std)
            times = np.clip(times, 0.0, h)
        else:
            raise ValueError(f"Unknown arrival mode: {self.arrival!r}")
        return np.sort(times)

    def generate(self) -> List[Order]:
        xmin, ymin, xmax, ymax = self.area
        times = self._request_times()
        orders: List[Order] = []
        for i in range(self.num_orders):
            origin = (
                float(self._rng.uniform(xmin, xmax)),
                float(self._rng.uniform(ymin, ymax)),
            )
            destination = (
                float(self._rng.uniform(xmin, xmax)),
                float(self._rng.uniform(ymin, ymax)),
            )
                        # Always draw to keep the RNG stream identical regardless of
            # fixed_party_size, so the spatial/temporal order layout is
            # decoupled from the party-size choice. Override only the value.
            drawn = int(self._rng.integers(1, self.max_party_size + 1))
            party = (
                self.fixed_party_size
                if self.fixed_party_size is not None
                else drawn
            )
            orders.append(
                Order(
                    order_id=i,
                    origin=origin,
                    destination=destination,
                    request_time=float(times[i]),
                    num_passengers=party,
                )
            )
        return orders


class OSMnxOrderGenerator(OrderGenerator):
    """Generate orders whose origins/destinations lie on a real road network.

    Spatially, each order's origin and destination are sampled uniformly from
    the graph's nodes (via :meth:`OSMnxNetwork.random_node_coord`), guaranteeing
    every endpoint is exactly on the network and mutually reachable (the cached
    graph is the largest strongly-connected component). Temporally, request
    times follow the same ``uniform`` / ``poisson`` / ``peak`` distributions as
    :class:`RandomOrderGenerator`.

    Parameters
    ----------
    network:
        An :class:`~ridepool_sim.osmnx_network.OSMnxNetwork` providing the node
        set to sample real ``(lon, lat)`` coordinates from.
    horizon:
        Total simulation duration in minutes.
    num_orders:
        Total number of orders to generate.
    arrival / peak_centers / peak_std:
        Temporal-distribution controls, identical to RandomOrderGenerator.
    max_party_size / fixed_party_size:
        Party-size controls, identical to RandomOrderGenerator.
    rng:
        Optional numpy random generator / seed for reproducibility.
    """

    def __init__(
        self,
        network,
        horizon: float,
        num_orders: int,
        arrival: str = "uniform",
        max_party_size: int = 1,
        fixed_party_size: Optional[int] = None,
        peak_centers: Optional[Sequence[float]] = None,
        peak_std: float = 30.0,
        rng=None,
    ):
        self.network = network
        self.horizon = float(horizon)
        self.num_orders = int(num_orders)
        self.arrival = arrival
        self.max_party_size = int(max_party_size)
        self.fixed_party_size = (
            int(fixed_party_size) if fixed_party_size is not None else None
        )
        self.peak_centers = list(peak_centers) if peak_centers else [horizon / 2.0]
        self.peak_std = float(peak_std)
        self._rng = np.random.default_rng(rng)

    def reseed(self, rng) -> None:
        """Reset the internal RNG (used by ``env.reset(seed=...)``)."""
        self._rng = np.random.default_rng(rng)

    def _request_times(self) -> np.ndarray:
        n, h = self.num_orders, self.horizon
        if self.arrival == "uniform":
            times = self._rng.uniform(0.0, h, size=n)
        elif self.arrival == "poisson":
            gaps = self._rng.exponential(scale=h / max(n, 1), size=n)
            times = np.cumsum(gaps)
            times = np.clip(times, 0.0, h)
        elif self.arrival == "peak":
            centers = self._rng.choice(self.peak_centers, size=n)
            times = self._rng.normal(loc=centers, scale=self.peak_std)
            times = np.clip(times, 0.0, h)
        else:
            raise ValueError(f"Unknown arrival mode: {self.arrival!r}")
        return np.sort(times)

    def generate(self) -> List[Order]:
        times = self._request_times()
        orders: List[Order] = []
        for i in range(self.num_orders):
            origin = self.network.random_node_coord(self._rng)
            destination = self.network.random_node_coord(self._rng)
            # Always draw to keep the RNG stream identical regardless of
            # fixed_party_size (matches RandomOrderGenerator's decoupling).
            drawn = int(self._rng.integers(1, self.max_party_size + 1))
            party = (
                self.fixed_party_size
                if self.fixed_party_size is not None
                else drawn
            )
            orders.append(
                Order(
                    order_id=i,
                    origin=origin,
                    destination=destination,
                    request_time=float(times[i]),
                    num_passengers=party,
                )
            )
        return orders


class DataFrameOrderGenerator(OrderGenerator):
    """Build orders from a pandas DataFrame (e.g. NYC Taxi, DiDi GAIA).

    The frame must contain the columns named by ``columns`` mapping. Defaults
    expect: ``origin_x, origin_y, dest_x, dest_y, request_time`` and an optional
    ``num_passengers`` (defaults to 1 when absent).
    """

    DEFAULT_COLUMNS = {
        "origin_x": "origin_x",
        "origin_y": "origin_y",
        "dest_x": "dest_x",
        "dest_y": "dest_y",
        "request_time": "request_time",
        "num_passengers": "num_passengers",
        }

    def __init__(self, dataframe, columns: Optional[dict] = None):
        self.df = dataframe
        self.columns = {**self.DEFAULT_COLUMNS, **(columns or {})}

    def generate(self) -> List[Order]:
        c = self.columns
        df = self.df.sort_values(c["request_time"]).reset_index(drop=True)
        has_party = c["num_passengers"] in df.columns
        orders: List[Order] = []
        for i, row in df.iterrows():
            party = int(row[c["num_passengers"]]) if has_party else 1
            orders.append(
                Order(
                    order_id=int(i),
                    origin=(float(row[c["origin_x"]]), float(row[c["origin_y"]])),
                    destination=(float(row[c["dest_x"]]), float(row[c["dest_y"]])),
                    request_time=float(row[c["request_time"]]),
                    num_passengers=party,
                )
            )
        return orders


class NYCOrderGenerator(OrderGenerator):
    """Build orders from a preprocessed NYC FHVHV order file, snapped to a graph.

    The order file (produced by ``data/nyc/preprocess_orders.py``) already holds
    real trips as ``origin_x/origin_y/dest_x/dest_y`` (zone-centroid lon/lat),
    ``request_time`` (minutes from the episode start) and ``num_passengers``.
    Because those coordinates are zone CENTROIDS they need not sit exactly on a
    road node, so each endpoint is **snapped to its nearest network node** and
    replaced by that node's coordinate. This guarantees every origin/destination
    is exactly on the graph and mutually reachable (the cached graph is the
    largest strongly-connected component), matching the contract the graph-mode
    movement model in :class:`RidePoolEnv` relies on.

    Unlike the random generators this source is **deterministic**: the trips,
    their times and their party sizes all come from the historical file, so
    every episode replays the same real demand. ``reseed`` is accepted (so
    ``env.reset(seed=...)`` works uniformly) but is a no-op, since there is no
    randomness to reseed.

    Parameters
    ----------
    network:
        An :class:`~ridepool_sim.osmnx_network.OSMnxNetwork` (or any network
        exposing ``snap`` + ``node_coord``) covering the order region; used to
        snap each endpoint onto a real road node.
    order_path:
        Path to the preprocessed order parquet/csv. Parquet is read when the
        suffix is ``.parquet``; anything else is read as CSV.
    horizon:
        Episode duration in minutes. Orders whose ``request_time`` is at/after
        the horizon are dropped (they could never be injected).
    limit:
        Optional cap on the number of orders (earliest-by-request_time kept).
        ``None`` (default) keeps them all.
    """

    def __init__(
        self,
        network,
        order_path: str,
        horizon: float,
        limit: Optional[int] = None,
    ):
        self.network = network
        self.order_path = order_path
        self.horizon = float(horizon)
        self.limit = int(limit) if limit is not None else None

    def reseed(self, rng) -> None:
        """No-op: historical demand is deterministic (kept for a uniform API)."""
        return None

    def _load_df(self):
        import pandas as pd

        if self.order_path.endswith(".parquet"):
            return pd.read_parquet(self.order_path)
        return pd.read_csv(self.order_path)

    def generate(self) -> List[Order]:
        df = self._load_df()
        # Keep only orders that can actually be injected within the horizon.
        df = df[df["request_time"] < self.horizon]
        df = df.sort_values("request_time").reset_index(drop=True)
        if self.limit is not None:
            df = df.iloc[: self.limit]

        has_party = "num_passengers" in df.columns
        snap = self.network.snap
        node_coord = self.network.node_coord

        # Cache snaps so repeated zone centroids (there are only ~60 zones, so
        # the same few coordinates recur thousands of times) snap once each.
        snap_cache: dict = {}

        def on_node(x: float, y: float):
            key = (x, y)
            idx = snap_cache.get(key)
            if idx is None:
                idx = snap((x, y))
                snap_cache[key] = idx
            return node_coord(idx)

        orders: List[Order] = []
        for i, row in df.iterrows():
            party = int(row["num_passengers"]) if has_party else 1
            origin = on_node(float(row["origin_x"]), float(row["origin_y"]))
            destination = on_node(float(row["dest_x"]), float(row["dest_y"]))
            orders.append(
                Order(
                    order_id=int(i),
                    origin=origin,
                    destination=destination,
                    request_time=float(row["request_time"]),
                    num_passengers=party,
                )
            )
        return orders