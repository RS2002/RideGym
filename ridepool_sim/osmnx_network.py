"""Real road-network backend backed by a cached OSMnx graph (constant speed).

Implements the :class:`~ridepool_sim.road_network.RoadNetwork` interface on top
of a real OpenStreetMap drive network (downloaded and cached by
``data/build_network.py``). **Distances** follow the actual street topology
(real detours, one-way streets, etc.); **travel time** uses a single constant
driver speed -- per-segment OSM speed limits are intentionally ignored, so every
driver moves at the same pace and time is simply ``distance / speed``.

Performance design (scheme A -- precomputed all-pairs data)
----------------------------------------------------------
The hot path ``distance(a, b)`` is called on the order of millions of times per
episode, so it must be O(1). At construction we:

* relabel the graph's nodes to contiguous indices ``0..N-1``;
* precompute the dense all-pairs shortest-path **distance** matrix (metres) and
  a **predecessor** matrix once via repeated Dijkstra (length-weighted). For a
  ~900-node region the distance matrix is a few MB, the predecessor matrix is
  an int32 ``N x N`` (~3 MB), both built in ~1-2 s, then free forever;
* map every continuous ``(lon, lat)`` query point to its nearest graph node
  ("snap"), caching the result so repeated queries for the same coordinate are
  free.

With these in place ``distance(a, b)`` is two cached snaps plus one matrix
lookup, and the node-level shortest path between any two nodes is rebuilt in
O(path length) from the predecessor matrix (no per-query Dijkstra), which the
environment's movement step walks along.

Units: distances are in **metres**, travel times in **minutes**. Coordinates are
``(lon, lat)`` to match OSMnx's node ``x`` (=longitude) / ``y`` (=latitude) and
the env's ``(x, y)`` convention.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from ridepool_sim.road_network import RoadNetwork, PathResult

Coord = Tuple[float, float]  # (lon, lat)

DEFAULT_GRAPH = os.path.join(os.path.dirname(__file__), "..", "data", "guomao.gpickle")

# Sentinel for "no predecessor" (the source itself, or unreachable).
_NO_PRED = -1


class OSMnxNetwork(RoadNetwork):
    """Road network over a cached real OSM drive graph, constant driver speed.

    Parameters
    ----------
    graph_path:
        Path to the pickled annotated networkx graph produced by
        ``data/build_network.py``. Defaults to the bundled Guomao region.
    speed_kmh:
        Constant driver speed in km/h applied uniformly to all drivers on all
        roads. Travel time between two points is ``distance / speed`` (OSM
        per-segment speed limits are ignored by design).
    snap_cache:
        If ``True`` (default), nearest-node lookups for query coordinates are
        memoised. Query coordinates recur heavily (a driver's node position, an
        order's fixed origin/destination), so caching removes almost all snap
        cost after warmup.
    cache_matrices:
        If ``True`` (default), the precomputed all-pairs distance + predecessor
        matrices are cached to disk (``<graph_path>.matrices.npz`` by default,
        or ``matrix_cache_path`` if given) after the first build and reloaded on
        subsequent constructions. For large networks (e.g. the ~4900-node
        Manhattan graph) this turns a ~90 s repeated-Dijkstra build into a
        sub-second load. The cache is keyed on the exact node-id ordering, so a
        changed graph automatically invalidates it (a stale cache is never used
        silently).
    matrix_cache_path:
        Explicit path for the matrix cache. ``None`` (default) derives it from
        ``graph_path`` by appending ``.matrices.npz``.
    """

    def __init__(
        self,
        graph_path: str = DEFAULT_GRAPH,
        speed_kmh: float = 60.0,
        snap_cache: bool = True,
        cache_matrices: bool = True,
        matrix_cache_path: Optional[str] = None,
    ):
        with open(graph_path, "rb") as f:
            g: nx.MultiDiGraph = pickle.load(f)
        self.graph = g
        self.speed_kmh = float(speed_kmh)
        self._cache_matrices = bool(cache_matrices)
        self._matrix_cache_path = (
            matrix_cache_path
            if matrix_cache_path is not None
            else graph_path + ".matrices.npz"
        )

        # Contiguous node indexing.
        self._node_ids: List[int] = list(g.nodes)
        self._id_to_idx: Dict[int, int] = {
            nid: i for i, nid in enumerate(self._node_ids)
        }
        n = len(self._node_ids)
        self._n = n

        # Node coordinates as an [N, 2] array of (lon, lat) for vectorised snap.
        self._coords = np.array(
            [(g.nodes[nid]["x"], g.nodes[nid]["y"]) for nid in self._node_ids],
            dtype=np.float64,
        )

        # Precompute (or load from disk) the all-pairs distance matrix (metres)
        # and predecessor matrix (for O(path) node-path reconstruction).
        self._dist_m, self._pred = self._load_or_build_matrices(g)

        # Constant speed in metres per minute: km/h * 1000 / 60.
        speed_m_per_min = self.speed_kmh * 1000.0 / 60.0
        super().__init__(speed=max(speed_m_per_min, 1e-6))

        self._snap_cache_on = bool(snap_cache)
        self._snap_cache: Dict[Coord, int] = {}

    # --------------------------------------------------------------- build
    def _load_or_build_matrices(
        self, g: nx.MultiDiGraph
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the distance + predecessor matrices, using the disk cache.

        Tries to load a previously saved cache whose stored node-id ordering
        matches the current graph exactly; on any mismatch / missing / corrupt
        cache it falls back to a fresh :meth:`_precompute` and (if caching is
        enabled) saves the result for next time. The node-id ordering is the
        cache key because the matrices are indexed by it -- reusing matrices
        built for a different ordering would silently corrupt every lookup.
        """
        if self._cache_matrices and os.path.exists(self._matrix_cache_path):
            cached = self._try_load_matrices()
            if cached is not None:
                return cached

        dist, pred = self._precompute(g)

        if self._cache_matrices:
            try:
                np.savez(
                    self._matrix_cache_path,
                    dist=dist,
                    pred=pred,
                    node_ids=np.asarray(self._node_ids, dtype=np.int64),
                )
            except OSError as exc:
                # A failed save is non-fatal: we still have the in-memory
                # matrices; just warn so the user can fix permissions/space.
                print(
                    f"[OSMnxNetwork] warning: could not write matrix cache "
                    f"{self._matrix_cache_path!r}: {exc}"
                )
        return dist, pred

    def _try_load_matrices(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load + validate the cached matrices, or ``None`` if unusable."""
        try:
            with np.load(self._matrix_cache_path) as data:
                node_ids = data["node_ids"]
                # Validate against the CURRENT graph's node ordering; any change
                # (different nodes, different order, different count) invalidates
                # the cache so we never index matrices with the wrong mapping.
                if node_ids.shape[0] != self._n or not np.array_equal(
                    node_ids, np.asarray(self._node_ids, dtype=np.int64)
                ):
                    return None
                dist = data["dist"]
                pred = data["pred"]
                if dist.shape != (self._n, self._n) or pred.shape != (
                    self._n,
                    self._n,
                ):
                    return None
                # np.load returns lazily-loaded arrays bound to the file; copy
                # them into memory so they stay valid after the file closes.
                return dist.copy(), pred.copy()
        except (OSError, ValueError, KeyError):
            # Corrupt / unreadable / wrong-format cache -> rebuild from scratch.
            return None

    def _precompute(self, g: nx.MultiDiGraph) -> Tuple[np.ndarray, np.ndarray]:
        """Dense all-pairs distance (m) + predecessor matrix (node indices).

        One length-weighted Dijkstra per source returns both the distances and
        the shortest paths; from each path we record, for every target, the
        index of the node immediately BEFORE it on the path from the source.
        The graph is the largest strongly-connected component (guaranteed by the
        builder), so every entry is finite / reachable.
        """
        n = self._n
        dist = np.full((n, n), np.inf, dtype=np.float64)
        pred = np.full((n, n), _NO_PRED, dtype=np.int32)
        for nid in self._node_ids:
            i = self._id_to_idx[nid]
            lengths, paths = nx.single_source_dijkstra(g, nid, weight="length")
            for tid, dval in lengths.items():
                j = self._id_to_idx[tid]
                dist[i, j] = dval
                path = paths[tid]
                if len(path) >= 2:
                    # Node immediately before the target on the path src->tid.
                    pred[i, j] = self._id_to_idx[path[-2]]
                # else: target == source; leave _NO_PRED.
        np.fill_diagonal(dist, 0.0)
        return dist, pred

    # --------------------------------------------------------------- snap
    def snap(self, coord: Coord) -> int:
        """Return the contiguous index of the graph node nearest ``coord``.

        ``coord`` is ``(lon, lat)``. Squared-Euclidean nearest search over node
        coordinates (adequate at city scale), memoised.
        """
        if self._snap_cache_on:
            hit = self._snap_cache.get(coord)
            if hit is not None:
                return hit
        dx = self._coords[:, 0] - coord[0]
        dy = self._coords[:, 1] - coord[1]
        idx = int(np.argmin(dx * dx + dy * dy))
        if self._snap_cache_on:
            self._snap_cache[coord] = idx
        return idx

    # --------------------------------------------------------------- API
    def distance(self, origin: Coord, destination: Coord) -> float:
        """Shortest-path road distance in metres between two continuous points."""
        return float(self._dist_m[self.snap(origin), self.snap(destination)])

    def shortest_path(self, origin: Coord, destination: Coord) -> PathResult:
        """Distance + constant-speed travel time between two points.

        ``PathResult.nodes`` returns the endpoints only (the continuous node
        sequence is not materialised here; movement uses :meth:`node_path` on
        node indices instead).
        """
        dist = self.distance(origin, destination)
        return PathResult(
            distance=dist,
            travel_time=self.travel_time(dist),
            nodes=[tuple(origin), tuple(destination)],
        )

    # -------------------------------------------------- node-level routing
    def node_path(self, src_idx: int, dst_idx: int) -> List[int]:
        """Reconstruct the shortest path as a list of node INDICES, src..dst.

        Rebuilt in O(path length) from the predecessor matrix -- no per-call
        Dijkstra. Returns ``[src_idx]`` when src == dst. The path is inclusive
        of both endpoints and ordered from source to destination.
        """
        if src_idx == dst_idx:
            return [src_idx]
        rev = [dst_idx]
        cur = dst_idx
        while cur != src_idx:
            p = int(self._pred[src_idx, cur])
            if p == _NO_PRED:
                # Should not happen on a strongly-connected graph; guard anyway.
                break
            rev.append(p)
            cur = p
        rev.reverse()
        return rev

    def node_distance(self, i_idx: int, j_idx: int) -> float:
        """Road distance in metres between two node INDICES (O(1) matrix read)."""
        return float(self._dist_m[i_idx, j_idx])

    def node_coord(self, idx: int) -> Coord:
        """``(lon, lat)`` coordinate of node index ``idx``."""
        return float(self._coords[idx, 0]), float(self._coords[idx, 1])

    # --------------------------------------------------------------- meta
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(lon_min, lat_min, lon_max, lat_max)`` covering all graph nodes."""
        lon_min, lat_min = self._coords.min(axis=0)
        lon_max, lat_max = self._coords.max(axis=0)
        return float(lon_min), float(lat_min), float(lon_max), float(lat_max)

    @property
    def node_coords(self) -> np.ndarray:
        """``[N, 2]`` array of node ``(lon, lat)`` coordinates (read-only)."""
        return self._coords

    def random_node_coord(self, rng: np.random.Generator) -> Coord:
        """Sample a uniformly-random graph node's ``(lon, lat)`` coordinate.

        Sampling order origins/destinations on real nodes guarantees they are
        reachable (the graph is strongly connected) and exactly on the network.
        """
        i = int(rng.integers(0, self._n))
        return float(self._coords[i, 0]), float(self._coords[i, 1])
