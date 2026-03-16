"""
Real road network distance calculator using OSMnx with path matrix and real-time path queries.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import networkx as nx

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    ox = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .base import DistanceCalculator
from ..utils.spatial_index import SpatialIndex

logger = logging.getLogger(__name__)


class OSMnxDistance(DistanceCalculator):
    """
    Road network distance calculator based on OSMnx graph.

    Precomputes distance and path matrices between zone centers for efficiency.
    Supports real-time path queries for arbitrary points via nearest zone centers.

    Args:
        place_name: OSMnx place query (e.g., "Manhattan, New York, USA").
        zone_centers: List of (lat, lon) tuples representing zone centers.
        network_type: Type of street network ('drive', 'walk', 'bike'). Default 'drive'.
        cache_dir: Optional directory to cache the graph and matrices.
        show_progress: If True, display progress bars during matrix computation.
    """

    def __init__(
        self,
        place_name: str,
        zone_centers: List[Tuple[float, float]],
        network_type: str = 'drive',
        cache_dir: Optional[str] = None,
        show_progress: bool = True
    ) -> None:
        if not OSMNX_AVAILABLE:
            raise ImportError(
                "OSMnx is required for OSMnxDistance. "
                "Install with: pip install osmnx networkx"
            )

        self.zone_centers = zone_centers
        self.show_progress = show_progress

        # Configure caching if requested
        if cache_dir is not None:
            ox.settings.cache_folder = cache_dir
            ox.settings.use_cache = True

        # Download and project the graph
        logger.info(f"Downloading OSM graph for {place_name}...")
        self.graph = ox.graph_from_place(place_name, network_type=network_type)
        self.graph = ox.project_graph(self.graph)  # Project to UTM for accurate distances
        logger.info("Graph downloaded and projected.")

        # Find nearest graph nodes for each zone center
        lons = [p[1] for p in zone_centers]
        lats = [p[0] for p in zone_centers]
        self.zone_nodes = ox.nearest_nodes(self.graph, lons, lats)

        # Precompute distance and path matrices between zone centers
        logger.info("Computing distance and path matrices...")
        self.dist_matrix, self.path_matrix = self._compute_matrices(self.zone_nodes)
        logger.info("Matrices computed.")

        # Build spatial index for fast nearest zone lookup
        self.spatial_index = SpatialIndex(zone_centers)

    def _compute_matrices(
        self,
        nodes: List[int]
    ) -> Tuple[np.ndarray, List[List[Optional[List[int]]]]]:
        """
        Compute distance matrix and path matrix between all pairs of nodes.

        Args:
            nodes: List of OSMnx node IDs.

        Returns:
            dist_mat: n x n numpy array of shortest path distances (meters).
            path_mat: n x n list of node ID lists (paths), or None if no path exists.
        """
        n = len(nodes)
        dist_mat = np.full((n, n), np.inf, dtype=float)
        path_mat: List[List[Optional[List[int]]]] = [[None] * n for _ in range(n)]

        iterator = range(n)
        if TQDM_AVAILABLE and self.show_progress:
            iterator = tqdm(iterator, desc="Computing paths")

        for i in iterator:
            dist_mat[i, i] = 0.0
            path_mat[i][i] = [nodes[i]]
            for j in range(i + 1, n):
                try:
                    path = nx.shortest_path(self.graph, nodes[i], nodes[j], weight='length')
                    length = nx.shortest_path_length(self.graph, nodes[i], nodes[j], weight='length')
                except nx.NetworkXNoPath:
                    path = None
                    length = np.inf
                dist_mat[i, j] = dist_mat[j, i] = length
                path_mat[i][j] = path_mat[j][i] = path

        return dist_mat, path_mat

    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Return the road network distance between two points.

        The distance is approximated by the precomputed distance between the
        nearest zone centers to each point.

        Args:
            point1: (lat, lon) of first point.
            point2: (lat, lon) of second point.

        Returns:
            Distance in meters.
        """
        i = self._nearest_zone(point1)
        j = self._nearest_zone(point2)
        return self.dist_matrix[i, j]

    def get_path_between(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> Optional[List[int]]:
        """
        Compute the shortest path node list between two arbitrary points in real time.

        This method uses OSMnx to find the nearest graph nodes to each point
        and then computes the shortest path between them using NetworkX.

        Args:
            point1: (lat, lon) of start point.
            point2: (lat, lon) of end point.

        Returns:
            List of node IDs along the shortest path, or None if no path exists.
        """
        node1 = ox.nearest_nodes(self.graph, point1[1], point1[0])  # (lon, lat)
        node2 = ox.nearest_nodes(self.graph, point2[1], point2[0])

        if node1 == node2:
            # Start and end at the same node; path consists of that node only
            return [node1]

        try:
            path = nx.shortest_path(self.graph, node1, node2, weight='length')
            return path
        except nx.NetworkXNoPath:
            return None

    def node_location(self, node_id: int) -> Tuple[float, float]:
        """
        Return the geographic coordinates of a graph node.

        Args:
            node_id: OSMnx node ID.

        Returns:
            (lat, lon) tuple.
        """
        node_data = self.graph.nodes[node_id]
        return (node_data['y'], node_data['x'])

    def _nearest_zone(self, point: Tuple[float, float]) -> int:
        """
        Find the index of the zone center closest to the given point.

        Args:
            point: (lat, lon) query point.

        Returns:
            Index of the nearest zone center in self.zone_centers.
        """
        idx, _ = self.spatial_index.nearest(point, k=1)
        return idx[0]