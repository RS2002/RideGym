"""
Spatial indexing using KDTree for fast nearest neighbor queries.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial import cKDTree


class SpatialIndex:
    """KDTree-based spatial index for 2D points."""

    def __init__(self, points: List[Tuple[float, float]]):
        self.points = np.array(points)
        self.tree = cKDTree(self.points)

    def nearest(self, point: Tuple[float, float], k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Find k nearest neighbors.

        Args:
            point: Query point (x, y).
            k: Number of neighbors.

        Returns:
            (indices, distances)
        """
        dists, idxs = self.tree.query(point, k=k)
        if k == 1:
            return [int(idxs)], [float(dists)]
        return idxs.tolist(), dists.tolist()

    def radius(self, point: Tuple[float, float], radius: float) -> List[int]:
        """Find all points within given radius."""
        return self.tree.query_ball_point(point, radius).tolist()