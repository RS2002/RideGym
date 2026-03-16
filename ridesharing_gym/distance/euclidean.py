"""
Euclidean distance calculator.
Assumes coordinates are in a projected coordinate system (e.g., meters).
"""

import math
from typing import Tuple
from .base import DistanceCalculator


class EuclideanDistance(DistanceCalculator):
    """
    Euclidean (straight-line) distance.
    """

    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.hypot(dx, dy)