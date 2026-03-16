"""
Manhattan (city block) distance calculator.
Assumes coordinates are in a projected coordinate system.
"""

from typing import Tuple
from .base import DistanceCalculator


class ManhattanDistance(DistanceCalculator):
    """
    Manhattan distance (sum of absolute differences).
    """

    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])