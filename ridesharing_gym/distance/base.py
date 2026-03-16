"""
Abstract base class for distance calculators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List


class DistanceCalculator(ABC):
    """
    Abstract interface for calculating distance between two points.

    All distance calculators should implement the `distance` method.
    Coordinates are assumed to be in the same coordinate system (e.g., projected meters or lat/lon).
    """

    @abstractmethod
    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Return the distance between two points.

        Args:
            point1: (x, y) or (lat, lon) coordinates.
            point2: (x, y) or (lat, lon) coordinates.

        Returns:
            Distance in meters (or the unit consistent with the coordinate system).
        """
        pass

    def get_path_between(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> Optional[List[int]]:
        """
        Return a list of node IDs along the path from point1 to point2, if supported.
        For straight-line calculators, this returns None.

        Args:
            point1: (lat, lon) of start point.
            point2: (lat, lon) of end point.

        Returns:
            List of node IDs if network path is supported, otherwise None.
        """
        return None