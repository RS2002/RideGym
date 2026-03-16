"""
Haversine distance for coordinates in latitude/longitude (WGS84).
Returns distance in meters.
"""

import math
from typing import Tuple
from .base import DistanceCalculator


class HaversineDistance(DistanceCalculator):
    """
    Great-circle distance using the haversine formula.
    Input coordinates: (latitude, longitude) in degrees.
    Output: distance in meters.
    """

    # Earth radius in meters
    EARTH_RADIUS = 6371000

    def distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        # Convert latitude and longitude to radians
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.EARTH_RADIUS * c