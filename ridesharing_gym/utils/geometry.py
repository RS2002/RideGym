"""
Geometry utilities for coordinate transformations.
"""

import math
from typing import Tuple

try:
    import utm
    UTM_AVAILABLE = True
except ImportError:
    UTM_AVAILABLE = False


def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert latitude/longitude to UTM coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        (easting, northing) in meters.
    """
    if UTM_AVAILABLE:
        return utm.from_latlon(lat, lon)[:2]
    else:
        # Fallback: approximate conversion (1 deg ~ 111 km)
        return (lon * 111000 * math.cos(math.radians(lat)), lat * 111000)


def utm_to_latlon(easting: float, northing: float, zone: int = 18, northern: bool = True) -> Tuple[float, float]:
    """
    Convert UTM coordinates to latitude/longitude.

    Args:
        easting: UTM easting in meters.
        northing: UTM northing in meters.
        zone: UTM zone number.
        northern: True if northern hemisphere.

    Returns:
        (latitude, longitude) in degrees.
    """
    if UTM_AVAILABLE:
        return utm.to_latlon(easting, northing, zone, northern)
    else:
        lat = northing / 111000
        lon = easting / (111000 * math.cos(math.radians(lat)))
        return (lat, lon)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Point 1 in degrees.
        lat2, lon2: Point 2 in degrees.

    Returns:
        Distance in meters.
    """
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c