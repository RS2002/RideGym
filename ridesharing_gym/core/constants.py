"""
Constants for driver status and other enumerations.
"""

from enum import IntEnum


class DriverStatus(IntEnum):
    """Possible states of a driver."""
    IDLE = 0
    ENROUTE = 1      # Has a route with pickup/dropoff waypoints
    REPOSITIONING = 2  # Moving to a reposition target (no orders)