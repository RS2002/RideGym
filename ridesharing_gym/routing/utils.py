"""
Utility functions for route planning.
"""

from typing import List, Tuple
from ..core.waypoint import Waypoint, WaypointType
from ..distance.base import DistanceCalculator


def total_route_distance(
    start_location: Tuple[float, float],
    route: List[Waypoint],
    dist_calc: DistanceCalculator
) -> float:
    """
    Calculate total distance from start location through all waypoints in order.
    """
    if not route:
        return 0.0
    total = dist_calc.distance(start_location, route[0].location)
    for i in range(len(route) - 1):
        total += dist_calc.distance(route[i].location, route[i+1].location)
    return total


def is_valid_route(route: List[Waypoint]) -> bool:
    """
    Check if a route satisfies the precedence constraints:
    - For each order, pickup must occur before dropoff.
    - No other constraints (e.g., capacity is checked elsewhere).
    """
    pickup_seen = set()
    for wp in route:
        if wp.waypoint_type == WaypointType.PICKUP:
            if wp.order_id in pickup_seen:
                return False  # duplicate pickup? shouldn't happen but check
            pickup_seen.add(wp.order_id)
        elif wp.waypoint_type == WaypointType.DROPOFF:
            if wp.order_id not in pickup_seen:
                return False  # dropoff before pickup
            # No need to remove from set because we don't allow multiple pickups/dropoffs per order
    return True