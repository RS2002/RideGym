"""
Route planners using insertion heuristics.
"""

from typing import List, Tuple, Optional

from ..core.waypoint import Waypoint, WaypointType
from ..core.order import Order
from ..distance.base import DistanceCalculator
from .base_planner import RoutePlanner
from .utils import total_route_distance, is_valid_route


class GreedyInsertionPlanner(RoutePlanner):
    """
    Greedy insertion planner.

    Inserts orders one by one, each time choosing the best insertion positions
    for the current order. This is fast but may not be globally optimal.

    Args:
        distance_calc: Distance calculator instance.
    """

    def __init__(self, distance_calc: DistanceCalculator):
        self.dist_calc = distance_calc

    def plan(
        self,
        current_location: Tuple[float, float],
        current_route: List[Waypoint],
        new_orders: List[Order]
    ) -> Tuple[bool, Optional[List[Waypoint]], Optional[dict]]:
        if not new_orders:
            return True, current_route, {"distance": total_route_distance(current_location, current_route, self.dist_calc)}

        route = list(current_route)

        for order in new_orders:
            pickup_wp = Waypoint(order.pickup_location, WaypointType.PICKUP, order.order_id)
            dropoff_wp = Waypoint(order.dropoff_location, WaypointType.DROPOFF, order.order_id)

            best_distance = float('inf')
            best_route = None

            # Try all insertion positions for pickup and dropoff
            for i in range(len(route) + 1):
                route_with_pickup = route[:i] + [pickup_wp] + route[i:]
                for j in range(i + 1, len(route_with_pickup) + 1):
                    candidate = route_with_pickup[:j] + [dropoff_wp] + route_with_pickup[j:]
                    if is_valid_route(candidate):
                        dist = total_route_distance(current_location, candidate, self.dist_calc)
                        if dist < best_distance:
                            best_distance = dist
                            best_route = candidate

            if best_route is None:
                return False, None, None

            route = best_route

        return True, route, {"distance": best_distance}


class EnumerationPlanner(RoutePlanner):
    """
    Enumeration planner.

    Enumerates all possible insertion positions for all new orders simultaneously,
    guaranteeing global optimality. Feasible only for small numbers of new orders
    and small route lengths.

    Args:
        distance_calc: Distance calculator instance.
        max_orders_to_enumerate: Maximum number of new orders to enumerate.
                                 If exceeded, falls back to greedy.
    """

    def __init__(self, distance_calc: DistanceCalculator, max_orders_to_enumerate: int = 3):
        self.dist_calc = distance_calc
        self.max_orders_to_enumerate = max_orders_to_enumerate

    def plan(
        self,
        current_location: Tuple[float, float],
        current_route: List[Waypoint],
        new_orders: List[Order]
    ) -> Tuple[bool, Optional[List[Waypoint]], Optional[dict]]:
        if not new_orders:
            return True, current_route, {"distance": total_route_distance(current_location, current_route, self.dist_calc)}

        if len(new_orders) > self.max_orders_to_enumerate:
            greedy = GreedyInsertionPlanner(self.dist_calc)
            return greedy.plan(current_location, current_route, new_orders)

        best_distance = float('inf')
        best_route = None

        def backtrack(route_so_far: List[Waypoint], remaining: List[Order]):
            nonlocal best_distance, best_route
            if not remaining:
                dist = total_route_distance(current_location, route_so_far, self.dist_calc)
                if dist < best_distance:
                    best_distance = dist
                    best_route = route_so_far
                return

            order = remaining[0]
            pickup_wp = Waypoint(order.pickup_location, WaypointType.PICKUP, order.order_id)
            dropoff_wp = Waypoint(order.dropoff_location, WaypointType.DROPOFF, order.order_id)

            # Try all insertion positions for this order
            for i in range(len(route_so_far) + 1):
                route_with_pickup = route_so_far[:i] + [pickup_wp] + route_so_far[i:]
                for j in range(i + 1, len(route_with_pickup) + 1):
                    candidate = route_with_pickup[:j] + [dropoff_wp] + route_with_pickup[j:]
                    if is_valid_route(candidate):
                        backtrack(candidate, remaining[1:])

        backtrack(current_route, new_orders)

        if best_route is None:
            return False, None, None

        return True, best_route, {"distance": best_distance}