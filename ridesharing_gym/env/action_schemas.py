"""
Action space definitions and validation for ride-sharing environment.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from gym import spaces

from ..core.driver import Driver
from ..core.order import Order
from ..core.waypoint import Waypoint
from ..routing.base_planner import RoutePlanner


def create_action_space(num_regions: int, max_capacity: int) -> spaces.Dict:
    """
    Create a Dict action space for a driver.

    Args:
        num_regions: Number of discrete reposition regions.
        max_capacity: Maximum number of orders a driver can carry.

    Returns:
        Gym Dict space with keys:
            - order_ids: Box with shape (max_capacity,) and values in [-1, 2^63-1].
            - reposition_region: Discrete(num_regions).
    """
    return spaces.Dict({
        "order_ids": spaces.Box(
            low=-1,
            high=np.iinfo(np.int64).max,
            shape=(max_capacity,),
            dtype=np.int64
        ),
        "reposition_region": spaces.Discrete(num_regions)
    })


def validate_action(
    action: Dict[str, Any],
    driver: Driver,
    pending_orders: List[Order],
    num_regions: int,
    region_centers: Optional[List[Tuple[float, float]]] = None,
    distance_calc: Optional['DistanceCalculator'] = None,
    max_pickup_distance: Optional[float] = None,
    max_reposition_distance: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate a driver's action for feasibility.

    Args:
        action: Action dictionary from the driver.
        driver: Driver object.
        pending_orders: List of currently pending orders.
        num_regions: Number of regions (for reposition index check).
        region_centers: List of (x,y) for each region (required for reposition distance check).
        distance_calc: Distance calculator (required for distance constraints).
        max_pickup_distance: Maximum allowed direct distance to pickup (if None, no check).
        max_reposition_distance: Maximum allowed reposition distance (if None, no check).

    Returns:
        (valid, reason) tuple. If invalid, reason provides error message.
    """
    order_ids = action["order_ids"]
    reposition = action["reposition_region"]

    # Build set of pending order IDs for fast lookup
    pending_ids = {o.order_id for o in pending_orders}

    # Collect selected orders
    selected_orders = []
    for oid in order_ids:
        if oid != -1:
            if oid not in pending_ids:
                return False, f"Order ID {oid} not in pending orders"
            order = next((o for o in pending_orders if o.order_id == oid), None)
            if order is None:
                return False, f"Order ID {oid} not found (internal error)"
            selected_orders.append(order)

    # Check mutual exclusivity
    has_orders = len(selected_orders) > 0
    if has_orders and reposition != 0:
        return False, "Cannot both assign orders and reposition"

    if has_orders:
        # Check capacity
        total_passengers = sum(o.passenger_count for o in selected_orders)
        if total_passengers > driver.remaining_capacity:
            return False, f"Passenger count {total_passengers} exceeds remaining capacity {driver.remaining_capacity}"

        # Check pickup distance constraint
        if max_pickup_distance is not None and distance_calc is not None:
            for order in selected_orders:
                dist = distance_calc.distance(driver.current_location, order.pickup_location)
                if dist > max_pickup_distance:
                    return False, f"Pickup distance {dist:.2f} exceeds limit {max_pickup_distance:.2f}"

    else:
        # Reposition action
        if reposition < 0 or reposition >= num_regions:
            return False, f"Reposition region {reposition} out of range [0, {num_regions-1}]"

        # Check reposition distance constraint
        if max_reposition_distance is not None and distance_calc is not None and region_centers is not None:
            target = region_centers[reposition]
            dist = distance_calc.distance(driver.current_location, target)
            if dist > max_reposition_distance:
                return False, f"Reposition distance {dist:.2f} exceeds limit {max_reposition_distance:.2f}"

    return True, None


def apply_action(
    action: Dict[str, Any],
    driver: Driver,
    pending_orders: List[Order],
    region_centers: List[Tuple[float, float]],
    route_planner: RoutePlanner,
    distance_calc: 'DistanceCalculator',
    current_time: float
) -> Tuple[bool, Optional[List[Waypoint]], Optional[List[Order]]]:
    """
    Apply a validated action to a driver, updating driver state and removing assigned orders.

    Args:
        action: Validated action dictionary.
        driver: Driver object to modify.
        pending_orders: List of pending orders (will be mutated if orders assigned).
        region_centers: List of (x,y) for each region.
        route_planner: Route planner to compute new route.
        distance_calc: Distance calculator.
        current_time: Current simulation time.

    Returns:
        (success, new_route, assigned_orders) where assigned_orders is list of Order objects assigned.
        If route planning fails (should not happen if validation passed), returns (False, None, None).
    """
    order_ids = action["order_ids"]
    reposition = action["reposition_region"]

    # Find selected orders by ID
    selected_orders = []
    for oid in order_ids:
        if oid != -1:
            order = next((o for o in pending_orders if o.order_id == oid), None)
            if order is None:
                # This should not happen if validation passed
                return False, None, None
            selected_orders.append(order)

    if selected_orders:
        # Plan route
        success, new_route, _ = route_planner.plan(
            driver.current_location,
            driver.route,
            selected_orders
        )
        if not success:
            return False, None, None

        # Update driver
        passenger_counts = [o.passenger_count for o in selected_orders]
        order_ids_list = [o.order_id for o in selected_orders]
        driver.assign_orders(order_ids_list, passenger_counts, new_route)

        # Mark orders as assigned and remove from pending
        for order in selected_orders:
            order.assign(driver.driver_id, current_time)
            pending_orders.remove(order)

        return True, new_route, selected_orders

    else:
        # Reposition
        target = region_centers[reposition]
        driver.set_reposition(target)
        return True, driver.route, None