"""
Observation building utilities.
"""

from typing import List, Dict, Any
from ..core.driver import Driver
from ..core.order import Order


def build_global_state(
    drivers: List[Driver],
    pending_orders: List[Order],
    current_time: float
) -> Dict[str, Any]:
    """
    Build the global state dictionary.

    Args:
        drivers: List of all drivers.
        pending_orders: List of pending orders.
        current_time: Current simulation time.

    Returns:
        Dictionary with keys:
            - timestamp: current_time
            - drivers: list of driver dicts (from driver.to_dict())
            - pending_orders: list of order dicts (from order.to_dict())
    """
    return {
        "timestamp": current_time,
        "drivers": [d.to_dict() for d in drivers],
        "pending_orders": [o.to_dict() for o in pending_orders]
    }


def build_observation_for_driver(
    driver_id: int,
    global_state: Dict[str, Any],
    max_pending_orders: int = 100,
    max_enroute_orders: int = 4
) -> Dict[str, Any]:
    """
    Build an observation for a specific driver from the global state.
    This is a placeholder; users are expected to implement their own
    observation functions. The environment will return the full global state,
    and it's up to the learning algorithm to process it.

    Args:
        driver_id: ID of the driver.
        global_state: Full global state dict.
        max_pending_orders: Maximum number of pending orders (for padding).
        max_enroute_orders: Maximum number of enroute orders (for padding).

    Returns:
        The same global state (for full observability).
        Users can override this function via environment config.
    """
    return global_state