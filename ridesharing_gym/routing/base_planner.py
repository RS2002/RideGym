"""
Abstract base class for route planners.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..core.waypoint import Waypoint
from ..core.order import Order


class RoutePlanner(ABC):
    """
    Abstract interface for planning a driver's route given current state and new orders.
    """

    @abstractmethod
    def plan(
        self,
        current_location: Tuple[float, float],
        current_route: List[Waypoint],
        new_orders: List[Order]
    ) -> Tuple[bool, Optional[List[Waypoint]], Optional[dict]]:
        """
        Insert new orders into the existing route.

        Args:
            current_location: Driver's current position.
            current_route: List of remaining waypoints (excluding those already passed).
            new_orders: List of Order objects to be inserted.

        Returns:
            (success, new_route, metrics)
            success: True if insertion is possible.
            new_route: List of Waypoints for the updated route (if success).
            metrics: Optional dict with extra info (e.g., total distance, computation time).
        """
        pass