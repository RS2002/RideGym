"""
Waypoint class representing a point on a driver's route.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class WaypointType(Enum):
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    REPOSITION = "reposition"  # For empty movements


@dataclass
class Waypoint:
    """
    A point along a driver's planned route.

    Attributes:
        location: (x, y) coordinates of the waypoint.
        waypoint_type: Type of waypoint (pickup, dropoff, reposition).
        order_id: ID of the associated order (None for reposition).
    """
    location: Tuple[float, float]
    waypoint_type: WaypointType
    order_id: Optional[int] = None

    def __post_init__(self):
        if self.waypoint_type != WaypointType.REPOSITION and self.order_id is None:
            raise ValueError(f"order_id must be provided for {self.waypoint_type} waypoint")
        if self.waypoint_type == WaypointType.REPOSITION and self.order_id is not None:
            raise ValueError("reposition waypoint should not have order_id")