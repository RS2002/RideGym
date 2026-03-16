"""
Order module representing a ride request in the system.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Order:
    """
    Represents a ride request from a passenger.

    Attributes:
        order_id: Unique identifier for the order.
        pickup_location: (x, y) coordinates of pickup point.
        dropoff_location: (x, y) coordinates of dropoff point.
        request_time: Simulated time when the order was created.
        passenger_count: Number of passengers for this order (default 1).
        cancel_time: Time at which the order will be cancelled if not assigned.
                     If None, the order never cancels.
        assigned_driver_id: ID of driver who accepted this order, None if unassigned.
        confirmed_time: Time when order was accepted by a driver.
        pickup_time: Actual time when pickup occurs (set later).
        dropoff_time: Actual time when dropoff occurs (set later).
        is_cancelled: Whether the order was cancelled before assignment.
    """
    order_id: int
    pickup_location: Tuple[float, float]
    dropoff_location: Tuple[float, float]
    request_time: float
    passenger_count: int = 1
    cancel_time: Optional[float] = None

    # Fields that will be updated during simulation
    assigned_driver_id: Optional[int] = None
    confirmed_time: Optional[float] = None
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None
    is_cancelled: bool = False

    def __post_init__(self):
        """Validate inputs."""
        if self.passenger_count <= 0:
            raise ValueError("passenger_count must be positive")
        if self.cancel_time is not None and self.cancel_time <= self.request_time:
            raise ValueError("cancel_time must be > request_time")

    @property
    def is_assigned(self) -> bool:
        """Return True if the order has been assigned to a driver."""
        return self.assigned_driver_id is not None

    @property
    def is_completed(self) -> bool:
        """Return True if the order has been dropped off."""
        return self.dropoff_time is not None

    @property
    def waiting_time(self) -> Optional[float]:
        """
        Return waiting time (pickup_time - request_time) if pickup occurred.
        """
        if self.pickup_time is not None:
            return self.pickup_time - self.request_time
        return None

    @property
    def travel_time(self) -> Optional[float]:
        """
        Return travel time (dropoff_time - pickup_time) if completed.
        """
        if self.dropoff_time is not None and self.pickup_time is not None:
            return self.dropoff_time - self.pickup_time
        return None

    def assign(self, driver_id: int, current_time: float) -> None:
        """
        Assign this order to a driver.

        Args:
            driver_id: ID of the driver accepting the order.
            current_time: Simulation time of assignment.

        Raises:
            RuntimeError: If order is already assigned or cancelled.
        """
        if self.is_cancelled:
            raise RuntimeError(f"Order {self.order_id} has been cancelled.")
        if self.is_assigned:
            raise RuntimeError(f"Order {self.order_id} is already assigned to driver {self.assigned_driver_id}.")
        self.assigned_driver_id = driver_id
        self.confirmed_time = current_time

    def cancel(self, current_time: float) -> None:
        """
        Cancel the order (e.g., due to timeout).

        Args:
            current_time: Simulation time of cancellation.

        Raises:
            RuntimeError: If order is already assigned or cancelled.
        """
        if self.is_assigned:
            raise RuntimeError(f"Cannot cancel assigned order {self.order_id}.")
        if self.is_cancelled:
            raise RuntimeError(f"Order {self.order_id} already cancelled.")
        self.is_cancelled = True
        # Optionally store cancellation time if needed
        # self.cancellation_time = current_time

    def to_dict(self) -> dict:
        """
        Return a dictionary representation of the order's current state.
        Useful for building observations.
        """
        return {
            "id": self.order_id,
            "pickup": self.pickup_location,
            "dropoff": self.dropoff_location,
            "request_time": self.request_time,
            "passenger_count": self.passenger_count,
            "cancel_time": self.cancel_time,
            "assigned_driver_id": self.assigned_driver_id,
            "confirmed_time": self.confirmed_time,
            "pickup_time": self.pickup_time,
            "dropoff_time": self.dropoff_time,
            "is_cancelled": self.is_cancelled,
        }