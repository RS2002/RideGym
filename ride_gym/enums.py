"""Enumerations for driver and order lifecycle states."""

from enum import Enum


class DriverStatus(str, Enum):
    """Lifecycle status of a driver (agent)."""

    IDLE = "idle"
    TO_PICKUP = "to_pickup"
    TO_DROPOFF = "to_dropoff"
    RELOCATING = "relocating"


class OrderStatus(str, Enum):
    """Full lifecycle status of an order."""

    PENDING = "pending"        # waiting to be assigned
    ASSIGNED = "assigned"      # bound to a driver, not yet picked up
    ONBOARD = "onboard"        # passenger(s) picked up, in vehicle
    COMPLETED = "completed"    # dropped off
    CANCELLED = "cancelled"    # timed out before assignment