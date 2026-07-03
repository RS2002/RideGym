"""Core domain entities: Order and Driver.

These are deliberately lightweight, mutable dataclasses. The environment owns
and mutates them across the simulation; observations are built by *snapshotting*
them into plain dictionaries (see ``env.py``) so that external code cannot
accidentally corrupt internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ridepool_sim.enums import DriverStatus, OrderStatus

Coord = Tuple[float, float]


@dataclass
class Order:
    """A passenger ride request with full lifecycle tracking.

    Attributes
    ----------
    order_id:
        Unique, stable identifier.
    origin / destination:
        Pickup and drop-off coordinates in the continuous service area.
    request_time:
        Simulation time (minutes) at which the order becomes available in the
        pending pool.
    num_passengers:
        Party size; consumes this much vehicle capacity.
    status:
        Current lifecycle state (see :class:`OrderStatus`).
    assigned_driver:
        Driver id this order is bound to once assigned (else ``None``).
    pickup_time / dropoff_time / cancel_time:
        Event timestamps, populated as the lifecycle progresses.
    """

    order_id: int
    origin: Coord
    destination: Coord
    request_time: float
    num_passengers: int = 1

    status: OrderStatus = OrderStatus.PENDING
    assigned_driver: Optional[int] = None
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None
    cancel_time: Optional[float] = None

    # Estimated remaining time-to-delivery (minutes) under the driver's current
    # plan: the projected travel time from the driver's position to this order's
    # drop-off along the planned route. Set when the order is assigned / the
    # route is re-planned, decremented by dt each step the driver advances, and
    # cleared on completion. Maintained incrementally so the environment can
    # attribute pooling detour to a newly accepted order (how much later it
    # pushes already-committed deliveries) without re-walking the prior route.
    eta: Optional[float] = None

    def waiting_time(self, now: float) -> float:
        """How long the order has waited in the pending pool so far."""
        return now - self.request_time

    def is_pending(self) -> bool:
        return self.status == OrderStatus.PENDING


@dataclass
class TaskPoint:
    """A single stop in a driver's plan.

    ``kind`` is either ``"pickup"`` or ``"dropoff"`` and ``order_id`` ties the
    stop back to its order so the 'pickup-before-dropoff' precedence constraint
    can be enforced by the routing planner.
    """

    kind: str  # "pickup" | "dropoff"
    order_id: int
    location: Coord


@dataclass
class Driver:
    """A driver (agent) with continuous position and a task queue.

    Attributes
    ----------
    driver_id:
        Unique, stable identifier.
    location:
        Current continuous coordinate.
    capacity:
        Maximum simultaneous onboard passengers (per-vehicle, may differ
        across the fleet).
    speed:
        Per-vehicle travel speed (distance units / minute). ``None`` inherits
        the road network's default speed (homogeneous fleet).
    status:
        Current task status (see :class:`DriverStatus`).
    onboard_passengers:
        Passengers currently in the vehicle (picked up, not yet dropped off).
    assigned_orders:
        Ids of orders bound to this driver that are not yet completed.
    task_points:
        Ordered list of remaining stops (pickups/drop-offs) defining the
        service plan. Maintained by the routing planner under the
        pickup-before-dropoff constraint.
    route_nodes:
        Fine-grained coordinate sequence the driver physically follows this/next
        steps, produced by the road network. Consumed by physical movement.
    relocation_target:
        Active relocation destination when ``status == RELOCATING``.
    """

    driver_id: int
    location: Coord
    capacity: int = 4
    # Per-driver travel speed (distance units per minute). ``None`` means the
    # driver inherits the road network's default speed, preserving the original
    # homogeneous-fleet behaviour. A positive value lets a heterogeneous fleet
    # have per-vehicle speeds; the movement model uses this in place of
    # ``network.speed`` when set.
    speed: Optional[float] = None

    status: DriverStatus = DriverStatus.IDLE
    onboard_passengers: int = 0
    assigned_orders: List[int] = field(default_factory=list)
    task_points: List[TaskPoint] = field(default_factory=list)
    route_nodes: List[Coord] = field(default_factory=list)
    relocation_target: Optional[Coord] = None

    # --- In-edge position state (graph/OSMnx networks only) --------------
    # When moving along a real road graph the driver may stop part-way along an
    # edge rather than exactly on a node. These track that sub-edge position:
    #   edge_from / edge_to : node INDICES of the edge currently being traversed
    #                          (both None means the driver sits exactly on the
    #                          node given by ``node_idx``);
    #   edge_pos_m          : metres already travelled along that edge from
    #                          ``edge_from`` toward ``edge_to``.
    # ``location`` is kept in sync as the interpolated continuous (lon, lat) so
    # observations/features always see a real coordinate. Abstract Euclidean /
    # Manhattan networks ignore all of these (they interpolate on ``location``
    # directly).
    node_idx: Optional[int] = None
    edge_from: Optional[int] = None
    edge_to: Optional[int] = None
    edge_pos_m: float = 0.0

    # ----- capacity accounting -------------------------------------------
    def committed_passengers(self, orders_by_id) -> int:
        """Passengers onboard plus those assigned but not yet picked up.

        This is the capacity baseline against which new bids are checked, so a
        driver can never oversell a pooled ride.
        """
        pending_assigned = 0
        for oid in self.assigned_orders:
            order = orders_by_id[oid]
            if order.status == OrderStatus.ASSIGNED:
                pending_assigned += order.num_passengers
        return self.onboard_passengers + pending_assigned

    def is_idle(self) -> bool:
        return self.status == DriverStatus.IDLE

    def has_tasks(self) -> bool:
        return bool(self.task_points)