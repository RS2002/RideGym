"""
Driver class representing a ride-sharing vehicle with support for both straight-line and network-based movement.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from .constants import DriverStatus
from .waypoint import Waypoint, WaypointType
from ..distance.base import DistanceCalculator


@dataclass
class Driver:
    """
    Represents a driver (vehicle) in the ride-sharing system.

    Attributes:
        driver_id: Unique identifier.
        capacity: Maximum number of passengers.
        current_location: (lat, lon) coordinates.
        speed: Movement speed in m/s.
        status: Current driver status (IDLE, ENROUTE, REPOSITIONING).
        route: List of Waypoints representing planned stops.
        enroute_orders: IDs of orders currently assigned but not completed.
        occupied_capacity: Current number of passengers on board.
        total_distance_driven: Cumulative distance driven (for statistics).
        current_node: Current graph node ID (only for network movement).
        node_path: Remaining node path to the next waypoint (network mode).
        path_index: Index of next node in node_path to move towards.
    """
    driver_id: int
    capacity: int
    current_location: Tuple[float, float]
    speed: float = 10.0
    status: DriverStatus = DriverStatus.IDLE
    route: List[Waypoint] = field(default_factory=list)
    enroute_orders: List[int] = field(default_factory=list)
    occupied_capacity: int = 0
    total_distance_driven: float = 0.0
    current_node: Optional[int] = None
    node_path: List[int] = field(default_factory=list)
    path_index: int = 0

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        self._sync_status()

    def _sync_status(self):
        """Update status based on route content."""
        if not self.route:
            self.status = DriverStatus.IDLE
        elif self.route[-1].waypoint_type == WaypointType.REPOSITION:
            self.status = DriverStatus.REPOSITIONING
        else:
            self.status = DriverStatus.ENROUTE

    @property
    def remaining_capacity(self) -> int:
        """Calculate remaining passenger capacity."""
        return self.capacity - self.occupied_capacity

    def assign_orders(
        self,
        new_order_ids: List[int],
        new_passenger_counts: List[int],
        new_route: List[Waypoint],
        distance_calc: Optional[DistanceCalculator] = None
    ):
        """
        Assign new orders to the driver, updating route and capacity.

        The route is set to the provided new_route. No precomputation of network paths
        is performed; paths will be generated during movement as needed.

        Args:
            new_order_ids: List of order IDs being assigned.
            new_passenger_counts: List of passenger counts for these orders.
            new_route: The new planned route (list of Waypoints).
            distance_calc: Distance calculator (unused in assignment, but kept for interface consistency).
        """
        if len(new_order_ids) != len(new_passenger_counts):
            raise ValueError("new_order_ids and new_passenger_counts must have same length")
        total_passengers = sum(new_passenger_counts)
        if self.occupied_capacity + total_passengers > self.capacity:
            raise RuntimeError(f"Capacity exceeded: {self.occupied_capacity}+{total_passengers} > {self.capacity}")

        self.enroute_orders.extend(new_order_ids)
        self.occupied_capacity += total_passengers
        self.route = new_route
        # Reset network movement state
        self.node_path = []
        self.path_index = 0
        self._sync_status()

    def complete_dropoff(self, order_id: int, passenger_count: int):
        """
        Called when a dropoff occurs. Removes the order and frees capacity.

        Args:
            order_id: ID of the completed order.
            passenger_count: Number of passengers for that order.
        """
        if order_id not in self.enroute_orders:
            raise RuntimeError(f"Order {order_id} not in enroute_orders")
        self.enroute_orders.remove(order_id)
        self.occupied_capacity -= passenger_count
        if self.occupied_capacity < 0:
            raise RuntimeError("Occupied capacity became negative")

    def set_reposition(self, target: Tuple[float, float], distance_calc: Optional[DistanceCalculator] = None):
        """
        Set a reposition target, clearing any existing route.

        Args:
            target: (lat, lon) coordinates to move to.
            distance_calc: Distance calculator (unused, but kept for interface consistency).
        """
        self.route = [Waypoint(location=target, waypoint_type=WaypointType.REPOSITION)]
        self.node_path = []
        self.path_index = 0
        self._sync_status()

    def move(self, time_step: float, distance_calc: DistanceCalculator) -> List[tuple]:
        """
        Move the driver for a given time step.

        If the distance calculator supports network-based movement (i.e., its
        `get_path_between` method returns a non-None path), the driver will move
        along the road network. Otherwise, it moves in a straight line.

        Args:
            time_step: Duration of movement in seconds.
            distance_calc: Distance calculator.

        Returns:
            List of events that occurred during this move.
        """
        if self.status == DriverStatus.IDLE or not self.route:
            return []

        events = []
        distance_to_travel = self.speed * time_step
        remaining_distance = distance_to_travel
        time_elapsed = 0.0

        while remaining_distance > 1e-6 and self.route:
            # Check if network movement is available and we need a new path
            if hasattr(distance_calc, 'get_path_between') and (not self.node_path or self.path_index >= len(self.node_path)):
                next_wp = self.route[0]
                path = distance_calc.get_path_between(self.current_location, next_wp.location)
                if path is not None and len(path) > 0:
                    if len(path) == 1:
                        # Already at destination
                        wp = self.route.pop(0)
                        event_time = time_elapsed
                        if wp.waypoint_type == WaypointType.PICKUP:
                            events.append(('pickup', wp.order_id, wp.location, event_time))
                        elif wp.waypoint_type == WaypointType.DROPOFF:
                            events.append(('dropoff', wp.order_id, wp.location, event_time))
                        elif wp.waypoint_type == WaypointType.REPOSITION:
                            events.append(('reposition_arrived', None, wp.location, event_time))
                        continue

                    self.node_path = path
                    self.path_index = 1  # skip current node
                    self.current_node = path[0]

            # Now decide whether to use network or straight-line
            if self.node_path and self.path_index < len(self.node_path):
                # Network movement along current node path
                target_node = self.node_path[self.path_index]
                target_loc = distance_calc.node_location(target_node)

                # Get distance to target using edge length if available
                if hasattr(distance_calc, 'edge_length'):
                    try:
                        dist_to_target = distance_calc.edge_length(self.current_node, target_node)
                    except ValueError:
                        # Fallback to Euclidean between nodes
                        dist_to_target = distance_calc.distance(self.current_location, target_loc)
                else:
                    dist_to_target = distance_calc.distance(self.current_location, target_loc)

                if dist_to_target <= remaining_distance:
                    travel_time = dist_to_target / self.speed
                    time_elapsed += travel_time
                    remaining_distance -= dist_to_target
                    self.current_location = target_loc
                    self.current_node = target_node
                    self.total_distance_driven += dist_to_target
                    self.path_index += 1

                    if self.path_index >= len(self.node_path):
                        # Reached the waypoint at the end of this path
                        wp = self.route.pop(0)
                        event_time = time_elapsed
                        if wp.waypoint_type == WaypointType.PICKUP:
                            events.append(('pickup', wp.order_id, wp.location, event_time))
                        elif wp.waypoint_type == WaypointType.DROPOFF:
                            events.append(('dropoff', wp.order_id, wp.location, event_time))
                        elif wp.waypoint_type == WaypointType.REPOSITION:
                            events.append(('reposition_arrived', None, wp.location, event_time))
                        self.node_path = []
                        self.path_index = 0
                else:
                    travel_time = remaining_distance / self.speed
                    time_elapsed += travel_time
                    ratio = remaining_distance / dist_to_target
                    new_lat = self.current_location[0] + ratio * (target_loc[0] - self.current_location[0])
                    new_lon = self.current_location[1] + ratio * (target_loc[1] - self.current_location[1])
                    self.current_location = (new_lat, new_lon)
                    self.total_distance_driven += remaining_distance
                    remaining_distance = 0
            else:
                # Straight-line movement
                next_wp = self.route[0]
                dist_to_next = distance_calc.distance(self.current_location, next_wp.location)

                if dist_to_next <= remaining_distance:
                    travel_time = dist_to_next / self.speed
                    time_elapsed += travel_time
                    remaining_distance -= dist_to_next
                    self.current_location = next_wp.location
                    self.total_distance_driven += dist_to_next
                    self.route.pop(0)
                    event_time = time_elapsed
                    events.append((next_wp.waypoint_type.value, next_wp.order_id, next_wp.location, event_time))
                else:
                    travel_time = remaining_distance / self.speed
                    time_elapsed += travel_time
                    ratio = remaining_distance / dist_to_next
                    new_lat = self.current_location[0] + ratio * (next_wp.location[0] - self.current_location[0])
                    new_lon = self.current_location[1] + ratio * (next_wp.location[1] - self.current_location[1])
                    self.current_location = (new_lat, new_lon)
                    self.total_distance_driven += remaining_distance
                    remaining_distance = 0

        self._sync_status()
        return events

    def to_dict(self) -> dict:
        """
        Return a dictionary representation of the driver's current state.
        Useful for building observations.
        """
        return {
            "id": self.driver_id,
            "location": self.current_location,
            "capacity": self.capacity,
            "occupied_capacity": self.occupied_capacity,
            "status": self.status.value,
            "enroute_orders": self.enroute_orders.copy(),
            "route": [(wp.location, wp.waypoint_type.value, wp.order_id) for wp in self.route],
        }