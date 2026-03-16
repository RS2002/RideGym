"""
Default reward function implementation.
"""

from typing import Dict, List, Optional
from ..core.driver import Driver
from ..core.order import Order
from ..distance.base import DistanceCalculator
from .base import RewardFunction


class DefaultReward(RewardFunction):
    """
    Reward function combining order revenue, waiting penalty, travel time penalty, and empty cruising cost.

    Args:
        base_fare: Fixed reward per completed order.
        per_km_fare: Additional reward per km of trip distance.
        waiting_penalty_factor: Penalty per second of waiting time (request to pickup).
        travel_time_penalty_factor: Penalty per second of travel time (pickup to dropoff).
        empty_cruising_cost: Penalty per reposition event (simplified constant).
        distance_calc: Distance calculator for trip distance estimation.
    """

    def __init__(
        self,
        base_fare: float = 2.0,
        per_km_fare: float = 1.0,
        waiting_penalty_factor: float = 0.01,
        travel_time_penalty_factor: float = 0.0,
        empty_cruising_cost: float = 0.5,
        distance_calc: Optional[DistanceCalculator] = None
    ):
        self.base_fare = base_fare
        self.per_km_fare = per_km_fare
        self.waiting_penalty_factor = waiting_penalty_factor
        self.travel_time_penalty_factor = travel_time_penalty_factor
        self.empty_cruising_cost = empty_cruising_cost
        self.distance_calc = distance_calc

    def compute(
        self,
        drivers: List[Driver],
        orders: List[Order],
        current_time: float,
        events: Optional[Dict[int, List]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """
        Compute rewards based on dropoff events and reposition arrivals.

        Args:
            drivers: List of drivers.
            orders: List of all orders.
            current_time: Current simulation time.
            events: Dictionary mapping driver_id to list of events from the last step.
                    Each event is a tuple (event_type, order_id, location, [time_offset]).
                    The last element (time_offset) is optional and ignored.

        Returns:
            Dictionary of driver rewards.
        """
        rewards = {d.driver_id: 0.0 for d in drivers}

        if events is None:
            return rewards

        for driver_id, ev_list in events.items():
            for ev in ev_list:
                # Unpack safely: event may have 3 or 4 elements
                if len(ev) >= 3:
                    ev_type = ev[0]
                    order_id = ev[1]
                else:
                    continue  # malformed event

                if ev_type == 'dropoff':
                    order = next((o for o in orders if o.order_id == order_id), None)
                    if order is not None:
                        # Trip revenue
                        if self.distance_calc is not None:
                            trip_dist = self.distance_calc.distance(
                                order.pickup_location, order.dropoff_location
                            )
                        else:
                            # Fallback to direct Euclidean if no calculator provided
                            dx = order.dropoff_location[0] - order.pickup_location[0]
                            dy = order.dropoff_location[1] - order.pickup_location[1]
                            trip_dist = (dx**2 + dy**2)**0.5
                        revenue = self.base_fare + self.per_km_fare * trip_dist
                        rewards[driver_id] += revenue

                        # Waiting penalty
                        if order.pickup_time is not None:
                            waiting = order.pickup_time - order.request_time
                            rewards[driver_id] -= self.waiting_penalty_factor * waiting

                        # Travel time penalty
                        if order.dropoff_time is not None and order.pickup_time is not None:
                            travel = order.dropoff_time - order.pickup_time
                            rewards[driver_id] -= self.travel_time_penalty_factor * travel

                elif ev_type == 'reposition_arrived':
                    # Constant penalty per reposition arrival
                    rewards[driver_id] -= self.empty_cruising_cost

        return rewards