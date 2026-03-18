"""
Default reward function implementation with split reward at confirmation and dropoff.
"""

from typing import Dict, List, Optional, Set
from ..core.driver import Driver
from ..core.order import Order
from ..distance.base import DistanceCalculator
from .base import RewardFunction


class DefaultReward(RewardFunction):
    """
    Reward function that splits order revenue into confirmation and dropoff parts.
    Also includes waiting penalty, travel time penalty, and empty cruising cost.

    Args:
        base_fare: Fixed reward per completed order.
        per_km_fare: Additional reward per km of trip distance.
        confirm_reward_factor: Fraction of total revenue given at confirmation time (0 to 1).
        waiting_penalty_factor: Penalty per second of waiting time (request to pickup).
        travel_time_penalty_factor: Penalty per second of travel time (pickup to dropoff).
        empty_cruising_cost: Penalty per reposition event (simplified constant).
        distance_calc: Distance calculator for trip distance estimation.
    """

    def __init__(
        self,
        base_fare: float = 2.0,
        per_km_fare: float = 1.0,
        confirm_reward_factor: float = 0.5,
        waiting_penalty_factor: float = 0.01,
        travel_time_penalty_factor: float = 0.0,
        empty_cruising_cost: float = 0.5,
        distance_calc: Optional[DistanceCalculator] = None
    ):
        if not 0 <= confirm_reward_factor <= 1:
            raise ValueError("confirm_reward_factor must be between 0 and 1")
        self.base_fare = base_fare
        self.per_km_fare = per_km_fare
        self.confirm_reward_factor = confirm_reward_factor
        self.waiting_penalty_factor = waiting_penalty_factor
        self.travel_time_penalty_factor = travel_time_penalty_factor
        self.empty_cruising_cost = empty_cruising_cost
        self.distance_calc = distance_calc

        # Track orders that have already received confirmation reward
        self._confirmed_orders: Set[int] = set()

    def compute(
        self,
        drivers: List[Driver],
        orders: List[Order],
        current_time: float,
        events: Optional[Dict[int, List]] = None,
        **kwargs
    ) -> Dict[int, float]:
        """
        Compute rewards for each driver.

        For each order that has been confirmed in this step (i.e., its confirmed_time
        is set and not yet rewarded), give a confirmation reward equal to
        confirm_reward_factor * (base_fare + per_km_fare * trip_distance).

        For each dropoff event, give the remaining reward (1 - confirm_reward_factor) * total,
        and apply waiting and travel time penalties.

        Reposition arrivals incur a fixed penalty.
        """
        rewards = {d.driver_id: 0.0 for d in drivers}

        # 1. Confirmation rewards for newly assigned orders
        for order in orders:
            if order.confirmed_time is not None and order.order_id not in self._confirmed_orders:
                self._confirmed_orders.add(order.order_id)
                driver_id = order.assigned_driver_id
                if driver_id is None:
                    continue

                # Compute estimated trip distance
                if self.distance_calc is not None:
                    trip_dist = self.distance_calc.distance(
                        order.pickup_location, order.dropoff_location
                    )
                else:
                    dx = order.dropoff_location[0] - order.pickup_location[0]
                    dy = order.dropoff_location[1] - order.pickup_location[1]
                    trip_dist = (dx**2 + dy**2)**0.5

                total_revenue = self.base_fare + self.per_km_fare * trip_dist
                confirm_reward = self.confirm_reward_factor * total_revenue
                rewards[driver_id] += confirm_reward

        # 2. Process events (dropoff and reposition)
        if events is not None:
            for driver_id, ev_list in events.items():
                for ev in ev_list:
                    if len(ev) < 3:
                        continue
                    ev_type = ev[0]
                    order_id = ev[1]

                    if ev_type == 'dropoff':
                        order = next((o for o in orders if o.order_id == order_id), None)
                        if order is None:
                            continue

                        # Compute total revenue for this order
                        if self.distance_calc is not None:
                            trip_dist = self.distance_calc.distance(
                                order.pickup_location, order.dropoff_location
                            )
                        else:
                            dx = order.dropoff_location[0] - order.pickup_location[0]
                            dy = order.dropoff_location[1] - order.pickup_location[1]
                            trip_dist = (dx**2 + dy**2)**0.5

                        total_revenue = self.base_fare + self.per_km_fare * trip_dist

                        # Remaining reward after confirmation
                        remaining_reward = (1 - self.confirm_reward_factor) * total_revenue
                        rewards[driver_id] += remaining_reward

                        # Waiting penalty
                        if order.pickup_time is not None:
                            waiting = order.pickup_time - order.request_time
                            rewards[driver_id] -= self.waiting_penalty_factor * waiting

                        # Travel time penalty
                        if order.dropoff_time is not None and order.pickup_time is not None:
                            travel = order.dropoff_time - order.pickup_time
                            rewards[driver_id] -= self.travel_time_penalty_factor * travel

                    elif ev_type == 'reposition_arrived':
                        rewards[driver_id] -= self.empty_cruising_cost

        return rewards

    def reset(self):
        """
        Reset the internal state. Should be called at the beginning of each episode.
        """
        self._confirmed_orders.clear()