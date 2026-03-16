"""
Synthetic data loader for testing and development.
Generates random orders with configurable patterns.
"""

import numpy as np
from typing import List, Optional, Tuple
from ..core.order import Order
from .base_loader import DataLoader


class SyntheticDataLoader(DataLoader):
    """
    Generates synthetic orders on-the-fly.

    Orders are generated with request times following a Poisson process
    and locations uniformly distributed in a rectangle.

    Args:
        rate: Average number of orders per second (Poisson rate).
        total_duration: Total time span to generate orders for (seconds).
        area_bounds: (min_x, max_x, min_y, max_y) for pickup/dropoff locations.
        passenger_dist: Probability distribution for passenger counts (list of probabilities for 1..max_passengers).
        cancel_delay: Optional fixed delay after request after which order cancels (seconds).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        rate: float,
        total_duration: float,
        area_bounds: Tuple[float, float, float, float] = (0, 1000, 0, 1000),
        passenger_dist: Optional[List[float]] = None,
        cancel_delay: Optional[float] = None,
        seed: Optional[int] = None
    ):
        self.rate = rate
        self.total_duration = total_duration
        self.area_bounds = area_bounds
        if passenger_dist is None:
            passenger_dist = [0.6, 0.3, 0.1]  # 1,2,3 passengers
        self.passenger_dist = passenger_dist
        self.cancel_delay = cancel_delay
        self.rng = np.random.RandomState(seed)

        # Pre-generate inter-arrival times and compute arrival times
        # We'll generate a Poisson process
        n_expected = int(rate * total_duration) + 100  # overshoot a bit
        inter_arrivals = self.rng.exponential(1.0 / rate, size=n_expected)
        self.arrival_times = np.cumsum(inter_arrivals)
        # Keep only times within total_duration
        self.arrival_times = self.arrival_times[self.arrival_times < total_duration]
        self.num_orders = len(self.arrival_times)

    def load_orders(self, start_time: float, end_time: float) -> List[Order]:
        """Return orders with request times in [start_time, end_time)."""
        # Find indices within the window
        start_idx = np.searchsorted(self.arrival_times, start_time, side='left')
        end_idx = np.searchsorted(self.arrival_times, end_time, side='left')
        times = self.arrival_times[start_idx:end_idx]

        orders = []
        for i, t in enumerate(times):
            # Random pickup and dropoff within area
            pickup_x = self.rng.uniform(self.area_bounds[0], self.area_bounds[1])
            pickup_y = self.rng.uniform(self.area_bounds[2], self.area_bounds[3])
            dropoff_x = self.rng.uniform(self.area_bounds[0], self.area_bounds[1])
            dropoff_y = self.rng.uniform(self.area_bounds[2], self.area_bounds[3])

            # Passenger count: options are 1,2,...,max_passengers with given probabilities
            max_passengers = len(self.passenger_dist)
            pc = self.rng.choice(np.arange(1, max_passengers + 1), p=self.passenger_dist)

            cancel_time = t + self.cancel_delay if self.cancel_delay is not None else None

            order = Order(
                order_id=start_idx + i,  # unique within this loader
                pickup_location=(pickup_x, pickup_y),
                dropoff_location=(dropoff_x, dropoff_y),
                request_time=float(t),
                passenger_count=int(pc),
                cancel_time=cancel_time
            )
            orders.append(order)

        return orders

    def get_total_duration(self) -> float:
        return self.total_duration