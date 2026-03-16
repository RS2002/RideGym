"""
Abstract base class for reward functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from ..core.driver import Driver
from ..core.order import Order


class RewardFunction(ABC):
    """
    Interface for computing rewards for each driver.
    """

    @abstractmethod
    def compute(
        self,
        drivers: List[Driver],
        orders: List[Order],
        current_time: float,
        **kwargs
    ) -> Dict[int, float]:
        """
        Compute reward for each driver.

        Args:
            drivers: List of all drivers.
            orders: List of all orders.
            current_time: Current simulation time.
            **kwargs: Additional information (e.g., events from last step).

        Returns:
            Mapping from driver_id to reward.
        """
        pass