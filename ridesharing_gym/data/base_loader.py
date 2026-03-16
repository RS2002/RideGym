"""
Abstract base class for order data loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.order import Order


class DataLoader(ABC):
    """
    Abstract interface for loading orders from a data source.
    """

    @abstractmethod
    def load_orders(self, start_time: float, end_time: float) -> List[Order]:
        """
        Load all orders with request_time in [start_time, end_time).

        Args:
            start_time: Start of time window (inclusive).
            end_time: End of time window (exclusive).

        Returns:
            List of Order objects sorted by request_time.
        """
        pass

    @abstractmethod
    def get_total_duration(self) -> float:
        """
        Return the total time span covered by the data source.
        Used to set simulation end time.

        Returns:
            Total duration in seconds (from earliest to latest request).
        """
        pass