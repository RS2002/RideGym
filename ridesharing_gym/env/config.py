"""
Configuration dataclass for the ride-sharing environment.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Any, Dict
from ..data.base_loader import DataLoader
from ..distance.base import DistanceCalculator
from ..routing.base_planner import RoutePlanner
from ..reward.base import RewardFunction


@dataclass
class EnvConfig:
    """
    Configuration for RideSharingEnv.

    Args:
        data_loader: DataLoader instance for generating orders.
        distance_calc: DistanceCalculator instance.
        route_planner: RoutePlanner instance.
        reward_fn: RewardFunction instance.
        num_drivers: Number of drivers.
        driver_capacities: List of capacities for each driver (if None, all use same capacity).
        driver_speed: Speed of all drivers (distance units per second).
        region_centers: List of (x,y) for reposition regions.
        max_pickup_distance: Optional max distance to pickup (if None, no limit).
        max_reposition_distance: Optional max reposition distance (if None, no limit).
        order_cancel_time: Seconds after request when order cancels if unassigned (None = never).
        step_duration: Duration of each simulation step (seconds).
        strict_action_check: If True, raise error on invalid action; else apply penalty.
        invalid_action_penalty: Reward penalty for invalid actions (when strict=False).
        observation_fn: Function to build observation from global state and driver_id.
        total_duration: Optional total simulation duration. If None, uses data_loader.get_total_duration().
        seed: Random seed for reproducibility.
    """
    data_loader: DataLoader
    distance_calc: DistanceCalculator
    route_planner: RoutePlanner
    reward_fn: RewardFunction

    num_drivers: int
    driver_capacities: Optional[List[int]] = None
    driver_speed: float = 10.0

    region_centers: List[Tuple[float, float]] = field(default_factory=list)

    max_pickup_distance: Optional[float] = None
    max_reposition_distance: Optional[float] = None

    order_cancel_time: Optional[float] = None
    step_duration: float = 60.0

    strict_action_check: bool = True
    invalid_action_penalty: float = -10.0

    observation_fn: Optional[Callable[[int, Dict[str, Any]], Any]] = None
    total_duration: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.driver_capacities is None:
            self.driver_capacities = [4] * self.num_drivers
        if len(self.driver_capacities) != self.num_drivers:
            raise ValueError("driver_capacities must have length num_drivers")
        if self.observation_fn is None:
            from .observation import build_observation_for_driver
            self.observation_fn = build_observation_for_driver