"""
ride_gym: A multi-agent ride-pooling & dispatching simulation environment.

Gym-like (not Gym-dependent) interface for transportation gig-market research.
Initial focus: ride-pooling and vehicle relocation.
"""

from ride_gym.enums import DriverStatus, OrderStatus
from ride_gym.exceptions import (
    InvalidActionError,
    ConflictError,
    EnvironmentError as RideSimError,
)
from ride_gym.entities import Order, Driver
from ride_gym.road_network import (
    RoadNetwork,
    EuclideanNetwork,
    ManhattanNetwork,
)
from ride_gym.order_generator import (
    OrderGenerator,
    RandomOrderGenerator,
    DataFrameOrderGenerator,
    NYCOrderGenerator,
    MultiWindowNYCOrderGenerator,
)
from ride_gym.routing import RoutesPlanner, GreedyInsertionPlanner
from ride_gym.rewards import RewardFunction, DefaultRewardFunction
from ride_gym.env import RidePoolEnv
from ride_gym.wrappers import CentralizedWrapper

__version__ = "0.1.0"

__all__ = [
    "DriverStatus",
    "OrderStatus",
    "InvalidActionError",
    "ConflictError",
    "RideSimError",
    "Order",
    "Driver",
    "RoadNetwork",
    "EuclideanNetwork",
    "ManhattanNetwork",
    "OrderGenerator",
    "RandomOrderGenerator",
    "DataFrameOrderGenerator",
    "NYCOrderGenerator",
    "MultiWindowNYCOrderGenerator",
    "RoutesPlanner",
    "GreedyInsertionPlanner",
    "RewardFunction",
    "DefaultRewardFunction",
    "RidePoolEnv",
    "CentralizedWrapper",
]