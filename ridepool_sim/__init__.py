"""
ridepool_sim: A multi-agent ride-pooling & dispatching simulation environment.

Gym-like (not Gym-dependent) interface for transportation gig-market research.
Initial focus: ride-pooling and vehicle relocation.
"""

from ridepool_sim.enums import DriverStatus, OrderStatus
from ridepool_sim.exceptions import (
    InvalidActionError,
    ConflictError,
    EnvironmentError as RideSimError,
)
from ridepool_sim.entities import Order, Driver
from ridepool_sim.road_network import (
    RoadNetwork,
    EuclideanNetwork,
    ManhattanNetwork,
)
from ridepool_sim.order_generator import (
    OrderGenerator,
    RandomOrderGenerator,
    DataFrameOrderGenerator,
)
from ridepool_sim.routing import RoutesPlanner, GreedyInsertionPlanner
from ridepool_sim.rewards import RewardFunction, DefaultRewardFunction
from ridepool_sim.env import RidePoolEnv
from ridepool_sim.wrappers import CentralizedWrapper

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
    "RoutesPlanner",
    "GreedyInsertionPlanner",
    "RewardFunction",
    "DefaultRewardFunction",
    "RidePoolEnv",
    "CentralizedWrapper",
]