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
    CsvOrderGenerator,
)
from ride_gym.routing import RoutesPlanner, GreedyInsertionPlanner
from ride_gym.rewards import RewardFunction, DefaultRewardFunction
from ride_gym.env import RidePoolEnv
from ride_gym.wrappers import CentralizedWrapper
from ride_gym.labor_source import (
    LaborSource,
    LegacyLaborSource,
    ExplicitLaborSource,
    CsvLaborSource,
    ShiftPolicy,
    AlwaysOnShiftPolicy,
    WindowShiftPolicy,
    ShiftWindow,
)
from ride_gym.acceptance import (
    AcceptanceContext,
    AcceptancePolicy,
    AcceptanceProbabilities,
    PerDriverAcceptancePolicy,
)
from ride_gym.response import (
    ResponseTimePolicy,
    FixedResponseTimePolicy,
    UniformResponseTimePolicy,
    BetaResponseTimePolicy,
)
from ride_gym.warmup import capture_state, restore_state, save_snapshot, load_snapshot

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
    "CsvOrderGenerator",
    "RoutesPlanner",
    "GreedyInsertionPlanner",
    "RewardFunction",
    "DefaultRewardFunction",
    "RidePoolEnv",
    "CentralizedWrapper",
    "LaborSource",
    "LegacyLaborSource",
    "ExplicitLaborSource",
    "CsvLaborSource",
    "ShiftPolicy",
    "AlwaysOnShiftPolicy",
    "WindowShiftPolicy",
    "ShiftWindow",
    "AcceptanceContext",
    "AcceptancePolicy",
    "AcceptanceProbabilities",
    "PerDriverAcceptancePolicy",
    "ResponseTimePolicy",
    "FixedResponseTimePolicy",
    "UniformResponseTimePolicy",
    "BetaResponseTimePolicy",
    "capture_state",
    "restore_state",
    "save_snapshot",
    "load_snapshot",
]
