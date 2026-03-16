"""
RideSharing-Gym: A flexible multi-agent ride-sharing simulation environment.

This package provides a Gym-compatible environment for studying ride-sharing
dispatching, matching, and rebalancing problems in a multi-agent reinforcement
learning setting.
"""

from .env.ridesharing_env import RideSharingEnv
from .env.config import EnvConfig
from .distance.euclidean import EuclideanDistance
from .reward.default import DefaultReward
from .routing.insertion_planner import GreedyInsertionPlanner

__version__ = "0.1.0"
__all__ = ["RideSharingEnv", "EnvConfig"]