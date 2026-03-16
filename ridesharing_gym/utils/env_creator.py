"""
Utility functions for creating a ride-sharing environment with common defaults.
"""

import os
from typing import Optional

from ridesharing_gym.data.tlc_loader import TLCDataLoader
from ridesharing_gym.distance.base import DistanceCalculator
from ridesharing_gym.distance.osmnx import OSMnxDistance
from ridesharing_gym.distance.haversine import HaversineDistance
from ridesharing_gym.distance.euclidean import EuclideanDistance
from ridesharing_gym.distance.manhattan import ManhattanDistance
from ridesharing_gym.routing.insertion_planner import GreedyInsertionPlanner
from ridesharing_gym.reward.default import DefaultReward
from ridesharing_gym.env.config import EnvConfig
from ridesharing_gym.env.ridesharing_env import RideSharingEnv
from ridesharing_gym.data.zone_loader import load_default_zone_coords


def create_env(
    tlc_file: str,
    date: str = "2025-11-30",
    start_hour: int = 8,
    end_hour: int = 20,
    num_drivers: int = 500,
    driver_capacity: int = 4,
    driver_speed: float = 10.0,
    step_duration: int = 60,
    order_cancel_time: int = 300,
    borough_filter: str = "Manhattan",
    distance_type: str = "osmnx",
    strict_action_check: bool = True,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    **kwargs
) -> RideSharingEnv:
    """
    Create a ride-sharing environment with sensible defaults.

    Args:
        tlc_file: Path to the TLC Parquet file.
        date: Date to simulate (YYYY-MM-DD). Default "2025-11-30".
        start_hour: Start hour (inclusive). Default 8.
        end_hour: End hour (exclusive). Default 20.
        num_drivers: Number of drivers. Default 500.
        driver_capacity: Maximum passengers per driver. Default 4.
        driver_speed: Driver speed in m/s. Default 10.0.
        step_duration: Duration of each simulation step (seconds). Default 60.
        order_cancel_time: Order cancellation time (seconds). Default 300.
        borough_filter: Borough filter for zones (e.g., "Manhattan"). Default "Manhattan".
        distance_type: Distance calculation method. One of "osmnx", "haversine",
                       "euclidean", "manhattan". Default "osmnx".
        strict_action_check: Enable strict action checking. Default True.
        seed: Random seed. Default 42.
        cache_dir: Optional directory to cache OSMnx data.
        **kwargs: Additional arguments passed to EnvConfig.

    Returns:
        A configured RideSharingEnv instance.

    Raises:
        FileNotFoundError: If tlc_file does not exist.
        ValueError: If distance_type is invalid or OSMnx not available.
    """
    if not os.path.exists(tlc_file):
        raise FileNotFoundError(f"TLC data file not found: {tlc_file}")

    # Load zone coordinates
    zone_coords = load_default_zone_coords(borough_filter=borough_filter)
    zone_centers = list(zone_coords.values())
    print(f"Loaded {len(zone_centers)} zones from borough '{borough_filter}'.")

    # Create distance calculator
    if distance_type == "osmnx":
        try:
            distance_calc = OSMnxDistance(
                place_name=f"{borough_filter}, New York, USA",
                zone_centers=zone_centers,
                network_type="drive",
                cache_dir=cache_dir,
                show_progress=True
            )
        except ImportError as e:
            raise ImportError("OSMnx not installed. Please install osmnx.") from e
    elif distance_type == "haversine":
        distance_calc = HaversineDistance()
    elif distance_type == "euclidean":
        distance_calc = EuclideanDistance()
    elif distance_type == "manhattan":
        distance_calc = ManhattanDistance()
    else:
        raise ValueError(f"Invalid distance_type: {distance_type}. Choose from 'osmnx', 'haversine', 'euclidean', 'manhattan'.")

    print(f"Using distance calculator: {distance_type}")

    # Data loader
    data_loader = TLCDataLoader(
        file_path=tlc_file,
        use_location_id=True,
        date_filter=date,
        hour_range=(start_hour, end_hour),
        step_duration=step_duration,
        request_time_column='tpep_pickup_datetime',
        passenger_count_column='passenger_count',
        pickup_location_id_column='PULocationID',
        dropoff_location_id_column='DOLocationID',
        cancel_time_delta=order_cancel_time,
        start_time_shift=None,
        skip_nan_passenger=True
    )

    # Route planner
    route_planner = GreedyInsertionPlanner(distance_calc)

    # Reward function
    reward_fn = DefaultReward(distance_calc=distance_calc)

    total_seconds = (end_hour - start_hour) * 3600

    # Environment config
    config = EnvConfig(
        data_loader=data_loader,
        distance_calc=distance_calc,
        route_planner=route_planner,
        reward_fn=reward_fn,
        num_drivers=num_drivers,
        driver_capacities=[driver_capacity] * num_drivers,
        driver_speed=driver_speed,
        region_centers=zone_centers,
        step_duration=step_duration,
        order_cancel_time=order_cancel_time,
        strict_action_check=strict_action_check,
        total_duration=total_seconds,
        seed=seed,
        **kwargs
    )

    env = RideSharingEnv(config)
    return env