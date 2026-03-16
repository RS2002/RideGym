"""
Unified baseline for ride-sharing environment.
Supports both greedy (Hungarian) and random assignment methods.
Eligible drivers are those with remaining capacity > 0.
Each driver receives at most one order per step, each order at most once.
Supports ride-pooling: drivers can accept multiple orders over time as capacity permits.
Uses create_env utility for environment creation.
All output is saved to a file with detailed statistics, including per-order anomaly detection.
"""

import sys
import os
import argparse
import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ridesharing_gym.utils.env_creator import create_env
from ridesharing_gym.distance.osmnx import OSMnxDistance

import networkx as nx
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    ox = None


class Tee:
    """Duplicate output to both console and a file."""

    def __init__(self, filename: str, mode: str = 'w'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def write(self, message: str):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def parse_args():
    parser = argparse.ArgumentParser(description='Unified baseline with ride-pooling support.')
    parser.add_argument('--tlc-file', type=str, default='yellow_tripdata_2025-11.parquet',
                        help='Path to TLC Parquet file')
    parser.add_argument('--date', type=str, default='2025-11-30',
                        help='Date to simulate (YYYY-MM-DD)')
    parser.add_argument('--start-hour', type=int, default=19,
                        help='Start hour (inclusive)')
    parser.add_argument('--end-hour', type=int, default=20,
                        help='End hour (exclusive)')
    parser.add_argument('--num-drivers', type=int, default=500,
                        help='Number of drivers')
    parser.add_argument('--driver-capacity', type=int, default=4,
                        help='Maximum passengers per driver')
    parser.add_argument('--driver-speed', type=float, default=10.0,
                        help='Driver speed in m/s')
    parser.add_argument('--step-duration', type=int, default=60,
                        help='Duration of each simulation step (seconds)')
    parser.add_argument('--order-cancel-time', type=int, default=300,
                        help='Order cancellation time (seconds)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='baseline.txt',
                        help='Output file for simulation log')
    parser.add_argument('--borough-filter', type=str, default='Manhattan',
                        help='Borough filter for zones (e.g., Manhattan)')
    parser.add_argument('--strict', action='store_true', default=True,
                        help='Enable strict action checking')
    parser.add_argument('--no-strict', dest='strict', action='store_false',
                        help='Disable strict action checking')
    parser.add_argument('--distance', type=str, default='osmnx',
                        choices=['haversine', 'euclidean', 'manhattan', 'osmnx'],
                        help='Distance calculation method')
    parser.add_argument('--assignment', type=str, default='random',
                        choices=['greedy', 'random'],
                        help='Assignment method: greedy (Hungarian) or random')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache OSMnx data')
    return parser.parse_args()


def assign_greedy(eligible_drivers, available_orders, distance_calc, driver_capacity):
    """
    Greedy assignment using Hungarian algorithm.
    Returns a dictionary mapping driver_id to action.
    """
    actions = {}
    if not eligible_drivers or not available_orders:
        return actions

    n_drivers = len(eligible_drivers)
    n_orders = len(available_orders)

    # Build feasibility mask
    feasible = np.zeros((n_drivers, n_orders), dtype=bool)
    for i, driver in enumerate(eligible_drivers):
        for j, order in enumerate(available_orders):
            if order.passenger_count <= driver.remaining_capacity:
                feasible[i, j] = True

    if not np.any(feasible):
        return actions

    LARGE = 1e9
    cost_matrix = np.full((n_drivers, n_orders), LARGE, dtype=float)
    for i, driver in enumerate(eligible_drivers):
        for j, order in enumerate(available_orders):
            if feasible[i, j]:
                cost_matrix[i, j] = distance_calc.distance(
                    driver.current_location, order.pickup_location
                )

    driver_indices, order_indices = linear_sum_assignment(cost_matrix)

    for d_idx, o_idx in zip(driver_indices, order_indices):
        if cost_matrix[d_idx, o_idx] >= LARGE:
            continue  # skip infeasible matches
        driver = eligible_drivers[d_idx]
        order = available_orders[o_idx]
        order_ids_array = np.full(driver_capacity, -1, dtype=np.int64)
        order_ids_array[0] = order.order_id
        actions[driver.driver_id] = {
            "order_ids": order_ids_array,
            "reposition_region": 0
        }
    return actions


def assign_random(eligible_drivers, available_orders, driver_capacity):
    """
    Random assignment: shuffle drivers and assign randomly without replacement.
    Returns a dictionary mapping driver_id to action.
    """
    actions = {}
    if not eligible_drivers or not available_orders:
        return actions

    remaining_orders = available_orders.copy()
    shuffled_drivers = eligible_drivers.copy()
    random.shuffle(shuffled_drivers)

    for driver in shuffled_drivers:
        if not remaining_orders:
            break
        feasible = [o for o in remaining_orders if o.passenger_count <= driver.remaining_capacity]
        if not feasible:
            continue
        chosen = random.choice(feasible)
        order_ids_array = np.full(driver_capacity, -1, dtype=np.int64)
        order_ids_array[0] = chosen.order_id
        actions[driver.driver_id] = {
            "order_ids": order_ids_array,
            "reposition_region": 0
        }
        remaining_orders.remove(chosen)
    return actions


def main():
    args = parse_args()

    tee = Tee(args.output)
    sys.stdout = tee
    start_time = time.time()

    try:
        # Print configuration
        print("=" * 60)
        print("Unified Baseline Configuration")
        print("=" * 60)
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("=" * 60)
        print()

        # Create environment using the utility function
        env = create_env(
            tlc_file=args.tlc_file,
            date=args.date,
            start_hour=args.start_hour,
            end_hour=args.end_hour,
            num_drivers=args.num_drivers,
            driver_capacity=args.driver_capacity,
            driver_speed=args.driver_speed,
            step_duration=args.step_duration,
            order_cancel_time=args.order_cancel_time,
            borough_filter=args.borough_filter,
            distance_type=args.distance,
            strict_action_check=args.strict,
            seed=args.seed,
            cache_dir=args.cache_dir
        )
        env.reset()
        print("Action space:", env.action_space)

        random.seed(args.seed)
        np.random.seed(args.seed)

        total_rewards = []
        max_steps = int((args.end_hour - args.start_hour) * 3600 / args.step_duration)
        skipped_orders = 0

        # Use tqdm for progress bar
        pbar = tqdm.tqdm(total=max_steps, desc="Simulation", unit="step")
        step = 0
        while step < max_steps:
            eligible_drivers = [d for d in env.drivers if d.remaining_capacity > 0]
            pending = env.pending_orders
            available_orders = [o for o in pending if o.request_time <= env.current_time]

            actions = {}

            if eligible_drivers and available_orders:
                if args.assignment == 'greedy':
                    actions = assign_greedy(
                        eligible_drivers, available_orders,
                        env.config.distance_calc, args.driver_capacity
                    )
                else:  # random
                    actions = assign_random(
                        eligible_drivers, available_orders,
                        args.driver_capacity
                    )

            obs, rewards, terminated, info = env.step(actions)
            total_rewards.append(sum(rewards.values()))

            # Update progress bar
            pbar.update(1)

            # Print detailed stats every 10 steps
            if (step + 1) % 10 == 0:
                cum_completed = len(env.completed_orders)
                confirmed = len([o for o in env.all_orders if o.is_assigned])
                cum_reward = sum(total_rewards)
                pbar.set_postfix({
                    'completed': cum_completed,
                    'confirmed': confirmed,
                    'reward': f"{cum_reward:.2f}"
                })
                print(f"Step {step+1:4d} | completed {cum_completed:6d} | confirmed {confirmed:5d} | cumulative reward {cum_reward:10.2f}")

            step += 1
            if terminated:
                break

        pbar.close()

        # Collect detailed statistics from completed orders
        completed = env.completed_orders
        total_orders = len(env.all_orders)
        completed_count = len(completed)

        anomalies = []
        zero_travel_orders = 0
        pickup_times = []
        dropoff_times = []
        confirmation_times = []
        detour_times = []

        distance_calc = env.config.distance_calc

        for o in completed:
            if o.confirmed_time is not None and o.request_time is not None:
                ct = o.confirmed_time - o.request_time
                if ct < -1e-6:
                    anomalies.append(f"Order {o.order_id}: negative confirmation time ({ct:.2f}s)")
                confirmation_times.append(ct)

            if o.pickup_time is not None and o.request_time is not None:
                pt = o.pickup_time - o.request_time
                if pt < -1e-6:
                    anomalies.append(f"Order {o.order_id}: negative pickup time ({pt:.2f}s)")
                pickup_times.append(pt)

            if o.dropoff_time is not None and o.pickup_time is not None:
                travel = o.dropoff_time - o.pickup_time
                if travel <= 0:
                    zero_travel_orders += 1
                    anomalies.append(f"Order {o.order_id}: zero/negative travel time ({travel:.2f}s)")
                else:
                    dropoff_times.append(travel)

                    # Compute theoretical shortest path distance
                    if hasattr(distance_calc, 'graph'):
                        try:
                            node1 = ox.nearest_nodes(distance_calc.graph, o.pickup_location[1], o.pickup_location[0])
                            node2 = ox.nearest_nodes(distance_calc.graph, o.dropoff_location[1], o.dropoff_location[0])
                            path_dist = nx.shortest_path_length(distance_calc.graph, node1, node2, weight='length')
                        except (nx.NetworkXNoPath, KeyError, AttributeError):
                            path_dist = distance_calc.distance(o.pickup_location, o.dropoff_location)
                    else:
                        path_dist = distance_calc.distance(o.pickup_location, o.dropoff_location)

                    direct_time = path_dist / args.driver_speed
                    detour = travel - direct_time
                    if detour < -1e-6:
                        anomalies.append(f"Order {o.order_id}: negative detour time ({detour:.2f}s)")
                    detour_times.append(detour)

        print("\n" + "=" * 60)
        print("Simulation Results")
        print("=" * 60)
        print(f"Total orders in simulation period: {total_orders}")
        print(f"Completed orders: {completed_count}")
        print(f"Simulation duration: {(args.end_hour - args.start_hour) * 3600} seconds ({args.end_hour - args.start_hour:.1f} hours)")
        print(f"Steps executed: {step} (step duration = {args.step_duration}s)")
        print(f"Average reward per step: {np.mean(total_rewards):.2f}")
        print(f"Total reward accumulated: {sum(total_rewards):.2f}")

        if anomalies:
            print("\n--- Anomalies Detected ---")
            for msg in anomalies[:20]:
                print(msg)
            if len(anomalies) > 20:
                print(f"... and {len(anomalies)-20} more anomalies")
        else:
            print("\nNo anomalies detected in order times.")

        if pickup_times:
            print(f"\nAverage pickup time (request to pickup): {np.mean(pickup_times):.2f} s")
        if dropoff_times:
            print(f"Average dropoff time (pickup to dropoff): {np.mean(dropoff_times):.2f} s")
        if confirmation_times:
            print(f"Average confirmation time (request to assignment): {np.mean(confirmation_times):.2f} s")
        if detour_times:
            print(f"Average detour time (actual - direct): {np.mean(detour_times):.2f} s")
        if zero_travel_orders > 0:
            print(f"Warning: {zero_travel_orders} orders had zero or negative travel time.")

        if skipped_orders > 0:
            print(f"Warning: {skipped_orders} orders were skipped due to disappearance (should be zero).")
        print("=" * 60)

        elapsed = time.time() - start_time
        print(f"\nReal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    finally:
        tee.close()


if __name__ == "__main__":
    main()