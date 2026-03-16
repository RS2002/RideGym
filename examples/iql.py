"""
Deep SARSA / DDQN for ride-sharing environment with ride-pooling.
Eligible drivers are those with remaining capacity > 0.
Each driver receives at most one order per step, each order at most once.
Uses a shared neural network to estimate Q-values for each (driver, order) pair,
plus a learnable dummy order embedding for the "no order" action.
Action selection via Hungarian algorithm to maximize total Q-value.
Training and testing episodes are separated by date ranges.
User can choose algorithm via --algorithm {sarsa, ddqn} and reward type via --reward-type {global, individual}.
"""

import sys
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
import tqdm
from collections import deque
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ridesharing_gym.utils.env_creator import create_env
from ridesharing_gym.distance.osmnx import OSMnxDistance
from ridesharing_gym.core.order import Order
from ridesharing_gym.core.driver import Driver, DriverStatus

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
    parser = argparse.ArgumentParser(description='Deep RL for ride-sharing with ride-pooling.')
    parser.add_argument('--tlc-file', type=str, default='yellow_tripdata_2025-11.parquet',
                        help='Path to TLC Parquet file')
    parser.add_argument('--test-date', type=str, default='2025-11-30',
                        help='Date for testing (YYYY-MM-DD)')
    parser.add_argument('--train-start-date', type=str, default='2025-11-23',
                        help='Start date for training date range (YYYY-MM-DD)')
    parser.add_argument('--train-end-date', type=str, default='2025-11-29',
                        help='End date for training date range (YYYY-MM-DD)')
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
    parser.add_argument('--output', type=str, default='iddqn.txt',
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
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache OSMnx data')

    # Deep RL parameters
    parser.add_argument('--algorithm', type=str, default='ddqn', choices=['sarsa', 'ddqn'],
                        help='Algorithm to use: sarsa or ddqn')
    parser.add_argument('--reward-type', type=str, default='individual', choices=['global', 'individual'],
                        help='Reward type: global (sum of all driver rewards) or individual (per-driver)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension of Q-network')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for exploration (random actions)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay per episode')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                        help='Minimum epsilon')
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='Soft update coefficient for target network')
    parser.add_argument('--train-start', type=int, default=16,
                        help='Number of steps before starting training')
    parser.add_argument('--train-freq', type=int, default=1,
                        help='Training frequency (steps per update)')
    parser.add_argument('--max-orders', type=int, default=500,
                        help='Maximum number of orders to consider (for fixed-size state)')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--test-episode-freq', type=int, default=5,
                        help='Frequency of testing episodes')
    return parser.parse_args()


class QNetwork(nn.Module):
    """Neural network to estimate Q(s, d, o) for each driver-order pair, plus dummy order."""
    def __init__(self, driver_feat_dim, order_feat_dim, hidden_dim):
        super().__init__()
        self.driver_fc = nn.Linear(driver_feat_dim, hidden_dim)
        self.order_fc = nn.Linear(order_feat_dim, hidden_dim)
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.dummy_embed = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, driver_feats, order_feats):
        """
        Args:
            driver_feats: (batch, total_drivers, feat_dim)
            order_feats: (batch, max_orders, order_feat_dim)  # only real orders
        Returns:
            real_q: (batch, total_drivers, max_orders)
            dummy_q: (batch, total_drivers)
        """
        batch_size, total_drivers, _ = driver_feats.shape
        max_orders = order_feats.shape[1]

        driver_h = torch.relu(self.driver_fc(driver_feats))  # (batch, total_drivers, hidden)

        # Real orders
        order_h = torch.relu(self.order_fc(order_feats))  # (batch, max_orders, hidden)
        driver_expanded = driver_h.unsqueeze(2).expand(-1, -1, max_orders, -1)
        order_expanded = order_h.unsqueeze(1).expand(-1, total_drivers, -1, -1)
        combined = torch.cat([driver_expanded, order_expanded], dim=-1)
        real_q = self.combined_fc(combined).squeeze(-1)  # (batch, total_drivers, max_orders)

        # Dummy order: apply dummy_embed to driver_h
        dummy_q = self.dummy_embed(driver_h).squeeze(-1)  # (batch, total_drivers)

        return real_q, dummy_q


def normalize_location(lat, lon):
    lat_min, lat_max = 40.70, 40.88
    lon_min, lon_max = -74.02, -73.93
    lat_norm = (lat - lat_min) / (lat_max - lat_min)
    lon_norm = (lon - lon_min) / (lon_max - lon_min)
    return lat_norm, lon_norm


def get_enroute_info(driver, all_orders):
    enroute_info = []
    for order_id in driver.enroute_orders:
        order = next((o for o in all_orders if o.order_id == order_id), None)
        if order is not None:
            is_picked_up = order.pickup_time is not None
            enroute_info.append((
                order.pickup_location[0], order.pickup_location[1],
                order.dropoff_location[0], order.dropoff_location[1],
                float(is_picked_up)
            ))
    return enroute_info


def build_state_features(drivers, orders, all_orders, total_drivers, max_orders, max_capacity, device):
    """
    Args:
        drivers: list of eligible Driver objects (actual drivers)
        orders: list of pending Order objects (real orders)
        all_orders: all orders in environment
        total_drivers: total number of drivers in the environment (for padding)
        max_orders: maximum orders to consider
        max_capacity: driver capacity
        device: torch device
    Returns:
        driver_tensor: (1, total_drivers, feat_dim)
        order_tensor: (1, max_orders, order_feat_dim)
        driver_mask: (total_drivers,) boolean, True for actual drivers
        order_mask: (max_orders,) boolean, True for actual orders
        n_drivers_actual: len(drivers)
        n_orders_actual: min(len(orders), max_orders)
    """
    n_drivers_actual = len(drivers)
    n_orders_actual = min(len(orders), max_orders)
    driver_feat_dim = 2 + 1 + 3 + max_capacity * 5
    order_feat_dim = 6

    driver_feats = np.zeros((total_drivers, driver_feat_dim), dtype=np.float32)
    order_feats = np.zeros((max_orders, order_feat_dim), dtype=np.float32)

    for i, d in enumerate(drivers):
        lat_n, lon_n = normalize_location(d.current_location[0], d.current_location[1])
        rem_cap = d.remaining_capacity / d.capacity
        status_onehot = [0, 0, 0]
        status_onehot[d.status] = 1

        enroute_info = get_enroute_info(d, all_orders)
        enroute_part = np.zeros(max_capacity * 5, dtype=np.float32)
        for idx, (pu_lat, pu_lon, do_lat, do_lon, picked) in enumerate(enroute_info[:max_capacity]):
            base = idx * 5
            pu_lat_n, pu_lon_n = normalize_location(pu_lat, pu_lon)
            do_lat_n, do_lon_n = normalize_location(do_lat, do_lon)
            enroute_part[base:base+5] = [pu_lat_n, pu_lon_n, do_lat_n, do_lon_n, picked]

        driver_feats[i] = np.concatenate([[lat_n, lon_n], [rem_cap], status_onehot, enroute_part])

    for j, o in enumerate(orders[:max_orders]):
        pu_lat_n, pu_lon_n = normalize_location(o.pickup_location[0], o.pickup_location[1])
        do_lat_n, do_lon_n = normalize_location(o.dropoff_location[0], o.dropoff_location[1])
        pass_norm = o.passenger_count / 4.0
        time_norm = o.request_time / 3600.0
        order_feats[j] = [pu_lat_n, pu_lon_n, do_lat_n, do_lon_n, pass_norm, time_norm]

    driver_tensor = torch.tensor(driver_feats, device=device).unsqueeze(0)
    order_tensor = torch.tensor(order_feats, device=device).unsqueeze(0)
    driver_mask = torch.zeros(total_drivers, dtype=torch.bool, device=device)
    driver_mask[:n_drivers_actual] = True
    order_mask = torch.zeros(max_orders, dtype=torch.bool, device=device)
    order_mask[:n_orders_actual] = True
    return driver_tensor, order_tensor, driver_mask, order_mask, n_drivers_actual, n_orders_actual


def compute_feasibility_mask(drivers, orders, max_orders):
    """Return boolean matrix (n_drivers, n_orders) indicating feasible real order pairs."""
    n_drivers = len(drivers)
    n_orders = min(len(orders), max_orders)
    mask = np.zeros((n_drivers, n_orders), dtype=bool)
    for i, d in enumerate(drivers):
        for j, o in enumerate(orders[:n_orders]):
            if o.passenger_count <= d.remaining_capacity:
                mask[i, j] = True
    return mask


def select_action(real_q, dummy_q, feasible_real, epsilon):
    """
    Select action (matching) using epsilon-greedy, with dummy order included.
    Returns list of (driver_idx, order_idx) where order_idx = n_real means dummy.
    """
    n_drivers, n_real = real_q.shape
    total_q = np.zeros((n_drivers, n_real + 1))
    total_q[:, :n_real] = real_q
    total_q[:, n_real] = dummy_q

    feasible_total = np.zeros((n_drivers, n_real + 1), dtype=bool)
    feasible_total[:, :n_real] = feasible_real
    feasible_total[:, n_real] = True

    if epsilon > 0 and np.random.rand() < epsilon:
        driver_indices = list(range(n_drivers))
        np.random.shuffle(driver_indices)
        matched_pairs = []
        assigned_orders = set()
        for d_idx in driver_indices:
            feasible_orders = [j for j in range(n_real + 1) if feasible_total[d_idx, j] and j not in assigned_orders]
            if feasible_orders:
                o_idx = np.random.choice(feasible_orders)
                matched_pairs.append((d_idx, o_idx))
                assigned_orders.add(o_idx)
        return [(int(d), int(o)) for d, o in matched_pairs]
    else:
        cost_matrix = -total_q.copy()
        cost_matrix[~feasible_total] = 1e9
        driver_indices, order_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        for d_idx, o_idx in zip(driver_indices, order_indices):
            if feasible_total[d_idx, o_idx]:
                matched_pairs.append((d_idx, o_idx))
        return [(int(d), int(o)) for d, o in matched_pairs]


def compute_detour_times(completed_orders, distance_calc, driver_speed):
    detour_times = []
    for o in completed_orders:
        if o.dropoff_time is not None and o.pickup_time is not None:
            travel = o.dropoff_time - o.pickup_time
            if travel <= 0:
                continue
            if hasattr(distance_calc, 'graph'):
                try:
                    node1 = ox.nearest_nodes(distance_calc.graph, o.pickup_location[1], o.pickup_location[0])
                    node2 = ox.nearest_nodes(distance_calc.graph, o.dropoff_location[1], o.dropoff_location[0])
                    path_dist = nx.shortest_path_length(distance_calc.graph, node1, node2, weight='length')
                except (nx.NetworkXNoPath, KeyError, AttributeError):
                    path_dist = distance_calc.distance(o.pickup_location, o.dropoff_location)
            else:
                path_dist = distance_calc.distance(o.pickup_location, o.dropoff_location)
            direct_time = path_dist / driver_speed
            detour_times.append(travel - direct_time)
    return detour_times


def run_episode(env, q_net, epsilon, max_orders, max_capacity, device, training=True):
    """Generator that yields (transition, step_reward, done) for each step."""
    obs = env.reset()
    done = False
    step = 0
    prev_state = None
    prev_action = None

    max_steps = int(env.total_duration / env.config.step_duration)

    # Create progress bar
    pbar = tqdm.tqdm(total=max_steps, desc="Episode", unit="step", leave=False)

    while not done and step < max_steps:
        eligible_drivers = [d for d in env.drivers if d.remaining_capacity > 0]
        pending = env.pending_orders
        available_orders = [o for o in pending if o.request_time <= env.current_time]

        if len(eligible_drivers) == 0 or len(available_orders) == 0:
            obs, rewards, done, info = env.step({})
            step_reward = sum(rewards.values())
            step += 1
            pbar.update(1)
            if training and prev_state is not None:
                # No action taken, but we still need to handle transition? For simplicity, skip.
                pass
            continue

        driver_tensor, order_tensor, driver_mask, order_mask, n_drivers_actual, n_orders_actual = build_state_features(
            eligible_drivers, available_orders, env.all_orders,
            len(env.drivers), max_orders, max_capacity, device
        )

        with torch.no_grad():
            real_q, dummy_q = q_net(driver_tensor, order_tensor)
            real_q = real_q.squeeze(0).detach().cpu().numpy()[:n_drivers_actual, :n_orders_actual]
            dummy_q = dummy_q.squeeze(0).detach().cpu().numpy()[:n_drivers_actual]

        feasible_real = compute_feasibility_mask(eligible_drivers, available_orders, max_orders)

        matched_pairs = select_action(real_q, dummy_q, feasible_real, epsilon)

        actions = {}
        n_real = len(available_orders)
        for d_idx, o_idx in matched_pairs:
            if o_idx < n_real:
                driver = eligible_drivers[d_idx]
                order = available_orders[o_idx]
                order_ids_array = np.full(env.config.driver_capacities[0], -1, dtype=np.int64)
                order_ids_array[0] = order.order_id
                actions[driver.driver_id] = {
                    "order_ids": order_ids_array,
                    "reposition_region": 0
                }

        current_state = (driver_tensor.detach().cpu(), order_tensor.detach().cpu(), driver_mask.detach().cpu(), order_mask.detach().cpu(), n_drivers_actual, n_orders_actual)
        current_action = matched_pairs
        current_feasible_real = feasible_real
        current_n_real = n_real

        obs, rewards, done, info = env.step(actions)
        step_reward = sum(rewards.values())

        if training and prev_state is not None:
            individual_rewards = [rewards.get(d.driver_id, 0.0) for d in eligible_drivers]
            # Transition structure (15 elements):
            # 0: prev_driver_tensor
            # 1: prev_order_tensor
            # 2: prev_driver_mask
            # 3: prev_order_mask
            # 4: prev_action (list of pairs)
            # 5: global_reward
            # 6: individual_rewards (list)
            # 7: curr_driver_tensor
            # 8: curr_order_tensor
            # 9: curr_driver_mask
            # 10: curr_order_mask
            # 11: curr_action (list of pairs)
            # 12: curr_feasible_real (numpy array)
            # 13: curr_n_real (int)
            # 14: done (bool)
            transition = (
                prev_state[0], prev_state[1], prev_state[2], prev_state[3],
                prev_action,
                step_reward,
                individual_rewards,
                current_state[0], current_state[1], current_state[2], current_state[3],
                current_action,
                current_feasible_real,
                current_n_real,
                done
            )
            yield transition, step_reward, done
        elif not training:
            # For testing, yield a dummy transition (None) to keep the generator alive
            yield None, step_reward, done

        prev_state = current_state
        prev_action = current_action
        step += 1
        pbar.update(1)

    pbar.close()
    # No return value; statistics are collected from environment after iteration


def train_step_sarsa(q_net, target_net, optimizer, batch, gamma, device, reward_type):
    # Unpack batch according to transition structure
    s_driver = torch.cat([b[0] for b in batch], dim=0).to(device)
    s_order = torch.cat([b[1] for b in batch], dim=0).to(device)
    a_pairs = [b[4] for b in batch]  # prev_action
    global_r = torch.tensor([b[5] for b in batch], device=device, dtype=torch.float32)
    individual_r = [b[6] for b in batch]
    s_prime_driver = torch.cat([b[7] for b in batch], dim=0).to(device)
    s_prime_order = torch.cat([b[8] for b in batch], dim=0).to(device)
    a_prime_pairs = [b[11] for b in batch]  # curr_action
    s_prime_n_real = [b[13] for b in batch]  # curr_n_real
    done = torch.tensor([b[14] for b in batch], device=device, dtype=torch.float32)

    real_q, dummy_q = q_net(s_driver, s_order)
    real_q_prime, dummy_q_prime = target_net(s_prime_driver, s_prime_order)

    batch_size = real_q.shape[0]

    if reward_type == 'global':
        q_s_a = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            n_real = s_prime_n_real[i]
            for d_idx, o_idx in a_pairs[i]:
                d_idx = int(d_idx)
                o_idx = int(o_idx)
                if o_idx < n_real:
                    q_s_a[i] += real_q[i, d_idx, o_idx]
                else:
                    q_s_a[i] += dummy_q[i, d_idx]

        q_s_prime_a_prime = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            n_real = s_prime_n_real[i]
            for d_idx, o_idx in a_prime_pairs[i]:
                d_idx = int(d_idx)
                o_idx = int(o_idx)
                if o_idx < n_real:
                    q_s_prime_a_prime[i] += real_q_prime[i, d_idx, o_idx]
                else:
                    q_s_prime_a_prime[i] += dummy_q_prime[i, d_idx]

        target = global_r + gamma * q_s_prime_a_prime * (1 - done)
        loss = nn.MSELoss()(q_s_a, target)

    else:  # individual
        total_loss = 0.0
        count = 0
        for i in range(batch_size):
            n_drivers = len(individual_r[i])
            n_real = s_prime_n_real[i]
            curr_map = {}
            for d, o in a_pairs[i]:
                d = int(d)
                o = int(o)
                curr_map[d] = o
            next_map = {}
            for d, o in a_prime_pairs[i]:
                d = int(d)
                o = int(o)
                next_map[d] = o

            for d_idx in range(n_drivers):
                if d_idx not in curr_map:
                    continue
                o_idx = curr_map[d_idx]
                if o_idx < n_real:
                    q_s = real_q[i, d_idx, o_idx]
                else:
                    q_s = dummy_q[i, d_idx]

                if d_idx in next_map:
                    o_next = next_map[d_idx]
                    if o_next < n_real:
                        q_next = real_q_prime[i, d_idx, o_next]
                    else:
                        q_next = dummy_q_prime[i, d_idx]
                else:
                    q_next = 0.0

                target_val = individual_r[i][d_idx] + gamma * q_next * (1 - done[i])
                total_loss += (q_s - target_val) ** 2
                count += 1

        loss = total_loss / count if count > 0 else torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_ddqn(q_net, target_net, optimizer, batch, gamma, device, reward_type):
    # Unpack batch according to transition structure
    s_driver = torch.cat([b[0] for b in batch], dim=0).to(device)
    s_order = torch.cat([b[1] for b in batch], dim=0).to(device)
    a_pairs = [b[4] for b in batch]  # prev_action
    global_r = torch.tensor([b[5] for b in batch], device=device, dtype=torch.float32)
    individual_r = [b[6] for b in batch]
    s_prime_driver = torch.cat([b[7] for b in batch], dim=0).to(device)
    s_prime_order = torch.cat([b[8] for b in batch], dim=0).to(device)
    s_prime_feasible_real = [b[12] for b in batch]  # curr_feasible_real
    s_prime_n_real = [b[13] for b in batch]  # curr_n_real
    done = torch.tensor([b[14] for b in batch], device=device, dtype=torch.float32)

    real_q, dummy_q = q_net(s_driver, s_order)
    real_q_prime, dummy_q_prime = target_net(s_prime_driver, s_prime_order)
    real_q_online, dummy_q_online = q_net(s_prime_driver, s_prime_order)

    batch_size = real_q.shape[0]

    if reward_type == 'global':
        q_s_a = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            n_real = s_prime_n_real[i]
            for d_idx, o_idx in a_pairs[i]:
                d_idx = int(d_idx)
                o_idx = int(o_idx)
                if o_idx < n_real:
                    q_s_a[i] += real_q[i, d_idx, o_idx]
                else:
                    q_s_a[i] += dummy_q[i, d_idx]

        q_s_prime_max = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            n_drivers = len(individual_r[i])
            n_real_actual = s_prime_n_real[i]
            feasible = s_prime_feasible_real[i]
            real_online = real_q_online[i, :n_drivers, :n_real_actual].detach().cpu().numpy()
            dummy_online = dummy_q_online[i, :n_drivers].detach().cpu().numpy()
            total_q = np.zeros((n_drivers, n_real_actual + 1))
            total_q[:, :n_real_actual] = real_online
            total_q[:, n_real_actual] = dummy_online
            feasible_total = np.zeros((n_drivers, n_real_actual + 1), dtype=bool)
            feasible_total[:, :n_real_actual] = feasible
            feasible_total[:, n_real_actual] = True
            cost = -total_q
            cost[~feasible_total] = 1e9
            d_idx, o_idx = linear_sum_assignment(cost)
            total = 0.0
            for d, o in zip(d_idx, o_idx):
                d = int(d)
                o = int(o)
                if o < n_real_actual:
                    total += real_q_prime[i, d, o].item()
                else:
                    total += dummy_q_prime[i, d].item()
            q_s_prime_max[i] = total

        target = global_r + gamma * q_s_prime_max * (1 - done)
        loss = nn.MSELoss()(q_s_a, target)

    else:  # individual
        total_loss = 0.0
        count = 0
        for i in range(batch_size):
            n_drivers = len(individual_r[i])
            n_real_actual = s_prime_n_real[i]
            feasible = s_prime_feasible_real[i]
            curr_map = {}
            for d, o in a_pairs[i]:
                d = int(d)
                o = int(o)
                curr_map[d] = o

            # Compute optimal next actions using online network
            real_online = real_q_online[i, :n_drivers, :n_real_actual].detach().cpu().numpy()
            dummy_online = dummy_q_online[i, :n_drivers].detach().cpu().numpy()
            total_q = np.zeros((n_drivers, n_real_actual + 1))
            total_q[:, :n_real_actual] = real_online
            total_q[:, n_real_actual] = dummy_online
            feasible_total = np.zeros((n_drivers, n_real_actual + 1), dtype=bool)
            feasible_total[:, :n_real_actual] = feasible
            feasible_total[:, n_real_actual] = True
            cost = -total_q
            cost[~feasible_total] = 1e9
            d_idx, o_idx = linear_sum_assignment(cost)
            next_map = {}
            for d, o in zip(d_idx, o_idx):
                next_map[int(d)] = int(o)

            for d in range(n_drivers):
                if d not in curr_map:
                    continue
                o = curr_map[d]
                if o < n_real_actual:
                    q_s = real_q[i, d, o]
                else:
                    q_s = dummy_q[i, d]

                if d in next_map:
                    o_next = next_map[d]
                    if o_next < n_real_actual:
                        q_next = real_q_prime[i, d, o_next]
                    else:
                        q_next = dummy_q_prime[i, d]
                else:
                    q_next = 0.0

                target_val = individual_r[i][d] + gamma * q_next * (1 - done[i])
                total_loss += (q_s - target_val) ** 2
                count += 1

        loss = total_loss / count if count > 0 else torch.tensor(0.0, device=device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def soft_update(target, source, tau):
    """Soft update target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def generate_train_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = end - start
    dates = []
    for i in range(delta.days + 1):
        date = start + timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    return dates


def main():
    args = parse_args()

    tee = Tee(args.output)
    sys.stdout = tee
    start_time = time.time()

    try:
        print("=" * 60)
        print(f"Deep {args.algorithm.upper()} Configuration")
        print("=" * 60)
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("=" * 60)
        print()

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        driver_feat_dim = 2 + 1 + 3 + args.driver_capacity * 5
        order_feat_dim = 6

        q_network = QNetwork(driver_feat_dim, order_feat_dim, args.hidden_dim).to(device)
        target_network = QNetwork(driver_feat_dim, order_feat_dim, args.hidden_dim).to(device)
        target_network.load_state_dict(q_network.state_dict())
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

        replay_buffer = deque(maxlen=args.buffer_size)

        epsilon = args.epsilon
        global_step = 0
        total_train_steps = 0
        all_episode_losses = []

        train_dates = generate_train_dates(args.train_start_date, args.train_end_date)
        print(f"Training dates: {train_dates}")

        for episode in range(args.num_episodes):
            print(f"\n--- Episode {episode+1} ---")

            train_date = random.choice(train_dates)

            env = create_env(
                tlc_file=args.tlc_file,
                date=train_date,
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
                seed=args.seed + episode,
                cache_dir=args.cache_dir
            )

            episode_losses = []
            episode_reward = 0
            step_count = 0

            # Run episode step by step (training)
            for transition, step_reward, done in run_episode(env, q_network, epsilon, args.max_orders, args.driver_capacity, device, training=True):
                episode_reward += step_reward
                step_count += 1
                replay_buffer.append(transition)
                global_step += 1

                # Train if conditions met
                if len(replay_buffer) >= args.train_start and global_step % args.train_freq == 0:
                    batch = random.sample(replay_buffer, min(args.batch_size, len(replay_buffer)))
                    if args.algorithm == 'sarsa':
                        loss = train_step_sarsa(q_network, target_network, optimizer, batch, args.gamma, device, args.reward_type)
                    else:
                        loss = train_step_ddqn(q_network, target_network, optimizer, batch, args.gamma, device, args.reward_type)
                    episode_losses.append(loss)
                    total_train_steps += 1

                    # Soft update target network
                    soft_update(target_network, q_network, args.tau)

            # Episode finished, get final stats from env
            completed_count = len(env.completed_orders)
            confirmed_count = len([o for o in env.all_orders if o.is_assigned])

            pickup_times = [o.pickup_time - o.request_time for o in env.completed_orders if o.pickup_time is not None]
            delivery_times = [o.dropoff_time - o.pickup_time for o in env.completed_orders if o.dropoff_time is not None and o.pickup_time is not None]

            avg_pickup = np.mean(pickup_times) if pickup_times else 0.0
            avg_delivery = np.mean(delivery_times) if delivery_times else 0.0

            print(f"Training episode {episode+1}: date={train_date}, reward={episode_reward:.2f}, completed={completed_count}, confirmed={confirmed_count}, avg_pickup={avg_pickup:.2f}s, avg_delivery={avg_delivery:.2f}s")

            if episode_losses:
                avg_loss = np.mean(episode_losses)
                all_episode_losses.append(avg_loss)
                print(f"Training loss: {avg_loss:.6f}")

            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

            if (episode + 1) % args.test_episode_freq == 0:
                print("\n--- Testing Episode ---")
                test_env = create_env(
                    tlc_file=args.tlc_file,
                    date=args.test_date,
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
                    seed=args.seed + episode + 1000,
                    cache_dir=args.cache_dir
                )

                test_reward = 0
                for _, step_reward, _ in run_episode(test_env, q_network, epsilon=0.0, max_orders=args.max_orders,
                                                     max_capacity=args.driver_capacity, device=device, training=False):
                    test_reward += step_reward

                # Get test stats
                test_completed = len(test_env.completed_orders)
                test_confirmed = len([o for o in test_env.all_orders if o.is_assigned])

                test_pickup_times = [o.pickup_time - o.request_time for o in test_env.completed_orders if o.pickup_time is not None]
                test_delivery_times = [o.dropoff_time - o.pickup_time for o in test_env.completed_orders if o.dropoff_time is not None and o.pickup_time is not None]
                test_avg_pickup = np.mean(test_pickup_times) if test_pickup_times else 0.0
                test_avg_delivery = np.mean(test_delivery_times) if test_delivery_times else 0.0

                detour_times = compute_detour_times(test_env.completed_orders, test_env.config.distance_calc, args.driver_speed)
                test_avg_detour = np.mean(detour_times) if detour_times else 0.0

                print(f"Test episode: reward={test_reward:.2f}, completed={test_completed}, confirmed={test_confirmed}")
                print(f"Test avg pickup time: {test_avg_pickup:.2f}s, avg delivery time: {test_avg_delivery:.2f}s, avg detour time: {test_avg_detour:.2f}s")

        if all_episode_losses:
            print("\nEpisode-wise average loss:")
            for i, loss in enumerate(all_episode_losses):
                print(f"Episode {i+1}: {loss:.6f}")

        elapsed = time.time() - start_time
        print(f"\nTotal training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    finally:
        tee.close()


if __name__ == "__main__":
    main()