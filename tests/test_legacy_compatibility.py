"""Default-path and public positional-API compatibility regressions.

The trajectory fingerprints were obtained from the original public simulator
using only procedural demand and a seeded built-in policy.
"""

import hashlib
import inspect
import json
import unittest
from dataclasses import fields

import numpy as np

from ride_gym.entities import Driver, Order, TaskPoint
from ride_gym.enums import DriverStatus, OrderStatus
from ride_gym.env import RidePoolEnv
from ride_gym.order_generator import RandomOrderGenerator
from ride_gym.policies import RandomConflictFreePolicy
from ride_gym.rewards import DefaultRewardFunction
from ride_gym.road_network import ManhattanNetwork
from ride_gym.routing import GreedyInsertionPlanner


def canonical(value):
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def legacy_trajectory_hash(seed):
    area = (0, 0, 10, 10)
    env = RidePoolEnv(
        area=area,
        num_drivers=10,
        dt=0.5,
        horizon=10,
        order_timeout=3,
        order_generator=RandomOrderGenerator(
            area=area, horizon=10, num_orders=40, max_party_size=3
        ),
        road_network=ManhattanNetwork(speed=1),
    )
    policy = RandomConflictFreePolicy(seed=77)
    observations, info = env.reset(seed=seed)
    trace = [(observations, info)]
    for _ in range(20):
        actions = policy.act(observations)
        observations, rewards, dones, info = env.step(actions)
        trace.append((actions, observations, rewards, dones, info))
    encoded = json.dumps(canonical(trace), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class LegacyCompatibilityTests(unittest.TestCase):
    def test_default_seeded_rollouts_are_unchanged(self):
        expected = {
            0: "15c5bfddaf5710614961213524708500823e3d49f3569c1e1f8642fa10ef62ec",
            42: "fec8b4d1c64f16616d6dfa92477204f5f5c90d29fbf01c7986acaf02bbdb238c",
        }
        for seed, digest in expected.items():
            with self.subTest(seed=seed):
                self.assertEqual(legacy_trajectory_hash(seed), digest)

    def test_all_original_environment_positional_parameters_still_bind(self):
        names = (
            "area",
            "num_drivers",
            "driver_capacity",
            "driver_capacities",
            "driver_speeds",
            "dt",
            "horizon",
            "order_timeout",
            "order_generator",
            "road_network",
            "routes_planner",
            "reward_function",
            "relocation_points",
            "relocation_grid",
            "relocation_centroids",
            "relocation_adjacency",
            "relocation_neighbours_k",
            "seed",
        )
        markers = tuple(object() for _ in names)
        bound = inspect.signature(RidePoolEnv).bind(*markers)
        self.assertEqual(tuple(bound.arguments), names)
        for name, marker in zip(names, markers):
            self.assertIs(bound.arguments[name], marker)

        area = (0, 0, 4, 4)
        generator = RandomOrderGenerator(area, 2, 0)
        network = ManhattanNetwork()
        planner = GreedyInsertionPlanner()
        reward = DefaultRewardFunction()
        values = (
            area,
            2,
            4,
            [2, 3],
            [None, 1.5],
            0.5,
            2.0,
            None,
            generator,
            network,
            planner,
            reward,
            [(1, 1), (2, 2)],
            (3, 3),
            None,
            4,
            1,
            123,
        )
        env = RidePoolEnv(*values)
        self.assertIs(env.order_generator, generator)
        self.assertIs(env.network, network)
        self.assertIs(env.planner, planner)
        self.assertIs(env.reward_function, reward)
        self.assertEqual(env.driver_capacities, [2, 3])
        self.assertEqual(env.driver_speeds, [None, 1.5])
        self.assertEqual(env._seed, 123)

    def test_driver_original_dataclass_positional_semantics_are_preserved(self):
        names = (
            "driver_id",
            "location",
            "capacity",
            "speed",
            "status",
            "onboard_passengers",
            "assigned_orders",
            "task_points",
            "route_nodes",
            "relocation_target",
            "node_idx",
            "edge_from",
            "edge_to",
            "edge_pos_m",
        )
        values = (
            7,
            (1.0, 2.0),
            5,
            1.25,
            DriverStatus.TO_DROPOFF,
            2,
            [9],
            [TaskPoint("dropoff", 9, (3.0, 4.0))],
            [(2.0, 3.0)],
            None,
            3,
            None,
            None,
            0.0,
        )
        self.assertEqual(tuple(field.name for field in fields(Driver))[: len(names)], names)
        driver = Driver(*values)
        for name, value in zip(names, values):
            self.assertEqual(getattr(driver, name), value)

    def test_order_and_stop_original_dataclass_positionals_are_preserved(self):
        names = (
            "order_id",
            "origin",
            "destination",
            "request_time",
            "num_passengers",
            "status",
            "assigned_driver",
            "pickup_time",
            "dropoff_time",
            "cancel_time",
            "eta",
        )
        values = (9, (1.0, 2.0), (3.0, 4.0), 1.5, 2, OrderStatus.ASSIGNED, 7, None, None, None, 4.5)
        self.assertEqual(tuple(field.name for field in fields(Order))[: len(names)], names)
        order = Order(*values)
        for name, value in zip(names, values):
            self.assertEqual(getattr(order, name), value)
        stop = TaskPoint("pickup", 9, (1.0, 2.0))
        self.assertEqual((stop.kind, stop.order_id, stop.location), ("pickup", 9, (1.0, 2.0)))


if __name__ == "__main__":
    unittest.main()
