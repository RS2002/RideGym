"""Synthetic state round-trips; no external datasets or optional dependencies."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from ride_gym import RidePoolEnv
from ride_gym.entities import Order
from ride_gym.order_generator import OrderGenerator
from ride_gym.road_network import EuclideanNetwork, PathResult
from ride_gym.warmup import (
    capture_state,
    load_snapshot,
    restore_state,
    save_snapshot,
    warmup,
)


class SyntheticOrders(OrderGenerator):
    def generate(self):
        return [Order(0, (2.0, 0.0), (7.0, 0.0), 0.0), Order(1, (4.0, 1.0), (8.0, 1.0), 3.0)]


def make_env(**options):
    return RidePoolEnv(
        area=(0.0, 0.0, 10.0, 10.0),
        num_drivers=2,
        dt=0.5,
        horizon=12.0,
        order_timeout=None,
        order_generator=SyntheticOrders(),
        **options,
    )


class TinyGraph(EuclideanNetwork):
    node_coords = ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0))

    def snap(self, coord):
        return min(range(3), key=lambda i: abs(self.node_coords[i][0] - coord[0]))

    def node_coord(self, index):
        return self.node_coords[index]

    def node_distance(self, source, destination):
        return abs(self.node_coords[source][0] - self.node_coords[destination][0])

    def node_path(self, source, destination):
        if source == destination:
            return [source]
        step = 1 if source < destination else -1
        return list(range(source, destination + step, step))

    def shortest_path(self, origin, destination):
        indices = self.node_path(self.snap(origin), self.snap(destination))
        distance = self.node_distance(indices[0], indices[-1])
        return PathResult(
            distance, self.travel_time(distance), [self.node_coord(index) for index in indices]
        )


class GraphOrders(OrderGenerator):
    def generate(self):
        return [Order(0, (4.0, 0.0), (8.0, 0.0), 0.0)]


class WarmupTests(unittest.TestCase):
    def test_snapshot_is_isolated_and_json_roundtrips(self):
        env = make_env()
        env.reset(seed=7)
        env.step({0: {"orders": [0]}})
        snapshot = capture_state(env)
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        original = env.drivers[0].location
        snapshot["drivers"][0]["location"][0] = -100
        self.assertEqual(env.drivers[0].location, original)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            save_snapshot(path, capture_state(env))
            self.assertEqual(load_snapshot(path), capture_state(env))
            self.assertFalse(path.with_name("state.json.tmp").exists())

    def test_restore_continues_pending_and_future_orders_and_rng(self):
        env = make_env()
        env.reset(seed=11)
        env.step({0: {"orders": [0]}})
        snapshot = capture_state(env)
        resumed = make_env()
        resumed.reset(seed=99)
        restore_state(resumed, snapshot)
        self.assertEqual(capture_state(resumed), snapshot)
        for _ in range(15):
            action = {1: {"orders": [1]}} if 1 in env._pending_ids else {}
            self.assertEqual(env.step(action), resumed.step(action))
        np.testing.assert_array_equal(env._rng.random(10), resumed._rng.random(10))

    def test_invalid_configuration_and_lifecycle_leave_env_unchanged(self):
        env = make_env()
        env.reset(seed=5)
        original = capture_state(env)
        invalid = deepcopy(original)
        invalid["configuration"]["dt"] = 9.0
        with self.assertRaises(ValueError):
            restore_state(env, invalid)
        self.assertEqual(capture_state(env), original)
        invalid = deepcopy(original)
        invalid["orders"][0]["status"] = "assigned"
        invalid["orders"][0]["assigned_driver"] = 0
        with self.assertRaises(ValueError):
            restore_state(env, invalid)
        self.assertEqual(capture_state(env), original)

    def test_statistics_granularity_must_match_before_restore(self):
        for source_mode, target_mode in (
            ("decision", "continuous"),
            ("continuous", "decision"),
        ):
            with self.subTest(source=source_mode, target=target_mode):
                source = make_env(statistics_granularity=source_mode)
                source.reset(seed=5)
                source.step({0: {"orders": [0]}})
                snapshot = capture_state(source)
                self.assertEqual(snapshot["configuration"]["statistics_granularity"], source_mode)

                target = make_env(statistics_granularity=target_mode)
                target.reset(seed=9)
                original = capture_state(target)
                with self.assertRaisesRegex(ValueError, "configuration"):
                    restore_state(target, snapshot)
                self.assertEqual(capture_state(target), original)

    def test_continuous_statistics_snapshot_continues_identically(self):
        source = make_env(statistics_granularity="continuous")
        source.reset(seed=5)
        source.step({0: {"orders": [0]}})
        snapshot = capture_state(source)
        target = make_env(statistics_granularity="continuous")
        target.reset(seed=9)
        restore_state(target, snapshot)
        self.assertEqual(capture_state(target), snapshot)
        for _ in range(5):
            self.assertEqual(source.step({}), target.step({}))

    def test_invalid_pending_stream_is_rejected(self):
        env = make_env()
        env.reset(seed=1)
        invalid = capture_state(env)
        invalid["pending_order_ids"].append(1)
        with self.assertRaises(ValueError):
            restore_state(env, invalid)

    def test_graph_mid_edge_position_survives_restore(self):
        def graph_env():
            return RidePoolEnv(
                area=(0.0, 0.0, 8.0, 1.0),
                num_drivers=1,
                dt=1.0,
                horizon=10.0,
                order_timeout=None,
                road_network=TinyGraph(speed=1.0),
                order_generator=GraphOrders(),
            )

        env = graph_env()
        env.reset(seed=4)
        env.drivers[0].location = (0.0, 0.0)
        env.drivers[0].node_idx = 0
        env.step({0: {"orders": [0]}})
        self.assertEqual(env.drivers[0].edge_pos_m, 1.0)
        saved = capture_state(env)
        resumed = graph_env()
        resumed.reset(seed=20)
        restore_state(resumed, saved)
        self.assertEqual(resumed.drivers[0].edge_from, 0)
        self.assertEqual(resumed.drivers[0].edge_to, 1)
        for _ in range(8):
            self.assertEqual(env.step({}), resumed.step({}))

    def test_warmup_is_an_ordinary_episode_prefix(self):
        env = make_env()
        env.reset(seed=3)
        obs, info, snapshot = warmup(env, lambda _: {}, steps=4)
        self.assertEqual(env.time, 2.0)
        self.assertEqual(info["time"], 2.0)
        self.assertEqual(obs, env._build_observations())
        self.assertEqual(snapshot, capture_state(env))
        with self.assertRaises(ValueError):
            warmup(env, lambda _: {}, steps=100)
        with self.assertRaises(ValueError):
            warmup(env, lambda _: {}, steps=True)

    def test_behavior_offers_survive_snapshot_and_resume(self):
        from ride_gym.acceptance import AcceptanceProbabilities, PerDriverAcceptancePolicy
        from ride_gym.response import FixedResponseTimePolicy

        def behavior_env():
            return make_env(
                acceptance_policy=PerDriverAcceptancePolicy({0: AcceptanceProbabilities(0.7)}),
                response_time_policy=FixedResponseTimePolicy(90),
                acceptance_seed=19,
                response_seed=23,
            )

        env = behavior_env()
        env.reset(seed=8)
        env.step({0: {"orders": [0]}})
        self.assertTrue(env._pending_responses)
        snapshot = capture_state(env)
        resumed = behavior_env()
        resumed.reset(seed=100)
        restore_state(resumed, snapshot)
        self.assertEqual(capture_state(resumed), snapshot)
        for _ in range(6):
            self.assertEqual(env.step({}), resumed.step({}))

    def test_pending_response_flags_must_match_saved_offers(self):
        from ride_gym.response import FixedResponseTimePolicy

        env = make_env(response_time_policy=FixedResponseTimePolicy(90))
        env.reset(seed=3)
        env.step({0: {"orders": [0]}})
        original = capture_state(env)
        for driver_index, value in ((0, False), (1, True)):
            invalid = deepcopy(original)
            invalid["drivers"][driver_index]["awaiting_response"] = value
            with self.assertRaisesRegex(ValueError, "awaiting_response"):
                restore_state(env, invalid)
            self.assertEqual(capture_state(env), original)
        invalid = deepcopy(original)
        invalid["response"]["pending"] = []
        with self.assertRaisesRegex(ValueError, "awaiting_response"):
            restore_state(env, invalid)
        self.assertEqual(capture_state(env), original)

    def test_custom_shift_configuration_is_checked(self):
        class ConfiguredShift:
            def __init__(self, start):
                self.start = start

            def configuration(self):
                return {"start": self.start}

            def update(self, drivers, time):
                for driver in drivers:
                    driver.on_duty = time >= self.start

        env = make_env(shift_policy=ConfiguredShift(2.0))
        env.reset(seed=2)
        saved = capture_state(env)
        other = make_env(shift_policy=ConfiguredShift(3.0))
        other.reset(seed=2)
        original = capture_state(other)
        with self.assertRaisesRegex(ValueError, "configuration"):
            restore_state(other, saved)
        self.assertEqual(capture_state(other), original)

    def test_snapshot_retains_shift_windows_and_service_state(self):
        from ride_gym.labor_source import ExplicitLaborSource, ShiftWindow

        source = ExplicitLaborSource(
            [
                {
                    "x": 1.0,
                    "y": 0.0,
                    "availability_windows": [ShiftWindow(0, 5), ShiftWindow(8, 12)],
                },
                {"x": 2.0, "y": 0.0},
            ]
        )
        env = make_env(labor_source=source)
        env.reset(seed=2)
        saved = capture_state(env)
        env.step({})
        restore_state(env, saved)
        self.assertEqual(
            env.drivers[0].availability_windows, (ShiftWindow(0, 5), ShiftWindow(8, 12))
        )
        self.assertEqual(capture_state(env), saved)

    def test_in_progress_stop_service_survives_restore(self):
        from ride_gym.labor_source import ExplicitLaborSource

        source = ExplicitLaborSource(
            [
                {"x": 1.0, "y": 0.0, "pickup_service_time": 1.5},
                {"x": 0.0, "y": 1.0},
            ]
        )
        env = make_env(labor_source=source)
        env.reset(seed=1)
        env.step({0: {"orders": [0]}})
        env.step({})
        self.assertIsNotNone(env.drivers[0].service_until)
        saved = capture_state(env)
        resumed = make_env(labor_source=source)
        resumed.reset(seed=2)
        restore_state(resumed, saved)
        self.assertEqual(resumed.drivers[0].service_until, env.drivers[0].service_until)
        for _ in range(6):
            self.assertEqual(env.step({}), resumed.step({}))


if __name__ == "__main__":
    unittest.main()
