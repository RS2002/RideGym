"""Reporting modes on deterministic, synthetic movement scenarios."""

import unittest
from copy import deepcopy
from dataclasses import asdict

from benchmark.recorder import EpisodeRecorder
from ride_gym.entities import Order
from ride_gym.env import RidePoolEnv
from ride_gym.labor_source import ExplicitLaborSource
from ride_gym.order_generator import OrderGenerator
from ride_gym.response import UniformResponseTimePolicy
from ride_gym.road_network import EuclideanNetwork


class FixedOrders(OrderGenerator):
    def __init__(self, orders):
        self.orders = orders

    def generate(self):
        return deepcopy(self.orders)


class SmallGraph(EuclideanNetwork):
    """A short line graph whose edge lengths differ from the decision step."""

    positions = (0.0, 0.25, 0.75, 1.0, 2.0)
    node_coords = ((0.0, 0.0), (0.25, 0.0), (0.75, 0.0), (1.0, 0.0), (2.0, 0.0))

    def snap(self, coord):
        return min(range(len(self.positions)), key=lambda i: abs(self.positions[i] - coord[0]))

    def node_coord(self, node):
        return (self.positions[node], 0.0)

    def node_distance(self, source, destination):
        return abs(self.positions[destination] - self.positions[source])

    def node_path(self, source, destination):
        step = 1 if destination >= source else -1
        return list(range(source, destination + step, step))


def make_env(orders=None, **kwargs):
    if orders is None:
        orders = [Order(0, (0.25, 0.0), (0.75, 0.0), 0.0)]
    options = dict(
        num_drivers=1,
        dt=1.0,
        horizon=10.0,
        order_timeout=None,
        order_generator=FixedOrders(orders),
        road_network=EuclideanNetwork(speed=1.0),
    )
    options.update(kwargs)
    env = RidePoolEnv(**options)
    env.reset(seed=7)
    env.drivers[0].location = (0.0, 0.0)
    if env._is_graph_network():
        env.drivers[0].node_idx = 0
    return env


def event(result):
    return result[3]["events"][0]


class StatisticsGranularityTests(unittest.TestCase):
    def assert_same_execution(self, coarse, fine, first, second):
        self.assertEqual(first[:3], second[:3])
        self.assertEqual(coarse.time, fine.time)
        self.assertEqual(asdict(coarse.drivers[0]), asdict(fine.drivers[0]))
        self.assertEqual(coarse._rng.bit_generator.state, fine._rng.bit_generator.state)
        for oid in coarse.orders:
            before, after = asdict(coarse.orders[oid]), asdict(fine.orders[oid])
            for name in ("pickup_time", "dropoff_time"):
                before.pop(name)
                after.pop(name)
            self.assertEqual(before, after)
        fine_event = dict(event(second))
        fine_event.pop("empty_distance_moved")
        fine_event.pop("occupied_distance_moved")
        self.assertEqual(event(first), fine_event)

    def test_default_matches_explicit_decision_mode(self):
        default = make_env()
        explicit = make_env(statistics_granularity="decision")
        self.assertEqual(default.statistics_granularity, "decision")
        result = default.step({0: {"orders": [0]}})
        self.assertEqual(result, explicit.step({0: {"orders": [0]}}))
        self.assertEqual(default.orders[0].pickup_time, 0.0)
        self.assertEqual(default.orders[0].dropoff_time, 0.0)
        self.assertNotIn("empty_distance_moved", event(result))

    def test_invalid_granularity_is_rejected(self):
        for value in (None, False, 0, "", "Continuous", [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "statistics_granularity"):
                    RidePoolEnv(statistics_granularity=value)

    def test_within_step_timestamps_and_segment_distances_on_both_networks(self):
        for network_type in (EuclideanNetwork, SmallGraph):
            with self.subTest(network=network_type.__name__):
                coarse = make_env(road_network=network_type())
                fine = make_env(road_network=network_type(), statistics_granularity="continuous")
                first = coarse.step({0: {"orders": [0]}})
                second = fine.step({0: {"orders": [0]}})
                self.assert_same_execution(coarse, fine, first, second)
                self.assertEqual(fine.orders[0].pickup_time, 0.25)
                self.assertEqual(fine.orders[0].dropoff_time, 0.75)
                self.assertEqual(event(second)["empty_distance_moved"], 0.25)
                self.assertEqual(event(second)["occupied_distance_moved"], 0.5)

    def test_cross_step_timestamps_and_partial_edge_distance_conservation(self):
        for network_type in (EuclideanNetwork, SmallGraph):
            with self.subTest(network=network_type.__name__):
                coarse = make_env(road_network=network_type(), dt=0.2)
                fine = make_env(
                    road_network=network_type(),
                    dt=0.2,
                    statistics_granularity="continuous",
                )
                empty, occupied = 0.0, 0.0
                for step in range(4):
                    action = {0: {"orders": [0]}} if step == 0 else {}
                    first, second = coarse.step(action), fine.step(action)
                    self.assert_same_execution(coarse, fine, first, second)
                    item = event(second)
                    empty += item["empty_distance_moved"]
                    occupied += item["occupied_distance_moved"]
                    self.assertAlmostEqual(
                        item["distance_moved"],
                        item["empty_distance_moved"] + item["occupied_distance_moved"],
                    )
                self.assertAlmostEqual(fine.orders[0].pickup_time, 0.25)
                self.assertAlmostEqual(fine.orders[0].dropoff_time, 0.75)
                self.assertAlmostEqual(coarse.orders[0].pickup_time, 0.2)
                self.assertAlmostEqual(coarse.orders[0].dropoff_time, 0.6)
                self.assertAlmostEqual(empty, 0.25)
                self.assertAlmostEqual(occupied, 0.5)

    def test_continuous_timestamps_use_effective_vehicle_speed(self):
        fine = make_env(driver_speeds=[2.0], statistics_granularity="continuous")
        result = fine.step({0: {"orders": [0]}})
        self.assertEqual(fine.orders[0].pickup_time, 0.125)
        self.assertEqual(fine.orders[0].dropoff_time, 0.375)
        self.assertEqual(event(result)["distance_moved"], 0.75)
        self.assertEqual(event(result)["empty_distance_moved"], 0.25)

    def test_service_clock_is_precise_in_both_reporting_modes(self):
        def service_env(mode):
            return make_env(
                [Order(0, (0.25, 0.0), (0.75, 0.0), 0.0, ready_time=0.4)],
                statistics_granularity=mode,
                labor_source=ExplicitLaborSource(
                    [
                        {
                            "location": (0.0, 0.0),
                            "pickup_service_time": 0.2,
                            "dropoff_service_time": 0.1,
                        }
                    ]
                ),
            )

        coarse, fine = service_env("decision"), service_env("continuous")
        first, second = coarse.step({0: {"orders": [0]}}), fine.step({0: {"orders": [0]}})
        self.assert_same_execution(coarse, fine, first, second)
        self.assertEqual(coarse.orders[0].pickup_time, 0.0)
        self.assertAlmostEqual(fine.orders[0].pickup_time, 0.6)
        self.assertAlmostEqual(fine.drivers[0].location[0], 0.65)
        first, second = coarse.step({}), fine.step({})
        self.assert_same_execution(coarse, fine, first, second)
        self.assertEqual(coarse.orders[0].dropoff_time, 1.0)
        self.assertAlmostEqual(fine.orders[0].dropoff_time, 1.2)

    def test_waiting_does_not_count_as_traveled_distance(self):
        env = make_env(
            [Order(0, (0.0, 0.0), (0.75, 0.0), 0.0, ready_time=2.0)],
            statistics_granularity="continuous",
        )
        result = env.step({0: {"orders": [0]}})
        self.assertEqual(event(result)["distance_moved"], 0.0)
        self.assertEqual(event(result)["empty_distance_moved"], 0.0)
        self.assertEqual(event(result)["occupied_distance_moved"], 0.0)
        self.assertIsNone(env.orders[0].pickup_time)

    def test_graph_service_wait_and_partial_edges_keep_exact_distance_split(self):
        env = make_env(
            [Order(0, (0.25, 0.0), (0.75, 0.0), 0.0, ready_time=0.4)],
            road_network=SmallGraph(),
            statistics_granularity="continuous",
            labor_source=ExplicitLaborSource(
                [{"location": (0.0, 0.0), "pickup_service_time": 0.2, "dropoff_service_time": 0.1}]
            ),
        )
        first = env.step({0: {"orders": [0]}})
        second = env.step({})
        self.assertAlmostEqual(env.orders[0].pickup_time, 0.6)
        self.assertAlmostEqual(env.orders[0].dropoff_time, 1.2)
        self.assertAlmostEqual(
            sum(event(result)["empty_distance_moved"] for result in (first, second)), 0.25
        )
        self.assertAlmostEqual(
            sum(event(result)["occupied_distance_moved"] for result in (first, second)), 0.5
        )

    def test_empty_segment_after_dropoff_does_not_change_legacy_reward(self):
        orders = [Order(0, (0.0, 0.0), (0.75, 0.0), 0.0), Order(1, (1.0, 0.0), (2.0, 0.0), 0.0)]
        coarse = make_env(orders, dt=0.5)
        fine = make_env(orders, dt=0.5, statistics_granularity="continuous")
        coarse.step({0: {"orders": [0, 1]}})
        fine.step({0: {"orders": [0, 1]}})
        first, second = coarse.step({}), fine.step({})
        self.assert_same_execution(coarse, fine, first, second)
        self.assertEqual(event(second)["empty_distance_moved"], 0.25)
        self.assertEqual(event(second)["occupied_distance_moved"], 0.25)
        self.assertFalse(event(second)["is_empty_move"])

    def test_recorder_uses_selected_mode_and_supports_old_events(self):
        for mode, expected_empty in (("decision", 0.75), ("continuous", 0.25)):
            with self.subTest(mode=mode):
                env = make_env(statistics_granularity=mode)
                result = env.step({0: {"orders": [0]}})
                recorder = EpisodeRecorder("synthetic")
                recorder.record_step(env, result[3], rewards=result[1])
                recorder.finalize(env)
                self.assertEqual(recorder.step_rows[0]["empty_distance_km"], expected_empty)
                self.assertEqual(recorder.driver_rows[0]["empty_distance_km"], expected_empty)
                self.assertEqual(
                    recorder.driver_rows[0]["occupied_distance_km"], 0.75 - expected_empty
                )

    def test_reporting_does_not_consume_response_randomness(self):
        def response_env(mode):
            return make_env(
                dt=0.2,
                statistics_granularity=mode,
                response_time_policy=UniformResponseTimePolicy(max_seconds=20.0),
                response_seed=19,
            )

        coarse, fine = response_env("decision"), response_env("continuous")
        for step in range(6):
            action = {0: {"orders": [0]}} if step == 0 else {}
            first, second = coarse.step(action), fine.step(action)
            self.assert_same_execution(coarse, fine, first, second)
            self.assertEqual(
                coarse._response_rng.bit_generator.state, fine._response_rng.bit_generator.state
            )


if __name__ == "__main__":
    unittest.main()
