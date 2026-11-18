"""Prediction/execution consistency and transactional assignment regressions."""

import math
import pickle
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from benchmark.recorder import EpisodeRecorder
from ride_gym.entities import Order
from ride_gym.enums import OrderStatus
from ride_gym.env import RidePoolEnv
from ride_gym.exceptions import InvalidActionError
from ride_gym.order_generator import OrderGenerator
from ride_gym.response import FixedResponseTimePolicy
from ride_gym.rewards import DefaultRewardFunction
from ride_gym.road_network import EuclideanNetwork
from ride_gym.routing import GreedyInsertionPlanner, RoutesPlanner


class FixedOrders(OrderGenerator):
    def __init__(self, specs):
        self.specs = specs

    def generate(self):
        return [Order(*spec) for spec in self.specs]


class SnappingLineGraph(EuclideanNetwork):
    """Three nodes ten units apart, including node-snapping distance queries."""

    node_coords = ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0))

    def snap(self, coord):
        return min(range(3), key=lambda node: abs(self.node_coords[node][0] - coord[0]))

    def node_coord(self, node):
        return self.node_coords[node]

    def node_distance(self, source, destination):
        return abs(destination - source) * 10.0

    def node_path(self, source, destination):
        step = 1 if destination >= source else -1
        return list(range(source, destination + step, step))

    def distance(self, origin, destination):
        return self.node_distance(self.snap(origin), self.snap(destination))

    def shortest_path(self, origin, destination):
        result = super().shortest_path(origin, destination)
        result.distance = self.distance(origin, destination)
        result.travel_time = self.travel_time(result.distance)
        return result


def make_env(specs=None, **kwargs):
    env = RidePoolEnv(
        num_drivers=1,
        horizon=60,
        order_timeout=None,
        order_generator=FixedOrders(specs or [(0, (0.0, 0.0), (10.0, 0.0), 0.0)]),
        **kwargs,
    )
    env.reset(seed=0)
    driver = env.drivers[0]
    driver.location = (0.0, 0.0)
    if env._is_graph_network():
        driver.node_idx = 0
    return env


def assignment_state(env, event):
    return deepcopy((env.orders, env.drivers, env._pending_ids, event))


class RoutePredictionTests(unittest.TestCase):
    def identical_graph_orders(self, network=None, **kwargs):
        return make_env(
            [(0, (10, 0), (20, 0), 0), (1, (10, 0), (20, 0), 0)],
            road_network=network or SnappingLineGraph(),
            **kwargs,
        )

    def test_identical_inserted_order_has_no_mid_edge_detour(self):
        env = self.identical_graph_orders()
        env.step({0: {"orders": [0]}})
        self.assertEqual(env.drivers[0].edge_pos_m, 1)
        _, _, _, info = env.step({0: {"orders": [1]}})
        event = info["events"][0]
        self.assertEqual(event["extra_detour_time"], 0)
        self.assertEqual(event["assigned_service_times"][1], 20)

    def test_prediction_keeps_remaining_edge_after_nearest_node_changes(self):
        env = self.identical_graph_orders(dt=6)
        env.step({0: {"orders": [0]}})
        self.assertEqual(env.drivers[0].edge_pos_m, 6)
        _, _, _, info = env.step({0: {"orders": [1]}})
        self.assertEqual(info["events"][0]["assigned_service_times"][1], 20)
        self.assertEqual(info["events"][0]["extra_detour_time"], 0)

    def test_mid_edge_vehicle_cannot_reverse_toward_new_pickup(self):
        env = make_env(
            [(0, (20, 0), (20, 0), 0), (1, (0, 0), (0, 0), 0)],
            road_network=SnappingLineGraph(),
        )
        env.step({0: {"orders": [0]}})
        _, times = env._prepare_invitation(env.drivers[0], [1])
        # Even a route visiting node 0 first must finish the active edge (9)
        # and return from node 10 (10), rather than reverse one unit to node 0.
        self.assertGreaterEqual(times[1]["pickup"], 19)
        env.step({0: {"orders": [1]}})
        self.assertEqual(env.drivers[0].edge_from, 0)
        self.assertEqual(env.drivers[0].edge_to, 1)
        self.assertEqual(env.drivers[0].edge_pos_m, 2)

    def test_mid_edge_prediction_uses_vehicle_speed(self):
        env = self.identical_graph_orders(driver_speeds=[2])
        env.step({0: {"orders": [0]}})
        _, _, _, info = env.step({0: {"orders": [1]}})
        self.assertEqual(info["events"][0]["assigned_service_times"][1], 10)
        self.assertEqual(info["events"][0]["extra_detour_time"], 0)
        self.assertEqual(info["events"][0]["assigned_solo_times"][1], 5)

    def test_fast_and_slow_vehicle_prediction_eta_and_reward(self):
        for speed in (2.0, 0.5):
            with self.subTest(speed=speed):
                env = make_env(
                    driver_speeds=[speed],
                    reward_function=DefaultRewardFunction(empty_move_penalty=0, idle_penalty=0),
                )
                _, rewards, _, info = env.step({0: {"orders": [0]}})
                event = info["events"][0]
                duration = 10 / speed
                self.assertEqual(event["assigned_service_times"][0], duration)
                self.assertEqual(event["assigned_solo_times"][0], duration)
                self.assertEqual(env.orders[0].eta, duration - 1)
                self.assertEqual(env.drivers[0].location, (speed, 0.0))
                self.assertAlmostEqual(rewards[0], 1 + 0.1 * duration)

    def test_service_timing_accounts_for_new_uncommitted_ready_order(self):
        env = make_env([(0, (0, 0), (10, 0), 0, 1)], driver_speeds=[2])
        env.orders[0].ready_time = 3
        _, _, _, info = env.step({0: {"orders": [0]}})
        self.assertEqual(info["events"][0]["assigned_service_times"][0], 8)
        self.assertEqual(env.orders[0].eta, 7)

    def test_invitation_uses_vehicle_speed_and_mid_edge_prefix(self):
        class Capture:
            def probability(self, context):
                self.context = context
                return 1

        acceptance = Capture()
        env = self.identical_graph_orders(driver_speeds=[2], acceptance_policy=acceptance)
        env.step({0: {"orders": [0]}})
        env.step({0: {"orders": [1]}})
        stop = next(s for s in acceptance.context.route if s.order_id == 1 and s.kind == "dropoff")
        self.assertEqual(stop.arrival_time, 9)

    def test_recorder_direct_baseline_uses_assigned_vehicle_speed(self):
        env = make_env(driver_speeds=[0.5])
        env.step({0: {"orders": [0]}})
        order = env.orders[0]
        # A 20-minute direct journey has two minutes of actual extra riding.
        order.pickup_time, order.dropoff_time = 1, 23
        recorder = EpisodeRecorder("synthetic")
        recorder._build_order_rows(env)
        self.assertEqual(recorder.order_rows[0]["detour_time"], 2)

    def test_real_graph_backend_mid_edge_regression(self):
        try:
            import networkx as nx

            from ride_gym.osmnx_network import OSMnxNetwork
        except ImportError:
            self.skipTest("networkx is an optional dependency")
        graph = nx.MultiDiGraph()
        for node in range(3):
            graph.add_node(node, x=node * 10.0, y=0.0)
        for first, second in ((0, 1), (1, 2)):
            graph.add_edge(first, second, length=10.0)
            graph.add_edge(second, first, length=10.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_graph.gpickle"
            with path.open("wb") as stream:
                pickle.dump(graph, stream)
            network = OSMnxNetwork(str(path), speed_kmh=0.06, cache_matrices=False)
            env = self.identical_graph_orders(network=network)
            env.step({0: {"orders": [0]}})
            _, _, _, info = env.step({0: {"orders": [1]}})
            self.assertEqual(info["events"][0]["extra_detour_time"], 0)
            self.assertEqual(info["events"][0]["assigned_service_times"][1], 20)


class AssignmentAtomicityTests(unittest.TestCase):
    def test_custom_plan_only_interface_remains_sufficient(self):
        class PlanOnly(RoutesPlanner):
            def plan(self, start, task_points, network):
                return GreedyInsertionPlanner().plan(start, task_points, network)

        env = make_env(routes_planner=PlanOnly())
        _, _, _, info = env.step({0: {"orders": [0]}})
        self.assertEqual(info["events"][0]["assigned_service_times"][0], 10)

    def test_custom_optimized_timing_override_remains_used(self):
        class Optimized(GreedyInsertionPlanner):
            calls = 0

            def plan_with_times(self, start, task_points, network):
                self.calls += 1
                return super().plan_with_times(start, task_points, network)

        planner = Optimized()
        env = make_env(routes_planner=planner)
        env.step({0: {"orders": [0]}})
        self.assertEqual(planner.calls, 1)

    def test_planner_exception_after_mutating_input_is_atomic(self):
        class Broken(RoutesPlanner):
            def plan(self, start, task_points, network):
                task_points[0].location = (999, 999)
                task_points.clear()
                raise RuntimeError("synthetic planner failure")

        env = make_env(routes_planner=Broken())
        event = env._new_event()
        before = assignment_state(env, event)
        with self.assertRaisesRegex(RuntimeError, "synthetic planner failure"):
            env._assign_orders(0, [0], event)
        self.assertEqual(assignment_state(env, event), before)

    def test_invalid_routes_and_times_do_not_commit(self):
        class Invalid(GreedyInsertionPlanner):
            mode = "missing"

            def plan_with_times(self, start, task_points, network):
                route, times = super().plan_with_times(start, task_points, network)
                if self.mode == "missing":
                    route.pop()
                elif self.mode == "duplicate":
                    route.append(route[0])
                elif self.mode == "precedence":
                    route.reverse()
                else:
                    times[0]["dropoff"] = self.mode
                return route, times

        for mode in ("missing", "duplicate", "precedence", -1, math.nan):
            with self.subTest(mode=mode):
                planner = Invalid()
                planner.mode = mode
                env = make_env(routes_planner=planner)
                event = env._new_event()
                before = assignment_state(env, event)
                with self.assertRaises(InvalidActionError):
                    env._assign_orders(0, [0], event)
                self.assertEqual(assignment_state(env, event), before)

    def test_direct_baseline_failure_does_not_commit(self):
        class BrokenBaseline(EuclideanNetwork):
            def shortest_path(self, origin, destination):
                raise RuntimeError("synthetic baseline failure")

        env = make_env(road_network=BrokenBaseline())
        event = env._new_event()
        before = assignment_state(env, event)
        with self.assertRaisesRegex(RuntimeError, "synthetic baseline failure"):
            env._assign_orders(0, [0], event)
        self.assertEqual(assignment_state(env, event), before)

    def test_known_response_replanning_failure_invalidates_entire_offer(self):
        env = make_env(
            [(0, (0, 0), (10, 0), 0), (1, (0, 0), (10, 0), 0)],
            response_time_policy=FixedResponseTimePolicy(120),
        )
        env.step({0: {"orders": [0, 1]}})

        class Broken(RoutesPlanner):
            def plan(self, start, task_points, network):
                raise InvalidActionError("synthetic invalid route")

        env.planner = Broken()
        env.time = 3
        events = {0: env._new_event()}
        env._settle_responses(events)
        self.assertFalse(env.pending_response_driver_ids)
        self.assertFalse(env.pending_response_order_ids)
        self.assertEqual(env.drivers[0].assigned_orders, [])
        self.assertTrue(all(o.status == OrderStatus.PENDING for o in env.orders.values()))
        self.assertEqual(events[0]["response_events"][0]["outcome"], "invalidated")

    def test_unknown_response_failure_preserves_offer_for_diagnosis(self):
        env = make_env(response_time_policy=FixedResponseTimePolicy(120))
        env.step({0: {"orders": [0]}})

        class Broken(RoutesPlanner):
            def plan(self, start, task_points, network):
                raise RuntimeError("synthetic unexpected failure")

        env.planner = Broken()
        env.time = 3
        pending = deepcopy(env._pending_responses)
        with self.assertRaisesRegex(RuntimeError, "synthetic unexpected failure"):
            env._settle_responses({0: env._new_event()})
        self.assertEqual(env._pending_responses, pending)
        self.assertTrue(env.drivers[0].awaiting_response)
        self.assertEqual(env.pending_response_order_ids, frozenset({0}))
        self.assertEqual(env.orders[0].status, OrderStatus.PENDING)


if __name__ == "__main__":
    unittest.main()
