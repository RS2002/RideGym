"""Synthetic integration coverage for opt-in fleet and offer lifecycles."""

import unittest
from copy import deepcopy

from ride_gym.acceptance import AcceptanceProbabilities, PerDriverAcceptancePolicy
from ride_gym.entities import Order
from ride_gym.enums import OrderStatus
from ride_gym.env import RidePoolEnv
from ride_gym.labor_source import ExplicitLaborSource
from ride_gym.order_generator import OrderGenerator, RandomOrderGenerator
from ride_gym.response import FixedResponseTimePolicy, UniformResponseTimePolicy
from ride_gym.rewards import DefaultRewardFunction
from ride_gym.road_network import EuclideanNetwork
from ride_gym.routing import RoutesPlanner


class FreshOrders(OrderGenerator):
    def __init__(self, orders):
        self.orders = deepcopy(orders)

    def generate(self):
        return deepcopy(self.orders)


class RecordingAcceptance:
    def __init__(self, probability=1.0):
        self.value = probability
        self.contexts = []

    def probability(self, context):
        self.contexts.append(context)
        return self.value


class LineGraph(EuclideanNetwork):
    """A tiny bidirectional three-node graph, with no external map assets."""

    node_coords = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    def snap(self, location):
        return min(
            range(len(self.node_coords)),
            key=lambda node: self.distance(location, self.node_coords[node]),
        )

    def node_coord(self, node):
        return self.node_coords[node]

    def node_distance(self, first, second):
        return abs(second - first)

    def node_path(self, first, second):
        step = 1 if second >= first else -1
        return list(range(first, second + step, step))


def make_env(orders, *, fleet=None, num_drivers=1, **kwargs):
    if fleet is None:
        fleet = [{"location": (0.0, 0.0), "capacity": 8} for _ in range(num_drivers)]
    values = dict(
        area=(0.0, 0.0, 30.0, 30.0),
        num_drivers=len(fleet),
        dt=0.5,
        horizon=20.0,
        order_timeout=None,
        order_generator=FreshOrders(orders),
        road_network=EuclideanNetwork(speed=1),
        labor_source=ExplicitLaborSource(fleet),
    )
    values.update(kwargs)
    env = RidePoolEnv(**values)
    env.reset(seed=12)
    return env


def default_orders(count=2):
    return [Order(i, (0.0, 0.0), (10.0 + i, 0.0), 0.0) for i in range(count)]


def event(result, driver_id=0):
    return result[3]["events"][driver_id]


class OfferLifecycleTests(unittest.TestCase):
    def test_fixed_twenty_seconds_commits_before_next_matching_without_moving_early(self):
        env = make_env(default_orders(), response_time_policy=FixedResponseTimePolicy(20))
        first = env.step({0: {"orders": [0]}})
        self.assertEqual(env.time, 0.5)
        self.assertEqual(env.drivers[0].location, (0.0, 0.0))
        self.assertEqual(env.orders[0].status, OrderStatus.ASSIGNED)
        self.assertIsNone(env.orders[0].pickup_time)
        self.assertEqual(event(first)["assignment_times"], {0: 0.5})
        self.assertEqual(env.pending_response_driver_ids, frozenset())
        self.assertTrue(first[0][0]["self"]["available"])

        second = env.step({0: {"orders": [1]}})
        self.assertEqual(event(second)["assignment_times"], {1: 1.0})
        self.assertEqual(env.drivers[0].location, (0.5, 0.0))
        self.assertEqual(env.orders[1].status, OrderStatus.ASSIGNED)

    def test_explicit_zero_waits_one_boundary_but_none_remains_immediate(self):
        immediate = make_env(default_orders(1))
        delayed = make_env(default_orders(1), response_time_policy=FixedResponseTimePolicy(0))
        first = immediate.step({0: {"orders": [0]}})
        second = delayed.step({0: {"orders": [0]}})
        self.assertEqual(immediate.orders[0].status, OrderStatus.ONBOARD)
        self.assertEqual(immediate.drivers[0].location, (0.5, 0.0))
        self.assertEqual(delayed.orders[0].status, OrderStatus.ASSIGNED)
        self.assertEqual(delayed.drivers[0].location, (0.0, 0.0))
        self.assertNotIn("response_events", event(first))
        self.assertEqual(event(second)["assignment_times"], {0: 0.5})

    def test_zero_probability_rejects_whole_offer_without_assignment_reward(self):
        reward = DefaultRewardFunction(
            assignment_bonus=5,
            revenue_coef=0,
            service_time_coef=0,
            detour_coef=0,
            empty_move_penalty=0,
            idle_penalty=0,
        )
        env = make_env(
            default_orders(),
            acceptance_policy=PerDriverAcceptancePolicy({0: {"general": 0}}),
            reward_function=reward,
        )
        result = env.step({0: {"orders": [0, 1]}})
        self.assertEqual(result[1][0], 0.0)
        self.assertEqual(event(result)["assigned_orders"], [])
        self.assertEqual(env.drivers[0].assigned_orders, [])
        self.assertEqual(env._acceptance_draws, 1)
        for order in env.orders.values():
            self.assertEqual(order.status, OrderStatus.PENDING)
            self.assertEqual(order.request_time, 0.0)
        self.assertFalse(event(result)["acceptance_events"][0]["accepted"])

    def test_one_probability_accepts_a_complete_bundle(self):
        env = make_env(
            default_orders(), acceptance_policy=PerDriverAcceptancePolicy({0: {"general": 1}})
        )
        result = env.step({0: {"orders": [0, 1]}})
        self.assertEqual(set(event(result)["assigned_orders"]), {0, 1})
        self.assertEqual(env._acceptance_draws, 1)
        self.assertTrue(event(result)["acceptance_events"][0]["accepted"])

    def test_single_and_bundle_only_count_new_orders(self):
        for new_ids, expected in (([1], False), ([1, 2], True)):
            with self.subTest(new_ids=new_ids):
                env = make_env(
                    default_orders(3),
                    acceptance_policy=PerDriverAcceptancePolicy(
                        {0: AcceptanceProbabilities(single=0, bundle=1)}
                    ),
                )
                env._assign_orders(0, [0], env._new_event(), _response_commit=True)
                result = env.step({0: {"orders": new_ids}})
                receipt = event(result)["acceptance_events"][0]
                self.assertEqual(receipt["kind"], "bundle" if len(new_ids) > 1 else "single")
                self.assertEqual(receipt["accepted"], expected)
                self.assertEqual(
                    set(event(result)["assigned_orders"]), set(new_ids) if expected else set()
                )

    def test_held_orders_and_waiting_driver_cannot_receive_duplicate_offers(self):
        policy = RecordingAcceptance()
        env = make_env(
            default_orders(),
            num_drivers=2,
            acceptance_policy=policy,
            response_time_policy=FixedResponseTimePolicy(90),
        )
        first = env.step({0: {"orders": [0]}})
        self.assertEqual(env.pending_response_order_ids, frozenset({0}))
        self.assertEqual(env.pending_response_driver_ids, frozenset({0}))
        self.assertEqual([o["order_id"] for o in first[0][0]["pending_orders"]], [1])
        self.assertNotIn("accepted", first[0][0]["self"]["pending_response"])
        env.step({0: {"orders": [1]}, 1: {"orders": [0]}})
        self.assertEqual(len(policy.contexts), 1)
        self.assertEqual(env.pending_response_order_ids, frozenset({0}))
        self.assertEqual(env.orders[1].status, OrderStatus.PENDING)

    def test_existing_work_moves_while_a_new_response_is_pending(self):
        env = make_env(default_orders(), response_time_policy=FixedResponseTimePolicy(45))
        env.step({0: {"orders": [0]}})
        env.step({})
        self.assertEqual(env.orders[0].status, OrderStatus.ASSIGNED)
        result = env.step({0: {"orders": [1]}})
        self.assertEqual(env.drivers[0].location, (0.5, 0.0))
        self.assertEqual(env.pending_response_order_ids, frozenset({1}))
        self.assertEqual(env.orders[0].status, OrderStatus.ONBOARD)
        self.assertEqual(env.orders[1].status, OrderStatus.PENDING)
        self.assertEqual(event(result)["assigned_orders"], [])
        env.step({})
        self.assertEqual(env.drivers[0].location, (1.0, 0.0))
        self.assertEqual(env.orders[1].status, OrderStatus.ASSIGNED)

    def test_cancellation_invalidates_complete_offer_without_partial_assignment(self):
        env = make_env(default_orders(), response_time_policy=FixedResponseTimePolicy(45))
        env.step({0: {"orders": [0, 1]}})
        env.orders[0].status = OrderStatus.CANCELLED
        env.orders[0].cancel_time = env.time
        env._pending_ids.remove(0)
        result = env.step({})
        self.assertEqual(event(result)["response_events"][-1]["outcome"], "invalidated")
        self.assertEqual(event(result)["assigned_orders"], [])
        self.assertEqual(env.orders[1].status, OrderStatus.PENDING)
        self.assertEqual(env.pending_response_order_ids, frozenset())

    def test_capacity_change_invalidates_whole_offer(self):
        env = make_env(default_orders(), response_time_policy=FixedResponseTimePolicy(45))
        env.step({0: {"orders": [0, 1]}})
        env.drivers[0].capacity = 1
        result = env.step({})
        self.assertEqual(event(result)["response_events"][-1]["outcome"], "invalidated")
        self.assertTrue(all(order.status == OrderStatus.PENDING for order in env.orders.values()))
        self.assertEqual(env.drivers[0].assigned_orders, [])

    def test_timeout_at_response_boundary_cannot_be_bypassed(self):
        env = make_env(
            default_orders(1), response_time_policy=FixedResponseTimePolicy(45), order_timeout=0.75
        )
        env.step({0: {"orders": [0]}})
        result = env.step({})
        self.assertEqual(event(result)["response_events"][-1]["outcome"], "invalidated")
        self.assertEqual(env.orders[0].status, OrderStatus.CANCELLED)
        self.assertEqual(event(result)["assigned_orders"], [])

    def test_early_timeout_releases_entire_offer_and_allows_snapshot_resume(self):
        from ride_gym.warmup import capture_state, restore_state

        env = make_env(
            default_orders(1), response_time_policy=FixedResponseTimePolicy(90), order_timeout=0.25
        )
        result = env.step({0: {"orders": [0]}})
        self.assertEqual(env.time, 0.5)
        receipt = event(result)["response_events"][-1]
        self.assertEqual(receipt["outcome"], "invalidated")
        self.assertLess(env.time, receipt["effective_time"])
        self.assertEqual(env.orders[0].status, OrderStatus.CANCELLED)
        self.assertEqual(env.pending_response_order_ids, frozenset())
        self.assertEqual(env.pending_response_driver_ids, frozenset())
        self.assertFalse(env.drivers[0].awaiting_response)
        saved = capture_state(env)
        resumed = make_env(
            default_orders(1), response_time_policy=FixedResponseTimePolicy(90), order_timeout=0.25
        )
        restore_state(resumed, saved)
        self.assertEqual(capture_state(resumed), saved)
        self.assertEqual(env.step({}), resumed.step({}))

    def test_shift_end_honors_existing_offer_but_stops_new_work(self):
        env = make_env(
            default_orders(),
            response_time_policy=FixedResponseTimePolicy(45),
            fleet=[{"location": (0.0, 0.0), "shift_end": 0.5}],
        )
        env.step({0: {"orders": [0]}})
        self.assertFalse(env.drivers[0].on_duty)
        second = env.step({0: {"orders": [1]}})
        self.assertEqual(event(second)["response_events"][-1]["outcome"], "committed")
        self.assertEqual(env.orders[0].status, OrderStatus.ASSIGNED)
        self.assertFalse(second[0][0]["self"]["available"])
        third = env.step({0: {"orders": [1]}})
        self.assertEqual(env.orders[1].status, OrderStatus.PENDING)
        self.assertEqual(event(third)["assigned_orders"], [])
        self.assertEqual(env.drivers[0].location, (0.5, 0.0))

    def test_response_rejection_preserves_original_request_time(self):
        env = make_env(
            default_orders(),
            response_time_policy=FixedResponseTimePolicy(20),
            acceptance_policy=PerDriverAcceptancePolicy({0: {"general": 0}}),
        )
        result = env.step({0: {"orders": [0, 1]}})
        self.assertEqual(event(result)["response_events"][-1]["outcome"], "rejected")
        for order in env.orders.values():
            self.assertEqual(order.status, OrderStatus.PENDING)
            self.assertEqual(order.request_time, 0.0)
        self.assertEqual(event(result)["assigned_orders"], [])


class ExtensionRandomnessTests(unittest.TestCase):
    def test_enabling_behavior_does_not_change_positions_demand_or_legacy_rng(self):
        values = dict(area=(0, 0, 10, 10), num_drivers=3, dt=0.5, horizon=5, order_timeout=None)
        environments = []
        for options in (
            {},
            {
                "acceptance_policy": PerDriverAcceptancePolicy({}),
                "response_time_policy": UniformResponseTimePolicy(),
            },
        ):
            env = RidePoolEnv(
                **values,
                **options,
                order_generator=RandomOrderGenerator(area=values["area"], horizon=5, num_orders=20),
            )
            env.reset(seed=77)
            environments.append(env)
        plain, behavior = environments
        self.assertEqual(
            [d.location for d in plain.drivers.values()],
            [d.location for d in behavior.drivers.values()],
        )
        self.assertEqual(plain._all_orders, behavior._all_orders)
        self.assertEqual(plain._rng.bit_generator.state, behavior._rng.bit_generator.state)

    def test_response_seed_does_not_change_acceptance_stream(self):
        first = make_env(
            default_orders(1),
            acceptance_seed=77,
            response_seed=1,
            acceptance_policy=PerDriverAcceptancePolicy({0: {"general": 0.5}}),
            response_time_policy=UniformResponseTimePolicy(),
        )
        second = make_env(
            default_orders(1),
            acceptance_seed=77,
            response_seed=2,
            acceptance_policy=PerDriverAcceptancePolicy({0: {"general": 0.5}}),
            response_time_policy=UniformResponseTimePolicy(),
        )
        first.step({0: {"orders": [0]}})
        second.step({0: {"orders": [0]}})
        self.assertEqual(
            first._acceptance_rng.bit_generator.state, second._acceptance_rng.bit_generator.state
        )
        self.assertEqual(first._acceptance_draws, 1)
        self.assertEqual(second._acceptance_draws, 1)

    def test_seeded_reset_repeats_offer_draws_and_response_durations(self):
        env = make_env(
            default_orders(6),
            acceptance_policy=PerDriverAcceptancePolicy({0: {"general": 0.5}}),
            response_time_policy=UniformResponseTimePolicy(10),
        )

        def rollout():
            records = []
            for _ in range(6):
                candidates = [
                    oid for oid in env._pending_ids if oid not in env.pending_response_order_ids
                ]
                action = {0: {"orders": candidates[:1]}}
                result = env.step(action)
                records.extend(deepcopy(event(result).get("response_events", [])))
            return records

        first = rollout()
        env.reset(seed=12)
        self.assertEqual(rollout(), first)


class ServiceAndPlannerTests(unittest.TestCase):
    def test_ready_before_request_or_episode_start_means_already_ready(self):
        env = make_env(
            [Order(0, (0.0, 0.0), (0.25, 0.0), 0.0, ready_time=-1)],
            statistics_granularity="continuous",
        )
        env.step({0: {"orders": [0]}})
        self.assertEqual(env.orders[0].pickup_time, 0.0)
        self.assertEqual(env.orders[0].dropoff_time, 0.25)

    def test_readiness_prevents_early_pickup_and_retains_remaining_step_time(self):
        env = make_env(
            [Order(0, (0.0, 0.0), (1.0, 0.0), 0.0, ready_time=1.25)],
            statistics_granularity="continuous",
        )
        env.step({0: {"orders": [0]}})
        env.step({})
        self.assertIsNone(env.orders[0].pickup_time)
        self.assertEqual(env.drivers[0].location, (0.0, 0.0))
        env.step({})
        self.assertEqual(env.orders[0].pickup_time, 1.25)
        self.assertEqual(env.drivers[0].location, (0.25, 0.0))

    def test_readiness_is_included_in_planned_service_time_and_eta(self):
        env = make_env([Order(0, (0.0, 0.0), (1.0, 0.0), 0.0, ready_time=1.25)])
        result = env.step({0: {"orders": [0]}})
        self.assertAlmostEqual(event(result)["assigned_service_times"][0], 2.25)
        self.assertAlmostEqual(env.orders[0].eta, 1.75)

    def test_zero_time_stops_at_boundary_release_capacity_before_next_action(self):
        for network in (EuclideanNetwork(), LineGraph()):
            with self.subTest(network=type(network).__name__):
                env = make_env(
                    [
                        Order(0, (0.0, 0.0), (0.0, 0.0), 0.0, ready_time=1.0),
                        Order(1, (0.0, 0.0), (1.0, 0.0), 1.0),
                    ],
                    fleet=[{"location": (0.0, 0.0), "capacity": 1}],
                    road_network=network,
                    statistics_granularity="continuous",
                )
                env.step({0: {"orders": [0]}})
                env.step({})
                self.assertEqual(env.time, 1.0)
                self.assertEqual(env.orders[0].status, OrderStatus.COMPLETED)
                self.assertEqual(env.orders[0].pickup_time, 1.0)
                self.assertEqual(env.orders[0].dropoff_time, 1.0)
                self.assertEqual(env.drivers[0].onboard_passengers, 0)
                self.assertEqual(env.drivers[0].location, (0.0, 0.0))
                result = env.step({0: {"orders": [1]}})
                self.assertEqual(event(result)["assigned_orders"], [1])

    def test_pickup_and_dropoff_service_span_ticks_on_abstract_and_graph_networks(self):
        for network in (EuclideanNetwork(), LineGraph()):
            with self.subTest(network=type(network).__name__):
                env = make_env(
                    [Order(0, (0.0, 0.0), (1.0, 0.0), 0.0)],
                    fleet=[
                        {
                            "location": (0.0, 0.0),
                            "pickup_service_time": 0.75,
                            "dropoff_service_time": 0.75,
                        }
                    ],
                    road_network=network,
                    statistics_granularity="continuous",
                )
                env.step({0: {"orders": [0]}})
                self.assertIsNone(env.orders[0].pickup_time)
                env.step({})
                self.assertEqual(env.orders[0].pickup_time, 0.75)
                self.assertEqual(env.drivers[0].location, (0.25, 0.0))
                env.step({})
                env.step({})
                self.assertIsNone(env.orders[0].dropoff_time)
                env.step({})
                self.assertEqual(env.orders[0].dropoff_time, 2.5)
                self.assertEqual(env.orders[0].status, OrderStatus.COMPLETED)
                self.assertEqual(env.drivers[0].onboard_passengers, 0)

    def test_service_durations_are_included_in_planned_times(self):
        env = make_env(
            [Order(0, (0.0, 0.0), (1.0, 0.0), 0.0)],
            fleet=[
                {"location": (0.0, 0.0), "pickup_service_time": 0.75, "dropoff_service_time": 0.75}
            ],
        )
        result = env.step({0: {"orders": [0]}})
        self.assertEqual(event(result)["assigned_service_times"][0], 2.5)
        self.assertEqual(env.orders[0].eta, 2.0)

    def test_new_assignment_cannot_interrupt_a_stop_already_in_service(self):
        env = make_env(
            [
                Order(0, (0.0, 0.0), (5.0, 0.0), 0.0),
                Order(1, (0.1, 0.0), (2.0, 0.0), 0.5),
            ],
            fleet=[{"location": (0.0, 0.0), "pickup_service_time": 1.25}],
            statistics_granularity="continuous",
        )
        env.step({0: {"orders": [0]}})
        self.assertEqual(env.drivers[0].service_order_id, 0)
        self.assertEqual(env.drivers[0].service_until, 1.25)
        env.step({0: {"orders": [1]}})
        self.assertEqual(env.drivers[0].task_points[0].order_id, 0)
        self.assertEqual(env.drivers[0].location, (0.0, 0.0))
        self.assertIsNone(env.orders[0].pickup_time)
        self.assertIsNone(env.orders[1].pickup_time)
        env.step({})
        self.assertEqual(env.orders[0].pickup_time, 1.25)
        self.assertIsNone(env.orders[1].pickup_time)

    def test_custom_planner_implementing_only_plan_remains_supported(self):
        class PlanOnly(RoutesPlanner):
            def plan(self, start, task_points, network):
                return list(task_points)

        env = RidePoolEnv(
            num_drivers=1,
            dt=0.5,
            horizon=3,
            order_timeout=None,
            routes_planner=PlanOnly(),
            order_generator=FreshOrders([Order(0, (0.0, 0.0), (1.0, 0.0), 0.0)]),
        )
        env.reset(seed=3)
        env.drivers[0].location = (0.0, 0.0)
        result = env.step({0: {"orders": [0]}})
        self.assertEqual(event(result)["assigned_orders"], [0])
        self.assertEqual(event(result)["assigned_service_times"][0], 1.0)
        self.assertEqual(env.orders[0].status, OrderStatus.ONBOARD)


if __name__ == "__main__":
    unittest.main()
