"""Synthetic unit tests for optional offer policies and response timing."""

import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from ride_gym.acceptance import (
    AcceptanceContext,
    AcceptanceDriverState,
    AcceptanceOrderState,
    AcceptanceProbabilities,
    AcceptanceStopState,
    PerDriverAcceptancePolicy,
    policy_configuration,
    validate_probability,
)
from ride_gym.response import (
    BetaResponseTimePolicy,
    FixedResponseTimePolicy,
    UniformResponseTimePolicy,
    next_response_boundary,
    response_policy_configuration,
    validate_response_seconds,
)


def make_context(driver_id=0, order_count=1, assigned_ids=()):
    driver = AcceptanceDriverState(driver_id, (1.0, 1.0), 4, None, assigned_ids, 0, True, "idle")
    orders = tuple(
        AcceptanceOrderState(i, (1.0, 2.0), (3.0, 2.0), 0.0, 1) for i in range(order_count)
    )
    route = tuple(
        AcceptanceStopState(order.order_id, "pickup", order.origin, 1.0) for order in orders
    )
    return AcceptanceContext(2.0, driver, orders, route)


class AcceptancePolicyTests(unittest.TestCase):
    def test_context_is_immutable_and_bundle_only_counts_new_orders(self):
        context = make_context(order_count=1, assigned_ids=(9, 10))
        self.assertEqual(context.driver_id, 0)
        self.assertEqual(context.order_ids, (0,))
        self.assertFalse(context.is_bundle)
        self.assertEqual(context.kind, "single")
        self.assertTrue(make_context(order_count=2).is_bundle)
        with self.assertRaises(FrozenInstanceError):
            context.driver.capacity = 9
        with self.assertRaises(FrozenInstanceError):
            context.time = 3.0

    def test_probability_overrides_and_fallback_are_explicit(self):
        policy = PerDriverAcceptancePolicy(
            {
                0: AcceptanceProbabilities(general=0.8, single=0.9, bundle=0.3),
                1: {"acceptance_probability": 0.4, "bundle_acceptance_probability": 0.2},
            },
            default=AcceptanceProbabilities(general=0.6),
        )
        self.assertEqual(policy.probability(make_context(0)), 0.9)
        self.assertEqual(policy.probability(make_context(0, 2)), 0.3)
        self.assertEqual(policy.probability(make_context(1)), 0.4)
        self.assertEqual(policy.probability(make_context(1, 2)), 0.2)
        self.assertEqual(policy.probability(make_context(8)), 0.6)
        self.assertEqual(PerDriverAcceptancePolicy({}).probability(make_context()), 1.0)

    def test_zero_and_one_probabilities_are_not_replaced_by_defaults(self):
        profile = AcceptanceProbabilities(general=0.5, single=0, bundle=1)
        self.assertEqual(profile.for_kind(False), 0.0)
        self.assertEqual(profile.for_kind(True), 1.0)

    def test_source_profiles_are_copied_and_cannot_be_mutated(self):
        source = {0: {"general": 0.2}}
        policy = PerDriverAcceptancePolicy(source)
        source[0]["general"] = 0.9
        source[1] = {"general": 0}
        self.assertEqual(policy.probability(make_context()), 0.2)
        self.assertEqual(policy.probability(make_context(1)), 1.0)
        with self.assertRaises(TypeError):
            policy.profiles[0] = AcceptanceProbabilities()

    def test_invalid_probability_and_profile_inputs_fail(self):
        for value in (-0.01, 1.01, math.nan, math.inf, -math.inf, "bad", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_probability(value)
        for key in (True, "0", 1.2):
            with self.subTest(key=key), self.assertRaises(ValueError):
                PerDriverAcceptancePolicy({key: {"general": 0.5}})
        with self.assertRaises(TypeError):
            PerDriverAcceptancePolicy({0: 0.5})
        with self.assertRaises(TypeError):
            PerDriverAcceptancePolicy({}, default=0.5)

    def test_configuration_is_order_independent_and_tracks_values(self):
        first = PerDriverAcceptancePolicy({0: {"general": 0.2}, 1: {"general": 0.6}})
        reordered = PerDriverAcceptancePolicy({1: {"general": 0.6}, 0: {"general": 0.2}})
        changed = PerDriverAcceptancePolicy({0: {"general": 0.3}, 1: {"general": 0.6}})
        self.assertEqual(policy_configuration(first), policy_configuration(reordered))
        self.assertNotEqual(policy_configuration(first), policy_configuration(changed))
        detached = policy_configuration(first)
        detached["configuration"]["default"]["general"] = 0
        self.assertEqual(first.default.general, 1.0)

    def test_checkpoint_configuration_is_required_only_when_requested(self):
        class ProbabilityOnly:
            def probability(self, context):
                return 0.4

        class InvalidConfiguration(ProbabilityOnly):
            def configuration(self):
                return {"bad": math.nan}

        self.assertEqual(ProbabilityOnly().probability(make_context()), 0.4)
        with self.assertRaises(TypeError):
            policy_configuration(ProbabilityOnly())
        with self.assertRaises(ValueError):
            policy_configuration(InvalidConfiguration())


class ResponsePolicyTests(unittest.TestCase):
    def test_fixed_response_does_not_consume_rng(self):
        rng = np.random.default_rng(5)
        reference = np.random.default_rng(5)
        self.assertEqual(FixedResponseTimePolicy().sample_seconds(make_context(), rng), 0.0)
        self.assertEqual(FixedResponseTimePolicy(12).sample_seconds(make_context(), rng), 12.0)
        self.assertEqual(rng.random(), reference.random())

    def test_uniform_and_beta_use_only_the_supplied_seeded_stream(self):
        for policy in (UniformResponseTimePolicy(), BetaResponseTimePolicy()):
            with self.subTest(policy=policy):
                first, second = np.random.default_rng(4), np.random.default_rng(4)
                values = [policy.sample_seconds(make_context(), first) for _ in range(20)]
                repeated = [policy.sample_seconds(make_context(), second) for _ in range(20)]
                self.assertEqual(values, repeated)
                self.assertTrue(all(0 <= value <= 60 for value in values))
                self.assertGreater(len(set(values)), 1)

    def test_beta_sampler_parameters_and_scale(self):
        class StubRng:
            def beta(self, alpha, beta):
                self.parameters = (alpha, beta)
                return 0.25

        rng = StubRng()
        self.assertEqual(BetaResponseTimePolicy(3, 4, 80).sample_seconds(make_context(), rng), 20)
        self.assertEqual(rng.parameters, (3.0, 4.0))

    def test_invalid_durations_and_beta_shapes_fail_at_construction(self):
        for value in (-1, math.nan, math.inf, None, "bad"):
            for factory in (FixedResponseTimePolicy, UniformResponseTimePolicy):
                with self.subTest(factory=factory, value=value), self.assertRaises(ValueError):
                    factory(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_response_seconds(value)
        for value in (0, -1, math.nan, math.inf, True, None, "bad"):
            for key in ("alpha", "beta"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    BetaResponseTimePolicy(**{key: value})

    def test_response_configuration_distinguishes_policies(self):
        fixed = response_policy_configuration(FixedResponseTimePolicy(10))
        uniform = response_policy_configuration(UniformResponseTimePolicy(10))
        beta = response_policy_configuration(BetaResponseTimePolicy(2, 1, 10))
        self.assertNotEqual(fixed, uniform)
        self.assertNotEqual(uniform, beta)
        self.assertEqual(fixed["parameters"], {"distribution": "fixed", "seconds": 10.0})
        with self.assertRaises(TypeError):
            response_policy_configuration(object())

    def test_responses_take_effect_at_strictly_next_global_boundary(self):
        self.assertEqual(next_response_boundary(0, 0.5), 0.5)
        self.assertEqual(next_response_boundary(20 / 60, 0.5), 0.5)
        self.assertEqual(next_response_boundary(0.5, 0.5), 1.0)
        self.assertAlmostEqual(next_response_boundary(0.3, 0.1), 0.4)
        self.assertEqual(next_response_boundary(0.5 - 1e-8, 0.5), 0.5)
        self.assertEqual(next_response_boundary(0.5 + 1e-8, 0.5), 1.0)
        for time, dt in ((math.nan, 1), (0, 0), (0, -1), (0, math.inf), ("bad", 1)):
            with self.subTest(time=time, dt=dt), self.assertRaises(ValueError):
                next_response_boundary(time, dt)


if __name__ == "__main__":
    unittest.main()
