"""Synthetic fleet, behavior and snapshot example; run from the repository root.

    python -m examples.optional_features

Only the core numpy dependency is needed. No files are read or written.
"""

from __future__ import annotations

import json

from ride_gym.acceptance import AcceptanceProbabilities, PerDriverAcceptancePolicy
from ride_gym.entities import Order
from ride_gym.env import RidePoolEnv
from ride_gym.labor_source import ExplicitLaborSource, ShiftWindow
from ride_gym.order_generator import OrderGenerator
from ride_gym.response import FixedResponseTimePolicy
from ride_gym.road_network import EuclideanNetwork
from ride_gym.warmup import warmup


class SyntheticOrders(OrderGenerator):
    """Return new lifecycle objects for every episode."""

    def generate(self):
        trips = (
            ((0.5, 0.5), (1.5, 1.0), 0.0, 0.5),
            ((0.6, 0.4), (1.8, 1.2), 0.0, 0.75),
            ((4.0, 1.0), (3.0, 1.5), 1.0, 1.5),
            ((3.8, 1.2), (2.8, 1.6), 1.5, 2.0),
            ((1.0, 1.0), (2.0, 2.0), 4.0, 4.5),
            ((0.7, 0.7), (1.5, 1.5), 7.0, 7.25),
        )
        return [
            Order(i, origin, destination, requested, ready_time=ready)
            for i, (origin, destination, requested, ready) in enumerate(trips)
        ]


def make_env(initial_state=None):
    fleet = ExplicitLaborSource(
        [
            {
                "location": (0.0, 0.0),
                "vehicle_type": "bicycle",
                "capacity": 2,
                "speed": 0.8,
                "pickup_service_time": 0.25,
                "dropoff_service_time": 0.15,
                "availability_windows": [
                    ShiftWindow(0.0, 5.0),
                    ShiftWindow(7.0, 12.0, location=(0.5, 0.5)),
                ],
            },
            {
                "location": (4.0, 1.0),
                "vehicle_type": "car",
                "capacity": 3,
                "speed": 1.2,
                "shift_start": 1.0,
                "shift_end": 12.0,
                "pickup_service_time": 0.1,
                "dropoff_service_time": 0.1,
            },
        ]
    )
    acceptance = PerDriverAcceptancePolicy(
        {
            0: AcceptanceProbabilities(general=0.85, single=0.9, bundle=0.75),
            1: AcceptanceProbabilities(general=0.95),
        }
    )
    return RidePoolEnv(
        area=(0.0, 0.0, 6.0, 6.0),
        num_drivers=2,
        dt=0.25,
        horizon=12.0,
        order_timeout=None,
        road_network=EuclideanNetwork(speed=1.0),
        order_generator=SyntheticOrders(),
        labor_source=fleet,
        acceptance_policy=acceptance,
        acceptance_seed=11,
        response_time_policy=FixedResponseTimePolicy(seconds=12.0),
        response_seed=23,
        initial_state=initial_state,
    )


def dispatch(observations):
    """A stateless centralized policy that respects availability and capacity."""
    actions, offered = {}, set()
    for did, observation in observations.items():
        vehicle = observation["self"]
        if not vehicle["available"]:
            continue
        free = vehicle["capacity"] - vehicle["committed_passengers"]
        selected = []
        for order in observation["pending_orders"]:
            oid, party = order["order_id"], order["num_passengers"]
            if oid not in offered and party <= free:
                selected.append(oid)
                offered.add(oid)
                free -= party
        actions[did] = {"orders": selected}
    return actions


def finish(env, observations):
    while env.time < env.horizon:
        observations, _, dones, _ = env.step(dispatch(observations))
        if dones["__all__"]:
            break
    return {
        "time": env.time,
        "orders": [
            {
                "order_id": order.order_id,
                "status": order.status.value,
                "pickup_time": order.pickup_time,
                "dropoff_time": order.dropoff_time,
            }
            for order in env.orders.values()
        ],
    }


def main():
    env = make_env()
    env.reset(seed=7)
    observations, info, state = warmup(env, dispatch, steps=4)
    original = finish(env, observations)

    restored_env = make_env(initial_state=state)
    restored_observations, _ = restored_env.reset(seed=7)
    restored = finish(restored_env, restored_observations)
    if original != restored:
        raise AssertionError("Snapshot continuation differs from the original rollout")
    print(
        json.dumps(
            {"warmup_end": info["time"], "snapshot_continuation_matches": True, "result": original},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
