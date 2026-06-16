"""Detailed episode recorder for cross-algorithm comparison.

Captures three granularities and a summary so different dispatch algorithms can
be compared on identical scenarios:

* **per-step**   : aggregate flow metrics each decision step.
* **per-order**  : full lifecycle timestamps and service quality per order.
* **per-driver** : utilisation, distance, and throughput per driver.
* **summary**    : scenario-level KPIs.

Results are written as CSV (one file per granularity) plus a JSON manifest that
includes the scenario config and the summary, giving each run self-contained
provenance.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Optional

from ridepool_sim.enums import OrderStatus


class EpisodeRecorder:
    """Collects detailed metrics over one episode and persists them.

    The recorder is algorithm-agnostic: it observes the environment and the
    per-step ``info`` event log. Call :meth:`record_step` once per step after
    ``env.step`` and :meth:`finalize` at the end.
    """

    def __init__(self, algorithm: str, config: Optional[dict] = None):
        self.algorithm = algorithm
        self.config = config or {}
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        # per-step rows
        self.step_rows: List[Dict] = []
        # Cumulative environment reward over the whole episode (summed across
        # drivers and steps). Reward is the headline metric for comparing the
        # overall performance of different dispatch methods, so it is tracked
        # here as a first-class quantity and surfaced in the summary.
        self.total_reward: float = 0.0
        # per-driver running accumulators
        self._driver_acc: Dict[int, Dict] = {}
        # filled at finalize()
        self.order_rows: List[Dict] = []
        self.driver_rows: List[Dict] = []
        self.summary: Dict = {}

    # ----------------------------------------------------------- per-step
    def record_step(
        self,
        env,
        info: Dict,
        assign_log: Optional[Dict] = None,
        rewards: Optional[Dict] = None,
    ) -> None:
        """Record one step.

        Parameters
        ----------
        env:
            The (unwrapped) :class:`RidePoolEnv`.
        info:
            The ``info`` dict returned by ``env.step`` (carries ``events``).
        assign_log:
            Optional ``{order_id: pickup_distance_km}`` produced by the
            dispatch algorithm this step. This is the *matched / intended*
            pickup distance measured at the moment of assignment (order origin
            -> driver's current location), i.e. the quantity the matcher
            optimises. It is NOT the actual distance the driver subsequently
            drives to reach the pickup: after assignment the planner re-orders
            all task points (pooling may insert other stops first), so the
            realised pickup path can be longer. Actual service quality (wait /
            ride time, driven distance) is derived independently from order
            lifecycle timestamps and per-driver distance in finalize().
        rewards:
            Optional ``{driver_id: reward}`` mapping returned by ``env.step``.
            When provided, the per-step total (sum over drivers) is recorded in
            the per-step row and accumulated into :attr:`total_reward`, making
            overall reward a first-class comparison metric in the summary.
        """
        events = info.get("events", {})
        assign_log = assign_log or {}

        step_assigned = 0
        step_completed = 0
        step_pickups = 0
        step_distance = 0.0
        step_empty_distance = 0.0
        # Per-step environment reward (sum over drivers); accumulated into the
        # episode total. Zero when the caller does not pass the reward dict.
        step_reward = float(sum(rewards.values())) if rewards else 0.0
        self.total_reward += step_reward

        for did, ev in events.items():
            n_assigned = len(ev["assigned_orders"])
            n_completed = len(ev["completed_orders"])
            n_pickups = len(ev["picked_up_orders"])
            dist = ev["distance_moved"]

            step_assigned += n_assigned
            step_completed += n_completed
            step_pickups += n_pickups
            step_distance += dist
            if ev["is_empty_move"]:
                step_empty_distance += dist

            acc = self._driver_acc.setdefault(
                did,
                {
                    "orders_served": 0,
                    "orders_assigned": 0,
                    "total_distance": 0.0,
                    "empty_distance": 0.0,
                    "busy_steps": 0,
                    "idle_steps": 0,
                    "total_reward": 0.0,
                },
            )
            acc["orders_served"] += n_completed
            acc["orders_assigned"] += n_assigned
            acc["total_distance"] += dist
            if ev["is_empty_move"]:
                acc["empty_distance"] += dist
            if ev["is_idle_wait"]:
                acc["idle_steps"] += 1
            else:
                acc["busy_steps"] += 1
            if rewards is not None and did in rewards:
                acc["total_reward"] += float(rewards[did])

        # pending / cancelled snapshot at this point in time
        pending = len(env._pending_ids)
        cancelled = sum(
            1 for o in env.orders.values() if o.status == OrderStatus.CANCELLED
        )
        onboard = sum(d.onboard_passengers for d in env.drivers.values())

        pickup_dists = list(assign_log.values())
        avg_pickup_dist = (
            sum(pickup_dists) / len(pickup_dists) if pickup_dists else 0.0
        )

        self.step_rows.append(
            {
                "time": info.get("time"),
                "assigned": step_assigned,
                "completed": step_completed,
                "picked_up": step_pickups,
                "pending": pending,
                "cancelled_cumulative": cancelled,
                "onboard_passengers": onboard,
                "total_distance_km": step_distance,
                "empty_distance_km": step_empty_distance,
                "avg_matched_pickup_distance_km": avg_pickup_dist,
                "step_reward": step_reward,
            }
        )

    # ----------------------------------------------------------- finalize
    def finalize(self, env) -> Dict:
        """Compute per-order and per-driver tables plus the summary."""
        self._build_order_rows(env)
        self._build_driver_rows(env)
        self._build_summary(env)
        return self.summary

    def _build_order_rows(self, env) -> None:
        rows = []
        for o in env.orders.values():
            wait = None
            if o.pickup_time is not None:
                wait = o.pickup_time - o.request_time
            elif o.cancel_time is not None:
                wait = o.cancel_time - o.request_time

            ride_time = None
            if o.pickup_time is not None and o.dropoff_time is not None:
                ride_time = o.dropoff_time - o.pickup_time

            direct_dist = env.network.distance(o.origin, o.destination)

            # Detour time: how much longer the passenger spent in the vehicle
            # than a direct, non-pooled trip would have taken. The direct ride
            # time is the straight (no-detour) travel time for the order's
            # origin->destination distance at network speed; the realised
            # ``ride_time`` exceeds it whenever pooling inserts other stops.
            # Only defined for completed trips (needs both pickup & dropoff);
            # clamped at zero to absorb tiny numerical/discretisation noise.
            detour_time = None
            if ride_time is not None:
                direct_ride_time = env.network.travel_time(direct_dist)
                detour_time = max(0.0, ride_time - direct_ride_time)

            rows.append(
                {
                    "order_id": o.order_id,
                    "status": o.status.value,
                    "origin_x": o.origin[0],
                    "origin_y": o.origin[1],
                    "dest_x": o.destination[0],
                    "dest_y": o.destination[1],
                    "num_passengers": o.num_passengers,
                    "request_time": o.request_time,
                    "pickup_time": o.pickup_time,
                    "dropoff_time": o.dropoff_time,
                    "cancel_time": o.cancel_time,
                    "assigned_driver": o.assigned_driver,
                    "wait_time": wait,
                    "ride_time": ride_time,
                    "detour_time": detour_time,
                    "direct_distance_km": direct_dist,
                }
            )
        self.order_rows = rows

    def _build_driver_rows(self, env) -> None:
        rows = []
        total_steps = len(self.step_rows)
        for did, d in env.drivers.items():
            acc = self._driver_acc.get(
                did,
                {
                    "orders_served": 0,
                    "orders_assigned": 0,
                    "total_distance": 0.0,
                    "empty_distance": 0.0,
                    "busy_steps": 0,
                    "idle_steps": 0,
                    "total_reward": 0.0,
                },
            )
            utilisation = acc["busy_steps"] / total_steps if total_steps else 0.0
            occupied = acc["total_distance"] - acc["empty_distance"]
            empty_ratio = (
                acc["empty_distance"] / acc["total_distance"]
                if acc["total_distance"] > 0
                else 0.0
            )
            rows.append(
                {
                    "driver_id": did,
                    "orders_served": acc["orders_served"],
                    "orders_assigned": acc["orders_assigned"],
                    "total_distance_km": acc["total_distance"],
                    "empty_distance_km": acc["empty_distance"],
                    "occupied_distance_km": occupied,
                    "empty_distance_ratio": empty_ratio,
                    "busy_steps": acc["busy_steps"],
                    "idle_steps": acc["idle_steps"],
                    "utilisation": utilisation,
                    "total_reward": acc["total_reward"],
                }
            )
        self.driver_rows = rows

    def _build_summary(self, env) -> None:
        total = len(env.orders)
        completed = sum(
            1 for o in env.orders.values() if o.status == OrderStatus.COMPLETED
        )
        cancelled = sum(
            1 for o in env.orders.values() if o.status == OrderStatus.CANCELLED
        )
        unserved = total - completed - cancelled
        # Confirmed orders: every order that was ever bound to a driver, i.e.
        # reached ASSIGNED or any later stage (ONBOARD / COMPLETED). Because
        # cancellation only happens to still-PENDING orders that time out
        # before assignment, "has a driver" is an exact confirmation test.
        confirmed = sum(
            1
            for o in env.orders.values()
            if o.status
            in (OrderStatus.ASSIGNED, OrderStatus.ONBOARD, OrderStatus.COMPLETED)
        )

        waits = [
            r["wait_time"]
            for r in self.order_rows
            if r["status"] == OrderStatus.COMPLETED.value and r["wait_time"] is not None
        ]
        ride_times = [
            r["ride_time"] for r in self.order_rows if r["ride_time"] is not None
        ]
        detour_times = [
            r["detour_time"]
            for r in self.order_rows
            if r["detour_time"] is not None
        ]
        total_distance = sum(r["total_distance_km"] for r in self.driver_rows)
        empty_distance = sum(r["empty_distance_km"] for r in self.driver_rows)
        utils = [r["utilisation"] for r in self.driver_rows]

        def _avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        n_drivers = len(env.drivers)
        n_steps = len(self.step_rows)
        self.summary = {
            "algorithm": self.algorithm,
            "created_at": self.created_at,
            "total_orders": total,
            "completed": completed,
            "cancelled": cancelled,
            "unserved": unserved,
            "confirmed": confirmed,
            # service_rate now measures dispatch coverage: the share of orders
            # that were ever assigned to a driver (confirmed), regardless of
            # whether the ride was ultimately completed within the horizon.
            "service_rate": confirmed / total if total else 0.0,
            # complete_rate is the former service_rate: the share of orders
            # actually delivered (dropped off).
            "complete_rate": completed / total if total else 0.0,
            "cancellation_rate": cancelled / total if total else 0.0,
            "avg_wait_time": _avg(waits),
            "avg_ride_time": _avg(ride_times),
            "avg_detour_time": _avg(detour_times),
            "total_distance_km": total_distance,
            "empty_distance_km": empty_distance,
            "empty_distance_ratio": (
                empty_distance / total_distance if total_distance > 0 else 0.0
            ),
            "avg_driver_utilisation": _avg(utils),
            "orders_per_driver": completed / n_drivers if n_drivers else 0.0,
            # Reward KPIs: total over the episode, plus per-driver and per-step
            # normalisations so reward is comparable across scenarios of
            # different size / length.
            "total_reward": self.total_reward,
            "avg_reward_per_driver": (
                self.total_reward / n_drivers if n_drivers else 0.0
            ),
            "avg_reward_per_step": (
                self.total_reward / n_steps if n_steps else 0.0
            ),
        }

    # ----------------------------------------------------------- persistence
    def save(self, out_dir: str) -> Dict[str, str]:
        """Write all tables + manifest under ``out_dir/<algorithm>/``."""
        run_dir = os.path.join(out_dir, self.algorithm)
        os.makedirs(run_dir, exist_ok=True)

        paths = {
            "steps": os.path.join(run_dir, "steps.csv"),
            "orders": os.path.join(run_dir, "orders.csv"),
            "drivers": os.path.join(run_dir, "drivers.csv"),
            "manifest": os.path.join(run_dir, "manifest.json"),
        }

        _write_csv(paths["steps"], self.step_rows)
        _write_csv(paths["orders"], self.order_rows)
        _write_csv(paths["drivers"], self.driver_rows)

        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(
                {
                    "algorithm": self.algorithm,
                    "created_at": self.created_at,
                    "config": self.config,
                    "summary": self.summary,
                },
                f,
                indent=2,
            )
        return paths


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        # Still create an empty file with no header to keep paths consistent.
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)