"""Shared route preparation and optional fleet, invitation and service support.

Route preparation is used by every assignment. Fleet, behavior and service
lifecycle helpers are enabled by their corresponding environment options.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from ride_gym.entities import Driver, TaskPoint
from ride_gym.enums import OrderStatus
from ride_gym.exceptions import InvalidActionError


class _VehicleNetwork:
    """Distance/path view with the configured vehicle's travel speed."""

    def __init__(self, network, speed):
        self.network, self.speed = network, speed

    def __getattr__(self, name):
        return getattr(self.network, name)

    def travel_time(self, distance):
        return distance / self.speed

    def shortest_path(self, origin, destination):
        from ride_gym.road_network import PathResult

        result = self.network.shortest_path(origin, destination)
        return PathResult(result.distance, self.travel_time(result.distance), result.nodes)


class RuntimeExtensions:
    """Implementation mixin; public entry points live on ``RidePoolEnv``."""

    def _configure_extensions(
        self,
        labor_source,
        shift_policy,
        initial_state,
        acceptance_policy,
        acceptance_seed,
        response_time_policy,
        response_seed,
        validate_coords,
    ):
        self.labor_source = labor_source
        self.shift_policy = shift_policy or getattr(labor_source, "shift_policy", None)
        self.initial_state = deepcopy(initial_state)
        self.acceptance_policy, self.acceptance_seed = acceptance_policy, acceptance_seed
        self.response_time_policy, self.response_seed = response_time_policy, response_seed
        self.validate_order_coords = bool(validate_coords)
        self._shift_enabled = self.shift_policy is not None
        self._behavior_enabled = acceptance_policy is not None or response_time_policy is not None
        if labor_source is not None and not callable(getattr(labor_source, "build", None)):
            raise TypeError("labor_source must implement build(num_drivers, rng, area, network)")
        if self._shift_enabled and not callable(getattr(self.shift_policy, "update", None)):
            raise TypeError("shift_policy must implement update(drivers, time)")
        if acceptance_policy is not None and not callable(
            getattr(acceptance_policy, "probability", None)
        ):
            raise TypeError("acceptance_policy must implement probability(context)")
        if response_time_policy is not None:
            if not callable(getattr(response_time_policy, "sample_seconds", response_time_policy)):
                raise TypeError("response_time_policy must implement sample_seconds(context, rng)")
            if not math.isfinite(self.dt) or self.dt <= 0:
                raise ValueError("response time requires finite positive dt")
        self._acceptance_rng = self._response_rng = None
        self._acceptance_draws = self._response_draws = self._next_offer_id = 0
        self._pending_responses, self._response_order_ids = {}, set()

    def _build_explicit_fleet(self):
        drivers = list(
            self.labor_source.build(self.num_drivers, self._rng, self.area, self.network)
        )
        if (
            len(drivers) != self.num_drivers
            or any(not isinstance(d, Driver) for d in drivers)
            or [d.driver_id for d in drivers] != list(range(self.num_drivers))
        ):
            raise ValueError("labor_source must return drivers numbered 0..num_drivers-1 in order")
        result = {d.driver_id: d for d in drivers}
        for driver in result.values():
            if (
                isinstance(driver.capacity, bool)
                or not isinstance(driver.capacity, Integral)
                or driver.capacity < 0
            ):
                raise ValueError("Explicit fleet capacity must be a nonnegative integer")
            if driver.speed is not None and (not math.isfinite(driver.speed) or driver.speed <= 0):
                raise ValueError("Explicit fleet speed must be finite and positive")
            if (
                len(driver.location) != 2
                or not all(math.isfinite(v) for v in driver.location)
                or any(
                    not math.isfinite(v) or v < 0
                    for v in (driver.pickup_service_time, driver.dropoff_service_time)
                )
            ):
                raise ValueError("Explicit fleet coordinates and service times must be finite")
            if self._is_graph_network():
                driver.node_idx = int(self.network.snap(driver.location))
                driver.location = self.network.node_coord(driver.node_idx)
        return result

    def _reset_extensions(self):
        self._pending_responses, self._response_order_ids = {}, set()
        self._acceptance_draws = self._response_draws = self._next_offer_id = 0
        if self.acceptance_policy is not None:
            base = self.acceptance_seed if self.acceptance_seed is not None else self._seed
            self._acceptance_rng = np.random.default_rng(
                np.random.SeedSequence(base, spawn_key=(0x41434345,))
            )
        if self.response_time_policy is not None:
            base = self.response_seed if self.response_seed is not None else self._seed
            self._response_rng = np.random.default_rng(
                np.random.SeedSequence(base, spawn_key=(0x5253504E,))
            )
        outside = 0
        for order in self._all_orders:
            if order.ready_time is not None and (
                isinstance(order.ready_time, bool)
                or not isinstance(order.ready_time, Real)
                or not math.isfinite(order.ready_time)
            ):
                raise ValueError("ready_time must be finite minutes or None")
            if self.validate_order_coords and (
                not self._in_area(order.origin) or not self._in_area(order.destination)
            ):
                outside += 1
        if outside:
            warnings.warn(
                f"{outside} order(s) have endpoints outside the service area",
                UserWarning,
                stacklevel=3,
            )
        self._update_shifts()
        if self.initial_state is not None:
            self.apply_initial_state(self.initial_state)

    def _update_shifts(self):
        if not self._shift_enabled:
            return
        self.shift_policy.update(self.drivers.values(), self.time)
        if self._is_graph_network():
            for driver in self.drivers.values():
                if driver.node_idx is None and driver.edge_to is None:
                    driver.node_idx = int(self.network.snap(driver.location))
                    driver.location = self.network.node_coord(driver.node_idx)

    def _driver_available(self, driver_id):
        driver = self.drivers[driver_id]
        if driver_id in self._pending_responses:
            return False
        if not self._shift_enabled:
            return True
        available = getattr(self.shift_policy, "is_available", None)
        return (
            bool(available(driver))
            if callable(available)
            else bool(driver.on_duty and not driver.departed)
        )

    @property
    def pending_response_driver_ids(self):
        return frozenset(self._pending_responses)

    @property
    def pending_response_order_ids(self):
        return frozenset(self._response_order_ids)

    def _driver_network(self, driver):
        """Use a vehicle's explicit speed without changing the network API."""
        if driver.speed is None:
            return self.network
        speed = self._driver_speed(driver)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Vehicle speed must be finite and positive")
        return _VehicleNetwork(self.network, speed)

    def _planning_start(self, driver):
        """Return the next reroutable position and its fixed travel prefix.

        Graph movement must finish an active directed edge before rerouting.
        Planning from the interpolated coordinate would snap it to a node and
        either charge already travelled distance or omit the remaining edge.
        """
        if self._is_graph_network():
            if driver.edge_to is not None:
                remaining = (
                    self.network.node_distance(driver.edge_from, driver.edge_to) - driver.edge_pos_m
                )
                return (
                    self.network.node_coord(driver.edge_to),
                    max(0.0, remaining) / self._driver_speed(driver),
                )
            if driver.node_idx is not None:
                return self.network.node_coord(driver.node_idx), 0.0
        return driver.location, 0.0

    def _extension_plan(self, driver, points):
        network = self._driver_network(driver)
        points = deepcopy(points)
        # An in-progress stop cannot be interrupted by inserting a new pickup.
        locked = None
        if driver.service_order_id is not None:
            locked = next(
                (
                    p
                    for p in points
                    if p.order_id == driver.service_order_id and p.kind == driver.service_kind
                ),
                None,
            )
            if locked is None:
                raise InvalidActionError("Active service stop is missing from the route")
            points.remove(locked)
        start, prefix = self._planning_start(driver)
        if locked is not None:
            start, prefix = locked.location, 0.0
        expected = sorted(
            (p.kind, p.order_id, tuple(p.location)) for p in points + ([locked] if locked else [])
        )
        ordered, times = self.planner.plan_with_times(start, points, network)
        if locked is not None:
            ordered = [locked] + list(ordered)
            times.setdefault(locked.order_id, {})[locked.kind] = 0.0
        # Validate all proposed stops, including the newly offered ones.
        actual = sorted((p.kind, p.order_id, tuple(p.location)) for p in ordered)
        if actual != expected:
            raise InvalidActionError("Planner must return each proposed stop exactly once")
        picked = {
            oid for oid in driver.assigned_orders if self.orders[oid].status == OrderStatus.ONBOARD
        }
        result_times = {}
        previous_time = 0.0
        for stop in ordered:
            if stop.kind == "pickup":
                picked.add(stop.order_id)
            elif stop.kind != "dropoff" or stop.order_id not in picked:
                raise InvalidActionError("Planner violates pickup-before-dropoff precedence")
            value = times.get(stop.order_id, {}).get(stop.kind)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < previous_time
            ):
                raise InvalidActionError(
                    "Planner must return finite nonnegative, chronological stop arrival times"
                )
            result_times.setdefault(stop.order_id, {})[stop.kind] = value + prefix
            previous_time = value
        return list(ordered), result_times

    def _prepare_invitation(self, driver, order_ids):
        points = deepcopy(driver.task_points)
        for oid in order_ids:
            order = self.orders[oid]
            points.extend(
                (
                    TaskPoint("pickup", oid, order.origin),
                    TaskPoint("dropoff", oid, order.destination),
                )
            )
        return self._extension_plan(driver, points)

    def _prepare_assignment(self, driver, order_ids, plan=None):
        """Compute a complete assignment without mutating simulator state."""
        route, travel_times = self._prepare_invitation(driver, order_ids) if plan is None else plan
        timed_service = (
            driver.pickup_service_time > 0
            or driver.dropoff_service_time > 0
            or driver.service_order_id is not None
            or any(self.orders[p.order_id].ready_time is not None for p in route)
        )
        arrival_times = self._service_plan_times(driver, route) if timed_service else travel_times
        network = self._driver_network(driver)
        solo_times, service_times = {}, {}
        for oid in order_ids:
            order = self.orders[oid]
            solo = network.shortest_path(order.origin, order.destination).travel_time
            if not isinstance(solo, Real) or not math.isfinite(solo) or solo < 0:
                raise InvalidActionError("Direct travel time must be finite and nonnegative")
            solo_times[oid] = solo
            service_times[oid] = max(
                0.0, self.time + arrival_times[oid]["dropoff"] - order.request_time
            )
        extra_detour = sum(
            arrival_times[oid]["dropoff"] - self.orders[oid].eta
            for oid in driver.assigned_orders
            if self.orders[oid].eta is not None
        )
        return {
            "order_ids": tuple(order_ids),
            "route": route,
            "travel_times": travel_times,
            "arrival_times": arrival_times,
            "solo_times": solo_times,
            "service_times": service_times,
            "extra_detour_time": extra_detour,
        }

    def _invitation_context(self, driver, order_ids, plan):
        from ride_gym.acceptance import (
            AcceptanceContext,
            AcceptanceDriverState,
            AcceptanceOrderState,
            AcceptanceStopState,
        )

        route, times = plan
        return AcceptanceContext(
            self.time,
            AcceptanceDriverState(
                driver.driver_id,
                tuple(driver.location),
                driver.capacity,
                self._driver_speed(driver),
                tuple(driver.assigned_orders),
                driver.onboard_passengers,
                driver.on_duty,
                driver.status.value,
            ),
            tuple(
                AcceptanceOrderState(
                    oid,
                    tuple(self.orders[oid].origin),
                    tuple(self.orders[oid].destination),
                    self.orders[oid].request_time,
                    self.orders[oid].num_passengers,
                )
                for oid in order_ids
            ),
            tuple(
                AcceptanceStopState(
                    p.order_id, p.kind, tuple(p.location), times[p.order_id][p.kind]
                )
                for p in route
            ),
        )

    def _consider_invitation(self, driver, order_ids, event, plan=None):
        from ride_gym.acceptance import validate_probability
        from ride_gym.response import next_response_boundary, validate_response_seconds

        if not order_ids:
            return False, None
        if plan is None:
            plan = self._prepare_invitation(driver, order_ids)
        context = self._invitation_context(driver, order_ids, plan)
        probability = (
            1.0
            if self.acceptance_policy is None
            else validate_probability(self.acceptance_policy.probability(context))
        )
        seconds = None
        if self.response_time_policy is not None:
            sample = getattr(self.response_time_policy, "sample_seconds", self.response_time_policy)
            before = deepcopy(self._response_rng.bit_generator.state)
            try:
                seconds = validate_response_seconds(sample(context, self._response_rng))
            except Exception:
                self._response_rng.bit_generator.state = before
                raise
        accepted = True
        if self.acceptance_policy is not None:
            accepted = bool(self._acceptance_rng.random() < probability)
            self._acceptance_draws += 1
        receipt = {
            "driver_id": driver.driver_id,
            "order_ids": list(order_ids),
            "kind": context.kind,
            "probability": probability,
            "offer_time": self.time,
        }
        if seconds is None:
            event.setdefault("acceptance_events", []).append({**receipt, "accepted": accepted})
            return accepted, plan
        self._response_draws += 1
        response_time = self.time + seconds / 60.0
        receipt.update(
            offer_id=self._next_offer_id,
            accepted=accepted,
            response_seconds=seconds,
            response_time=response_time,
            effective_time=next_response_boundary(response_time, self.dt),
        )
        self._next_offer_id += 1
        self._pending_responses[driver.driver_id] = receipt
        driver.awaiting_response = True
        self._response_order_ids.update(order_ids)
        event.setdefault("response_events", []).append(
            {k: deepcopy(v) for k, v in receipt.items() if k != "accepted"}
        )
        return False, None

    def _settle_responses(self, events):
        if self._pending_responses:
            # The clock has advanced since the normal start-of-step timeout
            # pass. A due reply must not revive an order expired at this boundary.
            self._cancel_timed_out()
        for did, pending in list(self._pending_responses.items()):
            driver, ids = self.drivers[did], pending["order_ids"]
            valid = (
                all(
                    oid in self._pending_ids and self.orders[oid].status == OrderStatus.PENDING
                    for oid in ids
                )
                and sum(self.orders[oid].num_passengers for oid in ids)
                + driver.committed_passengers(self.orders)
                <= driver.capacity
            )
            if valid and self.time + 1e-10 < pending["effective_time"]:
                continue
            outcome, prepared = "rejected", None
            if not valid:
                outcome = "invalidated"
            elif pending["accepted"]:
                try:
                    plan = self._prepare_invitation(driver, ids)
                    prepared = self._prepare_assignment(driver, ids, plan)
                except (ValueError, InvalidActionError):
                    outcome = "invalidated"
                else:
                    outcome = "committed"
            del self._pending_responses[did]
            driver.awaiting_response = False
            self._response_order_ids.difference_update(ids)
            if outcome == "committed":
                # An offer made on duty is a commitment even across shift end.
                driver.departed = False
                self._assign_orders(
                    did,
                    ids,
                    events[did],
                    _response_commit=True,
                    _prepared_assignment=prepared,
                )
                events[did].setdefault("assignment_times", {}).update(
                    {oid: self.time for oid in ids}
                )
            events[did].setdefault("response_events", []).append(
                {**deepcopy(pending), "outcome": outcome, "time": self.time}
            )

    def _extend_observations(self, obs, all_drivers):
        if (
            self.labor_source is not None
            or self._shift_enabled
            or self.response_time_policy is not None
        ):
            for did, driver in self.drivers.items():
                extra = {
                    "on_duty": driver.on_duty,
                    "departed": driver.departed,
                    "available": self._driver_available(did),
                    "vehicle_type": driver.vehicle_type,
                }
                all_drivers[did].update(extra)
                obs[did]["self"].update(extra)
                if self.response_time_policy is not None:
                    invitation = self._pending_responses.get(did)
                    obs[did]["self"]["pending_response"] = (
                        None
                        if invitation is None
                        else {
                            k: deepcopy(v)
                            for k, v in invitation.items()
                            if k not in ("accepted", "probability")
                        }
                    )

    def _timed_service(self, driver):
        return (
            driver.pickup_service_time > 0
            or driver.dropoff_service_time > 0
            or driver.service_order_id is not None
            or any(self.orders[oid].ready_time is not None for oid in driver.assigned_orders)
        )

    def _service_plan_times(self, driver, points):
        """Account for configured readiness and service in committed route ETAs.

        Invitation contexts remain travel-only. This accounting is applied
        after acceptance to the simulator's committed-order ETA and reward.
        """
        current, prefix = self._planning_start(driver)
        cursor, times = self.time + prefix, {}
        network = self._driver_network(driver)
        for index, point in enumerate(points):
            distance = network.distance(current, point.location)
            cursor += network.travel_time(distance)
            order = self.orders[point.order_id]
            if index == 0 and driver.service_order_id is not None:
                cursor = max(cursor, driver.service_until)
            else:
                if point.kind == "pickup" and order.ready_time is not None:
                    cursor = max(cursor, order.ready_time)
                cursor += (
                    driver.pickup_service_time
                    if point.kind == "pickup"
                    else driver.dropoff_service_time
                )
            times.setdefault(point.order_id, {})[point.kind] = cursor - self.time
            current = point.location
        return times

    def _advance_toward(self, driver, target, minutes):
        """Move for at most ``minutes``; return distance, elapsed, reached."""
        speed = self._driver_speed(driver)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Vehicle speed must be finite and positive")
        budget, moved = speed * minutes, 0.0
        if not self._is_graph_network():
            segment = self.network.shortest_path(driver.location, target)
            if segment.distance <= budget + 1e-12:
                driver.location = tuple(target)
                return segment.distance, segment.distance / speed, True
            fraction = budget / segment.distance
            driver.location = tuple(a + (b - a) * fraction for a, b in zip(driver.location, target))
            return budget, minutes, False
        net = self.network
        if driver.node_idx is None and driver.edge_to is None:
            driver.node_idx = int(net.snap(driver.location))
            driver.location = net.node_coord(driver.node_idx)
        if driver.edge_to is not None:
            length = net.node_distance(driver.edge_from, driver.edge_to)
            remaining = length - driver.edge_pos_m
            if budget + 1e-9 < remaining:
                driver.edge_pos_m += budget
                driver.location = self._edge_interp_location(driver)
                return budget, minutes, False
            moved += remaining
            budget -= remaining
            driver.node_idx = driver.edge_to
            driver.edge_from = driver.edge_to = None
            driver.edge_pos_m = 0.0
            driver.location = net.node_coord(driver.node_idx)
        destination = int(net.snap(target))
        if driver.node_idx == destination:
            return moved, moved / speed, True
        path = net.node_path(driver.node_idx, destination)
        if not path or path[-1] != destination:
            raise ValueError("No route to service stop")
        for origin, end in zip(path, path[1:]):
            length = net.node_distance(origin, end)
            if budget + 1e-9 >= length:
                moved += length
                budget -= length
                driver.node_idx = end
                driver.location = net.node_coord(end)
            else:
                driver.edge_from, driver.edge_to = origin, end
                driver.edge_pos_m, driver.node_idx = budget, None
                moved += budget
                driver.location = self._edge_interp_location(driver)
                return moved, moved / speed, False
        return moved, moved / speed, driver.node_idx == destination

    def _complete_service(self, driver, event, at):
        stop = driver.task_points.pop(0)
        order = self.orders[stop.order_id]
        recorded_time = at if self.statistics_granularity == "continuous" else self.time
        if stop.kind == "pickup":
            order.status, order.pickup_time = OrderStatus.ONBOARD, recorded_time
            driver.onboard_passengers += order.num_passengers
            event["picked_up_orders"].append(order.order_id)
        else:
            order.status, order.dropoff_time, order.eta = (
                OrderStatus.COMPLETED,
                recorded_time,
                None,
            )
            driver.onboard_passengers -= order.num_passengers
            driver.assigned_orders.remove(order.order_id)
            event["completed_orders"].append(order.order_id)
        driver.service_until = driver.service_order_id = driver.service_kind = None
        driver.waiting_until_ready = None
        self._refresh_driver_status(driver)

    def _move_with_service(self, driver, event):
        """Consume movement, readiness and stop service on a common minute clock.

        This path is used only when readiness/service constraints are supplied.
        The service clock is always precise. Statistics granularity controls
        recorded lifecycle times without changing completion scheduling.
        """
        cursor, end = self.time, self.time + self.dt
        started_onboard, moved = driver.onboard_passengers, 0.0
        while cursor <= end + 1e-12:
            target = self._current_target(driver)
            if target is None:
                break
            if driver.service_order_id is not None:
                stop = driver.task_points[0]
                if (stop.order_id, stop.kind) != (driver.service_order_id, driver.service_kind):
                    raise RuntimeError("Active service must remain the first planned stop")
                if driver.service_until > end + 1e-12:
                    break
                cursor = max(cursor, driver.service_until)
                self._complete_service(driver, event, cursor)
                continue
            if cursor >= end - 1e-12:
                at_target = (
                    driver.edge_to is None and driver.node_idx == self.network.snap(target)
                    if self._is_graph_network()
                    else self.network.distance(driver.location, target) <= 1e-12
                )
                if not at_target:
                    break
            distance, elapsed, reached = self._advance_toward(driver, target, end - cursor)
            moved += distance
            self._record_travel_segment(driver, event, distance)
            cursor += elapsed
            if not reached:
                break
            if not driver.task_points:
                driver.relocation_target = None
                self._refresh_driver_status(driver)
                continue
            stop = driver.task_points[0]
            order = self.orders[stop.order_id]
            ready = (
                cursor
                if stop.kind != "pickup" or order.ready_time is None
                else max(cursor, order.ready_time)
            )
            duration = (
                driver.pickup_service_time if stop.kind == "pickup" else driver.dropoff_service_time
            )
            driver.service_until = ready + duration
            driver.service_order_id, driver.service_kind = stop.order_id, stop.kind
            driver.waiting_until_ready = ready if ready > cursor else None
            # Loop once more even for coincident zero-duration stops. Each
            # completion shrinks the finite stop list, guaranteeing progress.
            if driver.service_until <= end + 1e-12:
                cursor = max(cursor, driver.service_until)
                self._complete_service(driver, event, cursor)
        event["distance_moved"] = moved
        event["time_moved"] = moved / self._driver_speed(driver)
        event["is_empty_move"] = moved > 0 and started_onboard == 0
        event["is_idle_wait"] = (
            moved == 0 and not driver.task_points and driver.relocation_target is None
        )
        for oid in driver.assigned_orders:
            if self.orders[oid].eta is not None:
                self.orders[oid].eta -= self.dt

    def capture_initial_state(self):
        """Capture physical, order and behavior state as a JSON-safe snapshot."""
        from ride_gym.warmup import capture_state

        return capture_state(self)

    def apply_initial_state(self, state):
        """Restore a compatible snapshot; does not reset its clock to zero."""
        from ride_gym.warmup import restore_state

        restore_state(self, state)

    def acceptance_configuration(self):
        if self.acceptance_policy is None:
            return {"enabled": False}
        from ride_gym.acceptance import policy_configuration

        return {
            "enabled": True,
            "seed": self.acceptance_seed,
            "effective_seed": self.acceptance_seed
            if self.acceptance_seed is not None
            else self._seed,
            "policy": policy_configuration(self.acceptance_policy),
        }

    def response_configuration(self):
        if self.response_time_policy is None:
            return {"enabled": False}
        from ride_gym.response import response_policy_configuration

        return {
            "enabled": True,
            "seed": self.response_seed,
            "effective_seed": self.response_seed if self.response_seed is not None else self._seed,
            "policy": response_policy_configuration(self.response_time_policy),
        }

    def _export_behavior(self, kind):
        policy = self.acceptance_policy if kind == "acceptance" else self.response_time_policy
        if policy is None:
            return None
        state = getattr(policy, "state_dict", None)
        if callable(state) and not callable(getattr(policy, "load_state_dict", None)):
            raise TypeError("stateful behavior policy must implement load_state_dict")
        result = {
            "schema_version": 1,
            "configuration": getattr(self, kind + "_configuration")(),
            "rng_state": deepcopy(getattr(self, "_" + kind + "_rng").bit_generator.state),
            "draws": getattr(self, "_" + kind + "_draws"),
            "policy_state": deepcopy(state()) if callable(state) else None,
        }
        if kind == "response":
            result.update(
                pending=deepcopy(list(self._pending_responses.values())),
                next_offer_id=self._next_offer_id,
            )
        return result

    def _validate_behavior(self, kind, state):
        policy = self.acceptance_policy if kind == "acceptance" else self.response_time_policy
        if policy is None:
            if state is not None:
                raise ValueError("Cannot restore enabled behavior into a disabled environment")
            return None
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 1
            or state.get("configuration") != getattr(self, kind + "_configuration")()
        ):
            raise ValueError("Behavior snapshot configuration differs")
        if type(state.get("draws")) is not int or state["draws"] < 0:
            raise ValueError("Invalid behavior draw count")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        if state.get("policy_state") is not None and not callable(
            getattr(policy, "load_state_dict", None)
        ):
            raise TypeError("Behavior policy cannot restore its saved state")
        if kind == "response":
            from ride_gym.acceptance import validate_probability
            from ride_gym.response import next_response_boundary, validate_response_seconds

            if (
                type(state.get("next_offer_id")) is not int
                or state["next_offer_id"] < 0
                or not isinstance(state.get("pending"), list)
            ):
                raise ValueError("Invalid pending response snapshot")
            drivers, orders, offers = set(), set(), set()
            for item in state["pending"]:
                if not isinstance(item, dict):
                    raise ValueError("Pending response must be an object")
                did, ids, offer = item.get("driver_id"), item.get("order_ids"), item.get("offer_id")
                if (
                    type(did) is not int
                    or did not in self.drivers
                    or did in drivers
                    or not isinstance(ids, list)
                    or not ids
                    or any(type(oid) is not int for oid in ids)
                    or len(ids) != len(set(ids))
                    or any(oid not in self._pending_ids or oid in orders for oid in ids)
                    or type(offer) is not int
                    or not 0 <= offer < state["next_offer_id"]
                    or offer in offers
                    or type(item.get("accepted")) is not bool
                ):
                    raise ValueError("Invalid or duplicate pending response")
                if (
                    item.get("kind") != ("bundle" if len(ids) > 1 else "single")
                    or sum(self.orders[oid].num_passengers for oid in ids)
                    + self.drivers[did].committed_passengers(self.orders)
                    > self.drivers[did].capacity
                ):
                    raise ValueError("Pending response kind or reserved capacity differs")
                validate_probability(item["probability"])
                seconds = validate_response_seconds(item["response_seconds"])
                offered, response, effective = (
                    item[key] for key in ("offer_time", "response_time", "effective_time")
                )
                if (
                    not all(math.isfinite(v) for v in (offered, response, effective))
                    or offered < 0
                    or offered > self.time + 1e-10
                    or effective <= self.time + 1e-10
                    or not math.isclose(response, offered + seconds / 60, rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(
                        effective,
                        next_response_boundary(response, self.dt),
                        rel_tol=0,
                        abs_tol=1e-10,
                    )
                ):
                    raise ValueError("Invalid response timing in snapshot")
                drivers.add(did)
                orders.update(ids)
                offers.add(offer)
        return deepcopy(state)

    def _restore_behavior(self, kind, state):
        state = self._validate_behavior(kind, state)
        if state is None:
            return
        policy = self.acceptance_policy if kind == "acceptance" else self.response_time_policy
        if state["policy_state"] is not None:
            policy.load_state_dict(deepcopy(state["policy_state"]))
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        setattr(self, "_" + kind + "_rng", rng)
        setattr(self, "_" + kind + "_draws", state["draws"])
        if kind == "response":
            self._next_offer_id = state["next_offer_id"]
            self._pending_responses = {item["driver_id"]: item for item in state["pending"]}
            self._response_order_ids = {
                oid for item in state["pending"] for oid in item["order_ids"]
            }
            for did, driver in self.drivers.items():
                driver.awaiting_response = did in self._pending_responses

    def export_acceptance_state(self):
        return self._export_behavior("acceptance")

    def restore_acceptance_state(self, state):
        self._restore_behavior("acceptance", state)

    def export_response_state(self):
        return self._export_behavior("response")

    def validate_response_state(self, state):
        return self._validate_behavior("response", state)

    def restore_response_state(self, state):
        self._restore_behavior("response", state)
