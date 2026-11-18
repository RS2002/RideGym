"""Optional simulator snapshots and in-place pre-rolls.

Snapshots contain the simulator's physical state, the episode order stream and
its random/offer state. They are JSON documents, not Python pickles. Restore
into an environment with the same network and simulation configuration.

An external dispatcher, a stateful reward/planner/shift policy, and an order
generator's state for *future resets* are outside this format. Applications
using such components must checkpoint them separately. The already-generated
order stream for the current episode is included in full.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import copy, deepcopy
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ride_gym.entities import Driver, Order, TaskPoint
from ride_gym.enums import DriverStatus, OrderStatus

_SCHEMA = "ride_gym.simulator_state"
_VERSION = 1
_RNG_TYPES = ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64")


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, (str, Integral)) or isinstance(key, bool) for key in value):
            raise ValueError("Snapshot mappings require string or integer keys")
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Snapshot contains a non-JSON or non-finite value")


def _class_name(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def _array_identity(array):
    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def _configuration(env):
    network = {"class": _class_name(env.network), "speed": env.network.speed}
    configuration = getattr(env.network, "configuration", None)
    if callable(configuration):
        network["configuration"] = configuration()
    # The built-in graph backend exposes these immutable routing arrays.
    # Hash their contents without embedding a road graph in every snapshot.
    for name in ("_coords", "_dist_m", "_pred"):
        if isinstance(getattr(env.network, name, None), np.ndarray):
            network[name] = _array_identity(getattr(env.network, name))
    reward = {"class": _class_name(env.reward_function)}
    if type(env.reward_function).__module__ == "ride_gym.rewards":
        reward["parameters"] = vars(env.reward_function)
    shift = getattr(env, "shift_policy", None)
    shift_identity = None
    if shift is not None:
        shift_identity = {"class": _class_name(shift)}
        shift_configuration = getattr(shift, "configuration", None)
        if callable(shift_configuration):
            shift_identity["configuration"] = shift_configuration()
    return _jsonable(
        {
            "num_drivers": env.num_drivers,
            "area": env.area,
            "dt": env.dt,
            "horizon": env.horizon,
            "order_timeout": env.order_timeout,
            "statistics_granularity": getattr(env, "statistics_granularity", "decision"),
            "driver_capacities": env.driver_capacities,
            "driver_speeds": env.driver_speeds,
            "explicit_fleet_enabled": getattr(env, "labor_source", None) is not None,
            "network": network,
            "planner": _class_name(env.planner),
            "reward": reward,
            "relocation_points": env.relocation_points,
            "region_neighbours": env.region_neighbours,
            "shift_policy": shift_identity,
        }
    )


def _export_behavior(env, kind):
    method = getattr(env, "export_" + kind + "_state", None)
    if callable(method):
        return method()
    return None


def capture_state(env) -> Dict[str, Any]:
    """Capture an isolated, JSON-safe snapshot of an already-reset environment.

    Graph node/edge progress, readiness and stop-service waits, pending offers,
    behavior-policy state and all simulator-owned random streams are retained.
    Custom behavior policies must implement the environment's state protocol.
    """
    if set(env.drivers) != set(range(env.num_drivers)):
        raise ValueError("Reset the environment before capturing state")
    state = {
        "schema": _SCHEMA,
        "version": _VERSION,
        "configuration": _configuration(env),
        "time": env.time,
        "seed": env._seed,
        "rng": {
            "bit_generator": type(env._rng.bit_generator).__name__,
            "state": deepcopy(env._rng.bit_generator.state),
        },
        "drivers": [asdict(driver) for driver in env.drivers.values()],
        "orders": [asdict(order) for order in env.orders.values()],
        "all_order_ids": [order.order_id for order in env._all_orders],
        "pending_order_ids": list(env._pending_ids),
        "next_inject_idx": env._next_inject_idx,
        "acceptance": _export_behavior(env, "acceptance"),
        "response": _export_behavior(env, "response"),
    }
    result = _jsonable(state)
    # The encoder is also checked by save_snapshot; validate here so an in-memory
    # state has exactly the same value domain as a file round-trip.
    json.dumps(result, allow_nan=False)
    return result


def _finite(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    if nonnegative and value < 0:
        raise ValueError(name + " must be nonnegative")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(value)


def _coord(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + " must be an (x, y) coordinate")
    return (_finite(value[0], name), _finite(value[1], name))


def _rows(state, key):
    rows = state.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(key + " must be a list of objects")
    return rows


def _decode_order(row):
    allowed = {item.name for item in fields(Order)}
    if set(row) - allowed:
        raise ValueError("Snapshot has unsupported Order fields")
    data = deepcopy(row)
    data["order_id"] = _integer(data.get("order_id"), "order_id")
    data["origin"] = _coord(data.get("origin"), "origin")
    data["destination"] = _coord(data.get("destination"), "destination")
    data["request_time"] = _finite(data.get("request_time"), "request_time")
    data["num_passengers"] = _integer(data.get("num_passengers", 1), "num_passengers", 1)
    data["status"] = OrderStatus(data.get("status", OrderStatus.PENDING.value))
    if data.get("assigned_driver") is not None:
        data["assigned_driver"] = _integer(data["assigned_driver"], "assigned_driver")
    for key in ("pickup_time", "dropoff_time", "cancel_time", "eta", "ready_time"):
        if data.get(key) is not None:
            data[key] = _finite(data[key], key)
    return Order(**data)


def _decode_driver(row):
    allowed = {item.name for item in fields(Driver)}
    if set(row) - allowed:
        raise ValueError("Snapshot has unsupported Driver fields")
    data = deepcopy(row)
    data["driver_id"] = _integer(data.get("driver_id"), "driver_id")
    data["capacity"] = _integer(data.get("capacity"), "capacity", 1)
    data["location"] = _coord(data.get("location"), "location")
    data["status"] = DriverStatus(data.get("status", DriverStatus.IDLE.value))
    if data.get("speed") is not None:
        data["speed"] = _finite(data["speed"], "speed")
        if data["speed"] <= 0:
            raise ValueError("speed must be positive")
    data["onboard_passengers"] = _integer(data.get("onboard_passengers", 0), "onboard_passengers")
    for key in ("node_idx", "edge_from", "edge_to", "service_order_id", "active_shift_window"):
        if data.get(key) is not None:
            data[key] = _integer(data[key], key)
    data["edge_pos_m"] = _finite(data.get("edge_pos_m", 0.0), "edge_pos_m", True)
    for key in ("shift_start", "shift_end", "waiting_until_ready", "service_until"):
        if data.get(key) is not None:
            data[key] = _finite(data[key], key)
    for key in ("pickup_service_time", "dropoff_service_time"):
        if key in data:
            data[key] = _finite(data[key], key, True)
    for key in ("on_duty", "departed", "awaiting_response"):
        if key in data and type(data[key]) is not bool:
            raise ValueError(key + " must be boolean")
    if data.get("vehicle_type") is not None and not isinstance(data["vehicle_type"], str):
        raise ValueError("vehicle_type must be a string or None")
    if (
        data.get("shift_start") is not None
        and data.get("shift_end") is not None
        and data["shift_start"] >= data["shift_end"]
    ):
        raise ValueError("shift_start must precede shift_end")
    assigned = data.get("assigned_orders", [])
    if not isinstance(assigned, list):
        raise ValueError("assigned_orders must be a list")
    data["assigned_orders"] = [_integer(oid, "assigned order") for oid in assigned]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Duplicate assigned order")
    stops = data.get("task_points", [])
    if not isinstance(stops, list) or not all(isinstance(stop, dict) for stop in stops):
        raise ValueError("task_points must be a list of objects")
    data["task_points"] = []
    for stop in stops:
        if set(stop) != {"kind", "order_id", "location"} or stop["kind"] not in (
            "pickup",
            "dropoff",
        ):
            raise ValueError("Invalid task point")
        data["task_points"].append(
            TaskPoint(
                stop["kind"],
                _integer(stop["order_id"], "stop order"),
                _coord(stop["location"], "stop location"),
            )
        )
    data["route_nodes"] = [_coord(coord, "route node") for coord in data.get("route_nodes", [])]
    if data.get("relocation_target") is not None:
        data["relocation_target"] = _coord(data["relocation_target"], "relocation_target")
    if data.get("availability_windows"):
        from ride_gym.labor_source import ShiftWindow

        data["availability_windows"] = tuple(
            ShiftWindow(**window) for window in data["availability_windows"]
        )
    elif "availability_windows" in data:
        data["availability_windows"] = ()
    return Driver(**data)


def _validate_lifecycle(env, drivers, orders):
    owners = {}
    for did, driver in drivers.items():
        if driver.onboard_passengers > driver.capacity:
            raise ValueError("Driver onboard load exceeds capacity")
        stops = {}
        for point in driver.task_points:
            order = orders.get(point.order_id)
            if order is None or point.order_id not in driver.assigned_orders:
                raise ValueError("Task point refers to an unassigned order")
            expected = order.origin if point.kind == "pickup" else order.destination
            if point.location != expected:
                raise ValueError("Task point and order coordinates disagree")
            sequence = stops.setdefault(point.order_id, [])
            sequence.append(point.kind)
        onboard = 0
        committed = 0
        for oid in driver.assigned_orders:
            order = orders.get(oid)
            if order is None or oid in owners or order.assigned_driver != did:
                raise ValueError("Invalid or duplicate order ownership")
            owners[oid] = did
            if order.status == OrderStatus.ASSIGNED:
                expected = ["pickup", "dropoff"]
            elif order.status == OrderStatus.ONBOARD:
                expected = ["dropoff"]
                onboard += order.num_passengers
            else:
                raise ValueError("Driver owns an inactive order")
            if stops.get(oid) != expected:
                raise ValueError("Order stops violate pickup/dropoff lifecycle")
            committed += order.num_passengers
        if onboard != driver.onboard_passengers or committed > driver.capacity:
            raise ValueError("Driver load accounting is inconsistent")
        if env._is_graph_network():
            if (driver.edge_from is None) != (driver.edge_to is None):
                raise ValueError("Incomplete graph edge state")
            if driver.edge_to is not None:
                if driver.node_idx is not None:
                    raise ValueError("Driver cannot be both on a node and in an edge")
                length = env.network.node_distance(driver.edge_from, driver.edge_to)
                if not math.isfinite(length) or not 0 <= driver.edge_pos_m < length:
                    raise ValueError("Invalid graph edge progress")
                start = env.network.node_coord(driver.edge_from)
                end = env.network.node_coord(driver.edge_to)
                fraction = driver.edge_pos_m / length
                expected_location = tuple(a + (b - a) * fraction for a, b in zip(start, end))
            elif driver.node_idx is None:
                raise ValueError("Graph driver has neither a node nor an edge position")
            else:
                expected_location = env.network.node_coord(driver.node_idx)
            if any(
                not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)
                for a, b in zip(driver.location, expected_location)
            ):
                raise ValueError("Graph position and continuous location disagree")
        if getattr(driver, "service_until", None) is not None:
            if (
                not driver.task_points
                or driver.service_order_id != driver.task_points[0].order_id
                or driver.service_kind != driver.task_points[0].kind
            ):
                raise ValueError("Service wait does not match the next stop")
    for oid, order in orders.items():
        if order.status in (OrderStatus.ASSIGNED, OrderStatus.ONBOARD) and oid not in owners:
            raise ValueError("Active order has no driver")
        if order.status == OrderStatus.PENDING and order.assigned_driver is not None:
            raise ValueError("Pending order is already bound to a driver")


def _decode(env, state):
    if (
        not isinstance(state, dict)
        or state.get("schema") != _SCHEMA
        or state.get("version") != _VERSION
    ):
        raise ValueError("Unsupported simulator snapshot schema")
    if state.get("configuration") != _configuration(env):
        raise ValueError("Snapshot simulation configuration differs")
    now = _finite(state.get("time"), "snapshot time", True)
    drivers = {}
    for row in _rows(state, "drivers"):
        driver = _decode_driver(row)
        if driver.driver_id in drivers:
            raise ValueError("Duplicate driver id")
        drivers[driver.driver_id] = driver
    if set(drivers) != set(range(env.num_drivers)):
        raise ValueError("Snapshot driver ids must match the fixed fleet")
    orders = {}
    for row in _rows(state, "orders"):
        order = _decode_order(row)
        if order.order_id in orders:
            raise ValueError("Duplicate order id")
        orders[order.order_id] = order
    all_ids = state.get("all_order_ids")
    pending = state.get("pending_order_ids")
    if (
        not isinstance(all_ids, list)
        or not isinstance(pending, list)
        or any(isinstance(oid, bool) or not isinstance(oid, Integral) for oid in all_ids + pending)
        or len(all_ids) != len(set(all_ids))
        or set(all_ids) != set(orders)
        or len(pending) != len(set(pending))
        or not set(pending).issubset(orders)
    ):
        raise ValueError("Invalid snapshot order index")
    inject = _integer(state.get("next_inject_idx"), "next_inject_idx")
    if inject > len(all_ids):
        raise ValueError("Order injection index exceeds order stream")
    all_orders = [orders[oid] for oid in all_ids]
    if any(a.request_time > b.request_time for a, b in zip(all_orders, all_orders[1:])):
        raise ValueError("Order stream is not sorted by request time")
    for index, order in enumerate(all_orders):
        if index < inject and order.request_time > now:
            raise ValueError("A future order was already injected")
        if index >= inject and order.request_time <= now:
            raise ValueError("A due order is missing from the injected stream")
    expected_pending = {
        order.order_id for order in all_orders[:inject] if order.status == OrderStatus.PENDING
    }
    if set(pending) != expected_pending:
        raise ValueError("Pending pool does not match order lifecycle")
    _validate_lifecycle(env, drivers, orders)
    response = state.get("response")
    if response is None:
        response_drivers = set()
    else:
        if not isinstance(response, dict) or not isinstance(response.get("pending"), list):
            raise ValueError("Invalid snapshot response state")
        response_drivers = set()
        for offer in response["pending"]:
            if not isinstance(offer, dict):
                raise ValueError("Invalid snapshot response offer")
            did = _integer(offer.get("driver_id"), "response driver")
            if did not in drivers or did in response_drivers:
                raise ValueError("Invalid or duplicate response driver")
            response_drivers.add(did)
    if any(
        getattr(driver, "awaiting_response", False) != (did in response_drivers)
        for did, driver in drivers.items()
    ):
        raise ValueError("Driver awaiting_response flags disagree with pending offers")
    seed = state.get("seed")
    if seed is not None:
        seed = _integer(seed, "snapshot seed")
    rng = state.get("rng", {})
    if not isinstance(rng, dict) or rng.get("bit_generator") not in _RNG_TYPES:
        raise ValueError("Unsupported snapshot random generator")
    bitgen = getattr(np.random, rng["bit_generator"])()
    bitgen.state = deepcopy(rng["state"])
    return {
        "time": now,
        "drivers": drivers,
        "orders": orders,
        "_all_orders": all_orders,
        "_pending_ids": list(pending),
        "_next_inject_idx": inject,
        "_seed": seed,
        "_rng": np.random.Generator(bitgen),
    }


def restore_state(env, state) -> None:
    """Validate and restore a snapshot without resetting the episode clock.

    Entity/index/route invariants and behavior states are checked on an isolated
    trial environment before installation. Custom behavior policies are copied
    for validation, so a rejected snapshot cannot partially mutate live state.
    The caller remains responsible for matching custom network semantics and
    separately restoring external policy/planner/reward state.
    """
    state = deepcopy(state)
    candidate = _decode(env, state)
    trial = copy(env)
    trial.__dict__ = dict(env.__dict__)
    trial.__dict__.update(candidate)
    for kind in ("acceptance", "response"):
        attribute = "acceptance_policy" if kind == "acceptance" else "response_time_policy"
        if hasattr(env, attribute):
            policy = getattr(env, attribute)
            # Immutable built-ins may hold MappingProxyType tables, which are
            # intentionally not pickle/deepcopy-able. Only a stateful policy
            # needs an independent mutable clone for load_state_dict validation.
            state_method = getattr(policy, "state_dict", None)
            try:
                cloned = deepcopy(policy) if callable(state_method) else copy(policy)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Stateful behavior policies must support isolated copying"
                ) from error
            if callable(state_method) and cloned is policy:
                raise ValueError("Stateful behavior policy copy must be an independent instance")
            setattr(trial, attribute, cloned)
        restore = getattr(trial, "restore_" + kind + "_state", None)
        if callable(restore):
            restore(deepcopy(state.get(kind)))
        elif state.get(kind) is not None:
            raise ValueError("Environment does not support saved " + kind + " behavior")
    # Install the fully validated clone in one mapping update. Network, reward,
    # planner and order source are still the original objects; only the copied
    # behavior-policy instances and simulator-owned mutable state are replaced.
    env.__dict__.update(trial.__dict__)


def save_snapshot(path, state) -> None:
    """Atomically save a JSON snapshot; no object deserialization is involved."""
    path = Path(path)
    document = _jsonable(state)
    if (
        not isinstance(document, dict)
        or document.get("schema") != _SCHEMA
        or document.get("version") != _VERSION
    ):
        raise ValueError("Unsupported simulator snapshot schema")
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def load_snapshot(path) -> Dict[str, Any]:
    """Read a snapshot document; full compatibility is checked on restore."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or state.get("schema") != _SCHEMA
        or state.get("version") != _VERSION
    ):
        raise ValueError("Unsupported simulator snapshot schema")
    return _jsonable(state)


def warmup(env, policy, steps):
    """Advance an already-reset env and return ``(obs, info, snapshot)``.

    ``policy`` may be callable or expose ``act(observations)``. The clock keeps
    advancing normally: a warm-up is an ordinary prefix of the same episode,
    without an implicit reset, rebase, reward filter or evaluation boundary.
    """
    steps = _integer(steps, "warmup steps")
    act = getattr(policy, "act", policy)
    if not callable(act):
        raise TypeError("warmup policy must be callable or expose act(observations)")
    if set(env.drivers) != set(range(env.num_drivers)):
        raise ValueError("Reset the environment before warmup")
    remaining = max(0.0, env.horizon - env.time)
    if steps * env.dt > remaining + 1e-10:
        raise ValueError("Warmup would advance beyond the episode horizon")
    observations = env._build_observations()
    info = {"time": env.time}
    for _ in range(steps):
        observations, _, _, info = env.step(act(observations))
    return observations, info, capture_state(env)
