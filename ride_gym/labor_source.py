"""Optional explicit fleets and availability schedules.

Times are simulation minutes. Speeds use the network's distance units per minute,
not an implicit geographic unit. A shift ending prevents new assignments; existing
commitments continue until completion. No schedule changes the driver status enum.
"""

from __future__ import annotations

import csv
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from ride_gym.entities import Driver

Coord = Tuple[float, float]


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _optional(value: Any, name: str, *, positive: bool = False):
    return None if value is None or value == "" else _number(value, name, positive=positive)


def _coord(value, name: str = "location") -> Coord:
    try:
        if (
            isinstance(value, (str, bytes))
            or len(value) != 2
            or any(isinstance(item, bool) for item in value)
        ):
            raise ValueError
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain two finite coordinates") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain two finite coordinates")
    return result


@dataclass(frozen=True)
class ShiftWindow:
    """An online interval ``[start, end)`` and optional entry location.

    An entry location is applied only upon entering the interval with no
    committed work. A busy driver continues their existing route in place.
    """

    start: float
    end: float
    location: Optional[Coord] = None

    def __post_init__(self):
        object.__setattr__(self, "start", _number(self.start, "shift start"))
        object.__setattr__(self, "end", _number(self.end, "shift end"))
        if self.end <= self.start:
            raise ValueError("shift end must be greater than shift start")
        if self.location is not None:
            object.__setattr__(self, "location", _coord(self.location, "shift location"))


def _windows(values) -> Tuple[ShiftWindow, ...]:
    if values is None or values == "":
        return ()
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except ValueError as exc:
            raise ValueError("availability_windows must be a JSON array") from exc
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("availability_windows must be a sequence of windows")
    parsed = []
    try:
        for value in values:
            if isinstance(value, ShiftWindow):
                window = value
            elif isinstance(value, Mapping):
                window = ShiftWindow(value["start"], value["end"], value.get("location"))
            else:
                window = ShiftWindow(*value)
            parsed.append(window)
    except (TypeError, KeyError) as exc:
        raise ValueError("each availability window needs start, end and optional location") from exc
    parsed.sort(key=lambda window: window.start)
    for previous, current in zip(parsed, parsed[1:]):
        if current.start < previous.end:
            raise ValueError("availability windows must not overlap or repeat")
    return tuple(parsed)


class ShiftPolicy(ABC):
    """Update new-work eligibility without interrupting committed work."""

    @abstractmethod
    def update(self, drivers: Iterable[Driver], time: float) -> None:
        """Refresh availability at simulation ``time`` in minutes."""

    def is_available(self, driver: Driver) -> bool:
        return bool(driver.on_duty) and not driver.departed


class AlwaysOnShiftPolicy(ShiftPolicy):
    """Explicitly keep all drivers available throughout the episode."""

    def update(self, drivers: Iterable[Driver], time: float) -> None:
        for driver in drivers:
            driver.on_duty = True
            driver.departed = False


class WindowShiftPolicy(ShiftPolicy):
    """Apply one shift or several nonoverlapping online intervals per driver.

    A driver without any schedule stays available. Off-duty drivers keep their
    commitments; ``departed`` becomes true only after they are drained. A later
    window can bring the same driver back online. For graph networks, callers
    must re-snap a driver whose entry location reset ``node_idx`` to ``None``.
    """

    def update(self, drivers: Iterable[Driver], time: float) -> None:
        time = _number(time, "time")
        for driver in drivers:
            windows = _windows(driver.availability_windows)
            start = _optional(driver.shift_start, "shift_start")
            end = _optional(driver.shift_end, "shift_end")
            if windows and (start is not None or end is not None):
                raise ValueError("use availability_windows or shift_start/shift_end, not both")
            if end is not None and end <= (0.0 if start is None else start):
                raise ValueError("shift_end must be greater than shift_start")
            busy = bool(
                driver.assigned_orders
                or driver.task_points
                or driver.onboard_passengers
                or driver.service_until is not None
                or driver.awaiting_response
            )
            if windows:
                active = next(
                    (i for i, window in enumerate(windows) if window.start <= time < window.end),
                    None,
                )
                previous = driver.active_shift_window
                if active is not None and active != previous and not busy:
                    location = windows[active].location
                    if location is not None:
                        driver.location = location
                        driver.node_idx = driver.edge_from = driver.edge_to = None
                        driver.edge_pos_m = 0.0
                        driver.route_nodes = []
                        driver.relocation_target = None
                driver.active_shift_window = active
                available = active is not None
                before_start = time < windows[0].start
            else:
                available = (start is None or time >= start) and (end is None or time < end)
                before_start = start is not None and time < start
            driver.on_duty = available
            driver.departed = not available and not before_start and not busy
            if driver.departed:
                driver.relocation_target = None


class LaborSource(ABC):
    """Construct a fleet with contiguous integer ids starting at zero."""

    @abstractmethod
    def build(self, num_drivers: int, rng, area, network) -> list[Driver]:
        """Return fresh drivers without mutating an existing environment."""


class LegacyLaborSource(LaborSource):
    """Opt-in version of the original uniform fleet placement algorithm."""

    def __init__(self, capacity=4, capacities=None, speeds=None, shift_policy=None):
        self.capacity = _integer(capacity, "capacity")
        self.capacities = (
            None
            if capacities is None
            else tuple(_integer(value, "capacity") for value in capacities)
        )
        self.speeds = (
            None
            if speeds is None
            else tuple(_optional(value, "speed", positive=True) for value in speeds)
        )
        self.shift_policy = shift_policy or AlwaysOnShiftPolicy()

    def build(self, num_drivers: int, rng, area, network) -> list[Driver]:
        count = _integer(num_drivers, "num_drivers")
        for name in ("capacities", "speeds"):
            values = getattr(self, name)
            if values is not None and len(values) != count:
                raise ValueError(f"{name} must contain num_drivers values")
        xmin, ymin, xmax, ymax = area
        graph = hasattr(network, "node_coords") and hasattr(network, "node_coord")
        drivers = []
        for did in range(count):
            node = int(rng.integers(0, len(network.node_coords))) if graph else None
            location = (
                network.node_coord(node)
                if graph
                else (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
            )
            drivers.append(
                Driver(
                    driver_id=did,
                    location=location,
                    capacity=self.capacity if self.capacities is None else self.capacities[did],
                    speed=None if self.speeds is None else self.speeds[did],
                    node_idx=node,
                )
            )
        return drivers


class ExplicitLaborSource(LaborSource):
    """A user-provided fleet with optional vehicle settings and schedules.

    Each mapping needs ``x``/``y``, ``lon``/``lat`` or a two-item ``location``.
    Optional fields are ``capacity``, ``speed``, ``vehicle_type``,
    ``pickup_service_time``, ``dropoff_service_time``, ``shift_start``,
    ``shift_end`` and ``availability_windows``. Times use minutes and speed uses
    the same distance units as the network per minute. No hidden unit conversion
    or empirical defaults are applied.
    """

    def __init__(
        self,
        specs: Sequence[Mapping[str, Any]],
        default_capacity=4,
        default_speed=None,
        shift_policy=None,
    ):
        self.default_capacity = _integer(default_capacity, "default_capacity")
        self.default_speed = _optional(default_speed, "default_speed", positive=True)
        self.shift_policy = shift_policy or WindowShiftPolicy()
        self.specs = tuple(self._parse(spec) for spec in specs)

    def _parse(self, spec):
        if not isinstance(spec, Mapping):
            raise TypeError("each fleet specification must be a mapping")
        coordinates = []
        if spec.get("location") is not None:
            coordinates.append(_coord(spec["location"]))
        for x, y in (("x", "y"), ("lon", "lat")):
            if x in spec or y in spec:
                if x not in spec or y not in spec:
                    raise ValueError(f"fleet location needs both {x} and {y}")
                coordinates.append(_coord((spec[x], spec[y])))
        if not coordinates or any(location != coordinates[0] for location in coordinates):
            raise ValueError("each fleet specification needs one consistent finite location")
        capacity = spec.get("capacity")
        capacity = (
            self.default_capacity
            if capacity is None or capacity == ""
            else _integer(capacity, "capacity")
        )
        speed = _optional(spec.get("speed"), "speed", positive=True)
        if speed is None:
            speed = self.default_speed
        start = _optional(spec.get("shift_start"), "shift_start")
        end = _optional(spec.get("shift_end"), "shift_end")
        if end is not None and end <= (0.0 if start is None else start):
            raise ValueError("shift_end must be greater than shift_start")
        windows = _windows(spec.get("availability_windows"))
        if windows and (start is not None or end is not None):
            raise ValueError("use availability_windows or shift_start/shift_end, not both")
        vehicle = spec.get("vehicle_type")
        if vehicle == "":
            vehicle = None
        if vehicle is not None and (not isinstance(vehicle, str) or not vehicle.strip()):
            raise ValueError("vehicle_type must be a nonempty string or None")
        service = {}
        for key in ("pickup_service_time", "dropoff_service_time"):
            value = spec.get(key)
            service[key] = 0.0 if value is None or value == "" else _number(value, key)
        return dict(
            location=coordinates[0],
            capacity=capacity,
            speed=speed,
            shift_start=start,
            shift_end=end,
            availability_windows=windows,
            vehicle_type=vehicle,
            **service,
        )

    def build(self, num_drivers: int, rng, area, network) -> list[Driver]:
        if _integer(num_drivers, "num_drivers") != len(self.specs):
            raise ValueError("fleet specification count must equal num_drivers")
        return [Driver(driver_id=did, **spec) for did, spec in enumerate(self.specs)]


class CsvLaborSource(ExplicitLaborSource):
    """Read the explicit-fleet schema from a UTF-8 CSV, one row per driver.

    Blank optional cells inherit defaults. ``availability_windows`` may contain
    a JSON array of window objects. Extra columns are ignored. No external data
    or filesystem-specific defaults are bundled with this source.
    """

    def __init__(self, path, default_capacity=4, default_speed=None, shift_policy=None):
        self.path = Path(path)
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        super().__init__(rows, default_capacity, default_speed, shift_policy)
