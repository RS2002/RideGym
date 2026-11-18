"""Optional acceptance policies for complete dispatch offers.

Policies receive immutable snapshots and return a probability. The environment
owns the random draw and the offer lifecycle, so a policy never needs access to
mutable environment objects or its random-number generators.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

Coord = Tuple[float, float]


def validate_probability(value: Any, *, name: str = "probability") -> float:
    """Convert a finite probability in ``[0, 1]`` without clipping it."""
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")
    return probability


@dataclass(frozen=True)
class AcceptanceDriverState:
    """Driver state at the time of an offer."""

    driver_id: int
    location: Coord
    capacity: int
    speed: Optional[float]
    assigned_order_ids: Tuple[int, ...]
    onboard_passengers: int
    on_duty: bool
    status: str


@dataclass(frozen=True)
class AcceptanceOrderState:
    """One newly offered order; previous commitments are not included."""

    order_id: int
    origin: Coord
    destination: Coord
    request_time: float
    num_passengers: int


@dataclass(frozen=True)
class AcceptanceStopState:
    """Proposed stop and arrival time in minutes relative to this offer."""

    order_id: int
    kind: str
    location: Coord
    arrival_time: float


@dataclass(frozen=True)
class AcceptanceContext:
    """Read-only input to an acceptance or response-time policy.

    ``orders`` contains only newly offered orders. ``route`` may also include
    previous commitments. A bundle means more than one *new* order in this
    offer, regardless of the driver's existing workload.
    """

    time: float
    driver: AcceptanceDriverState
    orders: Tuple[AcceptanceOrderState, ...]
    route: Tuple[AcceptanceStopState, ...]

    @property
    def driver_id(self) -> int:
        return self.driver.driver_id

    @property
    def order_ids(self) -> Tuple[int, ...]:
        return tuple(order.order_id for order in self.orders)

    @property
    def is_bundle(self) -> bool:
        return len(self.orders) > 1

    @property
    def kind(self) -> str:
        return "bundle" if self.is_bundle else "single"


@runtime_checkable
class AcceptancePolicy(Protocol):
    """Policy interface; the environment owns its Bernoulli random stream.

    Policies used with checkpoints must also implement ``configuration()``
    with JSON-compatible immutable parameters. Stateful policies additionally
    implement ``state_dict()`` and ``load_state_dict(state)``; their state must
    include any private random streams they own.
    """

    def probability(self, context: AcceptanceContext) -> float:
        """Return an acceptance probability for the complete offered set."""
        ...


@dataclass(frozen=True)
class AcceptanceProbabilities:
    """General acceptance probability with optional single/bundle overrides."""

    general: float = 1.0
    single: Optional[float] = None
    bundle: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "general", validate_probability(self.general, name="general"))
        for name in ("single", "bundle"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_probability(value, name=name))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AcceptanceProbabilities":
        """Accept concise keys or descriptive probability column names."""
        if not isinstance(row, Mapping):
            raise TypeError("acceptance probabilities must be a mapping")
        return cls(
            general=row.get("general", row.get("acceptance_probability", 1.0)),
            single=row.get("single", row.get("single_acceptance_probability")),
            bundle=row.get("bundle", row.get("bundle_acceptance_probability")),
        )

    def for_kind(self, is_bundle: bool) -> float:
        specific = self.bundle if is_bundle else self.single
        return self.general if specific is None else specific


class PerDriverAcceptancePolicy:
    """Fixed probabilities keyed by integer simulator driver IDs.

    Missing drivers use ``default`` (always accept if omitted). Input mappings
    are copied and profiles are immutable; changing a caller's source table
    after construction cannot silently change a running simulation.
    """

    def __init__(
        self,
        profiles: Mapping[int, Any],
        default: Optional[AcceptanceProbabilities] = None,
    ):
        parsed = {}
        for driver_id, values in profiles.items():
            if isinstance(driver_id, bool) or not isinstance(driver_id, Integral):
                raise ValueError("acceptance profile keys must be integer driver IDs")
            parsed[int(driver_id)] = (
                values
                if isinstance(values, AcceptanceProbabilities)
                else AcceptanceProbabilities.from_mapping(values)
            )
        self.profiles = MappingProxyType(parsed)
        self.default = default if default is not None else AcceptanceProbabilities()
        if not isinstance(self.default, AcceptanceProbabilities):
            raise TypeError("default must be AcceptanceProbabilities")

    def probability(self, context: AcceptanceContext) -> float:
        probabilities = self.profiles.get(context.driver_id, self.default)
        return probabilities.for_kind(context.is_bundle)

    def configuration(self) -> dict:
        """Compact, stable profile identity without duplicating the fleet table."""
        rows = [[driver_id, asdict(values)] for driver_id, values in sorted(self.profiles.items())]
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        return {
            "profiles_sha256": sha256(encoded).hexdigest(),
            "profiles_count": len(rows),
            "default": asdict(self.default),
        }


def policy_configuration(policy: AcceptancePolicy) -> dict:
    """Return an isolated JSON-compatible policy identity for checkpoints."""
    configuration = getattr(policy, "configuration", None)
    if not callable(configuration):
        raise TypeError("acceptance policy must implement configuration() for checkpoints")
    value = json.loads(json.dumps(configuration(), allow_nan=False))
    return {
        "class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "configuration": value,
    }
