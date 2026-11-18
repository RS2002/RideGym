"""Optional response-time policies for dispatch offers.

Policies return seconds using the environment's separate response RNG. The
simulation clock is in minutes. When response timing is enabled, replies become
effective at the strictly following decision boundary, while existing service
continues. Omitting a response policy preserves immediate assignment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol

from ride_gym.acceptance import AcceptanceContext


class ResponseTimePolicy(Protocol):
    """Implement ``sample_seconds`` using only the supplied random stream."""

    def sample_seconds(self, context: AcceptanceContext, rng) -> float:
        """Return a finite nonnegative duration in seconds."""
        ...


def validate_response_seconds(value) -> float:
    """Convert a finite nonnegative response duration without clipping."""
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("response time must be finite nonnegative seconds") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("response time must be finite nonnegative seconds")
    return seconds


@dataclass(frozen=True)
class FixedResponseTimePolicy:
    """A deterministic response duration, in seconds."""

    seconds: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "seconds", validate_response_seconds(self.seconds))

    def sample_seconds(self, context: AcceptanceContext, rng) -> float:
        return self.seconds

    def configuration(self) -> dict:
        return {"distribution": "fixed", "seconds": self.seconds}


@dataclass(frozen=True)
class UniformResponseTimePolicy:
    """A continuous uniform duration in ``[0, max_seconds]`` seconds."""

    max_seconds: float = 60.0

    def __post_init__(self):
        object.__setattr__(self, "max_seconds", validate_response_seconds(self.max_seconds))

    def sample_seconds(self, context: AcceptanceContext, rng) -> float:
        return float(rng.uniform(0.0, self.max_seconds))

    def configuration(self) -> dict:
        return {"distribution": "uniform", "max_seconds": self.max_seconds}


@dataclass(frozen=True)
class BetaResponseTimePolicy:
    """Duration ``max_seconds * Beta(alpha, beta)`` with positive shapes.

    The mean is ``max_seconds * alpha / (alpha + beta)`` seconds. Randomness
    comes exclusively from the response stream supplied by the environment.
    """

    alpha: float = 2.0
    beta: float = 1.0
    max_seconds: float = 60.0

    def __post_init__(self):
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            try:
                shape = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be finite and positive") from exc
            if isinstance(value, bool) or not math.isfinite(shape) or shape <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, shape)
        object.__setattr__(self, "max_seconds", validate_response_seconds(self.max_seconds))

    def sample_seconds(self, context: AcceptanceContext, rng) -> float:
        return self.max_seconds * float(rng.beta(self.alpha, self.beta))

    def configuration(self) -> dict:
        return {
            "distribution": "beta",
            "alpha": self.alpha,
            "beta": self.beta,
            "max_seconds": self.max_seconds,
        }


def response_policy_configuration(policy) -> dict:
    """Return an isolated JSON-compatible response-policy identity."""
    configuration = getattr(policy, "configuration", None)
    if not callable(configuration):
        raise TypeError("response policy must implement configuration() for checkpoints")
    value = json.loads(json.dumps(configuration(), allow_nan=False))
    return {"class": f"{type(policy).__module__}.{type(policy).__qualname__}", "parameters": value}


def next_response_boundary(response_time: float, dt: float) -> float:
    """Strictly next global boundary for an absolute response time in minutes.

    ULP-scale errors around exact multiples are normalized before choosing the
    next tick. ``dt`` must be finite and positive; the boundary must be finite.
    """
    try:
        response_time, dt = float(response_time), float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("response time and positive dt must be finite") from exc
    if not math.isfinite(response_time) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("response time and positive dt must be finite")
    quotient = response_time / dt
    if not math.isfinite(quotient):
        raise ValueError("response boundary cannot be represented with this dt")
    nearest = round(quotient)
    if abs(quotient - nearest) <= 8 * math.ulp(max(abs(quotient), 1.0)):
        quotient = float(nearest)
    result = (math.floor(quotient) + 1) * dt
    if not math.isfinite(result):
        raise ValueError("response boundary must be finite")
    return result
