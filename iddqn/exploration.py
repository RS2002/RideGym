"""Exploration via Q-magnitude-scaled noise on the matching inputs.

The behaviour policy perturbs the Q-values *before* the bipartite matching, so
exploration always respects the matching structure (no order conflicts ever
arise) while still being random. The noise scale adapts to the magnitude of the
current Q-values and is annealed by a temperature ``t`` toward zero:

    Q_perturbed = Q + noise * scale * t

where ``scale = noise_coef * stat(legal Q-values, including dummy) * t`` makes
the noise self-adapting to the Q magnitude (constant-scale noise would swamp the
small early-training Q-values and be negligible later). ``stat`` defaults to the
standard deviation of the Q pool (``scale_stat="std"``), which measures the
spread of the values the matcher chooses between and -- unlike a mean -- does
not collapse to ~0 when the pool straddles zero (sign cancellation); a
``"mean_abs"`` option is kept for reproducing reference work. A ``scale_floor``
guards against the noise vanishing when the Q values are tiny. As ``t -> 0`` the
behaviour policy collapses to the greedy matching.

Only legal (driver, order) entries and the dummy column are perturbed; illegal
entries stay at NEG_INF so exploration can never select an illegal action.

The TD target's next-action selection must NOT use this noise: it uses the plain
greedy matching (Double DQN greedy target). Hence noise lives only in the
behaviour/acting path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from iddqn.matching import NEG_INF


@dataclass
class AnnealSchedule:
    """Temperature annealing schedule for exploration noise.

    Attributes
    ----------
    t0:
        Initial temperature.
    mode:
        ``"exponential"`` -> ``t = t0 * decay ** step``;
        ``"linear"``      -> ``t = t0 * max(0, 1 - step / decay_steps)``.
    decay:
        Per-step multiplicative factor (exponential mode).
    decay_steps:
        Steps to reach zero (linear mode).
    t_min:
        Floor temperature (exponential mode never goes below this).
    """

    t0: float = 1.0
    mode: str = "exponential"
    decay: float = 0.9995
    decay_steps: int = 10000
    t_min: float = 0.0

    def temperature(self, step: int) -> float:
        if self.mode == "exponential":
            return max(self.t_min, self.t0 * (self.decay ** step))
        if self.mode == "linear":
            return self.t0 * max(0.0, 1.0 - step / max(self.decay_steps, 1))
        raise ValueError(f"unknown anneal mode {self.mode!r}")


class QNoiseExplorer:
    """Adds Q-magnitude-scaled, temperature-annealed noise to matching inputs."""

    def __init__(
        self,
        schedule: AnnealSchedule = None,
        noise_coef: float = 1.0,
        scale_stat: str = "std",
        scale_floor: float = 1e-3,
        rng: np.random.Generator = None,
    ):
        """
        Parameters
        ----------
        schedule:
            Temperature anneal schedule.
        noise_coef:
            Constant multiplier ``c`` on the Q-magnitude-derived scale.
        scale_stat:
            How the Q-magnitude scale is derived from the pool of legal real Q
            values plus all dummy Q values:

            * ``"std"``      -> ``std(pool)`` (default). Principled: it measures
              the *spread* of the Q values the matcher is choosing between, so
              the noise is comparable to the gaps it must perturb. Crucially it
              does NOT suffer the sign-cancellation of a mean: a pool with large
              positive and negative Q values has a near-zero mean (collapsing
              the noise to ~0) yet a large, correct std.
            * ``"mean_abs"`` -> ``|mean(pool)|`` (the older behaviour, kept for
              comparison / reproducing reference work). Susceptible to
              sign-cancellation when the pool straddles zero.
        scale_floor:
            Lower bound on the derived statistic before applying ``noise_coef``
            and the temperature, so exploration never silently dies just because
            the current Q values happen to be tiny / perfectly cancelling.
        rng:
            Optional numpy Generator for reproducibility.
        """
        if scale_stat not in ("std", "mean_abs"):
            raise ValueError(
                f"scale_stat must be 'std' or 'mean_abs', got {scale_stat!r}"
            )
        self.schedule = schedule or AnnealSchedule()
        self.noise_coef = float(noise_coef)
        self.scale_stat = scale_stat
        self.scale_floor = float(scale_floor)
        self.rng = rng or np.random.default_rng()

    def perturb(
        self,
        q_real: np.ndarray,
        q_dummy: np.ndarray,
        legal_mask: np.ndarray,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return noised copies of ``(q_real, q_dummy)`` for the behaviour policy.

        Illegal entries of ``q_real`` (NEG_INF) are left untouched so they can
        never be selected. The noise scale is ``noise_coef * |mean(legal Q)| *
        t`` with ``t`` from the schedule at ``step``.
        """
        t = self.schedule.temperature(step)
        if t <= 0.0:
            return q_real, q_dummy  # greedy

        # Q magnitude scale over the pool of legal real entries + all dummies.
        legal_vals = q_real[legal_mask] if legal_mask.any() else np.array([])
        pool = np.concatenate([legal_vals, q_dummy])
        if pool.size:
            if self.scale_stat == "std":
                stat = float(pool.std())
            else:  # "mean_abs"
                stat = float(abs(pool.mean()))
        else:
            stat = 1.0
        stat = max(stat, self.scale_floor)  # guard against vanishing noise
        scale = self.noise_coef * stat * t
        if scale <= 0.0:
            return q_real, q_dummy

        q_real_out = q_real.copy()
        if legal_mask.any():
            noise_real = self.rng.standard_normal(size=legal_mask.sum()) * scale
            q_real_out[legal_mask] = q_real[legal_mask] + noise_real
        noise_dummy = self.rng.standard_normal(size=q_dummy.shape) * scale
        q_dummy_out = q_dummy + noise_dummy
        return q_real_out, q_dummy_out