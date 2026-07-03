"""Feature engineering for (driver, order) pairs.

The shared Q-network consumes a concatenation of a driver feature vector and an
order feature vector. A special *dummy order* (all-zero order features plus a
dummy flag) represents the 'take no order' action, so the network learns a
baseline Q-value for staying idle/continuing the current plan.

All coordinates are normalised by the service-area extent so the network sees
values in roughly [0, 1]; this keeps training stable and is invariant to the
absolute scale of the area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Coord = Tuple[float, float]
Area = Tuple[float, float, float, float]

# Driver status strings -> index for one-hot encoding.
_STATUS_ORDER = ["idle", "to_pickup", "to_dropoff", "relocating"]
_STATUS_INDEX = {s: i for i, s in enumerate(_STATUS_ORDER)}


@dataclass
class FeatureConfig:
    """Dimensions and normalisation constants for feature encoding.

    Attributes
    ----------
    area:
        Service-area bounds used to normalise coordinates.
    max_capacity:
        Capacity used to normalise capacity/onboard counts.
    max_wait:
        Wait-time (minutes) used to normalise an order's waiting time.
    horizon:
        Episode horizon (minutes) used to normalise the global clock.
    """

    area: Area
    max_capacity: int
    max_wait: float = 10.0
    horizon: float = 60.0
    # Max en-route orders kept as a sequence (Assignment-Net only); longer
    # plans are truncated, shorter ones masked-padded.
    max_seq_len: int = 6

    # Feature widths (kept as properties so the net can size its input layer).
    @property
    def driver_dim(self) -> int:
        # x, y, status one-hot(4), true_free_cap, onboard, committed, time
        return 2 + len(_STATUS_ORDER) + 1 + 1 + 1 + 1

    @property
    def order_dim(self) -> int:
        # ox, oy, dx, dy, num_passengers, waiting_time, dummy_flag
        return 4 + 1 + 1 + 1

    @property
    def pair_dim(self) -> int:
        return self.driver_dim + self.order_dim

    @property
    def non_seq_dim(self) -> int:
        # Driver non-sequence features = the flat driver vector above.
        return self.driver_dim

    @property
    def seq_token_dim(self) -> int:
        # Per en-route order token: ox, oy, dx, dy, party, onboard_flag, eta.
        return 4 + 1 + 1 + 1


class FeatureEncoder:
    """Encodes drivers and orders into normalised numpy feature vectors."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        xmin, ymin, xmax, ymax = cfg.area
        self._x0, self._y0 = xmin, ymin
        self._xspan = max(xmax - xmin, 1e-9)
        self._yspan = max(ymax - ymin, 1e-9)

    # ----------------------------------------------------------- coordinates
    def _norm_xy(self, coord: Coord) -> Tuple[float, float]:
        return (
            (coord[0] - self._x0) / self._xspan,
            (coord[1] - self._y0) / self._yspan,
        )

    # ----------------------------------------------------------- drivers
    def encode_driver(self, driver_obs: Dict, time: float) -> np.ndarray:
        """Encode a single driver's ``self`` observation block.

        Uses ``committed_passengers`` (onboard + assigned-not-yet-picked-up) to
        expose the *true* remaining capacity and future committed load, so the
        network can distinguish a genuinely idle driver from one that looks idle
        (onboard == 0) but is already en route to several pickups.
        """
        x, y = self._norm_xy(driver_obs["location"])
        status = np.zeros(len(_STATUS_ORDER), dtype=np.float32)
        status[_STATUS_INDEX[driver_obs["status"]]] = 1.0
        cap = self.cfg.max_capacity
        committed = driver_obs["committed_passengers"]
        true_free_cap = (cap - committed) / max(cap, 1)
        onboard = driver_obs["onboard_passengers"] / max(cap, 1)
        committed_norm = committed / max(cap, 1)
        t = time / max(self.cfg.horizon, 1e-9)
        return np.concatenate(
            [[x, y], status, [true_free_cap, onboard, committed_norm, t]]
        ).astype(np.float32)

    def encode_driver_struct(self, driver_obs: Dict, time: float):
        """Structured driver encoding for Assignment-Net.

        Returns ``(non_seq [non_seq_dim], seq [max_seq_len, seq_token_dim],
        mask [max_seq_len])``. ``non_seq`` reuses the flat driver vector;
        ``seq`` tokens come from ``assigned_order_details`` (en-route orders),
        truncated/padded to ``max_seq_len`` with a 1/0 validity ``mask``.
        """
        non_seq = self.encode_driver(driver_obs, time)
        L = self.cfg.max_seq_len
        tok_dim = self.cfg.seq_token_dim
        seq = np.zeros((L, tok_dim), dtype=np.float32)
        mask = np.zeros((L, ), dtype=np.float32)
        details = driver_obs.get("assigned_order_details", []) or []
        for i, od in enumerate(details[:L]):
            ox, oy = self._norm_xy(od["origin"])
            dx, dy = self._norm_xy(od["destination"])
            party = od["num_passengers"] / max(self.cfg.max_capacity, 1)
            onboard = 1.0 if od.get("onboard") else 0.0
            eta = od.get("eta")
            eta_n = 0.0 if eta is None else eta / max(self.cfg.horizon, 1e-9)
            seq[i] = [ox, oy, dx, dy, party, onboard, eta_n]
            mask[i] = 1.0
        return non_seq, seq, mask

    # ----------------------------------------------------------- orders
    def encode_order(self, order_obs: Dict) -> np.ndarray:
        """Encode a single real (non-dummy) order."""
        ox, oy = self._norm_xy(order_obs["origin"])
        dx, dy = self._norm_xy(order_obs["destination"])
        party = order_obs["num_passengers"] / max(self.cfg.max_capacity, 1)
        wait = order_obs["waiting_time"] / max(self.cfg.max_wait, 1e-9)
        dummy_flag = 0.0
        return np.array(
            [ox, oy, dx, dy, party, wait, dummy_flag], dtype=np.float32
        )

    def dummy_order(self) -> np.ndarray:
        """The 'take no order' pseudo-order: zeros plus the dummy flag set."""
        vec = np.zeros(self.cfg.order_dim, dtype=np.float32)
        vec[-1] = 1.0  # dummy flag
        return vec

    # ----------------------------------------------------------- batch helpers
    def encode_drivers(self, observations: Dict[int, Dict], time: float):
        """Encode all drivers; return (driver_ids, matrix [N, driver_dim])."""
        ids = list(observations.keys())
        mat = np.stack(
            [self.encode_driver(observations[d]["self"], time) for d in ids]
        )
        return ids, mat

    def encode_drivers_struct(self, observations: Dict[int, Dict], time: float):
        """Structured-encode all drivers for Assignment-Net.

        Returns ``(driver_ids, non_seq [N, non_seq_dim], seq [N, L, tok_dim],
        mask [N, L])``.
        """
        ids = list(observations.keys())
        non_seqs, seqs, masks = [], [], []
        for d in ids:
            ns, sq, mk = self.encode_driver_struct(observations[d]["self"], time)
            non_seqs.append(ns)
            seqs.append(sq)
            masks.append(mk)
        if ids:
            non_seq = np.stack(non_seqs)
            seq = np.stack(seqs)
            mask = np.stack(masks)
        else:
            non_seq = np.zeros((0, self.cfg.non_seq_dim), dtype=np.float32)
            seq = np.zeros(
                (0, self.cfg.max_seq_len, self.cfg.seq_token_dim),
                dtype=np.float32,
            )
            mask = np.zeros((0, self.cfg.max_seq_len), dtype=np.float32)
        return ids, non_seq, seq, mask

    def encode_orders(self, pending: List[Dict]):
        """Encode all pending orders; return (order_ids, matrix [M, order_dim])."""
        ids = [o["order_id"] for o in pending]
        if pending:
            mat = np.stack([self.encode_order(o) for o in pending])
        else:
            mat = np.zeros((0, self.cfg.order_dim), dtype=np.float32)
        return ids, mat