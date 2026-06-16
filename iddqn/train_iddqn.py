"""Formal IDDQN trainer.

End-to-end training loop for the Independent Double DQN with bipartite-matching
agent on the standard ride-pooling benchmark scenario. It

* collects whole-step snapshots with Q-magnitude-scaled annealed exploration,
  keeping the reward<->next_state alignment that the matching TD target relies
  on (rewards[i] of the current step align to next_state row i, valid because
  the env returns the same driver set in the same order every step);
* trains the shared PairQNet via the Double-DQN matching target;
* periodically runs a GREEDY evaluation episode (no exploration) and compares
  it against the nearest-distance and Hungarian baselines on identical
  scenarios, reusing :func:`benchmark.runner.run_episode` so every metric is
  computed with exactly the same recorder / definitions;
* logs reward / loss / evaluation KPI curves to CSV and saves checkpoints.

Run:

    python -m iddqn.train_iddqn

All hyper-parameters are in :class:`TrainConfig`.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from benchmark.config import BenchmarkConfig, make_benchmark_env, _make_network
from benchmark.baselines import NearestDistanceDispatch, HungarianDispatch
from benchmark.runner import run_episode

from iddqn.features import FeatureConfig, FeatureEncoder
from iddqn.qnet import PairQNet
from iddqn.inference import IDDQNActor
from iddqn.exploration import QNoiseExplorer, AnnealSchedule
from iddqn.replay import StepSnapshot, ReplayBuffer
from iddqn.agent import IDDQNAgent, IDDQNConfig


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """IDDQN training hyper-parameters and run controls."""

    # Scenario (defaults = full-scale standard benchmark).
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Optimisation / agent.
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 8
    # Target-network update. tau > 0 -> Polyak soft update every gradient step
    # (recommended). tau == 0 -> hard sync every ``target_sync_every`` steps.
    tau: float = 0.01
    target_sync_every: int = 20
    grad_clip: float = 10.0
    hidden: tuple = (128, 128)

    # Replay / schedule. With the full-scale benchmark (horizon=60 -> ~60
    # snapshots/episode) a 50k buffer holds ~800 episodes; warmup ~2 episodes.
    replay_capacity: int = 50_000
    warmup_snapshots: int = 120       # min snapshots before any gradient step
    updates_per_step: int = 1         # gradient updates per env step (after warmup)

    # Episodes.
    num_episodes: int = 500
    eval_every: int = 10              # run a greedy eval episode every N episodes
    eval_baselines: bool = True       # also run nearest/hungarian for comparison

    # Exploration anneal (steps counted globally across the whole run).
    anneal_t0: float = 1.0
    anneal_mode: str = "exponential"  # "exponential" | "linear"
    anneal_decay: float = 0.9995      # exponential per-step factor
    anneal_decay_steps: int = 20_000  # linear: steps to reach zero
    anneal_t_min: float = 0.0
    noise_coef: float = 1.0
    # Q-magnitude scale for exploration noise: "std" (principled, fixes the
    # mean's sign-cancellation) or "mean_abs" (older reference behaviour).
    scale_stat: str = "std"
    scale_floor: float = 1e-3

    # Featurisation / candidate pruning.
    # use_knn=False (default) -> dense, fully-connected (driver, order) matching:
    # every driver may match any capacity-feasible order. Pruning to each order's
    # k nearest drivers structurally removes most of the action space the agent
    # learns over; enable only for very large scenarios where dense matching is
    # the bottleneck. k_nearest controls the pruning width when use_knn is True.
    use_knn: bool = False
    k_nearest: int = 20

    # Infra.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    out_dir: str = "iddqn/runs"
    run_name: Optional[str] = None    # default: timestamp
    save_every: int = 10              # checkpoint every N episodes
    # If True, each greedy evaluation also persists the full per-step /
    # per-order / per-driver tables + manifest under the run's eval_details dir.
    save_eval_details: bool = True
    verbose: bool = True


# --------------------------------------------------------------------------- #
# Greedy-eval adapter: wrap IDDQNActor as a benchmark dispatch algorithm
# --------------------------------------------------------------------------- #
class _GreedyActorDispatch:
    """Adapts an :class:`IDDQNActor` to the benchmark ``act(observations)`` API.

    Evaluation is always greedy (``explore_step=None``), so the recorded metrics
    reflect the learned policy, not the behaviour policy. Exposes
    ``last_assignment_distances`` (matched pickup distance per assigned order) so
    the recorder's ``avg_matched_pickup_distance_km`` is populated on the same
    footing as the baselines.
    """

    def __init__(self, actor: IDDQNActor, network):
        self._actor = actor
        self._network = network
        self.last_assignment_distances: Dict[int, float] = {}

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        actions, _state, _apf, _dbg = self._actor.act(observations, explore_step=None)

        self.last_assignment_distances = {}
        if observations:
            for did, act in actions.items():
                for oid in act.get("orders", []):
                    drv_loc = observations[did]["self"]["location"]
                    origin = next(
                        o["origin"]
                        for o in observations[did]["pending_orders"]
                        if o["order_id"] == oid
                    )
                    self.last_assignment_distances[oid] = self._network.distance(
                        origin, drv_loc
                    )
        return actions


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class IDDQNTrainer:
    """Owns the env, agent, replay buffer, explorer, and the run loop."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        bm = cfg.benchmark

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = make_benchmark_env(bm)
        self.network = _make_network(bm)

        # True coordinate range of the scenario. On graph scenarios (osmnx /
        # nyc) this is the network's geographic bounding box, NOT bm.area's
        # default rectangle; using the wrong range would misplace the spatial
        # index and wreck feature normalisation. Read from the already-built
        # network so no second precompute is paid.
        if bm.network_kind in ("osmnx", "nyc"):
            area = self.network.bounds
        else:
            area = bm.area
        self.area = area

        self.fc = FeatureConfig(
            area=area,
            max_capacity=bm.driver_capacity,
            max_wait=bm.order_timeout or float(bm.horizon),
            horizon=bm.horizon,
        )
        self.encoder = FeatureEncoder(self.fc)

        self.agent = IDDQNAgent(
            self.fc.pair_dim,
            IDDQNConfig(
                gamma=cfg.gamma,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                target_sync_every=cfg.target_sync_every,
                grad_clip=cfg.grad_clip,
                device=cfg.device,
            ),
            qnet=PairQNet(self.fc.pair_dim, hidden=cfg.hidden),
        )

        self.explorer = QNoiseExplorer(
            schedule=AnnealSchedule(
                t0=cfg.anneal_t0,
                mode=cfg.anneal_mode,
                decay=cfg.anneal_decay,
                decay_steps=cfg.anneal_decay_steps,
                t_min=cfg.anneal_t_min,
            ),
            noise_coef=cfg.noise_coef,
            scale_stat=cfg.scale_stat,
            scale_floor=cfg.scale_floor,
            rng=np.random.default_rng(cfg.seed),
        )

        self.actor = IDDQNActor(
            self.agent.online,
            self.encoder,
            area,
            self.network.speed,
            k_nearest=cfg.k_nearest,
            use_knn=cfg.use_knn,
            device=cfg.device,
            explorer=self.explorer,
        )

        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.global_step = 0

        run_name = cfg.run_name or time.strftime("iddqn_%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(cfg.out_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.train_log_path = os.path.join(self.run_dir, "train_log.csv")
        self.eval_log_path = os.path.join(self.run_dir, "eval_log.csv")
        self.eval_details_dir = os.path.join(self.run_dir, "eval_details")
        os.makedirs(self.eval_details_dir, exist_ok=True)
        self._train_rows: List[Dict] = []
        self._eval_rows: List[Dict] = []

        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(_config_to_jsonable(cfg), f, indent=2)

        self._baseline_cache: Optional[Dict[str, Dict]] = None

    # --------------------------------------------------------------- collect
    def collect_episode(self) -> Dict:
        """Run one behaviour-policy episode, push snapshots, and train."""
        cfg = self.cfg
        obs, _ = self.env.reset(seed=cfg.benchmark.seed)

        losses: List[float] = []
        ep_reward = 0.0
        steps = 0
        # prev holds (state, action_pair_feats, reward_vec) of the PREVIOUS step;
        # paired with the CURRENT step's state as next_state. rewards[i] aligns to
        # next_state row i (same driver order every step).
        prev = None

        while True:
            actions, state, apf, _dbg = self.actor.act(
                obs, explore_step=self.global_step
            )
            nobs, rew, dones, _info = self.env.step(actions)

            rvec = np.array([rew[d] for d in obs.keys()], dtype=np.float32)
            ep_reward += float(rvec.sum())

            done = dones["__all__"]
            if prev is not None:
                self.buffer.push(
                    StepSnapshot(prev[0], prev[1], prev[2], state, False)
                )
            prev = (state, apf, rvec)

            if self.buffer.can_sample(cfg.batch_size) and (
                len(self.buffer) >= cfg.warmup_snapshots
            ):
                for _ in range(cfg.updates_per_step):
                    loss = self.agent.update(self.buffer.sample(cfg.batch_size))
                    losses.append(loss)

            obs = nobs
            self.global_step += 1
            steps += 1

            if done:
                self.buffer.push(
                    StepSnapshot(prev[0], prev[1], prev[2], state, True)
                )
                break

        return {
            "steps": steps,
            "ep_reward": ep_reward,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "num_updates": len(losses),
            "temperature": self.explorer.schedule.temperature(self.global_step),
            "replay_size": len(self.buffer),
        }

    # --------------------------------------------------------------- evaluate
    def evaluate(self, episode: int) -> Dict:
        """Greedy evaluation + (optional) baseline comparison; logs and returns."""
        cfg = self.cfg
        dispatch = _GreedyActorDispatch(self.actor, self.network)
        eval_out = self.eval_details_dir if cfg.save_eval_details else None
        summary, _rec = run_episode(
            dispatch,
            algorithm_name=f"iddqn_ep{episode:04d}",
            cfg=cfg.benchmark,
            out_dir=eval_out,
            verbose=False,
        )

        row = {
            "episode": episode,
            "global_step": self.global_step,
            # Reward is the headline overall-performance metric, logged first.
            "total_reward": summary["total_reward"],
            "avg_reward_per_driver": summary["avg_reward_per_driver"],
            "avg_reward_per_step": summary["avg_reward_per_step"],
            # service_rate = confirmed/total (orders ever assigned to a driver);
            # complete_rate = completed/total (orders actually delivered).
            "service_rate": summary["service_rate"],
            "complete_rate": summary["complete_rate"],
            "avg_wait_time": summary["avg_wait_time"],
            "avg_ride_time": summary["avg_ride_time"],
            "avg_detour_time": summary["avg_detour_time"],
            "empty_distance_ratio": summary["empty_distance_ratio"],
            "avg_driver_utilisation": summary["avg_driver_utilisation"],
            "completed": summary["completed"],
            "cancelled": summary["cancelled"],
        }
        self._eval_rows.append(row)
        _write_csv(self.eval_log_path, self._eval_rows)

        if cfg.verbose:
            base = self._baselines() if cfg.eval_baselines else {}
            self._print_eval_table(episode, summary, base)

        return row

    def _baselines(self) -> Dict[str, Dict]:
        """Compute (and cache) baseline summaries on the fixed eval scenario."""
        if self._baseline_cache is not None:
            return self._baseline_cache
        cfg = self.cfg
        out: Dict[str, Dict] = {}
        base_out = self.eval_details_dir if cfg.save_eval_details else None
        nearest = NearestDistanceDispatch.from_config(
            cfg.benchmark, k_nearest=cfg.k_nearest, use_knn=cfg.use_knn
        )
        s_near, _ = run_episode(
            nearest, "nearest", cfg=cfg.benchmark, out_dir=base_out, verbose=False
        )
        out["nearest"] = s_near
        hungarian = HungarianDispatch.from_config(
            cfg.benchmark, k_nearest=cfg.k_nearest, use_knn=cfg.use_knn
        )
        s_hun, _ = run_episode(
            hungarian, "hungarian", cfg=cfg.benchmark, out_dir=base_out, verbose=False
        )
        out["hungarian"] = s_hun
        self._baseline_cache = out
        return out

    @staticmethod
    def _print_eval_table(episode: int, iddqn: Dict, baselines: Dict[str, Dict]):
        cols = [
            ("total_reward", "reward"),
            ("service_rate", "service"),
            ("complete_rate", "complete"),
            ("avg_wait_time", "wait"),
            ("avg_ride_time", "ride"),
            ("avg_detour_time", "detour"),
            ("empty_distance_ratio", "empty%"),
            ("avg_driver_utilisation", "util"),
        ]
        print(f"\n--- eval @ episode {episode} ---")
        header = f"{'algo':12s}" + "".join(
            f"{lbl:>12s}" if k == "total_reward" else f"{lbl:>10s}"
            for k, lbl in cols
        )
        print(header)
        print("-" * len(header))

        def _line(name, s):
            cells = []
            for k, _ in cols:
                if k == "total_reward":
                    cells.append(f"{s[k]:>12.1f}")
                else:
                    cells.append(f"{s[k]:>10.4f}")
            print(f"{name:12s}" + "".join(cells))

        _line("iddqn", iddqn)
        for name, s in baselines.items():
            _line(name, s)
        print()

    # --------------------------------------------------------------- checkpoint
    def save_checkpoint(self, episode: int) -> str:
        path = os.path.join(self.ckpt_dir, f"iddqn_ep{episode}.pt")
        torch.save(
            {
                "episode": episode,
                "global_step": self.global_step,
                "online": self.agent.online.state_dict(),
                "target": self.agent.target.state_dict(),
                "optim": self.agent.optim.state_dict(),
                "pair_dim": self.fc.pair_dim,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.agent.online.load_state_dict(ckpt["online"])
        self.agent.target.load_state_dict(ckpt["target"])
        self.agent.optim.load_state_dict(ckpt["optim"])
        self.global_step = ckpt.get("global_step", 0)

    # --------------------------------------------------------------- main loop
    def train(self) -> None:
        cfg = self.cfg
        # Reset once up front so the order set is generated and ``_all_orders``
        # is populated before we report it (the per-episode loop resets again,
        # which is harmless / idempotent). Without this the count would read 0
        # because reset -- not __init__ -- is what builds the order list.
        self.env.reset(seed=cfg.benchmark.seed)
        if cfg.verbose:
            # Report the ACTUAL order count the env will replay. On NYC scenarios
            # orders come from the historical file (num_orders is ignored), so
            # print the real loaded count rather than the unused config field.
            n_orders = len(getattr(self.env, "_all_orders", []))
            order_src = (
                f"nyc-file:{n_orders}"
                if cfg.benchmark.network_kind == "nyc"
                else f"{cfg.benchmark.num_orders}"
            )
            print(
                f"IDDQN training: {cfg.num_episodes} episodes, "
                f"device={cfg.device}, pair_dim={self.fc.pair_dim}, "
                f"network={cfg.benchmark.network_kind}, "
                f"drivers={cfg.benchmark.num_drivers}, "
                f"orders={order_src}, use_knn={cfg.use_knn}"
            )
            print(f"run dir: {self.run_dir}")

        t_run = time.time()
        for ep in range(1, cfg.num_episodes + 1):
            t0 = time.time()
            stats = self.collect_episode()
            dt = time.time() - t0

            row = {"episode": ep, **stats, "wall_seconds": dt}
            self._train_rows.append(row)
            _write_csv(self.train_log_path, self._train_rows)

            if cfg.verbose:
                print(
                    f"[ep {ep:3d}/{cfg.num_episodes}] "
                    f"reward={stats['ep_reward']:10.2f} "
                    f"loss={stats['mean_loss']:.4f} "
                    f"updates={stats['num_updates']:4d} "
                    f"temp={stats['temperature']:.4f} "
                    f"replay={stats['replay_size']:6d} "
                    f"({dt:.1f}s)"
                )

            if ep % cfg.eval_every == 0 or ep == cfg.num_episodes:
                self.evaluate(ep)

            if ep % cfg.save_every == 0 or ep == cfg.num_episodes:
                path = self.save_checkpoint(ep)
                if cfg.verbose:
                    print(f"  checkpoint -> {path}")

        if cfg.verbose:
            print(f"\nDONE in {time.time() - t_run:.1f}s. Logs in {self.run_dir}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _config_to_jsonable(cfg: TrainConfig) -> Dict:
    """Flatten a TrainConfig (incl. nested BenchmarkConfig) for JSON dump."""
    return asdict(cfg)


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(cfg: Optional[TrainConfig] = None) -> IDDQNTrainer:
    """Entry point: build a trainer from ``cfg`` and run it."""
    trainer = IDDQNTrainer(cfg or TrainConfig())
    trainer.train()
    return trainer


if __name__ == "__main__":
    train()