"""Formal IDDQN trainer.

End-to-end training loop for the Independent Double DQN with bipartite-matching
agent on the standard ride-pooling benchmark scenario.

Run:

    python -m iddqn.train_iddqn

All hyper-parameters are in :class:`TrainConfig`.
"""

from __future__ import annotations

import csv
import json
import os
import time
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch

from benchmark.config import BenchmarkConfig, make_benchmark_env, _make_network
from benchmark.baselines import NearestDistanceDispatch, HungarianDispatch
from benchmark.runner import run_episode

from iddqn.features import FeatureConfig, FeatureEncoder
from iddqn.qnet import PairQNet
from iddqn.cv_qnet import CVNet, HEX_DEFAULT_RESOLUTIONS
from iddqn.inference import IDDQNActor
from iddqn.assignment_net import AssignmentNet
from iddqn.assignment_inference import AssignmentActor
from iddqn.exploration import QNoiseExplorer, AnnealSchedule
from iddqn.replay import StepSnapshot, ReplayBuffer

from iddqn.agent import IDDQNAgent, AssignmentAgent, IDDQNConfig


@dataclass
class TrainConfig:
    """IDDQN training hyper-parameters and run controls."""

    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 8
    tau: float = 0.005
    target_sync_every: int = 20
    grad_clip: float = 1.0
    hidden: tuple = (128, 128)

    net_arch: str = "assignment"
    embed_dim: int = 64
    tf_heads: int = 4
    max_seq_len: int = 6

    # --- CV-Net (net_arch == "cvnet") multi-scale position embedding ---------
    # Spatial-cell SHAPE for the position embedding:
    #   "square" -> axis-aligned square grids (cv_resolutions below);
    #   "hex"    -> H3-style pointy-top hexagonal grids (cv_hex_* below), each
    #              level 1/7 the area of its parent. Hexagons are equidistant to
    #              all six neighbours (isotropic generalisation), as in the
    #              original CV-Net paper -- unlike squares whose diagonal
    #              neighbours are sqrt(2) farther.
    cv_grid_type: str = "hex"
    # (square only) Cells-per-axis of the small / medium / high-granularity
    # grids. A location is discretised on each grid and its per-grid embeddings
    # are averaged into one position embedding (see iddqn.cv_qnet.CVNet).
    cv_resolutions: tuple = (4, 7, 10)
        # (hex only) Per-level hexagons-per-axis (coarse -> fine), the hex analogue
    # of cv_resolutions. Default (4, 4*sqrt(7), 28) ~= (4, 10.58, 28) bakes in
    # the paper's H3-style 1/7-area hierarchy (each level sqrt(7)x the per-axis
    # resolution -> 1/7 the hex area); override with any resolutions you like.
    cv_hex_resolutions: tuple = HEX_DEFAULT_RESOLUTIONS
    # Per-grid / per-level position-embedding width.
    cv_pos_embed_dim: int = 32
    # Multi-scale aggregation: "mean" (default) or "concat".
    cv_aggregate: str = "mean"

    replay_capacity: int = 6_000
    warmup_snapshots: int = 120
    updates_per_step: int = 1

    num_episodes: int = 500
    eval_every: int = 10
    eval_baselines: bool = True
    # Train/val/test split control (NYC multi-window scenarios). Periodic
    # evaluation runs on eval_split (held-out "val"); a final evaluation after
    # training runs on test_split ("test"). Ignored without a split pool.
    eval_split: str = "val"
    test_split: str = "test"
    final_test: bool = True

    anneal_t0: float = 1.0
    anneal_mode: str = "exponential"
    anneal_decay: float = 0.9998
    anneal_decay_steps: int = 20_000
    anneal_t_min: float = 0.001
    noise_coef: float = 1.0
    scale_stat: str = "std"  # "std" or "mean_abs"
    scale_floor: float = 1e-3

    use_knn: bool = False
    k_nearest: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    out_dir: str = "iddqn/runs"
    run_name: Optional[str] = None
    save_every: int = 10
    save_eval_details: bool = True
    verbose: bool = True


class _GreedyActorDispatch:
    """Adapts an :class:`IDDQNActor` to the benchmark ``act(observations)`` API."""

    def __init__(self, actor: IDDQNActor, network):
        self._actor = actor
        self._network = network
        self.last_assignment_distances: Dict[int, float] = {}

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        out = self._actor.act(observations, explore_step=None)
        actions = out[0]
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


class IDDQNTrainer:
    """Owns the env, agent, replay buffer, explorer, and the run loop."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        bm = cfg.benchmark

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = make_benchmark_env(bm)
        self.network = _make_network(bm)

        if bm.network_kind in ("osmnx", "nyc"):
            area = self.network.bounds
        else:
            area = bm.area
        self.area = area

        if bm.network_kind in ("osmnx", "nyc"):
            _lat0 = 0.5 * (area[1] + area[3])
            self._coord_to_km = (
                111.0 * float(np.cos(np.radians(_lat0))),
                111.0,
            )
            self._net_dist_metres = True
        else:
            self._coord_to_km = (1.0, 1.0)
            self._net_dist_metres = False

        self.fc = FeatureConfig(
            area=area,
            max_capacity=bm.driver_capacity,
            max_wait=bm.order_timeout or float(bm.horizon),
            horizon=bm.horizon,
            max_seq_len=cfg.max_seq_len,
        )
        self.encoder = FeatureEncoder(self.fc)

        self.net_arch = cfg.net_arch
        agent_cfg = IDDQNConfig(
            gamma=cfg.gamma,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            target_sync_every=cfg.target_sync_every,
            grad_clip=cfg.grad_clip,
            device=cfg.device,
            # Idling rule for the Q-target's next-state action selection; must
            # match the actor below so target and behaviour agree.
            allow_idle=bm.allow_idle,
        )
        if cfg.net_arch == "assignment":
            net = AssignmentNet(
                non_seq_dim=self.fc.non_seq_dim,
                seq_token_dim=self.fc.seq_token_dim,
                order_dim=self.fc.order_dim,
                embed_dim=cfg.embed_dim,
                tf_heads=cfg.tf_heads,
            )
            self.agent = AssignmentAgent(net, agent_cfg)
        elif cfg.net_arch == "cvnet":
            # CV-Net: same two-tower matching pipeline as the MLP path, but
            # positions are encoded by multi-scale grid embeddings instead of
            # raw continuous values. Reuses IDDQNAgent unchanged (drop-in for
            # PairQNet).
            self.agent = IDDQNAgent(
                self.fc.pair_dim,
                agent_cfg,
                                qnet=CVNet(
                    self.fc.pair_dim,
                    driver_dim=self.fc.driver_dim,
                                        grid_type=cfg.cv_grid_type,
                    resolutions=cfg.cv_resolutions,
                    hex_resolutions=cfg.cv_hex_resolutions,
                    pos_embed_dim=cfg.cv_pos_embed_dim,
                    hidden=cfg.hidden,
                    embed_dim=cfg.embed_dim,
                    aggregate=cfg.cv_aggregate,
                ),
            )
        else:
                self.agent = IDDQNAgent(
                self.fc.pair_dim,
                agent_cfg,
                # Two-tower PairQNet: encode the driver half and the order half
                # separately (split at driver_dim), then fuse. driver_dim is the
                # concatenation split point every caller uses (driver first).
                qnet=PairQNet(
                    self.fc.pair_dim,
                    hidden=cfg.hidden,
                    driver_dim=self.fc.driver_dim,
                    embed_dim=cfg.embed_dim,
                ),
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

        actor_cls = (
            AssignmentActor if cfg.net_arch == "assignment" else IDDQNActor
        )
        self.actor = actor_cls(
            self.agent.online,
            self.encoder,
            area,
            self.network.speed,
            k_nearest=cfg.k_nearest,
            use_knn=cfg.use_knn,
            device=cfg.device,
            explorer=self.explorer,
            pickup_distance_threshold=bm.pickup_distance_threshold,
            distance_fn=(
                self.network.distance
                if bm.pickup_distance_metric == "network"
                                else None
            ),
            coord_to_km=self._coord_to_km,
            network_distance_is_metres=self._net_dist_metres,
            # When False, a driver actively takes no order only as a passive
            # fallback (assigned a legal order whenever one is available).
            allow_idle=bm.allow_idle,
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

        # Per-split baseline caches (val baselines differ from test baselines).
        self._baseline_cache: Dict[str, Dict[str, Dict]] = {}

        # Pre-build eval benchmark configs. With a split pool, val/test draw
        # from their own held-out window pools via nyc_split. Without one these
        # are just the training benchmark (single-file replay).
        self._uses_splits = getattr(bm, "nyc_splits_dir", None) is not None
        if self._uses_splits:
            self._eval_bm = dataclasses.replace(bm, nyc_split=cfg.eval_split)
            self._test_bm = dataclasses.replace(bm, nyc_split=cfg.test_split)
        else:
            self._eval_bm = bm
            self._test_bm = bm

    def _make_snapshot(self, prev, next_state, done: bool) -> StepSnapshot:
        state, act_data, rvec = prev
        if self.net_arch == "assignment":
            aof, adummy = act_data
            return StepSnapshot(
                state, None, rvec, next_state, done,
                action_order_feats=aof, action_is_dummy=adummy,
            )
        return StepSnapshot(state, act_data, rvec, next_state, done)

    def collect_episode(self, episode: int = 0) -> Dict:
        cfg = self.cfg
        # Per-episode reset seed so train-mode window sampling AND random
        # party sizes vary across episodes, while staying reproducible for a
        # given start seed (seed + episode is deterministic in episode).
        obs, _ = self.env.reset(seed=cfg.benchmark.seed + episode)

        losses: List[float] = []
        ep_reward = 0.0
        steps = 0
        prev = None

        assignment = self.net_arch == "assignment"
        while True:
            if assignment:
                actions, state, aof, adummy, _dbg = self.actor.act(
                    obs, explore_step=self.global_step
                )
                act_data = (aof, adummy)
            else:
                actions, state, apf, _dbg = self.actor.act(
                    obs, explore_step=self.global_step
                )
                act_data = apf
            nobs, rew, dones, _info = self.env.step(actions)

            rvec = np.array([rew[d] for d in obs.keys()], dtype=np.float32)
            ep_reward += float(rvec.sum())

            done = dones["__all__"]
            if prev is not None:
                self.buffer.push(self._make_snapshot(prev, state, False))
            prev = (state, act_data, rvec)

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
                self.buffer.push(self._make_snapshot(prev, state, True))
                break

        return {
            "steps": steps,
            "ep_reward": ep_reward,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "num_updates": len(losses),
            "temperature": self.explorer.schedule.temperature(self.global_step),
            "replay_size": len(self.buffer),
        }

    def evaluate(self, episode: int, split: Optional[str] = None) -> Dict:
        """Greedy evaluation on a given split (val by default, or test)."""
        cfg = self.cfg
        split = split or cfg.eval_split
        is_test = split == cfg.test_split
        eval_bm = self._test_bm if is_test else self._eval_bm
        tag = "test" if is_test else "val"
        dispatch = _GreedyActorDispatch(self.actor, self.network)
        eval_out = self.eval_details_dir if cfg.save_eval_details else None
        summary, _rec = run_episode(
            dispatch,
            algorithm_name=f"iddqn_{tag}_ep{episode:04d}",
            cfg=eval_bm,
            out_dir=eval_out,
            verbose=False,
        )

        row = {
            "episode": episode,
            "split": tag,
            "global_step": self.global_step,
            "total_reward": summary["total_reward"],
            "avg_reward_per_driver": summary["avg_reward_per_driver"],
            "avg_reward_per_step": summary["avg_reward_per_step"],
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
            base = self._baselines(split) if cfg.eval_baselines else {}
            self._print_eval_table(episode, tag, summary, base)

        return row

    def _baselines(self, split: Optional[str] = None) -> Dict[str, Dict]:
        """Compute (and cache PER SPLIT) baseline summaries on the eval scenario."""
        cfg = self.cfg
        split = split or cfg.eval_split
        is_test = split == cfg.test_split
        tag = "test" if is_test else "val"
        if tag in self._baseline_cache:
            return self._baseline_cache[tag]
        eval_bm = self._test_bm if is_test else self._eval_bm
        out: Dict[str, Dict] = {}
        base_out = self.eval_details_dir if cfg.save_eval_details else None
        nearest = NearestDistanceDispatch.from_config(
            eval_bm, k_nearest=cfg.k_nearest, use_knn=cfg.use_knn
        )
        s_near, _ = run_episode(
            nearest, f"nearest_{tag}", cfg=eval_bm, out_dir=base_out, verbose=False
        )
        out["nearest"] = s_near
        hungarian = HungarianDispatch.from_config(
            eval_bm, k_nearest=cfg.k_nearest, use_knn=cfg.use_knn
        )
        s_hun, _ = run_episode(
            hungarian, f"hungarian_{tag}", cfg=eval_bm, out_dir=base_out, verbose=False
        )
        out["hungarian"] = s_hun
        self._baseline_cache[tag] = out
        return out

    @staticmethod
    def _print_eval_table(episode: int, tag: str, iddqn: Dict, baselines: Dict[str, Dict]):
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
        print(f"\n--- {tag} eval @ episode {episode} ---")
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

    def save_checkpoint(self, episode: int) -> str:
        path = os.path.join(self.ckpt_dir, f"iddqn_ep{episode}.pt")
        torch.save(
            {
                "episode": episode,
                "global_step": self.global_step,
                "online": self.agent.online.state_dict(),
                "target": self.agent.target.state_dict(),
                "optim": self.agent.optim.state_dict(),
                "net_arch": self.net_arch,
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

    def train(self) -> None:
        cfg = self.cfg
        self.env.reset(seed=cfg.benchmark.seed)
        if cfg.verbose:
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
            stats = self.collect_episode(ep)
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
                # Periodic evaluation on the VALIDATION split (held-out windows).
                self.evaluate(ep, split=cfg.eval_split)

            if ep % cfg.save_every == 0 or ep == cfg.num_episodes:
                path = self.save_checkpoint(ep)
                if cfg.verbose:
                    print(f"  checkpoint -> {path}")

        # Final held-out TEST evaluation (disjoint from train and val windows).
        if cfg.final_test:
            if cfg.verbose:
                print("\n=== final held-out TEST evaluation ===")
            self.evaluate(cfg.num_episodes, split=cfg.test_split)

        if cfg.verbose:
            print(f"\nDONE in {time.time() - t_run:.1f}s. Logs in {self.run_dir}")


def _config_to_jsonable(cfg: TrainConfig) -> Dict:
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
    trainer = IDDQNTrainer(cfg or TrainConfig())
    trainer.train()
    return trainer


if __name__ == "__main__":
    train()