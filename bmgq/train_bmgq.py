"""Formal BMG-Q (GATDDQN) trainer.

A near-exact mirror of :mod:`mfddqn.train_mfddqn` / :mod:`iddqn.train_iddqn` --
the SAME ``BenchmarkConfig``, data-collection / eval cadence, nearest + Hungarian
baseline comparison, CSV logging and checkpointing -- so BMG-Q, MFDDQN and IDDQN
numbers are directly comparable. The only methodological difference is the
agent: each driver's state is enriched by graph attention over its top-K nearest
neighbours before pairing with orders; see :mod:`bmgq.gat`.

Run:

    python -m bmgq.train_bmgq

All hyper-parameters are in :class:`BMGTrainConfig`.
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
from iddqn.exploration import QNoiseExplorer, AnnealSchedule
from iddqn.replay import ReplayBuffer

from bmgq.gat_qnet import GATQNet
from bmgq.bmgq_inference import BMGQActor
from bmgq.bmgq_agent import BMGQAgent, BMGQConfig
from bmgq.bmgq_replay import BMGStepSnapshot


@dataclass
class BMGTrainConfig:
    """BMG-Q training hyper-parameters. Shares every field with TrainConfig for
    parity; the GAT-specific additions are ``neighbours_k``, ``embed_dim``,
    ``num_heads`` and ``gat_layers``."""

    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Optimisation / agent.
    gamma: float = 0.9998
    lr: float = 5e-4
    batch_size: int = 8
    tau: float = 0.005
    target_sync_every: int = 20
    grad_clip: float = 1.0
    hidden: tuple = (128,)

    # GAT specifics.
    neighbours_k: int = 20
    embed_dim: int = 64
    num_heads: int = 1
    gat_layers: int = 1

    # Replay / schedule.
    replay_capacity: int = 6_000
    warmup_snapshots: int = 120
    updates_per_step: int = 1

    # Episodes.
    num_episodes: int = 500
    eval_every: int = 10
    eval_baselines: bool = True
    # Train/val/test split control (NYC multi-window scenarios). When the
    # benchmark uses a split pool (benchmark.nyc_splits_dir set), periodic
    # evaluation runs on eval_split (held-out "val") and a final evaluation
    # after training runs on test_split ("test"). Ignored without a split pool.
    eval_split: str = "val"
    test_split: str = "test"
    final_test: bool = True

    # Exploration anneal.
    anneal_t0: float = 1.0
    anneal_mode: str = "exponential"
    anneal_decay: float = 0.99
    anneal_decay_steps: int = 20_000
    anneal_t_min: float = 0.001
    noise_coef: float = 1.0
    scale_stat: str = "std"
    scale_floor: float = 1e-3

    # Candidate pruning (kept for parity; dense by default).
    use_knn: bool = False
    k_nearest: int = 20

    # Infra.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    out_dir: str = "bmgq/runs"
    run_name: Optional[str] = None
    save_every: int = 10
    save_eval_details: bool = True
    verbose: bool = True


class _GreedyActorDispatch:
    """Adapts a :class:`BMGQActor` to the benchmark ``act(observations)`` API.

    Always greedy (``explore_step=None``); exposes ``last_assignment_distances``
    for the recorder, on the same footing as the baselines.
    """

    def __init__(self, actor: BMGQActor, network):
        self._actor = actor
        self._network = network
        self.last_assignment_distances: Dict[int, float] = {}

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        actions, _state, _nb, _cc, _dbg = self._actor.act(
            observations, explore_step=None
        )
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


class BMGQTrainer:
    """Owns the env, agent, replay buffer, explorer, and the run loop."""

    def __init__(self, cfg: BMGTrainConfig):
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

        # Pickup-distance gate unit conversion. The threshold is in km.
        # Abstract scenarios already use km coordinates -> (1, 1) and the
        # network metric is km. Graph scenarios (nyc / osmnx) use (lon,
        # lat) degrees -> scale to km with a lat-linear correction
        # (111 km/deg lat; 111*cos(lat0) km/deg lon at the area's centre
        # latitude), and OSMnx returns metres so the network gate scales
        # the threshold to metres.
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
        )
        self.encoder = FeatureEncoder(self.fc)

        qnet = GATQNet(
            self.fc.driver_dim,
            self.fc.order_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            hidden=cfg.hidden,
            gat_layers=cfg.gat_layers,
        )
        self.agent = BMGQAgent(
            self.fc.driver_dim,
            self.fc.order_dim,
            cfg.neighbours_k,
            BMGQConfig(
                gamma=cfg.gamma,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                target_sync_every=cfg.target_sync_every,
                grad_clip=cfg.grad_clip,
                device=cfg.device,
                allow_idle=bm.allow_idle,
            ),
            qnet=qnet,
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

        self.actor = BMGQActor(
            self.agent.online,
            self.encoder,
            area,
            self.network.speed,
            neighbours_k=cfg.neighbours_k,
            k_nearest=cfg.k_nearest,
            use_knn=cfg.use_knn,
            device=cfg.device,
            explorer=self.explorer,
            pickup_distance_threshold=bm.pickup_distance_threshold,
            distance_fn=(
                self.network.distance if bm.pickup_distance_metric == "network" else None
            ),
            coord_to_km=self._coord_to_km,
            network_distance_is_metres=self._net_dist_metres,
            allow_idle=bm.allow_idle,
        )

        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.global_step = 0

        run_name = cfg.run_name or time.strftime("bmgq_%Y%m%d_%H%M%S")
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

        with open(
            os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(cfg), f, indent=2)

        # Per-split baseline caches (val vs test draw different windows).
        self._baseline_cache: Dict[str, Dict[str, Dict]] = {}

        # Pre-build eval benchmark configs. With a split pool, val/test draw
        # from their own held-out window pools via nyc_split; without one
        # these are just the training benchmark (single-file replay).
        import dataclasses as _dc
        self._uses_splits = getattr(bm, "nyc_splits_dir", None) is not None
        if self._uses_splits:
            self._eval_bm = _dc.replace(bm, nyc_split=cfg.eval_split)
            self._test_bm = _dc.replace(bm, nyc_split=cfg.test_split)
        else:
            self._eval_bm = bm
            self._test_bm = bm

    # --------------------------------------------------------------- collect
    def collect_episode(self, episode: int = 0) -> Dict:
        """Run one behaviour-policy episode, push snapshots, and train."""
        cfg = self.cfg
        # Per-episode reset seed so train-mode window sampling AND random
        # party sizes vary across episodes, while staying reproducible for a
        # given start seed (seed + episode is deterministic in episode).
        obs, _ = self.env.reset(seed=cfg.benchmark.seed + episode)

        losses: List[float] = []
        ep_reward = 0.0
        steps = 0
        # prev = (state, neighbours, chosen_col, reward_vec) of the PREVIOUS step.
        prev = None

        while True:
            actions, state, neighbours, chosen_col, _dbg = self.actor.act(
                obs, explore_step=self.global_step
            )
            nobs, rew, dones, _info = self.env.step(actions)

            rvec = np.array([rew[d] for d in obs.keys()], dtype=np.float32)
            ep_reward += float(rvec.sum())

            done = dones["__all__"]
            if prev is not None:
                self.buffer.push(
                    BMGStepSnapshot(
                        state=prev[0],
                        state_neighbours=prev[1],
                        chosen_col=prev[2],
                        rewards=prev[3],
                        next_state=state,
                        next_state_neighbours=neighbours,
                        done=False,
                    )
                )
            prev = (state, neighbours, chosen_col, rvec)

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
                    BMGStepSnapshot(
                        state=prev[0],
                        state_neighbours=prev[1],
                        chosen_col=prev[2],
                        rewards=prev[3],
                        next_state=state,
                        next_state_neighbours=neighbours,
                        done=True,
                    )
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
    def evaluate(self, episode: int, split: Optional[str] = None) -> Dict:
        """Greedy evaluation on a split (val by default, or test)."""
        cfg = self.cfg
        split = split or cfg.eval_split
        is_test = split == cfg.test_split
        eval_bm = self._test_bm if is_test else self._eval_bm
        tag = "test" if is_test else "val"
        dispatch = _GreedyActorDispatch(self.actor, self.network)
        eval_out = self.eval_details_dir if cfg.save_eval_details else None
        summary, _rec = run_episode(
            dispatch,
            algorithm_name=f"bmgq_{tag}_ep{episode:04d}",
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
    def _print_eval_table(episode: int, tag: str, bmgq: Dict, baselines: Dict[str, Dict]):
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

        _line("bmgq", bmgq)
        for name, s in baselines.items():
            _line(name, s)
        print()

    # --------------------------------------------------------------- checkpoint
    def save_checkpoint(self, episode: int) -> str:
        path = os.path.join(self.ckpt_dir, f"bmgq_ep{episode}.pt")
        torch.save(
            {
                "episode": episode,
                "global_step": self.global_step,
                "online": self.agent.online.state_dict(),
                "target": self.agent.target.state_dict(),
                "optim": self.agent.optim.state_dict(),
                "driver_dim": self.fc.driver_dim,
                "order_dim": self.fc.order_dim,
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
        self.env.reset(seed=cfg.benchmark.seed)
        if cfg.verbose:
            n_orders = len(getattr(self.env, "_all_orders", []))
            order_src = (
                f"nyc-file:{n_orders}"
                if cfg.benchmark.network_kind == "nyc"
                else f"{cfg.benchmark.num_orders}"
            )
            print(
                f"BMG-Q training: {cfg.num_episodes} episodes, "
                f"device={cfg.device}, driver_dim={self.fc.driver_dim}, "
                f"order_dim={self.fc.order_dim}, embed={cfg.embed_dim}, "
                f"heads={cfg.num_heads}, K={cfg.neighbours_k}, "
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


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(cfg: Optional[BMGTrainConfig] = None) -> BMGQTrainer:
    """Entry point: build a trainer from ``cfg`` and run it."""
    trainer = BMGQTrainer(cfg or BMGTrainConfig())
    trainer.train()
    return trainer


if __name__ == "__main__":
    train()