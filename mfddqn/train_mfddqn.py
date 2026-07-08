"""Formal MF-DDQN (Mean-Field Double DQN) trainer.

Mirrors bmgq/train_bmgq.py and iddqn/train_iddqn.py (same BenchmarkConfig,
collection/eval cadence, nearest+Hungarian baselines, CSV logging, checkpoints,
and the held-out val/test split evaluation) so MF-DDQN, BMG-Q and IDDQN numbers
are directly comparable. The methodological difference is the agent: every
Q-value is conditioned on a per-driver mean action field, solved by a Hungarian
fixed-point loop (see mfddqn.mean_field).

Run:

    python -m mfddqn.train_mfddqn
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
from iddqn.exploration import QNoiseExplorer, AnnealSchedule
from iddqn.replay import ReplayBuffer

from mfddqn.mf_qnet import MeanFieldPairQNet
from mfddqn.mf_inference import MFDDQNActor
from mfddqn.mf_agent import MFDDQNAgent, MFDDQNConfig
from mfddqn.mf_replay import MFStepSnapshot
from mfddqn.mean_field import MeanFieldConfig


@dataclass
class MFTrainConfig:
    """MF-DDQN training hyper-parameters. Shares every field with the IDDQN /
    BMG-Q configs for parity; the mean-field additions are ``neighbours_k``,
    ``mf_iters`` and ``simplified``."""

    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Optimisation / agent.
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 8
    tau: float = 0.01
    target_sync_every: int = 20
    grad_clip: float = 10.0
    hidden: tuple = (128, 128)

    # Mean-field specifics.
    # How each driver's mean-field neighbourhood is chosen:
    #   "knn"    -> the ``neighbours_k`` nearest drivers (fixed count).
    #   "radius" -> every other driver within ``neighbour_radius_km`` km
    #               (variable count; a true metric radius even on lon/lat).
    neighbour_mode: str = "knn"
    neighbours_k: int = 30
    neighbour_radius_km: float = 1.0
    mf_iters: int = 2
    simplified: bool = True

    # Replay / schedule.
    replay_capacity: int = 6_000
    warmup_snapshots: int = 120
    updates_per_step: int = 1

    # Episodes.
    num_episodes: int = 500
    eval_every: int = 10
    eval_baselines: bool = True
    # Train/val/test split control (NYC multi-window scenarios). Periodic eval
    # runs on eval_split (held-out "val"); a final eval after training runs on
    # test_split ("test"). Ignored without a split pool.
    eval_split: str = "val"
    test_split: str = "test"
    final_test: bool = True

    # Exploration anneal.
    anneal_t0: float = 1.0
    anneal_mode: str = "exponential"
    anneal_decay: float = 0.9998
    anneal_decay_steps: int = 20_000
    anneal_t_min: float = 0.001
    noise_coef: float = 1.0
    scale_stat: str = "std"  # "std" or "mean_abs"
    scale_floor: float = 1e-3

    # Candidate pruning (kept for parity; dense by default).
    use_knn: bool = False
    k_nearest: int = 20

    # Infra.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    out_dir: str = "mfddqn/runs"
    run_name: Optional[str] = None
    save_every: int = 10
    save_eval_details: bool = True
    verbose: bool = True


class _GreedyActorDispatch:
    """Adapts an MFDDQNActor to the benchmark act(observations) API.

    Always greedy (explore_step=None). For the simplified mean-field variant the
    carried-over a_bar is threaded across the episode's steps (reset per episode
    via reset_episode()). Exposes last_assignment_distances for the recorder.
    """

    def __init__(self, actor: MFDDQNActor, network):
        self._actor = actor
        self._network = network
        self.last_assignment_distances: Dict[int, float] = {}
        self._a_bar = None  # carried mean field (simplified variant)

    def reset_episode(self) -> None:
        self._a_bar = None

    def act(self, observations: Dict[int, Dict]) -> Dict[int, Dict]:
        actions, _state, _nb, _triples, a_bar_out, _dbg = self._actor.act(
            observations, explore_step=None, a_bar_in=self._a_bar
        )
        # Simplified variant carries a_bar across steps; full variant returns None.
        self._a_bar = a_bar_out
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


class MFDDQNTrainer:
    """Owns the env, agent, replay buffer, explorer, and the run loop."""

    def __init__(self, cfg: MFTrainConfig):
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
        )
        self.encoder = FeatureEncoder(self.fc)

                # Mean-field block width == order feature width.
        mean_field_dim = self.fc.order_dim
        self.mf_cfg = MeanFieldConfig(
            neighbour_mode=cfg.neighbour_mode,
            neighbours_k=cfg.neighbours_k,
            neighbour_radius_km=cfg.neighbour_radius_km,
            # Convert coordinate deltas to km for the radius test (identity for
            # abstract km scenarios; local lon/lat factors for osmnx/nyc).
            coord_to_km=self._coord_to_km,
            iters=cfg.mf_iters,
            simplified=cfg.simplified,
            # Shared by the actor and the agent's Q-target so idling behaviour
            # is identical in acting and bootstrapping. False -> a driver takes
            # no order only as a passive fallback.
            allow_idle=bm.allow_idle,
        )

        qnet = MeanFieldPairQNet(
            self.fc.pair_dim, mean_field_dim, hidden=cfg.hidden
        )
        self.agent = MFDDQNAgent(
            self.fc.pair_dim,
            mean_field_dim,
            MFDDQNConfig(
                gamma=cfg.gamma,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                target_sync_every=cfg.target_sync_every,
                grad_clip=cfg.grad_clip,
                device=cfg.device,
            ),
            self.mf_cfg,
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

        self.actor = MFDDQNActor(
            self.agent.online,
            self.encoder,
            area,
            self.network.speed,
            self.mf_cfg,
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
        )

        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.global_step = 0

        run_name = cfg.run_name or time.strftime("mfddqn_%Y%m%d_%H%M%S")
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
        self._uses_splits = getattr(bm, "nyc_splits_dir", None) is not None
        if self._uses_splits:
            self._eval_bm = dataclasses.replace(bm, nyc_split=cfg.eval_split)
            self._test_bm = dataclasses.replace(bm, nyc_split=cfg.test_split)
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
        # prev = (state, neighbours, action_triple_feats, a_bar_out, reward_vec)
        # of the PREVIOUS step.
        prev = None
        a_bar_in = None  # simplified variant: carried mean field

        while True:
            actions, state, neighbours, triples, a_bar_out, _dbg = self.actor.act(
                obs, explore_step=self.global_step, a_bar_in=a_bar_in
            )
            nobs, rew, dones, _info = self.env.step(actions)

            rvec = np.array([rew[d] for d in obs.keys()], dtype=np.float32)
            ep_reward += float(rvec.sum())

            done = dones["__all__"]
            if prev is not None:
                p_state, p_nb, p_triples, p_abar_out, p_rvec = prev
                self.buffer.push(
                    MFStepSnapshot(
                        state=p_state,
                        action_triple_feats=p_triples,
                        rewards=p_rvec,
                        next_state=state,
                        next_state_neighbours=(
                            None if cfg.simplified else neighbours
                        ),
                        done=False,
                        a_bar_out=(p_abar_out if cfg.simplified else None),
                    )
                )
            prev = (state, neighbours, triples, a_bar_out, rvec)
            a_bar_in = a_bar_out  # carry for simplified variant (None if full)

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
                p_state, p_nb, p_triples, p_abar_out, p_rvec = prev
                self.buffer.push(
                    MFStepSnapshot(
                        state=p_state,
                        action_triple_feats=p_triples,
                        rewards=p_rvec,
                        next_state=state,
                        next_state_neighbours=(
                            None if cfg.simplified else neighbours
                        ),
                        done=True,
                        a_bar_out=(p_abar_out if cfg.simplified else None),
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
        dispatch.reset_episode()
        eval_out = self.eval_details_dir if cfg.save_eval_details else None
        summary, _rec = run_episode(
            dispatch,
            algorithm_name=f"mfddqn_{tag}_ep{episode:04d}",
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
    def _print_eval_table(episode: int, tag: str, mfddqn: Dict, baselines: Dict[str, Dict]):
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

        _line("mfddqn", mfddqn)
        for name, s in baselines.items():
            _line(name, s)
        print()

    # --------------------------------------------------------------- checkpoint
    def save_checkpoint(self, episode: int) -> str:
        path = os.path.join(self.ckpt_dir, f"mfddqn_ep{episode}.pt")
        torch.save(
            {
                "episode": episode,
                "global_step": self.global_step,
                "online": self.agent.online.state_dict(),
                "target": self.agent.target.state_dict(),
                "optim": self.agent.optim.state_dict(),
                "pair_dim": self.fc.pair_dim,
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
                f"MF-DDQN training: {cfg.num_episodes} episodes, "
                f"device={cfg.device}, pair_dim={self.fc.pair_dim}, "
                f"order_dim={self.fc.order_dim}, K={cfg.neighbours_k}, "
                f"mf_iters={cfg.mf_iters}, simplified={cfg.simplified}, "
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


def train(cfg: Optional[MFTrainConfig] = None) -> MFDDQNTrainer:
    """Entry point: build a trainer from cfg and run it."""
    trainer = MFDDQNTrainer(cfg or MFTrainConfig())
    trainer.train()
    return trainer


if __name__ == "__main__":
    train()
