"""Vectorized (async-lane) Go-Explore phase 1.

Drives N lanes of a batched vecenv so every native step does useful work. Each
lane is in one of two phases:

    REPLAY  -- feeding the stored action sequence to return to a sampled cell
    EXPLORE -- taking random actions and inserting visited states into the archive

Lanes are ASYNCHRONOUS: a short-trajectory lane finishes its replay and flips to
EXPLORE while a long-trajectory lane is still replaying. When a lane finishes its
explore budget (or the episode ends) it RETIRES -- we re-home it to the game seed
and assign it a freshly sampled cell to replay. No lane ever idles, so the single
batched ``vec.step(actions)`` call (native C for the real env) stays saturated.

This is the same archive/selection logic as the single-env GoExplore; only the
driver changes. Validated against MockVecEnv in tests/test_vectorized.py.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .archive import Archive
from .cells import CellConfig, cell_descriptor, game_depth, game_score
from .goexplore import Stats

REPLAY, EXPLORE = 0, 1


@dataclass
class VecGoExploreConfig:
    num_envs: int = 64
    max_env_steps: int = 200_000   # total lane-steps budget (N per batched step)
    explore_steps: int = 60        # random steps per explore round
    repeat_prob: float = 0.0       # sticky-action probability
    search_seed: int = 0           # RNG seed for the search
    game_seed: int = 0             # NLE CORE seed all lanes share (single-seed GE)
    log_every: int = 20_000


class VectorizedGoExplore:
    def __init__(self, vec, cell_cfg: CellConfig, cfg: VecGoExploreConfig):
        self.vec = vec
        self.cell_cfg = cell_cfg
        self.cfg = cfg
        self.archive = Archive(rng=random.Random(cfg.search_seed + 1))
        # per-lane exploration RNGs keep the search reproducible
        self.lane_rng = [random.Random(cfg.search_seed * 100003 + i)
                         for i in range(vec.num_envs)]
        self.stats = Stats()
        self.rounds = 0

    def _cell(self, obs):
        return cell_descriptor(obs, self.cell_cfg)

    def _assign(self, i, phase, traj, rptr, expl, last):
        """Re-home lane i: sample a cell and set it up to replay then explore."""
        _key, entry = self.archive.sample()
        traj[i] = list(entry.traj)
        rptr[i] = 0
        expl[i] = 0
        last[i] = None
        phase[i] = EXPLORE if not traj[i] else REPLAY

    def _explore_action(self, i, last):
        rng = self.lane_rng[i]
        if last[i] is not None and rng.random() < self.cfg.repeat_prob:
            return last[i]
        return rng.randrange(self.vec.n_actions)

    def run(self) -> Stats:
        N = self.vec.num_envs
        obs = self.vec.reset(self.cfg.game_seed)

        # Seed the archive root (all lanes share the seed, so lane 0 suffices).
        self.archive.consider(self._cell(obs[0]), (), game_score(obs[0]),
                              game_depth(obs[0]), 0)

        phase = [None] * N
        traj = [None] * N
        rptr = [0] * N
        expl = [0] * N
        last = [None] * N
        for i in range(N):
            self._assign(i, phase, traj, rptr, expl, last)

        env_steps = 0
        next_log = self.cfg.log_every
        t0 = time.time()

        while env_steps < self.cfg.max_env_steps:
            actions = [0] * N
            for i in range(N):
                if phase[i] == REPLAY:
                    actions[i] = traj[i][rptr[i]]
                else:
                    actions[i] = self._explore_action(i, last)

            obs, _rew, done, _info = self.vec.step(actions)
            env_steps += N

            to_reset = []
            for i in range(N):
                if phase[i] == REPLAY:
                    rptr[i] += 1
                    if done[i]:
                        to_reset.append(i)            # replay broke -> re-home
                    elif rptr[i] >= len(traj[i]):
                        phase[i] = EXPLORE            # arrived at the cell
                        expl[i] = 0
                else:  # EXPLORE
                    last[i] = actions[i]
                    traj[i].append(actions[i])
                    o = obs[i]
                    score, depth = game_score(o), game_depth(o)
                    self.archive.consider(self._cell(o), traj[i], score, depth, self.rounds)
                    self.stats.best_score = max(self.stats.best_score, score)
                    expl[i] += 1
                    if done[i] or expl[i] >= self.cfg.explore_steps:
                        to_reset.append(i)

            if to_reset:
                fresh = self.vec.reset_lanes(to_reset, self.cfg.game_seed)
                for i in to_reset:
                    obs[i] = fresh[i]
                    self.rounds += 1
                    self._assign(i, phase, traj, rptr, expl, last)

            self.stats.env_steps = env_steps
            self.stats.iterations = self.rounds
            self.stats.cells = len(self.archive)
            self.stats.max_depth = self.archive.max_depth
            if self.cfg.log_every and env_steps >= next_log:
                sps = env_steps / max(1e-9, time.time() - t0)
                print(
                    f"[steps {env_steps:>9}] rounds={self.rounds:>6} "
                    f"cells={len(self.archive):>6} max_depth={self.archive.max_depth:>2} "
                    f"best_score={self.stats.best_score:>6} ({sps:,.0f}/s)"
                )
                next_log += self.cfg.log_every

        return self.stats

    def best_trajectory(self):
        return self.archive.best().traj
