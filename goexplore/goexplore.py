"""Vanilla Go-Explore, phase 1 (explore) for NLE.

The loop is the canonical "first return, then explore":

    1. sample a promising cell from the archive
    2. RETURN to it (replay its trajectory from the seeded reset)
    3. EXPLORE from there with random actions for a fixed budget,
       inserting every state visited into the archive as a cell

Determinism makes step (2) exact and cheap to reason about. Phase 2
(robustification into a neural policy) lives in robustify.py.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .archive import Archive
from .cells import CellConfig, cell_descriptor, game_depth, game_score
from .env import DeterministicEnv


@dataclass
class GoExploreConfig:
    iterations: int = 2000        # number of return-then-explore rounds
    explore_steps: int = 60       # random steps taken per round
    repeat_prob: float = 0.0      # prob. of repeating the last action (sticky)
    search_seed: int = 0          # RNG seed for the *search* (not the game)
    log_every: int = 100


@dataclass
class Stats:
    iterations: int = 0
    env_steps: int = 0
    cells: int = 0
    max_depth: int = 1
    best_score: int = 0


class GoExplore:
    def __init__(self, env: DeterministicEnv, cell_cfg: CellConfig, cfg: GoExploreConfig):
        self.env = env
        self.cell_cfg = cell_cfg
        self.cfg = cfg
        self.rng = random.Random(cfg.search_seed)
        self.archive = Archive(rng=random.Random(cfg.search_seed + 1))
        self.stats = Stats()

    def _cell(self, obs):
        return cell_descriptor(obs, self.cell_cfg)

    def _act(self, last_action):
        if last_action is not None and self.rng.random() < self.cfg.repeat_prob:
            return last_action
        return self.rng.randrange(self.env.n_actions)

    def _seed_root(self):
        obs = self.env.reset()
        self.archive.consider(
            self._cell(obs), traj=(), score=game_score(obs),
            depth=game_depth(obs), iteration=0,
        )
        self.stats.best_score = game_score(obs)

    def run(self) -> Stats:
        self._seed_root()
        t0 = time.time()
        for it in range(1, self.cfg.iterations + 1):
            _key, entry = self.archive.sample()
            obs, done = self.env.reset_to(entry.traj)
            self.stats.env_steps += len(entry.traj)
            if done:
                continue  # trajectory terminates; nothing to explore from here

            traj = list(entry.traj)
            last_action = None
            for _ in range(self.cfg.explore_steps):
                action = self._act(last_action)
                obs, _reward, done, _info = self.env.step(action)
                last_action = action
                traj.append(action)
                self.stats.env_steps += 1

                score, depth = game_score(obs), game_depth(obs)
                self.archive.consider(
                    self._cell(obs), tuple(traj), score, depth, it,
                )
                self.stats.best_score = max(self.stats.best_score, score)
                if done:
                    break

            self.stats.iterations = it
            self.stats.cells = len(self.archive)
            self.stats.max_depth = self.archive.max_depth
            if self.cfg.log_every and it % self.cfg.log_every == 0:
                sps = self.stats.env_steps / max(1e-9, time.time() - t0)
                print(
                    f"[iter {it:>6}] cells={len(self.archive):>6} "
                    f"max_depth={self.archive.max_depth:>2} "
                    f"best_score={self.stats.best_score:>6} "
                    f"env_steps={self.stats.env_steps:>9} ({sps:,.0f}/s)"
                )
        return self.stats

    def best_trajectory(self):
        """Return the highest-value action sequence found (for robustification)."""
        return self.archive.best().traj
