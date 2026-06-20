"""Smoke test: run Go-Explore on the deterministic mock and assert it works.

Runs with plain `python tests/test_smoke.py` (no pytest / numpy / torch needed).
Verifies the three properties that matter:
  1. replay-based return is exact (determinism),
  2. the archive grows and reaches depth Go-Explore is supposed to dig to,
  3. Go-Explore beats an equal-budget pure-random baseline on max depth.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goexplore.cells import CellConfig, game_depth
from goexplore.env import DeterministicEnv, make_env
from goexplore.goexplore import GoExplore, GoExploreConfig


def test_determinism():
    env = DeterministicEnv(make_env("mock"), seed_core=7)
    traj = [1, 1, 1, 1, 4, 1, 1, 1, 1, 4]  # EAST*4, DESCEND, ...
    o1, _ = env.reset_to(traj)
    o2, _ = env.reset_to(traj)
    assert o1["blstats"] == o2["blstats"], "replay must be exact (determinism)"
    print("ok: determinism / replay-based return is exact")


def _random_baseline_depth(total_steps, seed=0):
    import random
    rng = random.Random(seed)
    env = DeterministicEnv(make_env("mock"), seed_core=0)
    obs = env.reset()
    best = game_depth(obs)
    for _ in range(total_steps):
        obs, _, done, _ = env.step(rng.randrange(env.n_actions))
        best = max(best, game_depth(obs))
        if done:
            obs = env.reset()
    return best


def test_goexplore_digs_deeper():
    env = DeterministicEnv(make_env("mock"), seed_core=0)
    cfg = GoExploreConfig(iterations=600, explore_steps=40, search_seed=0, log_every=0)
    ge = GoExplore(env, CellConfig(cell_w=1, cell_h=1), cfg)
    stats = ge.run()

    assert len(ge.archive) > 5, "archive should discover many cells"
    assert stats.max_depth >= 4, f"expected to dig to depth>=4, got {stats.max_depth}"

    baseline = _random_baseline_depth(stats.env_steps, seed=0)
    assert stats.max_depth > baseline, (
        f"Go-Explore depth {stats.max_depth} should beat random baseline {baseline}"
    )
    print(f"ok: Go-Explore reached depth {stats.max_depth} "
          f"({len(ge.archive)} cells) vs random baseline depth {baseline}")


if __name__ == "__main__":
    test_determinism()
    test_goexplore_digs_deeper()
    print("\nALL SMOKE TESTS PASSED")
