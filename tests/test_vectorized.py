"""Smoke test for vectorized (async-lane) Go-Explore on the mock vecenv.

Runs with plain `python tests/test_vectorized.py`. Verifies:
  1. the async-lane driver digs deep (archive logic survives vectorization),
  2. it beats an equal-budget random baseline,
  3. it is reproducible given the search seed,
  4. per-lane replay-based return is exact (the determinism Go-Explore needs).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goexplore.cells import CellConfig, game_depth
from goexplore.env import DeterministicEnv, make_env
from goexplore.vecenv import MockVecEnv
from goexplore.vectorized import VecGoExploreConfig, VectorizedGoExplore


def _run(num_envs=16, max_env_steps=60_000, search_seed=0):
    vec = MockVecEnv(num_envs)
    cfg = VecGoExploreConfig(
        num_envs=num_envs, max_env_steps=max_env_steps, explore_steps=40,
        search_seed=search_seed, game_seed=0, log_every=0,
    )
    ge = VectorizedGoExplore(vec, CellConfig(cell_w=1, cell_h=1), cfg)
    return ge, ge.run()


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


def test_vectorized_digs_deeper():
    ge, stats = _run()
    assert len(ge.archive) > 5, "archive should discover many cells"
    assert stats.max_depth >= 4, f"expected depth>=4, got {stats.max_depth}"
    baseline = _random_baseline_depth(stats.env_steps, seed=0)
    assert stats.max_depth > baseline, (
        f"vectorized GE depth {stats.max_depth} should beat random {baseline}"
    )
    print(f"ok: vectorized GE reached depth {stats.max_depth} "
          f"({len(ge.archive)} cells, {stats.iterations} rounds) vs random {baseline}")


def test_reproducible():
    ge1, s1 = _run(search_seed=3)
    ge2, s2 = _run(search_seed=3)
    assert s1.max_depth == s2.max_depth and len(ge1.archive) == len(ge2.archive), (
        "same search seed must reproduce the run"
    )
    print(f"ok: reproducible (depth {s1.max_depth}, {len(ge1.archive)} cells)")


def test_lane_replay_exact():
    vec = MockVecEnv(4)
    vec.reset(seed=5)
    traj = [1, 1, 1, 1, 4, 1, 1, 1, 1, 4]
    o1 = vec.reset_lanes([2], seed=5)[2]
    for a in traj:
        obs, _, _, _ = vec.step([a] * 4)
        o1 = obs[2]
    again = vec.reset_lanes([2], seed=5)[2]
    for a in traj:
        obs, _, _, _ = vec.step([a] * 4)
        o2 = obs[2]
    assert o1["blstats"] == o2["blstats"], "per-lane replay must be exact"
    print("ok: per-lane replay-based return is exact")


if __name__ == "__main__":
    test_lane_replay_exact()
    test_vectorized_digs_deeper()
    test_reproducible()
    print("\nALL VECTORIZED SMOKE TESTS PASSED")
