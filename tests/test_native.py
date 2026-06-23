"""Determinism + sanity tests for the REAL NetHack engine (libge_nethack.so).

Skips cleanly if the native lib isn't built or NETHACKDIR isn't set, so the
mock-only suite still runs anywhere. Run with:

    source scripts/env.sh
    python tests/test_native.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _available():
    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "goexplore", "native", "libge_nethack.so")
    return os.path.exists(lib) and "NETHACKDIR" in os.environ


def test_replay_determinism():
    """Fixed-seed reset + replay must reproduce exactly — the property
    Go-Explore's replay-based 'return' depends on."""
    import random
    from goexplore.native_env import NativeNLE

    env = NativeNLE(core=0x111, disp=0x222)
    rng = random.Random(0)
    actions = [rng.randrange(env.action_space.n) for _ in range(200)]

    def rollout(e):
        e.reset()  # reset() uses the env's own stored seed (set at construction)
        seq = []
        for a in actions:
            o, _, done, _ = e.step(a)
            seq.append(tuple(o["blstats"]))
            if done:
                break
        return seq

    a = rollout(env)
    b = rollout(env)
    assert a == b, "same-env replay must be exact"

    env2 = NativeNLE(core=0x111, disp=0x222)
    c = rollout(env2)
    assert a == c, "cross-env same-seed must be exact (lanes share a seed)"

    env3 = NativeNLE(core=0x999, disp=0x888)
    d = rollout(env3)
    assert a != d, "different seed should produce a different game"
    for e in (env, env2, env3):
        e.close()
    print(f"ok: native replay determinism exact ({len(a)} steps)")


def test_goexplore_runs_on_real_nethack():
    """A short vectorized phase-1 run on the real engine grows the archive and
    descends past the starting level."""
    import random
    from goexplore.cells import CellConfig
    from goexplore.vecenv import make_vecenv
    from goexplore.vectorized import VecGoExploreConfig, VectorizedGoExplore

    vec = make_vecenv("native", num_envs=16)
    cfg = VecGoExploreConfig(num_envs=16, max_env_steps=40_000, explore_steps=40,
                             search_seed=0, game_seed=0, log_every=0)
    ge = VectorizedGoExplore(vec, CellConfig(cell_w=6, cell_h=3), cfg)
    stats = ge.run()
    vec.close()
    assert len(ge.archive) > 20, f"archive should grow, got {len(ge.archive)} cells"
    assert stats.max_depth >= 2, f"should descend past dlvl 1, got {stats.max_depth}"
    print(f"ok: real-NetHack GE reached depth {stats.max_depth} "
          f"({len(ge.archive)} cells, {stats.iterations} rounds)")


if __name__ == "__main__":
    if not _available():
        print("SKIP: libge_nethack.so not built or NETHACKDIR unset "
              "(run scripts/build_native.sh and source scripts/env.sh)")
        sys.exit(0)
    test_replay_determinism()
    test_goexplore_runs_on_real_nethack()
    print("\nALL NATIVE TESTS PASSED")
