"""A tiny deterministic NLE stand-in for testing Go-Explore offline.

It mimics the parts of the NLE gym contract that Go-Explore touches:
  * ``action_space.n``
  * ``seed(core, disp, reseed)``  -> like NLE, sets the game seed
  * ``reset()``                   -> dict obs with a ``blstats`` vector
  * ``step(a)``                   -> (obs, reward, done, info), 4-tuple gym style

The "game": a corridor of width W. You start each level at x=0 and must walk
EAST to x=W-1 and then DESCEND to reach the next level (score += 100). HP ticks
down stochastically (seeded), bounding episode length. Crucially the dynamics
are a pure function of (seed, action sequence), so replay-based return is exact
-- exactly the property the real engine guarantees with reseed=False.

Reaching depth N requires building on the path to depth N-1, so random rollouts
alone get stuck shallow while Go-Explore (returning to the frontier) digs deeper.
"""

from __future__ import annotations

# Action ids (a reduced, NLE-like set; only a few are meaningful)
WEST, EAST, NORTH, SOUTH, DESCEND, WAIT = range(6)
WIDTH = 5
START_HP = 25
MAX_STEPS = 400


class Discrete:
    def __init__(self, n):
        self.n = n


class _LCG:
    """Reproducible PRNG seeded at reset, advanced every step (like ISAAC's role)."""
    def __init__(self, seed):
        self.state = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.state >> 33


class MockNLE:
    BLSTATS_LEN = 27

    def __init__(self, width: int = WIDTH):
        self.action_space = Discrete(6)
        self.width = width
        self._seed = 0

    def seed(self, core=0, disp=0, reseed=False):
        self._seed = int(core)
        return [core, disp]

    def reset(self):
        self.rng = _LCG(self._seed)
        self.x = 0
        self.level = 1
        self.hp = START_HP
        self.score = 0
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        if action == EAST:
            self.x = min(self.width - 1, self.x + 1)
        elif action == WEST:
            self.x = max(0, self.x - 1)
        elif action == DESCEND and self.x == self.width - 1:
            self.level += 1
            self.score += 100
            self.x = 0
        # NORTH/SOUTH/WAIT are no-ops in this corridor world

        # seeded "damage": ~1 in 6 steps costs an HP
        if self.rng.next() % 6 == 0:
            self.hp -= 1

        reward = 0.0
        done = self.hp <= 0 or self.t >= MAX_STEPS
        return self._obs(), reward, done, {}

    def _obs(self):
        bl = [0] * self.BLSTATS_LEN
        from . import nle_constants as C
        bl[C.BL_X] = self.x
        bl[C.BL_Y] = 0
        bl[C.BL_SCORE] = self.score
        bl[C.BL_HP] = max(0, self.hp)
        bl[C.BL_HPMAX] = START_HP
        bl[C.BL_DEPTH] = self.level
        bl[C.BL_DLEVEL] = self.level
        bl[C.BL_DNUM] = 0
        bl[C.BL_TIME] = self.t
        # a trivial "map": position marker, enough to exercise map-hashing
        glyphs = tuple(tuple(1 if c == self.x else 0 for c in range(self.width))
                       for _ in range(1))
        return {"blstats": bl, "glyphs": glyphs}
