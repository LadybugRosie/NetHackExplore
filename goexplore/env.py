"""Environment adapters.

``DeterministicEnv`` is the contract Go-Explore relies on: a fixed-seed env that
can be returned to *any* previously visited state by replaying the stored action
sequence from reset(). This is the "vanilla" (replay-based) form of Go-Explore's
"return" step and is correct *because* NLE is a pure deterministic function of
(core_seed, disp_seed, actions) when built with reseed=False -- see the engine's
rnd.c / hacklib.c (init_isaac64, reseed_random guarded by has_strong_rngseed).

The wrapper normalizes gym (4-tuple) vs gymnasium (5-tuple) step/reset APIs so
the same code drives the real NLE env, the PufferLib NLE env, or the mock.
"""

from __future__ import annotations


class ReplayError(RuntimeError):
    """Raised when a stored trajectory no longer reproduces (env not deterministic)."""


class DeterministicEnv:
    def __init__(self, env, seed_core: int = 0, seed_disp: int = 0):
        self.env = env
        self.core = seed_core
        self.disp = seed_disp
        self.n_actions = int(env.action_space.n)
        self._steps = 0

    # -- API normalization -------------------------------------------------
    def _apply_seed(self):
        target = getattr(self.env, "unwrapped", self.env)
        seed_fn = getattr(target, "seed", None)
        if seed_fn is not None:
            try:
                # NLE: seed(core, disp, reseed=False) -> fully reproducible game
                seed_fn(self.core, self.disp, False)
            except TypeError:
                try:
                    seed_fn(self.core)
                except TypeError:
                    pass

    def _reset_raw(self):
        out = self.env.reset()
        return out[0] if isinstance(out, tuple) else out

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, term, trunc, info = out
            done = bool(term or trunc)
        else:
            obs, reward, done, info = out
        self._steps += 1
        return obs, float(reward), bool(done), info

    # -- Go-Explore primitives --------------------------------------------
    def reset(self):
        """Reset to the fixed seed (the archive root)."""
        self._apply_seed()
        self._steps = 0
        return self._reset_raw()

    def reset_to(self, traj):
        """Return to a cell by replaying ``traj`` from the seeded reset.

        Returns (obs, done). ``done==True`` means the trajectory terminated
        before the end -- the caller should discard exploration from here.
        """
        obs = self.reset()
        done = False
        for action in traj:
            obs, _, done, _ = self.step(action)
            if done:
                break
        return obs, done

    @property
    def steps_taken(self) -> int:
        return self._steps


def make_env(kind: str = "mock", env_id: str = "NetHackScore-v0", **kwargs):
    """Construct a raw environment by name. Returns an unwrapped env object
    exposing reset()/step()/action_space.n (gym or gymnasium style)."""
    if kind == "mock":
        from .mock_nle import MockNLE

        return MockNLE(**kwargs)

    if kind == "native":
        # Real deterministic NetHack via libge_nethack.so (the fast fork).
        from .native_env import NativeNLE

        return NativeNLE(**kwargs)

    if kind == "nle":
        import nle  # noqa: F401  (registers gym ids)

        try:
            import gymnasium as gym
        except ImportError:
            import gym
        return gym.make(env_id)

    if kind == "puffer":
        # PufferLib wraps the very NLE engine this repo builds. The exact
        # creator path has moved across versions; try the common ones.
        try:
            from pufferlib.environments.nethack import (
                env_creator,
            )  # TODO: Check if this is the right place to import from

            return env_creator()()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not construct the PufferLib NetHack env. Check your "
                "pufferlib version's environment API."
            ) from exc

    raise ValueError(f"unknown env kind: {kind!r}")
