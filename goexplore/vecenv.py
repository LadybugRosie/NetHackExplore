"""Vectorized environment layer for Go-Explore.

Go-Explore needs three operations from a batch of envs:

  * ``reset(seed)``                -> reset ALL lanes to the fixed game seed
  * ``step(actions)``              -> step all lanes with one batched action array
  * ``reset_lanes(indices, seed)`` -> reset SOME lanes back to the seed (used when
                                      a lane retires and we re-home it to replay a
                                      new cell from the root)

``reset_lanes`` is the one operation a stock RL vecenv does NOT expose -- normal
vecenvs autoreset on episode end with whatever seeding policy they were built
with. Go-Explore's replay-based return *requires* re-homing a chosen lane to the
exact game seed on demand, so we make it a first-class method here.

``MockVecEnv`` is a fully-working synchronous implementation used to test the
async-lane driver offline. It loops in Python -- it makes NO throughput claim;
its only job is to validate the algorithm. ``PufferVecEnv`` is the real adapter:
it delegates ``step`` to PufferLib's native batched C step (where the throughput
actually lives) and documents the per-lane re-seed integration point.
"""

from __future__ import annotations


class MockVecEnv:
    """N independent MockNLE games stepped synchronously (test scaffolding)."""

    def __init__(self, num_envs: int, width: int = 5):
        from .mock_nle import MockNLE
        self.num_envs = num_envs
        self.envs = [MockNLE(width=width) for _ in range(num_envs)]
        self.n_actions = self.envs[0].action_space.n

    def reset(self, seed: int):
        out = []
        for e in self.envs:
            e.seed(seed)
            out.append(e.reset())
        return out

    def step(self, actions):
        obs, rew, done, info = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, i = e.step(a)
            obs.append(o); rew.append(r); done.append(d); info.append(i)
        return obs, rew, done, info

    def reset_lanes(self, indices, seed: int):
        fresh = {}
        for i in indices:
            self.envs[i].seed(seed)
            fresh[i] = self.envs[i].reset()
        return fresh

    def close(self):
        pass


def _resolve_backend(pv, name: str):
    """Pick a backend class that actually exists in this PufferLib version.

    Backend names have drifted across releases (and ``Native`` doesn't exist
    everywhere), so we try a fallback chain per friendly name and NEVER return
    None -- passing None to ``vector.make`` is what raised 'NoneType is not
    callable'.
    """
    chains = {
        "native": ["Native", "PufferEnv", "Multiprocessing", "Serial"],
        "multiprocessing": ["Multiprocessing", "Serial"],
        "serial": ["Serial"],
    }.get(name, ["Serial"])
    for attr in chains:
        cls = getattr(pv, attr, None)
        if cls is not None:
            return cls
    available = [a for a in dir(pv) if a[:1].isupper()]
    raise RuntimeError(
        f"No usable PufferLib vector backend for {name!r}. "
        f"Available in pufferlib.vector: {available}"
    )


class PufferVecEnv:
    """Adapter onto PufferLib's native vectorized NLE env.

    The batched ``step`` is where this fork's C throughput is realized: one call
    advances all lanes in native code. The exact construction + obs format are
    version-specific, so the version-sensitive bits are isolated here.
    """

    def __init__(self, num_envs: int, env_id: str = "NetHackChallenge-v0",
                 backend: str = "serial", num_workers: int | None = None, **kwargs):
        import pufferlib
        import pufferlib.vector

        # Importing nle is what registers the NetHack* gym ids (NetHackChallenge-v0,
        # NetHackScore-v0, ...). PufferLib's environment.py calls gym.make(name)
        # but doesn't import nle itself, so without this the id "doesn't exist".
        try:
            import nle  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "nle is not importable -- install the NetHack env (the fork's "
                "build) so the NetHack* gym ids get registered."
            ) from exc

        try:
            from pufferlib.environments.nethack import env_creator
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not import PufferLib's NetHack env_creator; check your "
                "pufferlib version's environment API."
            ) from exc

        backend_cls = _resolve_backend(pufferlib.vector, backend)

        # pufferlib.environments.nethack.env_creator is a FACTORY: calling it
        # returns the actual env-constructing callable (a functools.partial).
        # vector.make wants THAT callable, not the factory -- passing the factory
        # leaves pufferlib's driver_env a partial (-> 'no attribute num_agents').
        try:
            creator = env_creator(name=env_id)
        except TypeError:
            creator = env_creator()

        make_kwargs = dict(num_envs=num_envs, backend=backend_cls)
        # Multiprocessing needs workers and requires num_envs % num_workers == 0.
        if backend_cls is getattr(pufferlib.vector, "Multiprocessing", object()):
            import os
            num_workers = num_workers or min(num_envs, os.cpu_count() or 1)
            while num_workers > 1 and num_envs % num_workers:
                num_workers -= 1
            make_kwargs["num_workers"] = num_workers

        self.vecenv = pufferlib.vector.make(creator, **make_kwargs)
        self.num_envs = num_envs
        self.n_actions = int(self.vecenv.single_action_space.n)

    # PufferLib returns stacked/structured arrays; split them into per-lane dicts
    # so the rest of Go-Explore (cells.py) can index obs["blstats"] uniformly.
    def _split(self, batched_obs):
        out = []
        for i in range(self.num_envs):
            out.append({k: v[i] for k, v in batched_obs.items()}
                       if isinstance(batched_obs, dict)
                       else {"blstats": batched_obs[i]})
        return out

    def reset(self, seed: int):
        self._seed_all(seed)
        obs, _info = self.vecenv.reset(seed=seed)
        return self._split(obs)

    def step(self, actions):
        obs, rew, term, trunc, info = self.vecenv.step(actions)  # batched native step
        done = [bool(a or b) for a, b in zip(term, trunc)]
        return self._split(obs), list(rew), done, info

    def reset_lanes(self, indices, seed: int):
        # INTEGRATION POINT: PufferLib autoresets a lane on done; to make
        # replay-based return exact we must re-home the lane to *this* seed.
        # Depending on version this is done via per-env seeding before the
        # autoreset, or by an async send/recv to the worker. The robust,
        # version-independent alternative is C-level snapshot/restore
        # (nle_clone/nle_restore) -- see README "next steps". For now we
        # re-seed and pull a fresh observation for the requested lanes.
        self._seed_lanes(indices, seed)
        obs, _info = self.vecenv.reset(seed=seed)  # TODO: per-lane reset, not full
        split = self._split(obs)
        return {i: split[i] for i in indices}

    def _seed_all(self, seed):
        self._seed_lanes(range(self.num_envs), seed)

    def _seed_lanes(self, indices, seed):
        # Best-effort: the underlying NLE env exposes seed(core, disp, reseed).
        envs = getattr(self.vecenv, "envs", None)
        if envs is None:
            return
        for i in indices:
            target = getattr(envs[i], "unwrapped", envs[i])
            seed_fn = getattr(target, "seed", None)
            if seed_fn is not None:
                try:
                    seed_fn(seed, seed, False)
                except TypeError:
                    pass

    def close(self):
        self.vecenv.close()


class GymnasiumVecEnv:
    """N real NLE gymnasium envs stepped in a Python loop.

    Bypasses PufferLib's nethack wrapper (which uses legacy gym) so this works
    with modern gymnasium-based nle (1.x). Observations stay as dicts -- no
    PufferLib flattening -- so ``cell_descriptor`` reads ``blstats``/``glyphs``
    directly.

    TRADEOFF: this loops in Python, so it does NOT get the fork's C-native batched
    throughput. Use it to run Go-Explore on *real* NetHack and validate the
    algorithm; use ``PufferVecEnv`` (with a version-aligned install) for the
    high-throughput native path.

    DETERMINISM: replay-based return needs the game to be a pure function of
    (seed, actions). We seed every reset with the same game seed; on nle 1.x
    verify this reproduces (the engine must be in reseed=False mode), e.g. with
    the same exact-replay check used in tests/test_vectorized.py.
    """

    def __init__(self, num_envs: int, env_id: str = "NetHackScore-v0", **kwargs):
        import gymnasium
        import nle  # noqa: F401 - registers NetHack* ids into gymnasium

        self.envs = [gymnasium.make(env_id, **kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.n_actions = int(self.envs[0].action_space.n)

    def reset(self, seed: int):
        out = []
        for e in self.envs:
            obs, _info = e.reset(seed=seed)
            out.append(obs)
        return out

    def step(self, actions):
        obs, rew, done, info = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, i = e.step(int(a))
            obs.append(o); rew.append(r); done.append(bool(term or trunc)); info.append(i)
        return obs, rew, done, info

    def reset_lanes(self, indices, seed: int):
        fresh = {}
        for i in indices:
            obs, _info = self.envs[i].reset(seed=seed)
            fresh[i] = obs
        return fresh

    def close(self):
        for e in self.envs:
            e.close()


def make_vecenv(kind: str = "mock", num_envs: int = 64, env_id: str | None = None, **kwargs):
    if kind == "mock":
        return MockVecEnv(num_envs, **kwargs)
    if kind == "nle":
        # Direct gymnasium NLE (works with pip-installed nle 1.x); Python-loop.
        return GymnasiumVecEnv(num_envs, env_id=env_id or "NetHackScore-v0", **kwargs)
    if kind == "puffer":
        # Native PufferLib path; needs a version-aligned (fork) nle + matching gym.
        return PufferVecEnv(num_envs, env_id=env_id or "NetHackChallenge-v0", **kwargs)
    raise ValueError(f"unknown vecenv kind: {kind!r}")
