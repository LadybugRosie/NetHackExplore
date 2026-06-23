"""ctypes binding to ``libge_nethack.so`` — the real, deterministic NetHack
engine for Go-Explore phase 1.

``libge_nethack.so`` wraps PufferLib's Ocean NetHack env (the liujonathan24/NetHack
fork) with a FIXED-seed reset so replay-based "return" is exact. See
``goexplore/native/ge_nethack.c`` and ``build_ge.sh``.

Two adapters are exposed, mirroring the mock so the rest of Go-Explore is
unchanged:

* ``NativeNLE``  — single env with the gym-ish API ``DeterministicEnv`` expects
  (``action_space.n``, ``seed``, ``reset``, ``step``).
* ``NativeVecEnv`` — N envs stepped in one process (each ``ge_step`` re-anchors
  ``current_nle_ctx``, so a single-process loop is safe). Provides
  ``reset``/``step``/``reset_lanes`` like ``MockVecEnv``.

Observations are dicts: ``{"blstats": [27 ints], "glyphs": <list|None>}``. blstats
is snapshotted (copied) every step because the C buffer is reused in place.
"""

from __future__ import annotations

import ctypes
import os

# NLE geometry / obs sizes (must match vendor/nle/include/nleobs.h + nethack.h).
NH_ROWS, NH_COLS = 21, 79
NH_GRID = NH_ROWS * NH_COLS  # 1659
BLSTATS_LEN = 27
MESSAGE_LEN = 256

_DEFAULT_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "native", "libge_nethack.so")

_lib = None


def _load(lib_path: str | None = None):
    """Load (once) and configure the ctypes signatures."""
    global _lib
    if _lib is not None:
        return _lib
    path = lib_path or os.environ.get("GE_NETHACK_LIB", _DEFAULT_LIB)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"libge_nethack.so not found at {path}. Build it with "
            f"goexplore/native/build_ge.sh (and ensure libnethack.so is built)."
        )
    if "NETHACKDIR" not in os.environ:
        raise RuntimeError(
            "NETHACKDIR is not set. Point it at the dir containing nhdat, e.g. "
            "$PUFFERLIB_DIR/vendor/nle/src/build/dat (see scripts/env.sh)."
        )
    lib = ctypes.CDLL(path)
    lib.ge_make.restype = ctypes.c_void_p
    lib.ge_make.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    lib.ge_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
    lib.ge_reset.argtypes = [ctypes.c_void_p]
    lib.ge_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ge_step.restype = ctypes.c_int
    lib.ge_num_actions.restype = ctypes.c_int
    lib.ge_blstats_len.restype = ctypes.c_int
    lib.ge_blstats.argtypes = [ctypes.c_void_p]
    lib.ge_blstats.restype = ctypes.POINTER(ctypes.c_long)
    lib.ge_reward.argtypes = [ctypes.c_void_p]
    lib.ge_reward.restype = ctypes.c_float
    lib.ge_done.argtypes = [ctypes.c_void_p]
    lib.ge_done.restype = ctypes.c_int
    lib.ge_chars.argtypes = [ctypes.c_void_p]
    lib.ge_chars.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.ge_glyphs.argtypes = [ctypes.c_void_p]
    lib.ge_glyphs.restype = ctypes.POINTER(ctypes.c_short)
    lib.ge_message.argtypes = [ctypes.c_void_p]
    lib.ge_message.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.ge_free.argtypes = [ctypes.c_void_p]
    _lib = lib
    return lib


class _Discrete:
    def __init__(self, n):
        self.n = n


class NativeNLE:
    """A single deterministic NetHack env over libge_nethack.so.

    API matches what ``DeterministicEnv`` / the mock need: ``action_space.n``,
    ``seed(core, disp, reseed)``, ``reset() -> obs``, ``step(a) -> (obs, r, done, info)``.
    """

    def __init__(self, core: int = 0x111, disp: int = 0x222,
                 include_glyphs: bool = False, include_chars: bool = False,
                 include_message: bool = False, lib_path: str | None = None):
        self._lib = _load(lib_path)
        self._core = int(core)
        self._disp = int(disp)
        self.include_glyphs = include_glyphs
        # chars is the 1659-byte glyph-char grid; identical to PufferLib's default
        # (chars-only) training observation, so BC demos collected here transfer
        # directly to a `puffer train` policy.
        self.include_chars = include_chars
        # message is the top status line as a decoded str (for wiki/Motif rewards).
        self.include_message = include_message
        self.action_space = _Discrete(self._lib.ge_num_actions())
        self._h = self._lib.ge_make(ctypes.c_ulong(self._core), ctypes.c_ulong(self._disp))
        if not self._h:
            raise RuntimeError("ge_make returned NULL")

    # NLE-style seeding: (core, disp, reseed). reseed is ignored (always exact).
    def seed(self, core=0, disp=0, reseed=False):
        self._core, self._disp = int(core), int(disp)
        self._lib.ge_set_seed(self._h, ctypes.c_ulong(self._core), ctypes.c_ulong(self._disp))
        return [core, disp]

    def _obs(self):
        bl_ptr = self._lib.ge_blstats(self._h)
        blstats = bl_ptr[:BLSTATS_LEN]  # ctypes slice -> fresh python list (copy)
        obs = {"blstats": blstats}
        if self.include_glyphs:
            g_ptr = self._lib.ge_glyphs(self._h)
            obs["glyphs"] = bytes(ctypes.cast(
                g_ptr, ctypes.POINTER(ctypes.c_ubyte))[:NH_GRID * 2])
        else:
            obs["glyphs"] = None
        if self.include_chars:
            ch_ptr = self._lib.ge_chars(self._h)
            obs["chars"] = bytes(ch_ptr[:NH_GRID])  # 1659 bytes, matches puffer obs
        if self.include_message:
            m_ptr = self._lib.ge_message(self._h)
            raw = bytes(m_ptr[:MESSAGE_LEN])
            obs["message"] = raw.split(b"\x00", 1)[0].decode("latin1")
        return obs

    def reset(self):
        self._lib.ge_reset(self._h)
        return self._obs()

    def step(self, action):
        done = bool(self._lib.ge_step(self._h, int(action)))
        reward = float(self._lib.ge_reward(self._h))
        return self._obs(), reward, done, {}

    def close(self):
        if getattr(self, "_h", None):
            self._lib.ge_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NativeVecEnv:
    """N NativeNLE games stepped synchronously in one process.

    Mirrors ``MockVecEnv``: ``reset(seed)`` / ``step(actions)`` /
    ``reset_lanes(indices, seed)``. Single seed shared by all lanes (single-seed
    Go-Explore). Each ``ge_step`` re-anchors current_nle_ctx, so the in-process
    loop is correct. For more throughput run several NativeVecEnvs across
    processes (the engine is per-env; 208 cores here).
    """

    def __init__(self, num_envs: int, disp: int = 0x222,
                 include_glyphs: bool = False, lib_path: str | None = None):
        _load(lib_path)
        self.num_envs = num_envs
        self.include_glyphs = include_glyphs
        # core seeds set on reset(); start them all identical to a placeholder.
        self.envs = [NativeNLE(core=0x111, disp=disp, include_glyphs=include_glyphs,
                               lib_path=lib_path) for _ in range(num_envs)]
        self.n_actions = self.envs[0].action_space.n
        self._disp = disp

    def reset(self, seed: int):
        out = []
        for e in self.envs:
            e.seed(seed, self._disp, False)
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
            self.envs[i].seed(seed, self._disp, False)
            fresh[i] = self.envs[i].reset()
        return fresh

    def close(self):
        for e in self.envs:
            e.close()
