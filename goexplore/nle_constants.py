"""NLE blstats indices.

NLE packs the agent's scalar stats into a flat ``blstats`` vector. We try to
import the authoritative index enum from ``nle`` if it is installed; otherwise
we fall back to the constants that match NLE 0.9.x (the version this PufferLib
fork is based on). Keep these in sync with src/include/nle.h in the engine.
"""

try:  # pragma: no cover - only when the real package is present
    from nle.nethack import (
        NLE_BL_X as BL_X,
        NLE_BL_Y as BL_Y,
        NLE_BL_SCORE as BL_SCORE,
        NLE_BL_HP as BL_HP,
        NLE_BL_HPMAX as BL_HPMAX,
        NLE_BL_DEPTH as BL_DEPTH,
        NLE_BL_GOLD as BL_GOLD,
        NLE_BL_XP as BL_XP,
        NLE_BL_TIME as BL_TIME,
        NLE_BL_HUNGER as BL_HUNGER,
        NLE_BL_DNUM as BL_DNUM,
        NLE_BL_DLEVEL as BL_DLEVEL,
    )
except Exception:  # noqa: BLE001 - fall back to known-good indices
    BL_X = 0
    BL_Y = 1
    BL_SCORE = 9
    BL_HP = 10
    BL_HPMAX = 11
    BL_DEPTH = 12
    BL_GOLD = 13
    BL_XP = 18
    BL_TIME = 20
    BL_HUNGER = 21
    BL_DNUM = 23
    BL_DLEVEL = 24
