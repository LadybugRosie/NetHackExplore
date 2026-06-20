"""Cell descriptors and score extraction for Go-Explore on NLE.

A "cell" is a coarse, discrete fingerprint of a game state. Two states that map
to the same cell are treated as interchangeable by Go-Explore. The descriptor is
the single most important design choice:

  * too fine  -> the archive explodes and you never revisit anything
  * too coarse -> distinct situations collapse and progress stalls

The default fingerprint is the classic NetHack-friendly choice: dungeon
position (which branch + depth + coarse map location). Score and depth are
deliberately *not* part of the key -- they are the *value* used to rank rival
trajectories that reach the same cell.
"""

# TODO: Perhaps replace a cell if its blstats are better

from __future__ import annotations

from dataclasses import dataclass

from . import nle_constants as C


@dataclass(frozen=True)
class CellConfig:
    cell_w: int = 6  # x-bucket width (NLE map is 79 wide)
    cell_h: int = 3  # y-bucket height (NLE map is 21 tall)
    hp_buckets: int = 0  # if >0, include a coarse HP band in the key
    use_map_hash: bool = False  # include a hash of the explored glyph map


def _bl(obs, idx):
    """Index blstats whether it is a numpy array, list, or tuple."""
    return int(obs["blstats"][idx])


def game_score(obs) -> int:
    return _bl(obs, C.BL_SCORE)


def game_depth(obs) -> int:
    return _bl(obs, C.BL_DEPTH)


def is_alive(obs) -> bool:
    return _bl(obs, C.BL_HP) > 0


def _map_hash(obs) -> int:
    glyphs = obs.get("glyphs")
    if glyphs is None:
        return 0
    tobytes = getattr(glyphs, "tobytes", None)
    if tobytes is not None:  # numpy array
        return hash(tobytes())
    return hash(tuple(map(tuple, glyphs)))  # nested python sequence


def cell_descriptor(obs, cfg: CellConfig):
    """Return a hashable tuple identifying the cell for ``obs``."""
    dnum = _bl(obs, C.BL_DNUM)
    dlevel = _bl(obs, C.BL_DLEVEL)
    x = _bl(obs, C.BL_X)
    y = _bl(obs, C.BL_Y)
    key = [dnum, dlevel, x // cfg.cell_w, y // cfg.cell_h]
    if cfg.hp_buckets > 0:
        hp, hpmax = _bl(obs, C.BL_HP), max(1, _bl(obs, C.BL_HPMAX))
        key.append(min(cfg.hp_buckets - 1, (hp * cfg.hp_buckets) // (hpmax + 1)))
    if cfg.use_map_hash:
        key.append(_map_hash(obs))
    return tuple(key)
