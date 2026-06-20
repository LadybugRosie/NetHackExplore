"""The Go-Explore archive: the set of discovered cells and how we pick one.

Each cell stores the best trajectory found to reach it (an action sequence from
the fixed seed), plus the score/depth of the state it led to and bookkeeping
counts used for selection weighting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class CellEntry:
    traj: tuple = ()  # action sequence from reset() that reaches this cell
    score: int = 0  # game score at this cell (value, for ranking)
    depth: int = 1  # dungeon depth at this cell
    n_chosen: int = 0  # times selected as a restore point
    n_seen: int = 0  # times this cell was encountered
    last_improved: int = 0  # iteration at which traj/score last improved


# TODO: This is a rather naive score, might want to change later ... will this just get us stuck
def _better(new_score, new_depth, new_len, old: CellEntry) -> bool:
    """Prefer deeper, then higher score, then a shorter path (cheaper return)."""
    if new_depth != old.depth:
        return new_depth > old.depth
    if new_score != old.score:
        return new_score > old.score
    return new_len < len(old.traj)


# TODO: This is not nearly fine-grained enough
@dataclass
class Archive:
    rng: random.Random = field(default_factory=random.Random)
    cells: dict = field(default_factory=dict)
    frontier_bonus: float = 3.0  # extra weight for cells at the deepest depth
    max_depth: int = 1
    n_updates: int = 0

    def consider(self, cell_key, traj, score, depth, iteration):
        """Insert ``cell_key`` if new, or replace its trajectory if better.

        Returns True if the archive changed (new cell or improved trajectory).
        """
        self.n_updates += 1
        entry = self.cells.get(cell_key)
        if entry is None:
            self.cells[cell_key] = CellEntry(
                traj=tuple(traj),
                score=score,
                depth=depth,
                n_seen=1,
                last_improved=iteration,
            )
            self.max_depth = max(self.max_depth, depth)
            return True
        entry.n_seen += 1
        if _better(score, depth, len(traj), entry):
            entry.traj = tuple(traj)
            entry.score = score
            entry.depth = depth
            entry.last_improved = iteration
            self.max_depth = max(self.max_depth, depth)
            return True
        return False

    def _weight(self, entry: CellEntry) -> float:
        # Count-based novelty: prefer cells we've rarely explored from, and
        # bias hard toward the current frontier (deepest known depth).
        w = 1.0 / math.sqrt(entry.n_chosen + 1)
        if entry.depth >= self.max_depth:
            w *= self.frontier_bonus
        return w

    def sample(self):
        """Pick a cell to return to, weighted by novelty + frontier bonus."""
        keys = list(self.cells.keys())
        weights = [self._weight(self.cells[k]) for k in keys]
        key = self.rng.choices(keys, weights=weights, k=1)[0]
        entry = self.cells[key]
        entry.n_chosen += 1
        return key, entry

    def best(self) -> CellEntry:
        return max(self.cells.values(), key=lambda e: (e.depth, e.score, -len(e.traj)))

    def __len__(self):
        return len(self.cells)
