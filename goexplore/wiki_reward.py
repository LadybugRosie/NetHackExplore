"""Wiki-grounded intrinsic reward for NetHack (a Motif-flavored, knowledge-
distilled exploration signal).

Idea (the user's): reward the agent for reaching states that the game's own
knowledge base / wiki *describes* — i.e., for encountering new monsters, items,
and features that NetHack considers worth documenting. This is a dense,
semantically meaningful exploration reward, in contrast to the sparse score.

How it works:
  * Corpus = NetHack's encyclopedia ``dat/data.base`` (793 concept entries:
    monsters, items, dungeon features, with prose) — the same source the wiki is
    built from, and license-clean + local. (A full nethackwiki dump can be
    dropped in unchanged; see ``load_corpus``.)
  * Each step, the game ``message`` (e.g. "You kill the newt!", "You see here an
    orcish dagger.") is matched against the corpus (TF-IDF cosine by default; a
    sentence-embedding backend is optional).
  * The FIRST time (per episode) the agent matches a given concept above a
    threshold, it earns a bonus — "semantic novelty" / coverage of the knowledge
    base. Repeat encounters give nothing, so it can't farm one message.

This is the count-based analog in *concept space* of the env's per-tile ``scout``
bonus. The principled upgrade (true Motif) replaces the TF-IDF matcher with a
reward model trained from LLM/wiki *preferences* over message pairs — same plug
point (`WikiReward.score`).
"""

from __future__ import annotations

import os
import re

_DEFAULT_DATA_BASE = os.path.join(
    os.environ.get("PUFFERLIB_DIR", "/home/davidhovey/PufferLib"),
    "vendor", "nle", "src", "dat", "data.base",
)

# Progression-focused subset: concepts that mark *descent / milestones* rather
# than every monster/item. Matched as substrings against an entry's keys. The
# aim is a reward that pulls the agent DOWN the dungeon and toward goal states,
# instead of rewarding breadth on the current level (every newt). Game messages
# that name these ("There is a staircase down here.", "You see an altar.", "You
# enter the Gnomish Mines.", "You begin praying...") match these entries.
PROGRESSION_KEYWORDS = (
    # descent & dungeon features
    "stair", "ladder", "trap door", "trapdoor", "hole", "portal", "dungeon",
    "fountain", "throne", "sink", "grave", "altar", "temple",
    # branches / special levels
    "mine", "sokoban", "oracle", "big room", "quest", "ludios", "castle",
    "valley", "gehennom", "elemental", "astral", "plane", "moloch",
    # milestone items
    "amulet of yendor", "bell of opening", "candelabrum", "book of the dead",
    "luckstone", "invocation",
    # gate monsters
    "medusa", "vlad", "wizard of yendor", "high priest", "priest",
    "death", "pestilence", "famine", "nemesis",
    # progress actions
    "pray", "sacrifice",
)


def _is_progression(keys) -> bool:
    for k in keys:
        kl = k.lower()
        for kw in PROGRESSION_KEYWORDS:
            if kw in kl:
                return True
    return False


def load_corpus(data_base_path: str | None = None, subset: str = "all"):
    """Parse NetHack's ``data.base`` into a list of concept entries.

    Format: comment lines start with '#'; one or more key lines (not indented)
    name a concept (aliases, '*' wildcards, '~' negations); TAB-indented lines
    that follow are the description, shared by those keys.
    """
    path = data_base_path or _DEFAULT_DATA_BASE
    entries = []
    keys: list[str] = []
    body: list[str] = []

    def flush():
        if keys and body:
            # canonical name = first non-wildcard key, cleaned
            name = None
            for k in keys:
                if not k.startswith("~"):
                    name = k.replace("*", "").strip()
                    if name:
                        break
            if name:
                clean_keys = [k.replace("*", "").strip() for k in keys
                              if not k.startswith("~") and k.strip()]
                if subset == "progression" and not _is_progression(clean_keys):
                    return
                entries.append({
                    "name": name,
                    "keys": clean_keys,
                    "text": " ".join(body).strip(),
                })

    with open(path, encoding="latin1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            if line[0] in (" ", "\t"):           # description line
                body.append(line.strip())
            else:                                 # key line
                if body:                          # previous entry complete
                    flush()
                    keys, body = [], []
                keys.append(line.strip())
    flush()
    return entries


class WikiReward:
    """Per-episode semantic-novelty reward from game messages.

    backend='tfidf' (default, sklearn, CPU, no downloads) or 'st' (a
    sentence-transformers model name in `model`, optional dependency).
    """

    def __init__(self, corpus=None, *, threshold: float = 0.20, bonus: float = 1.0,
                 backend: str = "tfidf", analyzer: str = "word",
                 model: str = "BAAI/bge-small-en-v1.5",
                 data_base_path: str | None = None, subset: str = "all"):
        self.corpus = corpus if corpus is not None else load_corpus(data_base_path, subset=subset)
        self.threshold = threshold
        self.bonus = bonus
        self.backend = backend
        docs = [f"{c['name']} {' '.join(c['keys'])} {c['text']}" for c in self.corpus]
        if backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            if analyzer == "char":
                # char n-grams are robust to NetHack's morphology (staircase~stair,
                # praying~pray, Mines~mine) — important for the small progression
                # corpus where exact word tokens often miss.
                self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            else:
                self._vec = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=1)
            self._mat = self._vec.fit_transform(docs)  # rows l2-normalized
        elif backend == "st":
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer(model, device="cpu")
            self._emb = self._model.encode(docs, normalize_embeddings=True,
                                           convert_to_numpy=True)
            self._np = np
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.reset()

    def reset(self):
        """Clear per-episode discovered concepts (call on env reset)."""
        self.discovered: set[str] = set()

    # Raw similarity of a message to its nearest concept (the Motif plug point).
    def score(self, message: str):
        msg = (message or "").strip()
        # Strip pet/quantity noise that dilutes the match a little.
        msg = re.sub(r"\s+", " ", msg)
        if not msg:
            return 0.0, None
        if self.backend == "tfidf":
            q = self._vec.transform([msg])
            sims = (self._mat @ q.T).toarray().ravel()
        else:
            q = self._model.encode([msg], normalize_embeddings=True, convert_to_numpy=True)
            sims = (self._emb @ q[0])
        j = int(sims.argmax())
        return float(sims[j]), self.corpus[j]["name"]

    def reward(self, message: str):
        """Return (reward, matched_concept_or_None). Bonus only on the first
        above-threshold match of each concept this episode."""
        sim, name = self.score(message)
        if name is None or sim < self.threshold:
            return 0.0, None
        if name in self.discovered:
            return 0.0, name
        self.discovered.add(name)
        return self.bonus, name

    # ---- vectorized API for training (one set of discovered concepts per lane) ----
    def score_batch(self, messages):
        """Nearest-concept index + similarity for a list of messages (vectorized)."""
        import numpy as np
        msgs = [re.sub(r"\s+", " ", (m or "").strip()) for m in messages]
        if self.backend == "tfidf":
            Q = self._vec.transform(msgs)             # (B, V) l2-normalized rows
            sims = (Q @ self._mat.T).toarray()        # (B, N) cosine
        else:
            Q = self._model.encode(msgs, normalize_embeddings=True, convert_to_numpy=True)
            sims = Q @ self._emb.T
        idx = sims.argmax(axis=1)
        return idx, sims[np.arange(len(msgs)), idx]

    def reward_batch(self, messages, discovered):
        """Per-lane first-encounter bonus. ``discovered`` is a list of per-lane
        sets (mutated in place); reset a lane's set to {} on episode end.
        Returns a list of float rewards."""
        idx, sims = self.score_batch(messages)
        out = []
        for b in range(len(messages)):
            s = float(sims[b])
            if s < self.threshold:
                out.append(0.0); continue
            name = self.corpus[int(idx[b])]["name"]
            dset = discovered[b]
            if name in dset:
                out.append(0.0)
            else:
                dset.add(name); out.append(self.bonus)
        return out
