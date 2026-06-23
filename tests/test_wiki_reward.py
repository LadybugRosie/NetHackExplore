"""Tests for the wiki-grounded intrinsic reward (goexplore/wiki_reward.py).

Skips if NetHack's data.base isn't available (needs the built fork / PUFFERLIB_DIR).
Run: source scripts/env.sh && python tests/test_wiki_reward.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goexplore.wiki_reward import _DEFAULT_DATA_BASE, WikiReward, load_corpus


def _available():
    return os.path.exists(_DEFAULT_DATA_BASE)


def test_corpus_loads():
    corpus = load_corpus()
    assert len(corpus) > 300, f"expected a few hundred concepts, got {len(corpus)}"
    names = {c["name"] for c in corpus}
    for expect in ("newt", "altar", "dagger"):
        assert expect in names, f"{expect!r} missing from corpus"
    print(f"ok: corpus loaded ({len(corpus)} concepts)")


def test_matching_quality():
    wr = WikiReward(threshold=0.20)
    # Clear entity messages should match the right concept above threshold.
    for msg, concept in [("You kill the newt!", "newt"),
                         ("You see here an orcish dagger.", "dagger"),
                         ("The grid bug bites!", "grid bug")]:
        sim, name = wr.score(msg)
        assert sim >= wr.threshold and name == concept, \
            f"{msg!r} -> ({name}, {sim:.2f}), expected {concept}"
    # Non-entity flavor text should NOT cross the threshold.
    for msg in ("You feel hungry.", "This door is locked."):
        sim, _ = wr.score(msg)
        assert sim < wr.threshold, f"{msg!r} false-matched at {sim:.2f}"
    print("ok: matching quality (entities match, flavor text rejected)")


def test_novelty_first_encounter_only():
    wr = WikiReward(threshold=0.20, bonus=1.0)
    wr.reset()
    r1, n1 = wr.reward("You kill the newt!")
    r2, n2 = wr.reward("The newt bites!")          # same concept again
    assert r1 == 1.0 and n1 == "newt", (r1, n1)
    assert r2 == 0.0 and n2 == "newt", "repeat encounter must give no bonus"
    wr.reset()
    r3, _ = wr.reward("You kill the newt!")
    assert r3 == 1.0, "after reset the concept is novel again"
    print("ok: per-episode first-encounter novelty")


if __name__ == "__main__":
    if not _available():
        print(f"SKIP: {_DEFAULT_DATA_BASE} not found (build the fork / set PUFFERLIB_DIR)")
        sys.exit(0)
    test_corpus_loads()
    test_matching_quality()
    test_novelty_first_encounter_only()
    print("\nALL WIKI REWARD TESTS PASSED")
