"""Unit tests for the wiki-reward PPO injection (goexplore/wiki_ppo.py) that run
WITHOUT a GPU: obs message decoding, the monkeypatch applying, and the per-lane
novelty/reset logic. Full end-to-end PPO training is validated separately on GPU.

Run: source scripts/env.sh && python tests/test_wiki_ppo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_decode_messages():
    import numpy as np
    import torch
    from goexplore.wiki_ppo import MESSAGE_LEN, _decode_messages

    B, obs_size = 4, 1659 + MESSAGE_LEN
    o = torch.zeros(B, obs_size, dtype=torch.uint8)
    msgs = ["You kill the newt!", "The grid bug bites!", "", "It's a wall."]
    off = obs_size - MESSAGE_LEN
    for i, m in enumerate(msgs):
        b = m.encode("latin1")[:MESSAGE_LEN]
        o[i, off:off + len(b)] = torch.tensor(list(b), dtype=torch.uint8)
    decoded = _decode_messages(o, off)
    assert decoded == msgs, decoded
    print("ok: message slice decodes correctly")


def test_monkeypatch_applies():
    import pufferlib._C as _C
    if _C.precision_bytes != 4:
        print("SKIP monkeypatch test: need float build (precision_bytes==4)")
        return
    from goexplore import wiki_ppo
    from pufferlib import torch_pufferl as T
    wiki_ppo.enable(coef=0.5, threshold=0.2)
    assert getattr(T.PuffeRL, "_wiki_patched", False), "rollouts not patched"
    wiki_ppo.enable()  # idempotent
    print("ok: PuffeRL.rollouts monkeypatch applied (idempotent)")


def test_reward_injection_logic():
    """The exact per-lane bonus + episode-reset logic used in the rollout."""
    from goexplore.wiki_reward import WikiReward
    wr = WikiReward(threshold=0.20, bonus=1.0)
    disc = [set(), set()]
    msgs = ["You kill the newt!", "It's a wall."]
    b1 = wr.reward_batch(msgs, disc)
    assert b1[0] == 1.0 and b1[1] == 0.0, b1            # lane0 novel newt, lane1 no match
    b2 = wr.reward_batch(["The newt bites!", "You see here a dagger."], disc)
    assert b2[0] == 0.0, "lane0 newt already discovered this episode"
    assert b2[1] == 1.0, "lane1 dagger newly discovered"
    disc[0].clear()                                     # simulate episode end on lane0
    b3 = wr.reward_batch(["You kill the newt!", "The dagger hits!"], disc)
    assert b3[0] == 1.0, "newt novel again after lane0 reset"
    assert b3[1] == 0.0, "dagger still discovered on lane1"
    print("ok: per-lane novelty + episode-reset logic")


if __name__ == "__main__":
    test_decode_messages()
    test_reward_injection_logic()
    test_monkeypatch_applies()
    print("\nALL WIKI-PPO UNIT TESTS PASSED")
