"""A NetHack policy network shared by phase-2 BC and PPO.

The whole point of sharing one architecture is that the weights learned by
behavioral cloning (phase 2a) are loadable by PufferLib's PPO trainer for
fine-tuning (phase 2b). The encoder reads NLE's two core observation tensors --
the ``glyphs`` map (21x79 ints) through a small conv stack, and the ``blstats``
vector through an MLP -- and produces both action logits and a value estimate,
which is exactly what an actor-critic PPO needs.

torch is imported lazily; nothing here is needed for phase 1.
"""

from __future__ import annotations

MAX_GLYPH = 5991  # NLE has ~5976 glyph ids; round up for the embedding table


def build_policy(n_actions: int, embed_dim: int = 32, hidden: int = 256):
    import torch
    import torch.nn as nn

    class NetHackPolicy(nn.Module):
        """Actor-critic over (glyphs, blstats). Returns (logits, value)."""

        def __init__(self):
            super().__init__()
            self.glyph_embed = nn.Embedding(MAX_GLYPH, embed_dim)
            self.conv = nn.Sequential(
                nn.Conv2d(embed_dim, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(),
            )
            self.blstats_mlp = nn.Sequential(
                nn.LazyLinear(128), nn.ReLU(),
            )
            self.trunk = nn.Sequential(
                nn.LazyLinear(hidden), nn.ReLU(),
            )
            self.actor = nn.Linear(hidden, n_actions)
            self.critic = nn.Linear(hidden, 1)

        def encode(self, obs):
            glyphs = obs["glyphs"].long()                  # (B, H, W)
            g = self.glyph_embed(glyphs).permute(0, 3, 1, 2)  # (B, E, H, W)
            g = self.conv(g)
            b = self.blstats_mlp(obs["blstats"].float())
            return self.trunk(torch.cat([g, b], dim=-1))

        def forward(self, obs):
            h = self.encode(obs)
            return self.actor(h), self.critic(h).squeeze(-1)

    return NetHackPolicy()


def make_pufferlib_policy(env, **kwargs):
    """Wrap ``build_policy`` for PufferLib's PPO trainer.

    PufferLib expects a policy module whose forward returns (logits, value) for a
    batch of observations. ``build_policy`` already matches that contract; if your
    PufferLib version wants the ``pufferlib.models``/``LSTMWrapper`` interface or
    a flattened-obs ``Policy(env)`` signature, adapt the thin shim here rather
    than the network above.
    """
    n_actions = int(env.single_action_space.n)
    return build_policy(n_actions, **kwargs)
