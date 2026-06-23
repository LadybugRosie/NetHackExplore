"""Launch `puffer train nethack` with the wiki-grounded intrinsic reward added.

Applies the wiki-reward monkeypatch (goexplore/wiki_ppo.py) and then delegates to
PufferLib's training CLI, so all wandb/dashboard/checkpoint behavior is identical
to `puffer train` — only the rollout reward is augmented. Forces the torch
backend (--slowly), which is where a Python/CPU reward can be injected.

    python -m goexplore.train_wiki --vec.total-agents 2048 --train.gpus 1 \
        --train.total-timesteps 200000000 --wandb --wandb-project nethack-goexplore

Tunables (env vars): GE_WIKI_COEF, GE_WIKI_THRESHOLD, GE_WIKI_DATABASE.
Requires _C.so built with NETHACK_USE_MESSAGE=1 (scripts/train_wiki.sh handles it).
"""

from __future__ import annotations

import sys


def main():
    from . import wiki_ppo
    wiki_ppo.enable()

    import pufferlib._C as _C
    if _C.precision_bytes != 4:
        raise SystemExit("wiki-ppo needs the float32 build (bash build.sh nethack --float).")

    import pufferlib.pufferl as P

    extra = sys.argv[1:]
    if "--slowly" not in extra:
        extra.append("--slowly")
    sys.argv = ["puffer", "train", "nethack", *extra]
    P.main()


if __name__ == "__main__":
    main()
