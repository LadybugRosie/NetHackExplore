#!/bin/bash
# Train NetHack PPO with the wiki-grounded intrinsic reward (Motif-flavored).
# Ensures _C.so carries the message obs (NETHACK_USE_MESSAGE=1), then trains via
# the torch backend with the wiki bonus injected (goexplore/wiki_ppo.py).
#
#   scripts/train_wiki.sh --vec.total-agents 2048 --train.gpus 1 \
#       --train.total-timesteps 200000000 --wandb --wandb-project nethack-goexplore
#
# Tunables: GE_WIKI_COEF (default 0.5), GE_WIKI_THRESHOLD (0.20).
# NOTE: launch only when GPUs are free (torch backend runs the policy on GPU).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

# Ensure the installed _C.so has the message obs (rebuild if not 1915 wide).
NEED_MSG=$(python - <<'PY'
import pufferlib._C as C
from pufferlib.pufferl import load_config
import sys
a = load_config("nethack"); a["vec"]["total_agents"]=2; a["vec"]["num_threads"]=1; a["vec"]["num_buffers"]=1
try:
    v = C.create_vec(a, C.gpu); print(0 if v.obs_size == 1915 else 1)
except Exception:
    print(1)
PY
)
if [ "$NEED_MSG" != "0" ]; then
    echo "Rebuilding _C.so with NETHACK_USE_MESSAGE=1 ..."
    ( cd "$PUFFERLIB_DIR" && CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" \
        NVCC_ARCH="${NVCC_ARCH:-sm_90}" EXTRA_CFLAGS="-DNETHACK_USE_MESSAGE=1" \
        bash build.sh nethack --float )
fi

export GE_WIKI_COEF="${GE_WIKI_COEF:-0.5}"
export GE_WIKI_THRESHOLD="${GE_WIKI_THRESHOLD:-0.20}"
echo "wiki reward: coef=$GE_WIKI_COEF threshold=$GE_WIKI_THRESHOLD"
exec python -m goexplore.train_wiki "$@"
