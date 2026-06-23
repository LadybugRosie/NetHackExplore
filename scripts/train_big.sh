#!/bin/bash
# Large multi-GPU native PPO run on NetHack. Launch this ONLY when the GPUs are
# free (check `nvidia-smi`). Uses the fast native CUDA trainer (create_pufferl)
# with the config/nethack.ini tuned hyperparameters + the built-in
# depth/scout/score reward shaping. Logs to wandb.
#
#   scripts/train_big.sh                 # 8 GPUs, ~2B steps (defaults below)
#   GPUS=4 TIMESTEPS=1000000000 scripts/train_big.sh
#
# NOTE: not yet tested at --train.gpus 8 on this box (encoder DDP job owned the
# GPUs at build time). If multi-GPU NCCL errors, drop to GPUS=1 to confirm, then
# scale up.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

GPUS="${GPUS:-8}"
TIMESTEPS="${TIMESTEPS:-2000000000}"
AGENTS="${AGENTS:-4096}"          # per-process (per GPU)
THREADS="${THREADS:-16}"
PROJECT="${PROJECT:-nethack-goexplore}"

echo "Launching native PPO: gpus=$GPUS agents/gpu=$AGENTS threads=$THREADS steps=$TIMESTEPS"
exec puffer train nethack \
    --wandb --wandb-project "$PROJECT" --wandb-group "native-ppo-${GPUS}gpu" \
    --vec.total-agents "$AGENTS" --vec.num-threads "$THREADS" --vec.num-buffers 2 \
    --train.gpus "$GPUS" --train.total-timesteps "$TIMESTEPS" \
    "$@"
