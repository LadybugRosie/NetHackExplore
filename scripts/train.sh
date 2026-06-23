#!/bin/bash
# Convenience wrapper: native PPO training on NetHack via `puffer train`.
# Sources the build env, then forwards all args to `puffer train nethack`.
#
#   scripts/train.sh                                  # defaults from config/nethack.ini
#   scripts/train.sh --wandb --wandb-project nethack-goexplore
#   scripts/train.sh --vec.total-agents 4096 --train.gpus 1 --train.total-timesteps 1000000000
#   scripts/train.sh --slowly --load-model-path bc_init.pt   # PPO from a BC warm-start
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
exec puffer train nethack "$@"
