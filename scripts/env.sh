#!/bin/bash
# Source this to set up the environment for building/running the native NetHack
# stack (PufferLib 4.0 Ocean + the liujonathan24/NetHack fork) on this box.
#
#   source scripts/env.sh
#
# Idempotent; safe to source multiple times.

export PUFFERLIB_DIR="${PUFFERLIB_DIR:-/home/davidhovey/PufferLib}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

# NETHACKDIR must be an ABSOLUTE path to the dir containing nhdat (game data).
export NETHACKDIR="${NETHACKDIR:-$PUFFERLIB_DIR/vendor/nle/src/build/dat}"

# Activate the PufferLib venv (torch cu130 + pufferlib + wandb live here).
if [ -z "$VIRTUAL_ENV" ] && [ -f "$PUFFERLIB_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PUFFERLIB_DIR/.venv/bin/activate"
fi

# Runtime library path: the cu13 wheels (matching torch) win, then cudnn/nccl
# wheels, then the system CUDA toolkit (cudart/cublas/etc) as fallback.
_SP="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
if [ -n "$_SP" ]; then
    export LD_LIBRARY_PATH="$_SP/nvidia/cu13/lib:$_SP/nvidia/cudnn/lib:$_SP/nvidia/nccl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi
export PATH="$CUDA_HOME/bin:$PATH"
