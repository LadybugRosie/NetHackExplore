#!/bin/bash
# Reproducible build of the real native NetHack stack for Go-Explore on this box.
# Builds: PufferLib 4.0 + the liujonathan24/NetHack fork -> libnethack.so,
# PufferLib's _C.so (float32 GPU build), and libge_nethack.so (the deterministic
# Go-Explore shim).
#
# Idempotent-ish; safe to re-run. Assumes Debian 12 + NVIDIA driver + passwordless
# sudo (for apt + CUDA toolkit). Adjust PUFFERLIB_DIR / CUDA version as needed.
set -e

PUFFERLIB_DIR="${PUFFERLIB_DIR:-/home/davidhovey/PufferLib}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_VER="${CUDA_VER:-13-0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

echo "### 1. System build deps"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    cmake clang ccache unzip libbz2-dev libomp-dev python3.11-venv pkg-config flex bison wget

echo "### 2. CUDA toolkit $CUDA_VER (for nvcc; matches torch cu13x)"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-$CUDA_VER"
fi

echo "### 3. Clone PufferLib 4.0 + NetHack fork"
if [ ! -d "$PUFFERLIB_DIR" ]; then
    git clone --depth 1 -b 4.0 https://github.com/PufferAI/PufferLib.git "$PUFFERLIB_DIR"
fi
if [ ! -d "$PUFFERLIB_DIR/vendor/nle/src" ]; then
    git clone --depth 1 https://github.com/liujonathan24/NetHack.git "$PUFFERLIB_DIR/vendor/nle"
fi

echo "### 4. Python venv + PufferLib (torch cu13x, numpy, wandb, ...)"
if [ ! -d "$PUFFERLIB_DIR/.venv" ]; then
    python3 -m venv "$PUFFERLIB_DIR/.venv"
fi
# shellcheck disable=SC1091
source "$PUFFERLIB_DIR/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$PUFFERLIB_DIR"

echo "### 5. Build libnethack.so (the C engine)"
if [ ! -f "$PUFFERLIB_DIR/vendor/nle/src/build/libnethack.so" ]; then
    (cd "$PUFFERLIB_DIR/vendor/nle/src" && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release)
fi
make -C "$PUFFERLIB_DIR/vendor/nle/src/build" nethack -j"$(nproc)"

echo "### 6. Link symlinks the pip CUDA wheels omit (unversioned .so) + nvidia-ml stub"
SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
ln -sf libcudnn.so.9 "$SP/nvidia/cudnn/lib/libcudnn.so" 2>/dev/null || true
NCCL="$(ls "$SP"/nvidia/nccl/lib/libnccl.so.* 2>/dev/null | head -1)"
[ -n "$NCCL" ] && ln -sf "$(basename "$NCCL")" "$SP/nvidia/nccl/lib/libnccl.so" || true
sudo ln -sf "$CUDA_HOME/lib64/stubs/libnvidia-ml.so" "$CUDA_HOME/lib64/libnvidia-ml.so" 2>/dev/null || true

echo "### 7. Build PufferLib _C.so (float32 GPU build: native + torch backends)"
( cd "$PUFFERLIB_DIR" && CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" NVCC_ARCH="${NVCC_ARCH:-sm_90}" \
    bash build.sh nethack --float )

echo "### 8. Build libge_nethack.so (Go-Explore deterministic shim)"
PUFFERLIB_DIR="$PUFFERLIB_DIR" bash "$REPO_DIR/goexplore/native/build_ge.sh"

echo
echo "DONE. Now:  source scripts/env.sh  &&  python tests/test_native.py"
