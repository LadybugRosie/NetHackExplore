#!/bin/bash
# Build libge_nethack.so — the deterministic Go-Explore wrapper around
# PufferLib's Ocean NetHack env. Links against the SAME libnethack.so that
# PufferLib builds (vendor/nle/src/build/libnethack.so).
#
# Prereqs: libnethack.so must already be built (see scripts/build_native.sh or
# the PufferLib ocean/nethack README). Run from anywhere.
set -e

PUFFERLIB_DIR="${PUFFERLIB_DIR:-/home/davidhovey/PufferLib}"
NLE_BUILD="$PUFFERLIB_DIR/vendor/nle/src/build"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/libge_nethack.so"

if [ ! -f "$NLE_BUILD/libnethack.so" ]; then
    echo "ERROR: $NLE_BUILD/libnethack.so not found. Build it first:"
    echo "  (cd $PUFFERLIB_DIR/vendor/nle/src && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release)"
    echo "  make -C $PUFFERLIB_DIR/vendor/nle/src/build nethack -j\$(nproc)"
    exit 1
fi

CC="${CC:-clang}"
echo "Compiling ge_nethack.c -> $OUT"
"$CC" -O2 -fPIC -shared -DPLATFORM_DESKTOP \
    -Wno-incompatible-pointer-types-discards-qualifiers \
    -I"$PUFFERLIB_DIR/ocean/nethack" \
    -I"$PUFFERLIB_DIR/vendor/nle/include" \
    -I"$PUFFERLIB_DIR/src" \
    -I"$PUFFERLIB_DIR/vendor" \
    "$HERE/ge_nethack.c" \
    -L"$NLE_BUILD" -lnethack -Wl,-rpath,"$NLE_BUILD" \
    -ldl -lm -lpthread \
    -o "$OUT"
echo "Built: $OUT"
