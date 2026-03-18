#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export UV_PROJECT_ENVIRONMENT=/tmp/merlin-uv-env
rm -rf "${UV_PROJECT_ENVIRONMENT}"

# Start from clean release build trees so CMake caches do not clash between
# host paths (/scratch2/...) and container paths (/workspace/...).
rm -rf \
  build/host-vanilla-perf \
  build/host-merlin-perf \
  build/spacemit-merlin-perf \
  build/firesim-merlin-perf

python3 tools/setup.py submodules --submodules-profile core --submodule-sync
uv sync

# Build the real Linux host package first.
# This leaves host tools in build/host-merlin-perf/install/bin for reuse.
uv run tools/merlin.py build \
  --profile package-host \
  --clean \
  --no-use-ccache
mv dist/host-merlin-perf.tar.gz dist/merlin-host-linux-x86_64.tar.gz

# SpacemiT runtime package.
python3 tools/setup.py toolchain --toolchain-target spacemit
uv run tools/merlin.py build \
  --profile package-spacemit \
  --clean \
  --no-use-ccache
mv dist/spacemit-merlin-perf.tar.gz dist/merlin-runtime-spacemit.tar.gz

# FireSim / Saturn OPU runtime package.
bash /workspace/build_tools/firesim/setup_toolchain.sh

RISCV_TOOLCHAIN_ROOT="/workspace/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV"
RISCV="${RISCV_TOOLCHAIN_ROOT}"

echo "Using FireSim toolchain: ${RISCV_TOOLCHAIN_ROOT}"
test -d "${RISCV_TOOLCHAIN_ROOT}" || {
  echo "FireSim toolchain root not found"
  exit 1
}

env \
  RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}" \
  RISCV="${RISCV}" \
  python3 - <<'PY'
import os
print("PYTHON sees RISCV_TOOLCHAIN_ROOT =", os.environ.get("RISCV_TOOLCHAIN_ROOT"))
print("PYTHON sees RISCV =", os.environ.get("RISCV"))
PY

env \
  RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}" \
  RISCV="${RISCV}" \
  uv run tools/merlin.py build \
    --profile package-firesim \
    --clean \
    --no-use-ccache
