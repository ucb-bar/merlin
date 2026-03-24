#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# Ensure git trusts the bind-mounted workspace.
git config --global --add safe.directory /workspace

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

# ---------- 1. Host compiler package (Linux x86_64) ----------
echo ""
echo "=========================================="
echo "  Building host compiler package"
echo "=========================================="
uv run tools/merlin.py build \
  --profile package-host \
  --clean \
  --no-use-ccache
mv dist/host-merlin-perf.tar.gz dist/merlin-host-linux-x86_64.tar.gz

# ---------- 2. SpacemiT runtime package ----------
echo ""
echo "=========================================="
echo "  Building SpacemiT runtime package"
echo "=========================================="
python3 tools/setup.py toolchain --toolchain-target spacemit
uv run tools/merlin.py build \
  --profile package-spacemit \
  --clean \
  --no-use-ccache
mv dist/spacemit-merlin-perf.tar.gz dist/merlin-runtime-spacemit.tar.gz

# ---------- 3. FireSim / Saturn OPU runtime package ----------
echo ""
echo "=========================================="
echo "  Building FireSim runtime package"
echo "=========================================="
bash /workspace/build_tools/firesim/setup_toolchain.sh

RISCV_TOOLCHAIN_ROOT="/workspace/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV"
test -d "${RISCV_TOOLCHAIN_ROOT}" || {
  echo "FireSim toolchain root not found: ${RISCV_TOOLCHAIN_ROOT}"
  exit 1
}

env \
  RISCV_TOOLCHAIN_ROOT="${RISCV_TOOLCHAIN_ROOT}" \
  RISCV="${RISCV_TOOLCHAIN_ROOT}" \
  uv run tools/merlin.py build \
    --profile package-firesim \
    --clean \
    --no-use-ccache
mv dist/firesim-merlin-perf.tar.gz dist/merlin-runtime-saturnopu.tar.gz

# ---------- Summary ----------
echo ""
echo "=========================================="
echo "  Release artifacts"
echo "=========================================="
ls -lh dist/*.tar.gz
