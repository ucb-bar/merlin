#!/usr/bin/env bash
# Cross-build the board-LOCAL relax-VM runner (bypasses tvm_rpc) for the SpacemiT K1 (rv64gcv).
# Links against the cross-built libtvm_runtime.so (build/baselines/tvm-rv64/). Output binary is
# scp'd to the board alongside the model .so; it loads the module + runs the relax VM locally over
# plain ssh — no tracker, no RPC session (which flakes with kShutdown on large .so uploads).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
TC="${MERLIN_SPACEMIT_TC:-/path/to/merlin-iree/build_tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin}"
TVM="$REPO/third_party/baselines/tvm"
OUT="$REPO/build/baselines/tvm-rv64/board_runner"
mkdir -p "$OUT"
"$TC/clang++" -march=rv64gcv -mabi=lp64d -O2 -std=c++17 \
  -I "$TVM/include" -I "$TVM/3rdparty/dlpack/include" -I "$TVM/3rdparty/dmlc-core/include" \
  "$(dirname "$0")/board_runner.cc" \
  -L "$REPO/build/baselines/tvm-rv64" -ltvm_runtime -Wl,-rpath,'$ORIGIN' \
  -o "$OUT/board_runner"
echo "built $OUT/board_runner"
