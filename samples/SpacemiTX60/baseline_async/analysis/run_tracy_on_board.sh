#!/usr/bin/env bash
# run_tracy_on_board.sh — Build with Tracy, compile models, deploy, run.
#
# The binary runs with TRACY_NO_EXIT=1 so it waits for the Tracy profiler
# GUI to connect. Open tracy-profiler on your host and connect to
# localhost:8086 (forwarded via SSH).
#
# Usage (from repo root):
#   bash samples/SpacemiTX60/b_baseline_dual_model_async/analysis/run_tracy_on_board.sh
#
# Prerequisites:
#   - conda activate merlin-dev
#   - Board accessible at BOARD_HOST
#   - Tracy profiler GUI installed on host machine
#   - Host compiler available (build/host-merlin-release or host-vanilla-release)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

BOARD_HOST="${BOARD_HOST:-root@10.44.86.251}"
BOARD_DIR="${BOARD_DIR:-/home/baseline}"
CMAKE_TARGET="merlin_benchmark_baseline_dual_model_async"
BINARY_NAME="benchmark-baseline-dual-model-async-run"

GRAPH_ITERS="${GRAPH_ITERS:-3}"
PARALLELISM="${PARALLELISM:-6}"

# ── 1. Compile models for SpacemiT X60 with Tracy debug info ─────────────────
echo "=== Compiling dronet.q.int8 (RVV, with Tracy + benchmarks) ==="
conda run -n merlin-dev uv run tools/compile.py \
  models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 \
  --hw RVV \
  --quantized \
  --tracy \
  --build-benchmarks

echo "=== Compiling mlp.q.int8 (RVV, with Tracy + benchmarks) ==="
conda run -n merlin-dev uv run tools/compile.py \
  models/mlp/mlp.q.int8.mlir \
  --target spacemit_x60 \
  --hw RVV \
  --quantized \
  --tracy \
  --build-benchmarks

# ── 2. Build runtime binary with Tracy ────────────────────────────────────────
echo "=== Building ${CMAKE_TARGET} with Tracy ==="
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit \
  --config release \
  --enable-tracy \
  --cmake-target "${CMAKE_TARGET}"

BINARY=$(find build/spacemit-merlin-release -name "${BINARY_NAME}" -type f | head -1)
if [[ -z "$BINARY" ]]; then
  echo "ERROR: Binary not found in build tree" >&2
  exit 1
fi
echo "Built: ${BINARY}"

# ── 3. Deploy to board ───────────────────────────────────────────────────────
echo "=== Deploying to ${BOARD_HOST} ==="
scp "${BINARY}" "${BOARD_HOST}:${BOARD_DIR}/"

# Also deploy the compiled VMFBs (dronet + MLP full-model)
DRONET_VMFB=$(find build/compiled_models/dronet -name "dronet.q.int8.vmfb" -path "*RVV*" | head -1)
MLP_VMFB=$(find build/compiled_models/mlp -name "mlp.q.int8.vmfb" -path "*RVV*" | head -1)
if [[ -n "$DRONET_VMFB" ]]; then
  echo "  dronet VMFB: ${DRONET_VMFB}"
  scp "${DRONET_VMFB}" "${BOARD_HOST}:${BOARD_DIR}/dronet.q.int8.vmfb"
fi
if [[ -n "$MLP_VMFB" ]]; then
  echo "  MLP VMFB: ${MLP_VMFB}"
  scp "${MLP_VMFB}" "${BOARD_HOST}:${BOARD_DIR}/mlp.q.int8.vmfb"
fi

# ── 4. Run on board with Tracy ───────────────────────────────────────────────
echo ""
echo "=== Running on board ==="
echo "    The binary will wait for Tracy profiler to connect."
echo ""
echo "    On your host machine:"
echo "      1. Open a separate terminal and run:"
echo "           ssh -L 8086:localhost:8086 ${BOARD_HOST}"
echo "      2. Open tracy-profiler and connect to localhost:8086"
echo "      3. Then press Enter here to start the run."
echo ""
read -rp "Press Enter when Tracy profiler is ready..."

# shellcheck disable=SC2029
ssh "${BOARD_HOST}" \
  "TRACY_NO_EXIT=1 ${BOARD_DIR}/${BINARY_NAME} \
    ${BOARD_DIR}/dronet.q.int8_dispatch_graph.json \
    local-task ${GRAPH_ITERS} 1 1 \
    --parallelism=${PARALLELISM}"

echo ""
echo "=== Done ==="
echo "  Save the trace from Tracy profiler GUI."
