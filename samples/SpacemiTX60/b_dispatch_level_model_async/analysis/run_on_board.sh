#!/usr/bin/env bash
# run_on_board.sh — Build, deploy, run on SpacemiT X60, and plot results.
#
# Usage (from repo root):
#   bash samples/SpacemiTX60/b_dispatch_level_model_async/analysis/run_on_board.sh
#
# Prerequisites:
#   - conda activate merlin-dev
#   - Board accessible at BOARD_HOST (default: root@10.44.86.251)
#   - VMFBs + schedule JSON already on the board at BOARD_DIR
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

BOARD_HOST="${BOARD_HOST:-root@10.44.86.251}"
BOARD_DIR="${BOARD_DIR:-/home/baseline}"
BOARD_OUT="${BOARD_DIR}/run/out"

CMAKE_TARGET="merlin_benchmark_dispatch_level_model_async"
BINARY_NAME="benchmark-dispatch-level-model-async-run"
SCHEDULE_JSON="scheduled_networks_periodic_profile_profiled.json"

# ── 1. Build ─────────────────────────────────────────────────────────────────
echo "=== Building ${CMAKE_TARGET} ==="
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit \
  --cmake-target "${CMAKE_TARGET}"

BINARY=$(find build/spacemit-merlin-release -name "${BINARY_NAME}" -type f | head -1)
if [[ -z "$BINARY" ]]; then
  echo "ERROR: Binary not found in build tree" >&2
  exit 1
fi
echo "Built: ${BINARY}"

# ── 2. Deploy ────────────────────────────────────────────────────────────────
echo "=== Deploying to ${BOARD_HOST} ==="
scp "${BINARY}" "${BOARD_HOST}:${BOARD_DIR}/"

# ── 3. Run on board ──────────────────────────────────────────────────────────
echo "=== Running on board ==="
# shellcheck disable=SC2029
ssh "${BOARD_HOST}" "mkdir -p ${BOARD_OUT} && \
  ${BOARD_DIR}/${BINARY_NAME} \
    ${BOARD_DIR}/${SCHEDULE_JSON} \
    local-task 1 1 1 \
    --vmfb_dir=${BOARD_DIR}/dispatches \
    --cpu_p_cpu_ids=0,1,2,3 \
    --cpu_e_cpu_ids=4,5 \
    --visible_cores=8 \
    --trace_csv=${BOARD_OUT}/run_trace.csv \
    --out_json=${BOARD_OUT}/run_summary.json \
    --out_dot=${BOARD_OUT}/run_graph.dot"

# ── 4. Copy results back ────────────────────────────────────────────────────
LOCAL_OUT="samples/SpacemiTX60/b_dispatch_level_model_async/analysis/results"
mkdir -p "${LOCAL_OUT}"
echo "=== Copying results to ${LOCAL_OUT} ==="
scp "${BOARD_HOST}:${BOARD_OUT}/run_trace.csv"    "${LOCAL_OUT}/"
scp "${BOARD_HOST}:${BOARD_OUT}/run_summary.json"  "${LOCAL_OUT}/"
scp "${BOARD_HOST}:${BOARD_OUT}/run_graph.dot"     "${LOCAL_OUT}/"

# ── 5. Plot ──────────────────────────────────────────────────────────────────
PLOT_DIR="${LOCAL_OUT}/plots"
echo "=== Generating plots in ${PLOT_DIR} ==="
conda run -n merlin-dev uv run \
  samples/SpacemiTX60/b_dispatch_level_model_async/analysis/plot_dispatch_trace.py \
    --trace-csv "${LOCAL_OUT}/run_trace.csv" \
    --out-dir "${PLOT_DIR}"

echo "=== Done ==="
echo "  Trace:  ${LOCAL_OUT}/run_trace.csv"
echo "  Plot:   ${PLOT_DIR}/cluster_schedule_by_target_planned_vs_observed.png"
