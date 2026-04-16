#!/usr/bin/env bash
# Run the bare-metal OPU instruction-cost microbench on FireSim and
# extract the cyc/op numbers for VOPACC, OPMVINBCAST, VMV_VR.
#
# Usage:
#   bash build_tools/firesim/run_opu_microbench.sh
#
# Output: prints the CSV line from the uartlog and copies the full
# uartlog to /tmp/opu_microbench.uartlog.

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
BIN="bench_mt_opu_microbench"
WL="merlin-bench-${BIN}"
OUT="/tmp/opu_microbench.uartlog"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

if [ ! -f "$BIN_DIR/$BIN" ]; then
	echo "missing $BIN_DIR/$BIN — run:"
	echo "  env CHIPYARD_ROOT=$CHIPYARD_ROOT conda run -n merlin-dev uv run \\"
	echo "    tools/merlin.py build --profile firesim --cmake-target $BIN"
	exit 1
fi

(cd "$DEP" && firesim kill) 2>&1 | tail -2 || true
screen -S fsim0 -X quit 2>/dev/null || true

mkdir -p "$DEP/workloads/$WL"
cp "$BIN_DIR/$BIN" "$DEP/workloads/$WL/"
cat >"$DEP/workloads/$WL.json" <<EOF
{
  "benchmark_name": "$WL",
  "common_bootbinary": "$BIN",
  "common_rootfs": null,
  "common_outputs": [],
  "common_simulation_outputs": ["uartlog"]
}
EOF

CFG="$DEP/config_runtime.yaml"
sed -i -E "s#^(\s*workload_name:\s*).*#\1${WL}.json#" "$CFG"
sed -i -E "s#^(\s*enable:\s*)(yes|no)\b#\1no#" "$CFG"

echo "=== infrasetup ==="
(cd "$DEP" && firesim infrasetup) 2>&1 | tail -5
echo
echo "=== runworkload (cap 300s — this should take <10s) ==="
(cd "$DEP" && timeout --foreground 300 firesim runworkload) 2>&1 | tail -10
(cd "$DEP" && firesim kill) 2>&1 | tail -2 || true

RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
UARTLOG=""
[ -n "$RESULT" ] && UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	LIVE="${FIRESIM_RUNS_DIR:-/scratch2/agustin/FIRESIM_RUNS_DIR}/sim_slot_0/uartlog"
	[ -f "$LIVE" ] && UARTLOG="$LIVE"
fi
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	echo "NO UARTLOG"
	exit 1
fi
cp "$UARTLOG" "$OUT"
echo
echo "=== microbench results ==="
grep -E '^(VOPACC|OPMVINBCAST|VMV_VR|CSV, opu_microbench)' "$UARTLOG"
echo
echo "Full uartlog: $OUT"
