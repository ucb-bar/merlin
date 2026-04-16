#!/usr/bin/env bash
# Run the post-fix vit_small on FireSim with NO host timeout.
#
# Prerequisite: the patched host iree-compile (Option E f32-reduction
# per-function `-v` attribute) has been built, and vit_small has been
# recompiled with it. The recompile happens automatically when you run
#
#   CHIPYARD_ROOT=/scratch2/agustin/chipyard conda run -n merlin-dev \
#     uv run tools/merlin.py build --profile firesim --config release \
#     --cmake-target bench_model_opu_bench_vit_small_opu
#
# after the host rebuild finishes.
#
# Usage:
#   bash build_tools/firesim/run_vit_small_fixed.sh             # OPU, no timeout
#   bash build_tools/firesim/run_vit_small_fixed.sh rvv         # RVV variant
#
# Output:
#   /tmp/vit_small_fixed.uartlog

set -uo pipefail

VARIANT="${1:-opu}"
case "$VARIANT" in
opu | rvv) ;;
*)
	echo "Unknown variant: $VARIANT (use opu or rvv)"
	exit 1
	;;
esac

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
BIN="bench_model_opu_bench_vit_small_${VARIANT}"
SRC="$BIN_DIR/$BIN"
WL="merlin-bench-$BIN"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

if [ ! -f "$SRC" ]; then
	echo "Missing binary: $SRC"
	echo "Rebuild vit_small after the host iree-compile patch:"
	echo "  CHIPYARD_ROOT=$CHIPYARD_ROOT conda run -n merlin-dev uv run tools/merlin.py build \\"
	echo "    --profile firesim --config release --cmake-target $BIN"
	exit 1
fi
if ! command -v firesim >/dev/null 2>&1; then
	echo "firesim not on PATH. Check build_tools/firesim/env.sh."
	exit 1
fi

_cleanup_ran=0
_cleanup() {
	[ "$_cleanup_ran" = "1" ] && return
	_cleanup_ran=1
	echo
	echo "[cleanup] killing FireSim and detached screen session..."
	screen -S fsim0 -X quit 2>/dev/null || true
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
	exit 130
}
trap _cleanup INT TERM HUP

echo "=== config ==="
echo "  variant  : $VARIANT"
echo "  binary   : $SRC"
echo "  workload : $WL"
echo "  timeout  : NONE (run to natural completion; Ctrl-C to abort)"

echo
echo "--- kill stale firesim ---"
(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
if screen -ls 2>&1 | grep -q 'fsim0'; then
	screen -S fsim0 -X quit 2>/dev/null || true
fi

echo
echo "--- stage binary + workload ---"
mkdir -p "$DEP/workloads/$WL"
cp "$SRC" "$DEP/workloads/$WL/$BIN"
if [ ! -f "$DEP/workloads/$WL.json" ]; then
	cat >"$DEP/workloads/$WL.json" <<EOF
{
  "benchmark_name": "$WL",
  "common_bootbinary": "$BIN",
  "common_rootfs": null,
  "common_outputs": [],
  "common_simulation_outputs": ["uartlog"]
}
EOF
fi
sed -i -E "s#^(\s*workload_name:\s*).*#\1${WL}.json#" "$DEP/config_runtime.yaml"
sed -i -E "s#^(\s*enable:\s*)(yes|no)\b#\1no#" "$DEP/config_runtime.yaml"
ls -la "$DEP/workloads/$WL/$BIN"

echo
echo "--- firesim infrasetup ---"
(cd "$DEP" && firesim infrasetup) || {
	echo "infrasetup FAILED"
	exit 2
}

echo
echo "--- firesim runworkload (NO timeout) ---"
echo "To watch live, in another shell: tail -f /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog"
echo
(cd "$DEP" && firesim runworkload) || true
(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

echo
echo "--- collecting uartlog ---"
RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
UARTLOG=""
[ -n "$RESULT" ] && UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	LIVE="${FIRESIM_RUNS_DIR:-/scratch2/agustin/FIRESIM_RUNS_DIR}/sim_slot_0/uartlog"
	[ -f "$LIVE" ] && UARTLOG="$LIVE"
fi
[ -n "$UARTLOG" ] && cp "$UARTLOG" /tmp/vit_small_fixed.uartlog

echo
echo "===================== VERDICT ====================="
if grep -qE 'Bench iter [0-9]+/10' "$UARTLOG" 2>/dev/null; then
	echo "PASS — vit_small ran bench iterations to completion."
	echo
	echo "Last 10 markers:"
	grep -E '^Bench iter|^Warmup iter|Benchmark complete|DONE' "$UARTLOG" | tail -10
elif grep -qE 'Warmup iter [0-9]+/[0-9]+ done' "$UARTLOG" 2>/dev/null; then
	echo "WARMUP only — bench did not finish (probably Ctrl-C'd)."
	grep -E '^Warmup iter' "$UARTLOG" | tail -5
else
	echo "DID NOT COMPLETE WARMUP — last dispatch marker:"
	grep -E '^\[dn\]|^\[dc\]' "$UARTLOG" 2>/dev/null | tail -5
	echo
	echo "If you see a [dn] symbol with no matching [dc], that's a hang."
fi
echo
echo "Full log: /tmp/vit_small_fixed.uartlog"
