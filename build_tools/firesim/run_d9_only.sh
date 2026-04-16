#!/usr/bin/env bash
# Focused runner — just mt_vit_d9_matmul_opu, no timeout, to confirm
# end-to-end completion. Our partial results show d9 progressing through
# wg 10+ of d1's i8 reduction at ~35 k cyc/wg; this run lets it finish
# all 16 workgroups × 2 warmup + 10 bench iterations uninterrupted.
#
# Expected duration: ~5-15 min on FireSim. If it takes longer than 30 min,
# something is genuinely stuck and you can Ctrl-C with no harm.
#
# Usage:
#   bash build_tools/firesim/run_d9_only.sh              # OPU variant (default)
#   bash build_tools/firesim/run_d9_only.sh rvv          # RVV variant
#
# Output:
#   /tmp/d9_only.uartlog

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
BIN="bench_model_mt_vit_d9_matmul_${VARIANT}"
SRC="$BIN_DIR/$BIN"
WL="merlin-bench-$BIN"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

if [ ! -f "$SRC" ]; then
	echo "Missing binary: $SRC"
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
	echo "[cleanup] firesim kill..."
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
}
trap _cleanup INT TERM

(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

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

(cd "$DEP" && firesim infrasetup)
# No timeout — let it finish naturally.
(cd "$DEP" && firesim runworkload) || true
(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
UARTLOG=""
[ -n "$RESULT" ] && UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	LIVE="${FIRESIM_RUNS_DIR:-/scratch2/agustin/FIRESIM_RUNS_DIR}/sim_slot_0/uartlog"
	[ -f "$LIVE" ] && UARTLOG="$LIVE"
fi
[ -n "$UARTLOG" ] && cp "$UARTLOG" /tmp/d9_only.uartlog

echo
echo "===================== VERDICT ====================="
if grep -qE 'Bench iter [0-9]+/10' "$UARTLOG" 2>/dev/null; then
	echo "PASS — bench iterations completed"
	grep -E '^Bench iter|^Warmup iter' "$UARTLOG" | tail -5
elif grep -qE 'Warmup iter.*done' "$UARTLOG" 2>/dev/null; then
	echo "WARMUP only — bench did not finish"
	grep -E '^Warmup iter' "$UARTLOG" | tail -3
else
	echo "DID NOT COMPLETE WARMUP — checking last dispatch marker..."
	grep -E '^\[dn\]|^\[dc\]' "$UARTLOG" 2>/dev/null | tail -5
fi
echo
echo "Full log: /tmp/d9_only.uartlog"
