#!/usr/bin/env bash
# Run the post-fix tinyllama on FireSim with NO host timeout.
#
# Important: tinyllama uses FLAGS_MODEL_OPU_LLM which adds the preprocessing
# pass `--iree-preprocessing-collapse-multi-n-contractions`. The
# `mt_matmul_64x128_opu_llm` microtest showed a separate "early" failure
# (no [dn] ever printed) under that flag — meaning tinyllama may have a
# SECOND bug stacked on top of the f32-reduction hang. Option E fixes
# the f32-reduction hang; the collapse-multi-n issue is a separate
# investigation. If this run still hangs, report the last [dn] symbol.
#
# The VMFB is ~1 GB embedded; expect a ~5-15 min boot + 5-30 min inference.
#
# Usage:
#   bash build_tools/firesim/run_tinyllama_fixed.sh
#
# Output:
#   /tmp/tinyllama_fixed.uartlog

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
BIN="bench_model_tinyllama_opu"
SRC="$BIN_DIR/$BIN"
WL="merlin-bench-$BIN"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

if [ ! -f "$SRC" ]; then
	echo "Missing binary: $SRC"
	echo "Rebuild with the patched compiler:"
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
echo "  binary   : $SRC ($(du -h "$SRC" | cut -f1))"
echo "  workload : $WL"
echo "  timeout  : NONE (Ctrl-C to abort cleanly)"

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
echo "--- firesim runworkload (NO timeout, tinyllama ~5-30 min) ---"
echo "Live watch in another shell:"
echo "  tail -f /scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog"
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
[ -n "$UARTLOG" ] && cp "$UARTLOG" /tmp/tinyllama_fixed.uartlog

echo
echo "===================== VERDICT ====================="
if grep -qE 'Bench iter [0-9]+/[0-9]+|CSV, tinyllama' "$UARTLOG" 2>/dev/null; then
	echo "PASS — tinyllama ran benchmark iterations to completion."
	grep -E '^Bench iter|^Warmup iter|^CSV' "$UARTLOG" | tail -10
elif grep -qE 'Warmup iter [0-9]+/[0-9]+ done' "$UARTLOG" 2>/dev/null; then
	echo "WARMUP only — bench did not finish."
	grep -E '^Warmup iter' "$UARTLOG" | tail -5
elif grep -qE '\[dn\] o=' "$UARTLOG" 2>/dev/null; then
	echo "HANG — last dispatch seen:"
	grep -E '^\[dn\]|^\[dc\]' "$UARTLOG" 2>/dev/null | tail -5
	echo
	echo "If the last [dn] is a non-reduction/softmax symbol, that's a new hang"
	echo "(possibly the OPU_LLM preprocessing bug). Report it."
else
	echo "NEVER REACHED WARMUP — tinyllama-specific pre-dispatch crash."
	echo "Last 20 lines of uartlog:"
	tail -20 "$UARTLOG" 2>/dev/null
fi
echo
echo "Full log: /tmp/tinyllama_fixed.uartlog"
