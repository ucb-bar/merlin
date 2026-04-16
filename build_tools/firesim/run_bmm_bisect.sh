#!/usr/bin/env bash
# Phase-D flag bisection for the mt_bmm_4x64x64 hang.
# Runs the 4 variants compiled with progressively more aggressive
# de-vectorization so whichever one stops hanging pinpoints the
# broken lowering class.
#
# Variants:
#   opu           — baseline (known hang)
#   opu_scalar    — --iree-llvmcpu-target-vector-width-in-bytes=0
#   opu_novec     — drop +v feature (no RVV at all)
#   opu_nocustom  — --iree-llvmcpu-enable-vector-contract-custom-kernels=false
#   opu_noukernel — --iree-llvmcpu-enable-ukernels=none
#
# Usage:
#   bash build_tools/firesim/run_bmm_bisect.sh            # 600 s/test default
#   bash build_tools/firesim/run_bmm_bisect.sh 1200       # bump per-test cap
#
# Output:
#   /tmp/bmm_bisect.csv                   test,verdict,last_ord,last_symbol,cycles
#   /tmp/bmm_bisect_logs/*.uartlog

set -uo pipefail

TIMEOUT_SECONDS="${1:-600}"

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
OUT_DIR="/tmp/bmm_bisect_logs"
CSV="/tmp/bmm_bisect.csv"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

BINARIES=(
	bench_model_mt_bmm_4x64x64_opu
	bench_model_mt_bmm_4x64x64_opu_scalar
	bench_model_mt_bmm_4x64x64_opu_novec
	bench_model_mt_bmm_4x64x64_opu_nocustom
	bench_model_mt_bmm_4x64x64_opu_noukernel
)

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

if ! command -v firesim >/dev/null 2>&1; then
	echo "firesim not on PATH. Check build_tools/firesim/env.sh."
	exit 1
fi

mkdir -p "$OUT_DIR"
echo "test,verdict,last_ord,last_symbol,cycles,notes" >"$CSV"

CFG="$DEP/config_runtime.yaml"
if [ ! -f "$CFG.bak_bmm_bisect" ]; then
	cp "$CFG" "$CFG.bak_bmm_bisect"
fi

idx=0
total="${#BINARIES[@]}"
for BIN in "${BINARIES[@]}"; do
	idx=$((idx + 1))
	echo
	echo "========================================================================"
	echo "[$idx/$total] $BIN"
	echo "========================================================================"

	SRC="$BIN_DIR/$BIN"
	if [ ! -f "$SRC" ]; then
		echo "  MISSING — skipping"
		echo "${BIN},missing,,,0," >>"$CSV"
		continue
	fi

	WL="merlin-bench-$BIN"

	echo "--- kill stale firesim ---"
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

	echo "--- stage + infrasetup ---"
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
	sed -i -E "s#^(\s*workload_name:\s*).*#\1${WL}.json#" "$CFG"
	sed -i -E "s#^(\s*enable:\s*)(yes|no)\b#\1no#" "$CFG"
	if ! (cd "$DEP" && firesim infrasetup); then
		echo "  infrasetup FAILED"
		echo "${BIN},infrasetup_fail,,,0," >>"$CSV"
		continue
	fi

	echo "--- runworkload (timeout=${TIMEOUT_SECONDS}s) ---"
	timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
		bash -c "cd '$DEP' && firesim runworkload" \
		>/tmp/bmm_bisect_last_run.log 2>&1 || true
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

	RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
	UARTLOG=""
	if [ -n "$RESULT" ]; then
		UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
	fi
	if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
		LIVE="${FIRESIM_RUNS_DIR:-/scratch2/agustin/FIRESIM_RUNS_DIR}/sim_slot_0/uartlog"
		[ -f "$LIVE" ] && UARTLOG="$LIVE"
	fi
	if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
		echo "  NO UARTLOG"
		echo "${BIN},no_uartlog,,,0," >>"$CSV"
		continue
	fi
	cp "$UARTLOG" "$OUT_DIR/$BIN.uartlog"

	CYC=$(grep -E '^\[dc\] o=' "$UARTLOG" |
		sed -E 's/.*cyc=([0-9]+).*/\1/' |
		awk '{s+=$1} END {print s+0}')
	LAST_ORD=$(grep -E '^\[dn\] o=|^\[dc\] o=' "$UARTLOG" | tail -1 |
		sed -nE 's/.*o=([0-9]+).*/\1/p')
	LAST_SYM=$(grep -E '^\[dn\] o=' "$UARTLOG" | tail -1 |
		sed -nE 's/.*sym=([^ ]+).*/\1/p')
	LAST_ORD="${LAST_ORD:-}"
	LAST_SYM="${LAST_SYM:-}"

	if grep -q 'Iteration 1' "$UARTLOG"; then
		VERDICT=pass
	elif grep -qE '\[dn\] o=|\[dc\] o=' "$UARTLOG"; then
		VERDICT=hang
	else
		VERDICT=early
	fi
	NOTES=""
	if grep -q 'iree_status_abort\|PANIC\|Fatal' "$UARTLOG"; then
		NOTES="abort"
	fi
	echo "  -> $VERDICT ord=$LAST_ORD sym=$LAST_SYM cyc=$CYC $NOTES"
	echo "${BIN},${VERDICT},${LAST_ORD},${LAST_SYM},${CYC},${NOTES}" >>"$CSV"
done

echo
echo "========================================================================"
echo "bmm flag-bisect summary:"
echo "========================================================================"
column -t -s, "$CSV"
echo
echo "Interpretation:"
echo "  - If opu_scalar passes and opu hangs  → vector-lowering bug"
echo "  - If opu_novec passes and opu_scalar hangs → vsetvli/LMUL interaction"
echo "  - If opu_nocustom passes and opu hangs → custom vector-contract bug"
echo "  - If opu_noukernel passes and opu hangs → ukernel lowering bug"
echo "  - If all hang → deeper than lowering (memory/runtime/linking)"
