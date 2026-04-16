#!/usr/bin/env bash
# Focused pinpoint sweep — just the tests we need to localize the
# transformer hang given what we ALREADY know from partial results:
#   - f32 elementwise passes (mt_cvt, mt_divf, mt_erf confirmed)
#   - f32 reductions hang (mt_bmm, mt_matmul_64x128 confirmed)
#   - i8 reductions progress (mt_mm_i8 confirmed)
#
# This 7-run sweep nails down WHICH vit layer breaks and WHICH lowering
# class carries the bug. Total ~35 min at 300 s/test cap.
#
# Tier 1 — confirm vit root-cause shape matches bmm hang:
#   mt_vit_d1_layernorm_rvv   → must hang at reduction_*_f32 if hypothesis right
#   mt_softmax_64x64_rvv      → does softmax decomp hit same path?
#
# Tier 2 — flag bisection on the known-hang mt_bmm pattern:
#   mt_bmm_4x64x64_opu_scalar    (vector-width=0)
#   mt_bmm_4x64x64_opu_novec     (no +v)
#   mt_bmm_4x64x64_opu_nocustom  (no custom vector-contract)
#   mt_bmm_4x64x64_opu_noukernel (no ukernels)
#
# Tier 3 — verify i8 ukernel path is unbroken:
#   mt_mm_i8_64x128_opu       → full end-to-end i8 matmul run
#
# Usage:
#   bash build_tools/firesim/run_pinpoint.sh            # 300 s per test
#   bash build_tools/firesim/run_pinpoint.sh 600        # longer cap

set -uo pipefail

TIMEOUT_SECONDS="${1:-300}"

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
OUT_DIR="/tmp/pinpoint_logs"
CSV="/tmp/pinpoint.csv"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

BINARIES=(
	bench_model_mt_vit_d1_layernorm_rvv
	bench_model_mt_softmax_64x64_rvv
	bench_model_mt_bmm_4x64x64_opu_scalar
	bench_model_mt_bmm_4x64x64_opu_novec
	bench_model_mt_bmm_4x64x64_opu_nocustom
	bench_model_mt_bmm_4x64x64_opu_noukernel
	bench_model_mt_mm_i8_64x128_opu
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
[ -f "$CFG.bak_pinpoint" ] || cp "$CFG" "$CFG.bak_pinpoint"

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
		>/tmp/pinpoint_last_run.log 2>&1 || true
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

	RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
	UARTLOG=""
	[ -n "$RESULT" ] && UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
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

	if grep -qE 'Bench iter [0-9]+/[0-9]+' "$UARTLOG"; then
		VERDICT=pass
	elif grep -qE 'Warmup iter [0-9]+/[0-9]+ done' "$UARTLOG"; then
		VERDICT=warmup
	elif grep -qE '\[dc\] o=' "$UARTLOG"; then
		VERDICT=progress
	elif grep -qE '\[dn\] o=' "$UARTLOG"; then
		VERDICT=hang
	else
		VERDICT=early
	fi
	NOTES=""
	grep -q 'iree_status_abort\|PANIC\|Fatal' "$UARTLOG" && NOTES="abort"
	echo "  -> $VERDICT ord=$LAST_ORD sym=$LAST_SYM cyc=$CYC $NOTES"
	echo "${BIN},${VERDICT},${LAST_ORD},${LAST_SYM},${CYC},${NOTES}" >>"$CSV"
done

echo
echo "========================================================================"
echo "Pinpoint summary:"
echo "========================================================================"
column -t -s, "$CSV"
echo
echo "Decision matrix:"
echo "  vit_d1_layernorm hangs at reduction_*_f32  → LN root cause confirmed"
echo "  softmax_64x64 hangs at reduction_*_f32     → softmax hits same path"
echo "  bmm_opu_scalar passes, others hang          → any vector lowering breaks"
echo "  bmm_opu_nocustom passes, bmm_opu hangs      → custom vector-contract bug"
echo "  bmm_opu_noukernel passes, bmm_opu hangs     → ukernel lowering bug"
echo "  mm_i8_64x128_opu fully passes               → i8 path / opu_matmul unbroken"
