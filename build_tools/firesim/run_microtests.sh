#!/usr/bin/env bash
# Phase-D micro-test sweep: runs every pre-built mt_* micro-test binary
# through FireSim and records whether it passes, hangs, or crashes.
# Each test is tiny (one dispatch) so 60 s per run is the default cap.
#
# Usage:
#   bash build_tools/firesim/run_microtests.sh               # default 120 s
#   bash build_tools/firesim/run_microtests.sh 300           # override
#
# Output:
#   /tmp/microtests.csv                columns: test,variant,verdict,cycles,notes
#   /tmp/microtests_logs/*.uartlog     per-run uartlog

set -uo pipefail

# Microtests are small but FireSim is slow (~1M cycles/s). Wide matmuls
# with 64+ workgroups can legitimately take many minutes — err on the
# side of a generous timeout so a slow-but-correct test doesn't get
# mis-classified as a hang. Bump manually for genuinely runaway runs.
TIMEOUT_SECONDS="${1:-900}"

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
OUT_DIR="/tmp/microtests_logs"
CSV="/tmp/microtests.csv"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# Auto-discover every mt_* binary in the build dir.
mapfile -t BINARIES < <(ls -1 "$BIN_DIR"/bench_model_mt_* 2>/dev/null |
	xargs -n1 basename | sort)
if [ "${#BINARIES[@]}" -eq 0 ]; then
	echo "no mt_* binaries found under $BIN_DIR"
	echo "run: CHIPYARD_ROOT=$CHIPYARD_ROOT conda run -n merlin-dev uv run tools/merlin.py build --profile firesim --config release --cmake-target microtests"
	exit 1
fi

_cleanup_ran=0
_cleanup() {
	[ "$_cleanup_ran" = "1" ] && return
	_cleanup_ran=1
	echo
	echo "[cleanup] killing FireSim and detached screen session..."
	# The detached screen session "fsim0" holds the simulator binary;
	# without this it survives the shell trap and keeps the FPGA busy.
	screen -S fsim0 -X quit 2>/dev/null || true
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
	# Exit explicitly — bash does NOT terminate a script after a trap
	# handler returns; without this line the sweep would continue to
	# the next test after Ctrl-C.
	exit 130
}
trap _cleanup INT TERM HUP

if ! command -v firesim >/dev/null 2>&1; then
	echo "firesim not on PATH. Check build_tools/firesim/env.sh."
	exit 1
fi

mkdir -p "$OUT_DIR"
echo "test,variant,verdict,last_ord,last_symbol,cycles,notes" >"$CSV"

CFG="$DEP/config_runtime.yaml"
if [ ! -f "$CFG.bak_microtests" ]; then
	cp "$CFG" "$CFG.bak_microtests"
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
		continue
	fi

	# Parse test + variant from the binary basename.
	#   bench_model_mt_rsqrt_64xf32_opu         → mt_rsqrt_64xf32, opu
	#   bench_model_mt_matmul_64x128_opu_llm    → mt_matmul_64x128, opu_llm
	name="${BIN#bench_model_}"
	case "$name" in
	*_opu_llm) VARIANT=opu_llm; TEST="${name%_opu_llm}" ;;
	*_opu)     VARIANT=opu;     TEST="${name%_opu}" ;;
	*_rvv)     VARIANT=rvv;     TEST="${name%_rvv}" ;;
	*)         VARIANT=unknown; TEST="$name" ;;
	esac

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
		echo "${TEST},${VARIANT},infrasetup_fail,," >>"$CSV"
		continue
	fi

	echo "--- runworkload (timeout=${TIMEOUT_SECONDS}s) ---"
	timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
		bash -c "cd '$DEP' && firesim runworkload" \
		>/tmp/microtests_last_run.log 2>&1 || true
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
		echo "${TEST},${VARIANT},no_uartlog,," >>"$CSV"
		continue
	fi
	cp "$UARTLOG" "$OUT_DIR/$BIN.uartlog"

	# Total cycles = sum of every [dc] cyc= seen (any ordinal).
	CYC=$(grep -E '^\[dc\] o=' "$UARTLOG" |
		sed -E 's/.*cyc=([0-9]+).*/\1/' |
		awk '{s+=$1} END {print s+0}')

	# Last dispatch we saw enter — tells us where the hang is.
	LAST_LINE=$(grep -E '^\[dn\] o=|^\[dc\] o=' "$UARTLOG" | tail -1)
	LAST_ORD=$(echo "$LAST_LINE" | sed -nE 's/.*o=([0-9]+).*/\1/p')
	LAST_SYM=$(grep -E '^\[dn\] o=' "$UARTLOG" | tail -1 |
		sed -nE 's/.*sym=([^ ]+).*/\1/p')
	LAST_ORD="${LAST_ORD:-}"
	LAST_SYM="${LAST_SYM:-}"

	# "pass"     = completed ≥1 bench iteration (harness prints "Bench iter 1/10")
	# "warmup"   = finished warmup but timeout cut the bench (progress, not hang)
	# "progress" = i8 reductions ticking workgroups but timed out mid-matmul
	# "hang"     = [dn] printed but never any [dc] for the terminal dispatch
	# "early"    = not even a [dn] made it out
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
	if grep -q 'iree_status_abort\|PANIC\|Fatal' "$UARTLOG"; then
		NOTES="abort"
	fi
	echo "  -> $VERDICT ord=$LAST_ORD sym=$LAST_SYM cyc=$CYC $NOTES"
	echo "${TEST},${VARIANT},${VERDICT},${LAST_ORD},${LAST_SYM},${CYC},${NOTES}" >>"$CSV"
done

echo
echo "========================================================================"
echo "Microtest summary:"
echo "========================================================================"
column -t -s, "$CSV"
echo
echo "Full uartlogs: $OUT_DIR/"
echo "CSV:           $CSV"
