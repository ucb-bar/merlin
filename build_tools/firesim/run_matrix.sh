#!/usr/bin/env bash
# Phase A sanity sweep: run every pre-built SaturnOPU FireSim benchmark
# binary (all model × variant combinations we have binaries for) under a
# fixed host timeout and record pass/fail + last-seen dispatch into a CSV.
#
# The goal is a clean pass/fail matrix so we can confirm whether the
# remaining hangs are transformer-specific (vit_small, tinyllama) while
# the non-transformer models all pass on both OPU and RVV variants.
#
# Usage:
#   bash build_tools/firesim/run_matrix.sh                 # default 300s per run
#   bash build_tools/firesim/run_matrix.sh 600             # 600s per run
#
# Output:
#   /tmp/firesim_matrix.csv                                # summary rows
#   /tmp/firesim_matrix_logs/<binary>.uartlog              # per-run uartlog
#   /tmp/firesim_matrix_logs/<binary>.tail                 # last 80 lines

set -uo pipefail

# Default 0 = no host timeout. FireSim's own terminate_on_completion handles
# clean exits; runs that genuinely hang still need Ctrl-C. Pass a positive
# number to bound a single sweep.
TIMEOUT_SECONDS="${1:-0}"

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
OUT_DIR="/tmp/firesim_matrix_logs"
CSV="/tmp/firesim_matrix.csv"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# Model × variant list. Entries are the pre-built binary basenames (each
# one exists in $BIN_DIR). Known-good and known-bad cases share the same
# sweep so we detect environment drift in the same run.
BINARIES=(
	# non-transformers — expected PASS on both variants
	bench_model_opu_bench_large_mlp_opu
	bench_model_opu_bench_large_mlp_rvv
	bench_model_mlp_wide_opu
	bench_model_mlp_wide_rvv
	bench_model_dronet_opu
	bench_model_dronet_rvv
	bench_model_yolov8_nano_opu
	bench_model_yolov8_nano_rvv
	# transformers dropped — vit_small/vit/tinyllama all hang and add
	# no new signal to the sanity matrix.
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
echo "model,variant,verdict,last_dispatch,runtime_s" >"$CSV"

CFG="$DEP/config_runtime.yaml"
if [ ! -f "$CFG.bak_matrix" ]; then
	cp "$CFG" "$CFG.bak_matrix"
fi

total="${#BINARIES[@]}"
idx=0
for BIN in "${BINARIES[@]}"; do
	idx=$((idx + 1))
	echo
	echo "========================================================================"
	echo "[$idx/$total] $BIN"
	echo "========================================================================"

	SRC="$BIN_DIR/$BIN"
	if [ ! -f "$SRC" ]; then
		echo "  MISSING BINARY — skipping"
		echo "${BIN},,missing,," >>"$CSV"
		continue
	fi

	# Derive model + variant from the binary name for the CSV.
	# Binaries follow bench_model_<stem>_<variant> where variant ∈ {opu,rvv,rvvtest}.
	VARIANT="${BIN##*_}"
	STEM="${BIN%_*}"
	MODEL="${STEM#bench_model_}"

	WL="merlin-bench-$BIN"

	echo "--- kill stale firesim ---"
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
	if screen -ls 2>&1 | grep -q 'fsim0'; then
		screen -S fsim0 -X quit 2>/dev/null || true
	fi

	echo "--- stage binary into $WL ---"
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
	# Force tracing off for the matrix sweep — per-cycle PC trace is
	# irrelevant here and would bloat results-workload/.
	sed -i -E "s#^(\s*enable:\s*)(yes|no)\b#\1no#" "$CFG"

	echo "--- infrasetup ---"
	if ! (cd "$DEP" && firesim infrasetup); then
		echo "  infrasetup FAILED"
		echo "${MODEL},${VARIANT},infrasetup_fail,," >>"$CSV"
		continue
	fi

	if [ "$TIMEOUT_SECONDS" = "0" ]; then
		echo "--- runworkload (no host timeout) ---"
	else
		echo "--- runworkload (timeout=${TIMEOUT_SECONDS}s) ---"
	fi
	t_start=$(date +%s)
	if [ "$TIMEOUT_SECONDS" = "0" ]; then
		(cd "$DEP" && firesim runworkload) \
			>/tmp/firesim_matrix_last_run.log 2>&1 || true
	else
		timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
			bash -c "cd '$DEP' && firesim runworkload" \
			>/tmp/firesim_matrix_last_run.log 2>&1 || true
	fi
	t_end=$(date +%s)
	runtime_s=$((t_end - t_start))
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

	# Grab the uartlog: prefer results-workload (clean shutdown), fall
	# back to the live slot (matches run_phase_d.sh behavior).
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
		echo "${MODEL},${VARIANT},no_uartlog,,${runtime_s}" >>"$CSV"
		continue
	fi

	cp "$UARTLOG" "$OUT_DIR/$BIN.uartlog"
	tail -80 "$UARTLOG" >"$OUT_DIR/$BIN.tail"

	# Decide verdict. Preference order:
	#   pass    — "Iteration 1" appears (warmup completed)
	#   hang    — no "Iteration 1", but at least one [dn] or [dc] seen
	#   early   — not even a dispatch marker (crashed pre-warmup)
	LAST_D=$(grep -E '\[dn\] o=|\[dc\] o=' "$UARTLOG" | tail -1 |
		sed -E 's/.*o=([0-9]+).*/\1/' | head -c 8)
	LAST_D="${LAST_D:-}"
	if grep -q 'Iteration 1' "$UARTLOG"; then
		VERDICT=pass
	elif grep -qE '\[dn\] o=|\[dc\] o=' "$UARTLOG"; then
		VERDICT=hang
	else
		VERDICT=early
	fi
	echo "  -> verdict=$VERDICT last_dispatch=$LAST_D runtime=${runtime_s}s"
	echo "${MODEL},${VARIANT},${VERDICT},${LAST_D},${runtime_s}" >>"$CSV"
done

echo
echo "========================================================================"
echo "Matrix complete. Summary:"
echo "========================================================================"
column -t -s, "$CSV"
echo
echo "Full uartlogs: $OUT_DIR/"
echo "CSV:           $CSV"
