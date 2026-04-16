#!/usr/bin/env bash
# Final end-to-end OPU vs RVV benchmark — the authoritative measurement.
#
# Two-phase design:
#
#   Phase 1: CLEAN SWEEP (`--phase=clean`, default)
#     Runs every `bench_model_<name>_{opu,rvv}` binary. These were compiled
#     WITHOUT the MERLIN_PROFILE_CYCLES define, so the hot benchmark loop
#     has zero instrumentation — just the IREE VM invoke + the final CSV
#     line. Used for OPU-vs-RVV speedup comparison.
#
#   Phase 2: PROFILE SWEEP (`--phase=profile`)
#     Runs each `bench_model_<name>_opu_prof` binary (OPU only). These
#     wrap every dispatch with rdcycle and dump a CYC line per ordinal at
#     end of run. Used for the compute-share decomposition plot.
#     *Never* used for speedup numbers — the wrap adds ~8 cyc/dispatch
#     overhead which would contaminate the OPU timing.
#
#   `--phase=both` runs clean first then profile.
#
# Usage:
#   bash build_tools/firesim/run_final_sweep.sh                    # clean
#   bash build_tools/firesim/run_final_sweep.sh --phase=profile    # profile
#   bash build_tools/firesim/run_final_sweep.sh --phase=both       # full
#   bash build_tools/firesim/run_final_sweep.sh --only=vit_small   # filter
#
# Output:
#   /tmp/sweep_iters.csv    model,variant,iter1..iter5,avg,median,stddev
#   /tmp/sweep_cycles.csv   model,variant,ordinal,symbol,total_cycles,wg_count
#   /tmp/sweep_logs/*.uartlog   raw per-run uartlogs

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
OUT_DIR="/tmp/sweep_logs"
ITERS_CSV="/tmp/sweep_iters.csv"
CYCLES_CSV="/tmp/sweep_cycles.csv"

PHASE="clean"
ONLY=""
INCLUDE_PAIRS=""   # exact-match list of "model:variant model:variant ..."
for arg in "$@"; do
	case "$arg" in
	--phase=clean | --phase=profile | --phase=both) PHASE="${arg#--phase=}" ;;
	--only=*) ONLY="${arg#--only=}" ;;
	--include=*) INCLUDE_PAIRS="${arg#--include=}" ;;
	*)
		echo "Unknown arg: $arg"
		exit 1
		;;
	esac
done

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# Master model list. Each row: <bench_name> <iree_model> <variant>
# `opu_prof` variants are only run in the profile phase; `opu`/`rvv` in clean.
# Models that are OPU-only (no RVV binary exists) show only their OPU row.
MODELS_CLEAN=(
	"mlp_wide opu"
	"mlp_wide rvv"
	"opu_bench_large_mlp opu"
	"opu_bench_large_mlp rvv"
	"opu_bench_vit_small opu"
	"opu_bench_vit_small rvv"
	"opu_bench_vit opu"
	"opu_bench_vit rvv"
	"dronet opu"
	"dronet rvv"
	"yolov8_nano opu"
	"yolov8_nano rvv"
	"mlp opu"
	"mlp rvv"
	"opu_bench_hybrid opu"
	"opu_bench_hybrid rvv"
	"opu_bench_convnet opu"
	"opu_bench_convnet rvv"
	"tinyllama opu"
	"tinyllama rvv"
	"mlp_fast opu"
	"mlp_fast rvv"
)
MODELS_PROF=(
	"mlp_wide opu_prof"
	"opu_bench_large_mlp opu_prof"
	"opu_bench_vit_small opu_prof"
	"opu_bench_vit opu_prof"
	"dronet opu_prof"
	"mlp opu_prof"
	"opu_bench_hybrid opu_prof"
	"opu_bench_convnet opu_prof"
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
[ -f "$ITERS_CSV" ] || echo "model,variant,iter1,iter2,iter3,iter4,iter5,avg,median,stddev" >"$ITERS_CSV"
[ -f "$CYCLES_CSV" ] || echo "model,variant,ordinal,symbol,total_cycles,wg_count" >"$CYCLES_CSV"

CFG="$DEP/config_runtime.yaml"
[ -f "$CFG.bak_final_sweep" ] || cp "$CFG" "$CFG.bak_final_sweep"

_run_one() {
	local MODEL="$1"
	local VARIANT="$2"
	if [ -n "$ONLY" ] && [[ "$MODEL" != *"$ONLY"* ]]; then
		return
	fi
	# Exact (model,variant) include-list filter — supersedes --only.
	# Accepts both space- and comma-separated pair lists.
	if [ -n "$INCLUDE_PAIRS" ]; then
		local pair="${MODEL}:${VARIANT}"
		local norm=" ${INCLUDE_PAIRS//,/ } "
		if [[ "$norm" != *" $pair "* ]]; then
			return
		fi
	fi
	local BIN="bench_model_${MODEL}_${VARIANT}"
	local SRC="$BIN_DIR/$BIN"
	if [ ! -f "$SRC" ]; then
		echo "  MISSING $BIN — skip"
		echo "${MODEL},${VARIANT},missing,,,,,,," >>"$ITERS_CSV"
		return
	fi
	local WL="merlin-bench-$BIN"

	echo
	echo "========================================================================"
	echo "  $MODEL ($VARIANT)"
	echo "========================================================================"

	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
	screen -S fsim0 -X quit 2>/dev/null || true

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
		echo "${MODEL},${VARIANT},infrasetup_fail,,,,,,," >>"$ITERS_CSV"
		return
	fi

	# Per-run wall-clock timeout. Default 30 min; override via
	# MERLIN_RUN_TIMEOUT env. Hung runs (mlp narrow-M edge case, etc.)
	# get killed instead of stalling the whole sweep. Models in
	# NO_TIMEOUT_MODELS are exempt because their clean NO_TILING / LLM
	# run legitimately exceeds the global cap:
	#   tinyllama — 1.1GB VMFB + full transformer inference
	#   yolov8_nano — 320x320 YOLO backbone with pure-vector convs
	#     (no mmt4d ukernel once the OPU-contamination path was removed)
	local TIMEOUT="${MERLIN_RUN_TIMEOUT:-1800}"
	local rc
	case "$MODEL" in
	tinyllama | yolov8_nano | opu_bench_vit | mlp_fast)
		(cd "$DEP" && firesim runworkload) >/tmp/final_sweep_last.log 2>&1
		rc=$?
		;;
	*)
		(cd "$DEP" && timeout --foreground "$TIMEOUT" firesim runworkload) \
			>/tmp/final_sweep_last.log 2>&1
		rc=$?
		;;
	esac
	(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true
	if [ "$rc" -eq 124 ]; then
		echo "  TIMEOUT after ${TIMEOUT}s — recording as hung"
		echo "${MODEL},${VARIANT},timeout_${TIMEOUT}s,,,,,,," >>"$ITERS_CSV"
		return
	fi

	RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
	UARTLOG=""
	[ -n "$RESULT" ] && UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
	if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
		LIVE="${FIRESIM_RUNS_DIR:-/scratch2/agustin/FIRESIM_RUNS_DIR}/sim_slot_0/uartlog"
		[ -f "$LIVE" ] && UARTLOG="$LIVE"
	fi
	if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
		echo "  NO UARTLOG"
		echo "${MODEL},${VARIANT},no_uartlog,,,,,,," >>"$ITERS_CSV"
		return
	fi
	cp "$UARTLOG" "$OUT_DIR/$BIN.uartlog"

	# Parse the multi-iter CSV row.
	CSV_LINE=$(grep -E '^CSV, ' "$UARTLOG" | tail -1)
	if [ -z "$CSV_LINE" ]; then
		echo "  NO CSV line — hang or abort"
		echo "${MODEL},${VARIANT},no_csv,,,,,,," >>"$ITERS_CSV"
		return
	fi
	# CSV, model, variant, iter1, iter2, iter3, iter4, iter5, avg
	mapfile -t fields < <(echo "$CSV_LINE" | awk -F', *' '{for(i=1;i<=NF;++i) print $i}')
	I1=${fields[3]:-}
	I2=${fields[4]:-}
	I3=${fields[5]:-}
	I4=${fields[6]:-}
	I5=${fields[7]:-}
	AVG=${fields[8]:-}
	MED=$(python3 -c "s=sorted([int(x) for x in '$I1 $I2 $I3 $I4 $I5'.split() if x]);print(s[len(s)//2] if s else '')")
	STD=$(python3 -c "import statistics,sys;xs=[int(x) for x in '$I1 $I2 $I3 $I4 $I5'.split() if x];print(f'{statistics.pstdev(xs):.0f}' if len(xs)>1 else '0')")
	echo "  iters: $I1 $I2 $I3 $I4 $I5 | avg=$AVG med=$MED std=$STD"
	echo "${MODEL},${VARIANT},${I1},${I2},${I3},${I4},${I5},${AVG},${MED},${STD}" >>"$ITERS_CSV"

	# Parse CYC lines (profile builds only).
	grep -E '^CYC, [0-9]+, ' "$UARTLOG" | while IFS=, read -r _ ord sym cyc wg; do
		ord=$(echo "$ord" | tr -d ' ')
		sym=$(echo "$sym" | sed -E 's/^ +//; s/ +$//')
		cyc=$(echo "$cyc" | tr -d ' ')
		wg=$(echo "$wg" | tr -d ' ')
		echo "${MODEL},${VARIANT},${ord},${sym},${cyc},${wg}" >>"$CYCLES_CSV"
	done
}

if [ "$PHASE" = "clean" ] || [ "$PHASE" = "both" ]; then
	echo "===================== PHASE 1: CLEAN SWEEP ====================="
	for pair in "${MODELS_CLEAN[@]}"; do
		read -r MODEL VARIANT <<<"$pair"
		_run_one "$MODEL" "$VARIANT"
	done
fi
if [ "$PHASE" = "profile" ] || [ "$PHASE" = "both" ]; then
	echo
	echo "===================== PHASE 2: PROFILE SWEEP ====================="
	for pair in "${MODELS_PROF[@]}"; do
		read -r MODEL VARIANT <<<"$pair"
		_run_one "$MODEL" "$VARIANT"
	done
fi

echo
echo "========================================================================"
echo "OPU vs RVV speedup (clean sweep only, RVV_avg / OPU_avg):"
echo "========================================================================"
python3 - "$ITERS_CSV" <<'PY'
import csv, sys, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    v = r["variant"]
    avg = (r.get("avg") or "").strip()
    if v in ("opu_prof",) or not avg.isdigit():
        continue
    by.setdefault(r["model"], {})[v] = int(avg)
speedups = []
print(f"{'model':28s}  {'OPU cyc':>14s}  {'RVV cyc':>14s}  {'speedup':>8s}")
print("-" * 74)
for model, d in sorted(by.items()):
    o = d.get("opu"); r = d.get("rvv")
    if o and r:
        s = r / o
        speedups.append(s)
        print(f"{model:28s}  {o:>14d}  {r:>14d}  {s:>7.2f}x")
    elif o:
        print(f"{model:28s}  {o:>14d}  {'—':>14s}  {'OPU only':>8s}")
    elif r:
        print(f"{model:28s}  {'—':>14s}  {r:>14d}  {'RVV only':>8s}")
print("-" * 74)
if speedups:
    gmean = 1.0
    for s in speedups: gmean *= s
    gmean = gmean ** (1.0/len(speedups))
    print(f"{'geomean speedup':28s}  {'':>14s}  {'':>14s}  {gmean:>7.2f}x")
PY

echo
echo "Iter CSV:   $ITERS_CSV"
echo "Cycles CSV: $CYCLES_CSV"
echo "Uartlogs:   $OUT_DIR/"
