#!/usr/bin/env bash
# End-to-end OPU-vs-RVV speedup sweep.
#
# Runs each model × variant, parses the benchmark harness's CSV line
#   CSV, <model>, <variant>, <cycles>
# and produces a speedup report comparing OPU (largest tile wins via the
# compiler's chooseMatmulTile fallback) vs the RVV baseline.
#
# Requires: all target binaries already built (rebuild separately via
# `uv run tools/merlin.py build --profile firesim --config release
#  --cmake-target <name>` after any compiler patch).
#
# Usage:
#   bash build_tools/firesim/run_speedup_sweep.sh
#
# Output:
#   /tmp/speedup_sweep.csv         model,variant,cycles_per_inference,relative_speedup
#   /tmp/speedup_sweep_logs/*.uartlog

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN_DIR="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
OUT_DIR="/tmp/speedup_sweep_logs"
CSV="/tmp/speedup_sweep.csv"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# Every model that has both OPU and RVV variants. Pairs are ordered so
# the CSV has matching rows for speedup computation.
MODELS=(
	opu_bench_large_mlp
	mlp_wide
	dronet
	yolov8_nano
	opu_bench_vit_small
	# Transformers — now working post Option E fix
	# tinyllama only has OPU variant; runs separately below
)
VARIANTS=(opu rvv)

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
echo "model,variant,cycles_per_inference" >"$CSV"

CFG="$DEP/config_runtime.yaml"
[ -f "$CFG.bak_speedup" ] || cp "$CFG" "$CFG.bak_speedup"

_run_one() {
	local BIN="$1"
	local MODEL="$2"
	local VARIANT="$3"
	local WL="merlin-bench-$BIN"
	local SRC="$BIN_DIR/$BIN"

	if [ ! -f "$SRC" ]; then
		echo "  MISSING $SRC — skipping"
		echo "${MODEL},${VARIANT}," >>"$CSV"
		return
	fi

	echo
	echo "========================================================================"
	echo "  $MODEL ($VARIANT) — $BIN"
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
		echo "${MODEL},${VARIANT},infrasetup_fail" >>"$CSV"
		return
	fi

	(cd "$DEP" && firesim runworkload) >/tmp/speedup_last_run.log 2>&1 || true
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
		echo "${MODEL},${VARIANT},no_uartlog" >>"$CSV"
		return
	fi
	cp "$UARTLOG" "$OUT_DIR/$BIN.uartlog"

	# Parse the benchmark's CSV line: "CSV, <model>, <variant>, <cycles>"
	CYC=$(grep -E '^CSV, ' "$UARTLOG" | tail -1 | awk -F', *' '{print $NF}' | tr -d ' \r')
	if [ -z "$CYC" ]; then
		echo "  NO CSV line — run did not complete"
		echo "${MODEL},${VARIANT},no_csv" >>"$CSV"
		return
	fi
	echo "  -> ${CYC} cycles/inference"
	echo "${MODEL},${VARIANT},${CYC}" >>"$CSV"
}

for MODEL in "${MODELS[@]}"; do
	for VARIANT in "${VARIANTS[@]}"; do
		BIN="bench_model_${MODEL}_${VARIANT}"
		_run_one "$BIN" "$MODEL" "$VARIANT"
	done
done

# tinyllama — OPU only (no RVV build exists)
_run_one "bench_model_tinyllama_opu" "tinyllama" "opu"

echo
echo "========================================================================"
echo "Raw cycles per inference:"
echo "========================================================================"
column -t -s, "$CSV"

# Compute speedup = RVV_cycles / OPU_cycles per model
echo
echo "========================================================================"
echo "OPU vs RVV speedup (RVV_cyc / OPU_cyc; higher = OPU faster):"
echo "========================================================================"
python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    key = r["model"]
    v = r["variant"]
    try:
        c = int(r["cycles_per_inference"])
    except (ValueError, KeyError):
        c = None
    by.setdefault(key, {})[v] = c

speedups = []
print(f"{'model':25s}  {'OPU cyc':>14s}  {'RVV cyc':>14s}  {'speedup':>8s}")
print("-" * 68)
for model, d in by.items():
    o = d.get("opu")
    r = d.get("rvv")
    if o and r:
        s = r / o
        speedups.append(s)
        print(f"{model:25s}  {o:>14d}  {r:>14d}  {s:>7.2f}x")
    elif o:
        print(f"{model:25s}  {o:>14d}  {'—':>14s}  {'OPU only':>8s}")
    else:
        print(f"{model:25s}  {'—':>14s}  {str(r):>14s}  incomplete")
print("-" * 68)
if speedups:
    gmean = (1.0)
    for s in speedups: gmean *= s
    gmean = gmean ** (1.0/len(speedups))
    amean = sum(speedups) / len(speedups)
    print(f"{'geomean speedup':25s}  {'':>14s}  {'':>14s}  {gmean:>7.2f}x")
    print(f"{'arith mean speedup':25s}  {'':>14s}  {'':>14s}  {amean:>7.2f}x")
PY
echo
echo "Full uartlogs: $OUT_DIR/"
echo "CSV:           $CSV"
