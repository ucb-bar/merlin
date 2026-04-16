#!/usr/bin/env bash
# Phase D (dynamic verification of the +xopu scalarization fix):
#   Stage the post-fix vit_small_opu (or large_mlp_opu) binary as the active
#   FireSim workload and run end-to-end under a host timeout. Expected
#   result for vit_small: dispatch ord=9 returns both workgroups and the
#   full inference completes (vs. the pre-fix freeze after wg=8).
#
# Prerequisite: build_rvv_selftest.sh --phase-d-only has been run so the
# post-fix binary exists under build/firesim-merlin-release/.
#
# Usage:
#   bash build_tools/firesim/run_phase_d.sh                    # vit_small, 600s timeout
#   bash build_tools/firesim/run_phase_d.sh vit_small 900
#   bash build_tools/firesim/run_phase_d.sh large_mlp 600

set -uo pipefail

MODEL="${1:-vit_small}"
TIMEOUT_SECONDS="${2:-600}"

case "$MODEL" in
vit_small)
	BIN_NAME="bench_model_opu_bench_vit_small_opu"
	WL="merlin-bench-bench_model_opu_bench_vit_small_opu"
	;;
vit)
	BIN_NAME="bench_model_opu_bench_vit_opu"
	WL="merlin-bench-bench_model_opu_bench_vit_opu"
	;;
large_mlp)
	BIN_NAME="bench_model_opu_bench_large_mlp_opu"
	WL="merlin-bench-bench_model_opu_bench_large_mlp_opu"
	;;
tinyllama)
	BIN_NAME="bench_model_tinyllama_opu"
	WL="merlin-bench-bench_model_tinyllama_opu"
	;;
*)
	echo "Unknown model '$MODEL'. Use: vit_small | vit | large_mlp | tinyllama"
	exit 1
	;;
esac

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
SRC_BIN="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/$BIN_NAME"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# On Ctrl-C or script exit, always try to kill the simulation so the FPGA
# doesn't get left holding state. Runs exactly once.
_cleanup_ran=0
_cleanup() {
	[ "$_cleanup_ran" = "1" ] && return
	_cleanup_ran=1
	echo
	echo "[cleanup] running firesim kill..."
	(cd "$DEP" && firesim kill) 2>&1 | tail -5 || true
}
trap _cleanup INT TERM

if [ ! -f "$SRC_BIN" ]; then
	echo "Missing post-fix binary: $SRC_BIN"
	echo "Run:  bash $MERLIN_ROOT/build_tools/firesim/build_rvv_selftest.sh --phase-d-only"
	exit 1
fi
if [ ! -d "$DEP/workloads/$WL" ]; then
	echo "FireSim workload dir missing: $DEP/workloads/$WL"
	echo "Stage the workload manifest first (json + overlay) — see firesim marshal docs."
	exit 1
fi
if ! command -v firesim >/dev/null 2>&1; then
	echo "firesim not on PATH. Check build_tools/firesim/env.sh."
	exit 1
fi

echo "=== Phase D config ==="
echo "  model         : $MODEL"
echo "  host timeout  : ${TIMEOUT_SECONDS}s"
echo "  source binary : $SRC_BIN"
echo "  workload name : $WL"

echo
echo "=== [1/4] tear down stale FireSim run ==="
(cd "$DEP" && firesim kill) 2>&1 | tail -5 || true
screen -ls 2>&1 | grep -q 'fsim0' && screen -S fsim0 -X quit 2>/dev/null || true

echo
echo "=== [2/4] stage post-fix binary + point FireSim at this workload ==="
cp "$SRC_BIN" "$DEP/workloads/$WL/$BIN_NAME"
ls -la "$DEP/workloads/$WL/$BIN_NAME"

# Swap config_runtime.yaml to this workload (back up once).
CFG="$DEP/config_runtime.yaml"
if [ ! -f "$CFG.bak_phase_d" ]; then
	cp "$CFG" "$CFG.bak_phase_d"
fi
# Replace workload_name: ... line.
sed -i -E "s|^(\s*workload_name:\s*).*|\1${WL}.json|" "$CFG"
grep workload_name "$CFG" | head -1

echo
if [ "$TIMEOUT_SECONDS" = "0" ]; then
	echo "=== [3/4] firesim infrasetup + runworkload (NO host timeout) ==="
else
	echo "=== [3/4] firesim infrasetup + runworkload (timeout=${TIMEOUT_SECONDS}s) ==="
fi
(cd "$DEP" && firesim infrasetup) || {
	echo "infrasetup FAILED"
	exit 1
}
# Pass timeout=0 to disable the host-side wall-clock limit (useful for long
# runs like full ViT or TinyLlama where it's more important to get the real
# result than to bound the run). When the host timeout fires mid-run, firesim
# doesn't get to copy the uartlog back from FIRESIM_RUNS_DIR to
# results-workload/, so we have a fallback below.
if [ "$TIMEOUT_SECONDS" = "0" ]; then
	(cd "$DEP" && firesim runworkload) ||
		echo "  runworkload exited non-zero"
else
	timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
		bash -c "cd '$DEP' && firesim runworkload" ||
		echo "  runworkload exited non-zero or hit ${TIMEOUT_SECONDS}s timeout"
fi
(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

echo
echo "=== [4/4] parse uartlog ==="
RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
UARTLOG=""
if [ -n "$RESULT" ]; then
	UARTLOG=$(find "$RESULT" -name uartlog -print -quit 2>/dev/null)
fi
# Fallback: if firesim didn't copy the uartlog into results-workload/
# (happens when the host timeout SIGTERM'd firesim mid-run), the live log
# is still in the simulation slot dir.
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	LIVE_UART="${FIRESIM_RUNS_DIR:-/scratch2/agustin/FIRESIM_RUNS_DIR}/sim_slot_0/uartlog"
	if [ -f "$LIVE_UART" ]; then
		echo "NOTE: results-workload had no uartlog; falling back to live slot log"
		UARTLOG="$LIVE_UART"
	fi
fi
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	echo "NO uartlog found under $RESULT or $FIRESIM_RUNS_DIR/sim_slot_0/"
	exit 3
fi
echo "Results at: $RESULT"
echo "Uartlog:   $UARTLOG"
echo
echo "--- last 40 lines of uartlog ---"
tail -40 "$UARTLOG"
echo
echo "--- verdict ---"
if grep -q 'benchmark_done\|Benchmark complete\|DONE' "$UARTLOG"; then
	echo "POST-FIX RUN COMPLETED — LayerNorm no longer hangs."
else
	LAST_O9=$(grep -E 'o=9 wg=' "$UARTLOG" | tail -1)
	if [ -n "$LAST_O9" ]; then
		echo "Last ord=9 workgroup print: $LAST_O9"
	fi
	echo "NO 'DONE' marker found — check uartlog for where it stopped."
fi
