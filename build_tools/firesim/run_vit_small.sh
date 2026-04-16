#!/usr/bin/env bash
# Stage + run the pre-built Saturn vit_small binary on FireSim (OPU or RVV
# variant), impose a host-level timeout so a hung vector unit doesn't park
# the FPGA forever, then print the uartlog tail plus the [d]/[vm_invoke]
# markers we use to localize hangs.
#
# Variants (must match the CMake targets in
# samples/SaturnOPU/simple_embedding_ukernel/CMakeLists.txt):
#   opu — FLAGS_MODEL_OPU_NO_CUSTOM_VC: +v,+xopu, data tiling + ukernels,
#         --iree-llvmcpu-enable-vector-contract-custom-kernels=false
#   rvv — FLAGS_MODEL_RVV_NO_TILING:   +v, no data tiling, no ukernels
#
# Usage:
#   bash build_tools/firesim/run_vit_small.sh opu             # timeout=360s
#   bash build_tools/firesim/run_vit_small.sh rvv 600         # custom timeout

set -uo pipefail

VARIANT="${1:-opu}"
TIMEOUT_SECONDS="${2:-0}"
TRACE_MODE="${3:-off}"  # off | on   — enables FireSim TraceRV PC capture

case "$VARIANT" in
opu | rvv) ;;
*)
	echo "Unknown variant: $VARIANT (expected: opu | rvv)"
	exit 1
	;;
esac
case "$TRACE_MODE" in
on | off) ;;
*)
	echo "Unknown trace mode: $TRACE_MODE (expected: on | off)"
	exit 1
	;;
esac

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
BIN="bench_model_opu_bench_vit_small_${VARIANT}"
WL="merlin-bench-${BIN}"
BUILT="$MERLIN_ROOT/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/${BIN}"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

if [ ! -f "$BUILT" ]; then
	echo "Pre-built binary missing: $BUILT"
	echo "Run: bash build_tools/firesim/build_rvv_selftest.sh --phase-d-only"
	echo "  or: CHIPYARD_ROOT=$CHIPYARD_ROOT conda run -n merlin-dev uv run tools/merlin.py build \\"
	echo "        --profile firesim --cmake-target ${BIN}"
	exit 1
fi
if ! command -v firesim >/dev/null 2>&1; then
	echo "firesim not on PATH. Check build_tools/firesim/env.sh."
	exit 1
fi

echo "=== config ==="
echo "  variant         : $VARIANT"
echo "  host timeout    : ${TIMEOUT_SECONDS}s"
echo "  built ELF       : $BUILT"
echo "  workload name   : $WL"

echo
echo "=== [1/4] tear down stale FireSim run ==="
(cd "$DEP" && firesim kill) 2>&1 | tail -5 || true
if screen -ls 2>&1 | grep -q 'fsim0'; then
	screen -S fsim0 -X quit 2>/dev/null || true
fi

echo
echo "=== [2/4] stage binary + select workload ==="
mkdir -p "$DEP/workloads/$WL"
cp "$BUILT" "$DEP/workloads/$WL/$BIN"
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
	echo "  → created $DEP/workloads/$WL.json"
fi
sed -i "s|^\(    workload_name:\).*|\1 ${WL}.json|" "$DEP/config_runtime.yaml"
# Phase-4 PC trace toggle: flip tracing.enable in config_runtime.yaml.
# When on, FireSim's TraceRV writes per-cycle retired-PC data into a
# TRACEFILE0 inside the results-workload dir. Our verdict logic at
# the end of this script tails that file when present. Use selector=1
# (cycle-count trigger), start=0 end=-1 = full run; we'll grep for the
# last PC that was retired before the simulator gave up.
TRACE_ENABLE="no"
if [ "$TRACE_MODE" = "on" ]; then
	TRACE_ENABLE="yes"
fi
# Delimiter must not be `|`: sed treats `|` as its delimiter, and `(yes|no)`
# contains a `|` that then closes the `s` command prematurely → "unknown
# option to 's'". Use `#` instead so the alternation is untouched.
sed -i -E "s#^(\\s*enable:\\s*)(yes|no)\\b#\\1${TRACE_ENABLE}#" \
	"$DEP/config_runtime.yaml"
ls -la "$DEP/workloads/$WL/$BIN"
grep -m1 workload_name "$DEP/config_runtime.yaml"
grep -m1 -E '^\s*enable:\s*(yes|no)' "$DEP/config_runtime.yaml" | sed 's|^|tracing.|'

echo
echo "=== [3/4] firesim infrasetup + runworkload (timeout=${TIMEOUT_SECONDS}s) ==="
(cd "$DEP" && firesim infrasetup) || {
	echo "infrasetup FAILED"
	exit 1
}
if [ "$TIMEOUT_SECONDS" = "0" ]; then
	(cd "$DEP" && firesim runworkload) ||
		echo "  runworkload exited non-zero"
else
	timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
		bash -c "cd '$DEP' && firesim runworkload" ||
		echo "  runworkload exited non-zero or hit ${TIMEOUT_SECONDS}s timeout (expected if it hangs)"
fi
(cd "$DEP" && firesim kill) 2>&1 | tail -3 || true

echo
echo "=== [4/4] parse uartlog ==="
RESULT=$(ls -1td "$DEP/results-workload/"*"$WL"* 2>/dev/null | head -1)
if [ -z "$RESULT" ]; then
	echo "NO results directory found under $DEP/results-workload/"
	exit 2
fi
UARTLOG=$(find "$RESULT" -name uartlog -print -quit)
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
	echo "NO uartlog found under $RESULT"
	exit 3
fi

echo "Results at: $RESULT"
echo "Uartlog:    $UARTLOG"
echo
echo "--- tail(uartlog) ---"
tail -60 "$UARTLOG"
echo
echo "--- dispatch / vm_invoke / apply markers ---"
grep -E '\[d\]|\[d-enter\]|\[d-exit\]|\[vm_invoke\]|\[apply\]' "$UARTLOG" | tail -50
echo
echo "--- alignment + binding markers ---"
grep -E '\[align\]|\[binding\] o=[0-3]' "$UARTLOG" | head -40
echo
if [ "$TRACE_MODE" = "on" ]; then
	echo "--- TraceRV: last 80 lines of every TRACEFILE* ---"
	for tf in "$RESULT"/*/TRACEFILE* "$RESULT"/TRACEFILE*; do
		[ -f "$tf" ] || continue
		echo "→ $tf"
		tail -80 "$tf"
		echo "---"
	done
fi
echo
echo "--- verdict ---"
if grep -q 'Iteration 1' "$UARTLOG"; then
	echo "SUCCESS — end-to-end inference completed."
elif grep -qE '\[d\] ' "$UARTLOG"; then
	LAST=$(grep -E '\[d\] ' "$UARTLOG" | tail -1)
	echo "HUNG — last dispatch marker: $LAST"
else
	echo "NO dispatch markers — early hang or printf broke. Check tail(uartlog) above."
fi
