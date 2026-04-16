#!/usr/bin/env bash
# Stage + run a pre-built Saturn RVV selftest binary on FireSim, then parse
# the [rvv] checkpoints from the uartlog.
#
# A hang inside the running RISC-V ELF freezes the simulated core (the
# vector unit never retires the bad opcode), so a software timeout *inside*
# the binary cannot rescue it. This script therefore imposes a HOST-LEVEL
# `timeout` around `firesim runworkload` and follows up with `firesim kill`
# so the FPGA is clean for the next survey iteration.
#
# Prerequisite: the pre-built binary for the requested skip mask must
# already exist under build/firesim-rvvtest/skip_<MASK>/. Generate with:
#   bash build_tools/firesim/build_rvv_selftest.sh
#
# Usage:
#   bash run_rvv_selftest.sh                # skip=0x00,  timeout=360s
#   bash run_rvv_selftest.sh 0x18           # custom skip mask
#   bash run_rvv_selftest.sh 0x18 600       # custom skip and timeout

set -uo pipefail

SKIP_MASK="${1:-0x00}"
TIMEOUT_SECONDS="${2:-360}"

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
WL=merlin-bench-bench_model_opu_bench_vit_small_rvvtest
PREBUILT="$MERLIN_ROOT/build/firesim-rvvtest/skip_$SKIP_MASK/bench_model_opu_bench_vit_small_rvvtest"

# Source the shared env setup (conda, firesim PATH, ssh-agent).
# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

if [ ! -f "$PREBUILT" ]; then
	echo "Pre-built binary missing: $PREBUILT"
	echo "Run:  bash $MERLIN_ROOT/build_tools/firesim/build_rvv_selftest.sh $SKIP_MASK"
	exit 1
fi
if ! command -v firesim >/dev/null 2>&1; then
	echo "firesim not on PATH. Check build_tools/firesim/env.sh."
	exit 1
fi

echo "=== config ==="
echo "  skip mask       : $SKIP_MASK"
echo "  host timeout    : ${TIMEOUT_SECONDS}s"
echo "  pre-built ELF   : $PREBUILT"
echo "  workload name   : $WL"

echo
echo "=== [1/4] tear down stale FireSim run ==="
(cd "$DEP" && firesim kill) 2>&1 | tail -5 || true
if screen -ls 2>&1 | grep -q 'fsim0'; then
	screen -S fsim0 -X quit 2>/dev/null || true
fi

echo
echo "=== [2/4] stage pre-built binary ==="
cp "$PREBUILT" "$DEP/workloads/$WL/bench_model_opu_bench_vit_small_rvvtest"
ls -la "$DEP/workloads/$WL/bench_model_opu_bench_vit_small_rvvtest"
grep workload_name "$DEP/config_runtime.yaml" | head -1

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
		echo "  runworkload exited non-zero or hit ${TIMEOUT_SECONDS}s timeout (expected if a probe hangs)"
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
echo "Uartlog:   $UARTLOG"
echo
echo "--- [rvv] checkpoints ---"
grep -E '\[rvv\]' "$UARTLOG" || echo "(no [rvv] lines found — selftest did not run?)"
echo
echo "--- verdict ---"
LAST=$(grep -E '\[rvv\] cp=' "$UARTLOG" | tail -1)
if grep -q 'SELFTEST DONE' "$UARTLOG"; then
	echo "ALL ENABLED CHECKPOINTS PASSED."
elif [ -z "$LAST" ]; then
	echo "No checkpoint reached — vector unit unreachable OR printf broke. Check uartlog."
else
	LASTCP=$(echo "$LAST" | grep -oE 'cp=[0-9]+')
	case "$LASTCP" in
	cp=1) echo "Stopped after cp=1 (vlenb). Suspect: plain vadd.vv hangs." ;;
	cp=2) echo "Stopped after cp=2 (vadd). Suspect: next enabled cp hangs (vfredusum / vfsqrt / ...)." ;;
	cp=3) echo "Stopped after cp=3 (vfredusum). Suspect: vfsqrt.v hangs." ;;
	cp=4) echo "Stopped after cp=4 (vfsqrt). Suspect: vrgather.vi hangs." ;;
	cp=5) echo "Stopped after cp=5 (vrgather). Suspect: vfredmin.vs hangs." ;;
	cp=6) echo "Stopped after cp=6 (vfredmin). Suspect: vfredmax.vs hangs." ;;
	cp=7) echo "Stopped after cp=7 (vfredmax). Suspect: vfwredusum.vs hangs." ;;
	cp=8) echo "Stopped after cp=8 (vfwredusum). Suspect: vfslide1down.vf hangs." ;;
	cp=9) echo "Stopped after cp=9 (vfslide1down) but no DONE — selftest stopped mid-print." ;;
	*) echo "Unknown checkpoint state: $LAST" ;;
	esac
fi
