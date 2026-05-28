#!/usr/bin/env bash
# Smoke tests for the dual-extension spike (gemmini + saturn_opu).
#
# Runs three ELFs in increasing complexity:
#   1. Upstream gemmini-rocc-tests matmul_1x1x2048_os   (pure gemmini, no OPU)
#   2. IREE bench_gemmini_spike_matmul                  (gemmini via IREE bare-metal)
#   3. merlin_hetero_runner zephyr.elf                  (both extensions, 2 harts)
#
# Test 1 is portable across any chipyard checkout that has built
# gemmini-rocc-tests. Tests 2 and 3 use repo-specific ELFs and will
# SKIP if their paths aren't present; override via env vars
# (IREE_ELF, MERLIN_HETERO_ELF) to point at your own.
#
# Required:
#   CHIPYARD_ROOT   chipyard checkout with built spike + gemmini-rocc-tests
#
# Optional overrides:
#   RISCV               path to riscv-tools install (default: $CHIPYARD_ROOT/.conda-env/riscv-tools)
#   GEM_ROCC_ELF        custom path for test 1
#   IREE_ELF            custom path for test 2 (SKIP if unset and default missing)
#   MERLIN_HETERO_ELF   custom path for test 3 (SKIP if unset and default missing)
#   HETERO_TIMEOUT_SEC  seconds to allow for test 3 (default 600; full dronet may need >900s)

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [ -z "${CHIPYARD_ROOT:-}" ]; then
    echo "smoke.sh: CHIPYARD_ROOT must be set" >&2
    exit 2
fi

RISCV="${RISCV:-$CHIPYARD_ROOT/.conda-env/riscv-tools}"

# Defaults — present in any standard chipyard checkout with gemmini-rocc-tests built
GEM_ROCC_ELF="${GEM_ROCC_ELF:-$CHIPYARD_ROOT/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/matmul_1x1x2048_os-baremetal}"

# Repo-specific ELFs — caller may override or skip
IREE_ELF="${IREE_ELF:-}"
HETERO_ELF="${MERLIN_HETERO_ELF:-}"
HETERO_TIMEOUT_SEC="${HETERO_TIMEOUT_SEC:-600}"

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    local elf="$2"
    local timeout_sec="$3"
    shift 3
    local extra_env=("$@")
    if [ -z "$elf" ]; then
        echo "SKIP $name — no ELF path provided"
        SKIP=$((SKIP + 1))
        return
    fi
    if [ ! -f "$elf" ]; then
        echo "SKIP $name — $elf not found"
        SKIP=$((SKIP + 1))
        return
    fi
    echo "==== $name ===="
    echo "    ELF: $elf  (timeout ${timeout_sec}s)"
    local out
    out="$(env "${extra_env[@]}" timeout "$timeout_sec" "$ROOT/spike-hetero" "$elf" 2>&1)"
    local rc=$?
    echo "$out" | tail -15
    if [ $rc -eq 0 ]; then
        echo "    [PASS] rc=0"
        PASS=$((PASS + 1))
    else
        echo "    [FAIL] rc=$rc"
        FAIL=$((FAIL + 1))
    fi
    echo
}

export CHIPYARD_ROOT RISCV

# Test 1: gemmini-only, single hart suffices
run_test "gemmini-rocc matmul_1x1x2048_os (pure gemmini, 1 hart)" \
    "$GEM_ROCC_ELF" 120 SPIKE_HARTS=1

# Test 2: IREE bare-metal matmul via gemmini (uses HTIF for output)
run_test "bench_gemmini_spike_matmul (IREE bare-metal, 1 hart)" \
    "$IREE_ELF" 120 SPIKE_HARTS=1

# Test 3: full hetero Zephyr ELF (both extensions, 2 harts) — slow
run_test "merlin_hetero_runner (gemmini+opu, 2 harts)" \
    "$HETERO_ELF" "$HETERO_TIMEOUT_SEC" SPIKE_HARTS=2

echo "================================================================"
echo "spike-hetero smoke results: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
echo "================================================================"
exit $FAIL
