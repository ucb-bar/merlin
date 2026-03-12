#!/usr/bin/env bash
# run_benchmark.sh - Build and run all Buddy-MLIR Gemmini benchmarks on Spike
#
# Usage: ./scripts/run_benchmark.sh
#
# Prerequisites:
#   - RISCV, BUDDY, SPIKE env vars set (or defaults in Makefiles)
#   - gemmini-rocc-tests available at expected path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

PASS=0
FAIL=0
TOTAL=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_header() {
    echo ""
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""
}

log_result() {
    local name="$1"
    local status="$2"
    local cycles="$3"
    local checksum="$4"

    TOTAL=$((TOTAL + 1))
    if [ "$status" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}[PASS]${NC} $name  cycles=$cycles  checksum=$checksum"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}[FAIL]${NC} $name  cycles=$cycles  checksum=$checksum"
    fi
}

# ============================================================
# Step 1: Build kernel benchmarks
# ============================================================
log_header "Building kernel benchmarks"

cd "$ROOT_DIR/kernels"
make clean 2>/dev/null || true
make all 2>&1 | tail -5
echo "Kernel benchmarks built."

# ============================================================
# Step 2: Run kernel benchmarks on Spike
# ============================================================
log_header "Running kernel benchmarks on Spike"

SPIKE="${SPIKE:-${RISCV:-/home/eecs/ashvin.verma/toolchains/riscv}/bin/spike}"

declare -A EXPECTED_CHECKSUMS
EXPECTED_CHECKSUMS[conv]=950
EXPECTED_CHECKSUMS[conv-with-pool]=30827
EXPECTED_CHECKSUMS[mlp2]=252338
EXPECTED_CHECKSUMS[mlp2-os]=252338
EXPECTED_CHECKSUMS[mlp1]=258664
EXPECTED_CHECKSUMS[softmax-matmul]=3860
EXPECTED_CHECKSUMS[igelu-matmul]=-23260

for bench in conv conv-with-pool mlp2 mlp2-os mlp1 softmax-matmul igelu-matmul; do
    if [ ! -f "${bench}-baremetal" ]; then
        echo -e "  ${RED}[SKIP]${NC} $bench - binary not found"
        continue
    fi

    OUTPUT=$($SPIKE --extension=gemmini "${bench}-baremetal" 2>&1) || true

    # Extract cycles (look for "cycles:" in output)
    CYCLES=$(echo "$OUTPUT" | grep -i 'cycles:' | grep -oP '\d+' | tail -1 || echo "N/A")

    # Extract checksum (look for "output checksum:" in output)
    CHECKSUM=$(echo "$OUTPUT" | grep -i 'output checksum:' | grep -oP '[-]?\d+' | tail -1 || echo "N/A")

    EXPECTED="${EXPECTED_CHECKSUMS[$bench]:-UNKNOWN}"
    if [ "$CHECKSUM" = "$EXPECTED" ]; then
        log_result "$bench" "PASS" "$CYCLES" "$CHECKSUM"
    else
        log_result "$bench" "FAIL" "$CYCLES" "$CHECKSUM (expected $EXPECTED)"
    fi
done

# ============================================================
# Step 3: Build and run ResNet50 validation
# ============================================================
log_header "Building ResNet50 validation"

cd "$ROOT_DIR/resnet50"
make clean 2>/dev/null || true
make all 2>&1 | tail -5
echo "ResNet50 benchmarks built."

log_header "Running ResNet50 validation on Spike"

# Run Gemmini C reference
if [ -f "conv1-gemmini-baremetal" ]; then
    OUTPUT=$($SPIKE --extension=gemmini conv1-gemmini-baremetal 2>&1) || true
    GEMMINI_CYCLES=$(echo "$OUTPUT" | grep -i 'Conv1 cycles:' | grep -oP '\d+' | tail -1 || echo "N/A")
    GEMMINI_CHECKSUM=$(echo "$OUTPUT" | grep -i 'Output checksum:' | grep -oP '[-]?\d+' | tail -1 || echo "N/A")
    echo "  Gemmini C: cycles=$GEMMINI_CYCLES checksum=$GEMMINI_CHECKSUM"
fi

# Run Buddy
if [ -f "conv1-buddy-baremetal" ]; then
    OUTPUT=$($SPIKE --extension=gemmini conv1-buddy-baremetal 2>&1) || true
    BUDDY_CYCLES=$(echo "$OUTPUT" | grep -i 'conv1 cycles:' | grep -oP '\d+' | tail -1 || echo "N/A")
    BUDDY_CHECKSUM=$(echo "$OUTPUT" | grep -i 'Output checksum:' | grep -oP '[-]?\d+' | tail -1 || echo "N/A")

    if [ "$BUDDY_CHECKSUM" = "$GEMMINI_CHECKSUM" ]; then
        log_result "resnet50-conv1 (buddy)" "PASS" "$BUDDY_CYCLES" "$BUDDY_CHECKSUM"
    else
        log_result "resnet50-conv1 (buddy)" "FAIL" "$BUDDY_CYCLES" "$BUDDY_CHECKSUM (expected $GEMMINI_CHECKSUM)"
    fi
fi

# Run BAD test (should NOT match)
if [ -f "conv1-bad-buddy-baremetal" ]; then
    OUTPUT=$($SPIKE --extension=gemmini conv1-bad-buddy-baremetal 2>&1) || true
    BAD_CHECKSUM=$(echo "$OUTPUT" | grep -i 'Output checksum:' | grep -oP '[-]?\d+' | tail -1 || echo "N/A")

    TOTAL=$((TOTAL + 1))
    if [ "$BAD_CHECKSUM" != "$GEMMINI_CHECKSUM" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}[PASS]${NC} resnet50-conv1 (bad) correctly differs: checksum=$BAD_CHECKSUM"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}[FAIL]${NC} resnet50-conv1 (bad) unexpectedly matches reference!"
    fi
fi

# ============================================================
# Summary
# ============================================================
log_header "Summary"

echo "  Total tests: $TOTAL"
echo -e "  Passed:      ${GREEN}$PASS${NC}"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  Failed:      ${RED}$FAIL${NC}"
else
    echo -e "  Failed:      $FAIL"
fi
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed.${NC}"
    exit 0
fi
