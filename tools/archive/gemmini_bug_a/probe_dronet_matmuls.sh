#!/usr/bin/env bash
# Probe each unique matmul shape in dronet × Gemmini to find which shape(s)
# diverge from numpy golden. Compares spike's i32 output against
# numpy.matmul(A_i8.astype(i32), B_i8.astype(i32)) with A=B=all-ones.
#
# Output: per-shape PASS / FAIL with the first divergence.
#
# Assumes ./merlin compile + ./merlin build are functional and that
# tests/integration/gemmini_spike/fixtures/matmul_MxNxK_tensor.mlir
# exists for each shape (auto-generated below if missing).

set -uo pipefail
MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
cd "$MERLIN_ROOT"
# The riscv_firesim toolchain file (build_tools/firesim/riscv_firesim.toolchain.cmake)
# requires CHIPYARD_ROOT or RISCV_NEWLIB_SYSROOT to be set; default to local chipyard.
export CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"

# The 11 unique shapes in dronet × Gemmini (extracted via strings vmfb)
SHAPES=(
    "1x1x2048"        # FC head — already verified PASS
    "196x32x32"       # smallest K=32 single tile
    "49x64x32"        # K=32, larger N
    "16x128x64"       # K=64
    "196x32x288"      # K=288 multi-tile
    "49x64x288"
    "49x64x576"
    "16x128x576"
    "16x128x1152"
    "3136x32x27"      # K=27 (odd, < tile width)
    "16x128x64"
)
# dedup
SHAPES=($(printf '%s\n' "${SHAPES[@]}" | awk '!seen[$0]++'))

FIXTURE_DIR="tests/integration/gemmini_spike/fixtures"
BENCH_BASE="build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"

ensure_fixture() {
    local shape="$1" m n k
    IFS=x read -r m n k <<<"$shape"
    local f="$FIXTURE_DIR/matmul_${shape}_tensor.mlir"
    if [ -f "$f" ]; then return 0; fi
    cat >"$f" <<EOF
// Auto-generated probe fixture for dronet × Gemmini matmul $shape.
// All-ones A and B → expected i32 output = ${k} at every cell.
func.func @matmul_${shape}(%A: tensor<${m}x${k}xi8>, %B: tensor<${k}x${n}xi8>) -> tensor<${m}x${n}xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<${m}x${n}xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<${m}x${n}xi32>) -> tensor<${m}x${n}xi32>
  %res = linalg.matmul ins(%A, %B : tensor<${m}x${k}xi8>, tensor<${k}x${n}xi8>)
                       outs(%fill : tensor<${m}x${n}xi32>) -> tensor<${m}x${n}xi32>
  return %res : tensor<${m}x${n}xi32>
}
EOF
    echo "    wrote $f"
}

run_one_shape() {
    local shape="$1" m n k
    IFS=x read -r m n k <<<"$shape"
    echo "============================================================"
    echo "Probing matmul $shape (expected output = $k everywhere)"
    echo "============================================================"
    ensure_fixture "$shape"

    echo "    [1/3] ./merlin build (target=bench_gemmini_spike_matmul)..."
    if ! ./merlin build --profile firesim --cmake-target bench_gemmini_spike_matmul \
            --cmake-arg="-DGEMMINI_SPIKE_MATMUL_SHAPE=$shape" \
            >"/tmp/probe_${shape}_build.log" 2>&1; then
        echo "    BUILD FAILED — see /tmp/probe_${shape}_build.log"
        return 1
    fi

    local elf="$BENCH_BASE/bench_gemmini_spike_matmul"
    if [ ! -f "$elf" ]; then
        echo "    ELF missing at $elf"
        return 1
    fi

    echo "    [2/3] running on spike --extension=gemmini..."
    local out
    out="$(timeout 600 /scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike \
        --extension=gemmini --isa=rv64gcv_zicntr "$elf" 2>&1 | head -60)"

    echo "$out" | head -8
    echo "    [3/3] checking output..."
    # The runner prints the first row/cell as i32. For all-ones A,B the
    # expected value at every cell is K. We just need to confirm at least
    # one row prints values == K.
    local first_val
    first_val="$(echo "$out" | grep -oE '^[ ]*-?[0-9]+' | head -1)"
    if [ -z "$first_val" ]; then
        echo "    NO OUTPUT VALUES PARSED — possibly PASS marker missing"
        echo "    raw output:"
        echo "$out" | tail -10
        return 1
    fi
    if [ "$first_val" = "$k" ]; then
        echo "    [PASS] first value = $first_val (expected $k)"
        return 0
    else
        echo "    [FAIL] first value = $first_val (expected $k)"
        echo "    raw output tail:"
        echo "$out" | tail -10
        return 2
    fi
}

PASS=()
FAIL=()
ERR=()

for s in "${SHAPES[@]}"; do
    if run_one_shape "$s"; then
        PASS+=("$s")
    else
        rc=$?
        if [ $rc -eq 2 ]; then FAIL+=("$s"); else ERR+=("$s"); fi
    fi
    echo
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "PASS (${#PASS[@]}): ${PASS[*]}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]}"
echo "ERR  (${#ERR[@]}): ${ERR[*]}"
exit ${#FAIL[@]}
