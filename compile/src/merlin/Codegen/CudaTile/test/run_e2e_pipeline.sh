#!/usr/bin/env bash
# run_e2e_pipeline.sh — End-to-end test: linalg → cuda_tile text → tilebc → cubin
#
# Usage:
#   ./run_e2e_pipeline.sh [--iree-opt PATH] [--cuda-tile-opt PATH]
#                         [--cuda-tile-translate PATH] [--tileiras PATH]
#                         [--gpu-name GPU]
#
# Defaults assume tools are on PATH or at known locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERLIN_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"

# --- Tool defaults ---
IREE_OPT="${IREE_OPT:-${MERLIN_ROOT}/build-iree-merlin/tools/iree-opt}"
CUDA_TILE_OPT="${CUDA_TILE_OPT:-${MERLIN_ROOT}/third_party/cuda-tile/build/bin/cuda-tile-opt}"
CUDA_TILE_TRANSLATE="${CUDA_TILE_TRANSLATE:-${MERLIN_ROOT}/third_party/cuda-tile/build/bin/cuda-tile-translate}"
TILEIRAS="${TILEIRAS:-$(which tileiras 2>/dev/null || echo "")}"
GPU_NAME="${GPU_NAME:-sm_100}"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iree-opt) IREE_OPT="$2"; shift 2 ;;
    --cuda-tile-opt) CUDA_TILE_OPT="$2"; shift 2 ;;
    --cuda-tile-translate) CUDA_TILE_TRANSLATE="$2"; shift 2 ;;
    --tileiras) TILEIRAS="$2"; shift 2 ;;
    --gpu-name) GPU_NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# --- Validate tools ---
for tool_var in IREE_OPT CUDA_TILE_OPT CUDA_TILE_TRANSLATE; do
  tool_path="${!tool_var}"
  if [[ ! -x "$tool_path" ]]; then
    echo "ERROR: $tool_var not found at: $tool_path"
    echo "  Set $tool_var env var or pass --$(echo $tool_var | tr '_' '-' | tr 'A-Z' 'a-z') PATH"
    exit 1
  fi
done

if [[ -z "$TILEIRAS" || ! -x "$TILEIRAS" ]]; then
  echo "WARNING: tileiras not found. Will skip tilebc→cubin step."
  echo "  Install: conda install -c nvidia cuda-tileiras"
  HAS_TILEIRAS=false
else
  HAS_TILEIRAS=true
fi

TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

PASS=0
FAIL=0

# --- Test cases: (name, mlir_file, tile_m, tile_n, tile_k, expected_k_tiles) ---
run_test() {
  local name="$1"
  local mlir_file="$2"
  local tile_m="$3"
  local tile_n="$4"
  local tile_k="$5"

  local out_dir="$TMPDIR/$name"
  mkdir -p "$out_dir"

  echo "=== Test: $name (tile ${tile_m}x${tile_n}x${tile_k}) ==="

  # Step 1: iree-opt → cuda_tile text
  local cuda_tile_mlir="$out_dir/output.mlir"
  echo "  [1/4] iree-opt → cuda_tile text"
  if ! "$IREE_OPT" \
      --pass-pipeline="builtin.module(merlin-linalg-to-cuda-tile-text{output-path=${cuda_tile_mlir} tile-m=${tile_m} tile-n=${tile_n} tile-k=${tile_k}})" \
      "$mlir_file" 2>&1; then
    echo "  FAIL: iree-opt failed"
    FAIL=$((FAIL+1))
    return
  fi

  if [[ ! -s "$cuda_tile_mlir" ]]; then
    echo "  FAIL: output file is empty"
    FAIL=$((FAIL+1))
    return
  fi
  echo "  OK: generated $(wc -l < "$cuda_tile_mlir") lines"

  # Step 2: cuda-tile-opt → verify parse + verify
  echo "  [2/4] cuda-tile-opt (parse + verify)"
  if ! "$CUDA_TILE_OPT" "$cuda_tile_mlir" -o /dev/null 2>&1; then
    echo "  FAIL: cuda-tile-opt rejected the generated IR"
    FAIL=$((FAIL+1))
    return
  fi
  echo "  OK: parse + verify passed"

  # Step 3: cuda-tile-translate → tilebc
  local tilebc="$out_dir/output.tilebc"
  echo "  [3/4] cuda-tile-translate → tilebc"
  if ! "$CUDA_TILE_TRANSLATE" --mlir-to-cudatilebc --no-implicit-module \
      --bytecode-version=13.1 "$cuda_tile_mlir" -o "$tilebc" 2>&1; then
    echo "  FAIL: cuda-tile-translate failed"
    FAIL=$((FAIL+1))
    return
  fi
  echo "  OK: tilebc $(wc -c < "$tilebc") bytes"

  # Step 4: tileiras → cubin
  if $HAS_TILEIRAS; then
    local cubin="$out_dir/output.cubin"
    echo "  [4/4] tileiras → cubin"
    if ! "$TILEIRAS" --gpu-name "$GPU_NAME" "$tilebc" -o "$cubin" 2>&1; then
      echo "  FAIL: tileiras failed"
      FAIL=$((FAIL+1))
      return
    fi
    echo "  OK: cubin $(wc -c < "$cubin") bytes"
  else
    echo "  [4/4] SKIP (no tileiras)"
  fi

  echo "  PASS"
  PASS=$((PASS+1))
}

# --- Run test cases ---

# Test 1: 128x64 * 64x256 (original, default tiles)
run_test "128x256_default" \
  "$SCRIPT_DIR/linalg_to_cuda_tile_text.mlir" 64 64 32

# Test 2: 256x256 * 256x256 (square, default tiles)
run_test "256x256_default" \
  "$SCRIPT_DIR/matmul_256x256.mlir" 64 64 32

# Test 3: 64x32 * 32x64 (minimal, default tiles)
run_test "64x64_default" \
  "$SCRIPT_DIR/matmul_64x64.mlir" 64 64 32

# Test 4: 512x128 * 128x512 (larger, default tiles)
run_test "512x512_default" \
  "$SCRIPT_DIR/matmul_512x512.mlir" 64 64 32

# Test 5: 256x128 * 128x256 (custom tiles 128/128/64)
run_test "256x256_custom" \
  "$SCRIPT_DIR/matmul_custom_tiles.mlir" 128 128 64

echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo "========================================"

[[ $FAIL -eq 0 ]]
