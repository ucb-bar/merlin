#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
BIN_DIR="${1:-${ROOT_DIR}/build/host-merlin-release/install/bin}"
SYMBOL="${2:-}"
OUT_FILE="${3:-${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/outputs/npu_symbol_ukernel_isa.txt}"

if [[ -z "${SYMBOL}" ]]; then
  echo "usage: $0 <bin-dir> <ukernel-symbol> [out-file]" >&2
  exit 1
fi

IREE_OPT="${BIN_DIR}/iree-opt"
NPU_TRANSLATE="${BIN_DIR}/npu-translate"

if [[ ! -x "${IREE_OPT}" ]]; then
  echo "error: iree-opt not found at ${IREE_OPT}" >&2
  exit 1
fi
if [[ ! -x "${NPU_TRANSLATE}" ]]; then
  echo "error: npu-translate not found at ${NPU_TRANSLATE}" >&2
  exit 1
fi

LHS_TYPE=""
RHS_TYPE=""
OUT_TYPE=""

to_mlir_elem_type() {
  case "$1" in
    f8E4M3FN) echo "f8E4M3FN" ;;
    bf16) echo "bf16" ;;
    f32) echo "f32" ;;
    i8) echo "i8" ;;
    i32) echo "i32" ;;
    *)
      echo "error: unsupported element type token '$1' in symbol '${SYMBOL}'" >&2
      return 1
      ;;
  esac
}

case "${SYMBOL}" in
  npu_uk_matmul_*)
    SUFFIX="${SYMBOL#npu_uk_matmul_}"
    IFS='_' read -r LHS_ELEM RHS_ELEM OUT_ELEM <<< "${SUFFIX}"
    if [[ -z "${LHS_ELEM:-}" || -z "${RHS_ELEM:-}" || -z "${OUT_ELEM:-}" ]]; then
      echo "error: expected npu_uk_matmul_<lhs>_<rhs>_<out>, got '${SYMBOL}'" >&2
      exit 1
    fi
    LHS_ELEM="$(to_mlir_elem_type "${LHS_ELEM}")"
    RHS_ELEM="$(to_mlir_elem_type "${RHS_ELEM}")"
    OUT_ELEM="$(to_mlir_elem_type "${OUT_ELEM}")"
    LHS_TYPE="tensor<64x32x${LHS_ELEM}>"
    RHS_TYPE="tensor<32x16x${RHS_ELEM}>"
    OUT_TYPE="tensor<64x16x${OUT_ELEM}>"
    ;;
  npu_uk_gemma_mlp_*)
    SUFFIX="${SYMBOL#npu_uk_gemma_mlp_}"
    IFS='_' read -r LHS_ELEM RHS_ELEM OUT_ELEM <<< "${SUFFIX}"
    if [[ -z "${LHS_ELEM:-}" || -z "${RHS_ELEM:-}" || -z "${OUT_ELEM:-}" ]]; then
      echo "error: expected npu_uk_gemma_mlp_<lhs>_<rhs>_<out>, got '${SYMBOL}'" >&2
      exit 1
    fi
    LHS_ELEM="$(to_mlir_elem_type "${LHS_ELEM}")"
    RHS_ELEM="$(to_mlir_elem_type "${RHS_ELEM}")"
    OUT_ELEM="$(to_mlir_elem_type "${OUT_ELEM}")"
    LHS_TYPE="tensor<64x32x${LHS_ELEM}>"
    RHS_TYPE="tensor<32x16x${RHS_ELEM}>"
    OUT_TYPE="tensor<64x16x${OUT_ELEM}>"
    ;;
  npu_uk_gemma_attention_*)
    SUFFIX="${SYMBOL#npu_uk_gemma_attention_}"
    IFS='_' read -r Q_ELEM OUT_ELEM <<< "${SUFFIX}"
    if [[ -z "${Q_ELEM:-}" || -z "${OUT_ELEM:-}" ]]; then
      echo "error: expected npu_uk_gemma_attention_<q>_<out>, got '${SYMBOL}'" >&2
      exit 1
    fi
    Q_ELEM="$(to_mlir_elem_type "${Q_ELEM}")"
    OUT_ELEM="$(to_mlir_elem_type "${OUT_ELEM}")"
    LHS_TYPE="tensor<64x16x${Q_ELEM}>"
    RHS_TYPE="tensor<16x16x${Q_ELEM}>"
    OUT_TYPE="tensor<64x16x${OUT_ELEM}>"
    ;;
  *)
    echo "error: unsupported symbol '${SYMBOL}'" >&2
    echo "supported prefixes: npu_uk_matmul_, npu_uk_gemma_mlp_, npu_uk_gemma_attention_" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "${OUT_FILE}")"

TMP_INPUT="$(mktemp /tmp/npu_symbol_ukernel_XXXXXX.mlir)"
trap 'rm -f "${TMP_INPUT}"' EXIT

cat > "${TMP_INPUT}" <<MLIR
module {
  func.func @entry(%lhs: ${LHS_TYPE}, %rhs: ${RHS_TYPE}) -> ${OUT_TYPE} {
    %0 = npu_kernel.ukernel_generic "${SYMBOL}"(%lhs, %rhs) : ${LHS_TYPE}, ${RHS_TYPE} -> ${OUT_TYPE}
    return %0 : ${OUT_TYPE}
  }
}
MLIR

"${IREE_OPT}" "${TMP_INPUT}" \
  --iree-plugin=npu \
  --pass-pipeline='builtin.module(convert-npu-kernel-to-schedule,convert-npu-schedule-to-isa)' \
  | "${NPU_TRANSLATE}" --mlir-to-npu-text-isa > "${OUT_FILE}"

echo "wrote ${OUT_FILE}"
