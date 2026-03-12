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
OUT_TYPE="tensor<64x16xf32>"

case "${SYMBOL}" in
  npu_uk_matmul_*|npu_uk_gemma_mlp_*)
    LHS_TYPE="tensor<64x32xf8E4M3FN>"
    RHS_TYPE="tensor<32x16xf8E4M3FN>"
    ;;
  npu_uk_gemma_attention_*)
    LHS_TYPE="tensor<64x16xf8E4M3FN>"
    RHS_TYPE="tensor<16x16xf8E4M3FN>"
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
