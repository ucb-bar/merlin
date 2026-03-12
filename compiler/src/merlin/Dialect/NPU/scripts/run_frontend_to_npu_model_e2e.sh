#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
BIN_DIR="${1:-${ROOT_DIR}/build/host-merlin-release/install/bin}"
INPUT_FILE="${2:-${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/Transforms/tests/post-global-opt-hook.mlir}"
OUT_DIR="${3:-${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/outputs/e2e}"
PARITY_FLAG="${4:-}"

IREE_COMPILE="${BIN_DIR}/iree-compile"
NPU_TRANSLATE="${BIN_DIR}/npu-translate"

if [[ ! -x "${IREE_COMPILE}" ]]; then
  echo "error: iree-compile not found at ${IREE_COMPILE}" >&2
  exit 1
fi
if [[ ! -x "${NPU_TRANSLATE}" ]]; then
  echo "error: npu-translate not found at ${NPU_TRANSLATE}" >&2
  exit 1
fi
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "error: input file not found at ${INPUT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
GLOBAL_MLIR="${OUT_DIR}/global_optimization.mlir"
ISA_TXT="${OUT_DIR}/program.isa.txt"

"${IREE_COMPILE}" "${INPUT_FILE}" \
  --iree-input-type=none \
  --iree-hal-target-backends=llvm-cpu \
  --compile-to=global-optimization \
  --mlir-print-op-generic \
  --iree-plugin=npu \
  --iree-npu-enable \
  --iree-npu-enable-ukernel-verify \
  --iree-npu-strict-ukernel-verify \
  --iree-npu-allow-unknown-ukernel-fallback=false \
  > "${GLOBAL_MLIR}"

"${NPU_TRANSLATE}" --allow-unregistered-dialect --mlir-to-npu-text-isa "${GLOBAL_MLIR}" > "${ISA_TXT}"

uv run python "${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py" "${ISA_TXT}"
uv run python "${ROOT_DIR}/third_party/npu_model/compiler/scripts/run_simulator_smoke.py" "${ISA_TXT}"

if [[ "${PARITY_FLAG}" == "--parity" ]]; then
  uv run python "${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/run_numerical_parity.py" "${ISA_TXT}"
fi

echo "wrote:"
echo "  ${GLOBAL_MLIR}"
echo "  ${ISA_TXT}"
