#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
BIN_DIR="${1:-${ROOT_DIR}/build/host-merlin-release/install/bin}"
SYMBOL="${2:-npu_uk_gemma_mlp_f8E4M3FN_f8E4M3FN_f32}"
OUT_FILE="${3:-${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/outputs/${SYMBOL}.txt}"
PARITY_FLAG="${4:-}"

"${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/emit_symbol_ukernel_isa.sh" \
  "${BIN_DIR}" "${SYMBOL}" "${OUT_FILE}"

uv run python "${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py" "${OUT_FILE}"
uv run python "${ROOT_DIR}/third_party/npu_model/compiler/scripts/run_simulator_smoke.py" "${OUT_FILE}"

if [[ "${PARITY_FLAG}" == "--parity" ]]; then
  uv run python "${ROOT_DIR}/compiler/src/merlin/Dialect/NPU/scripts/run_numerical_parity.py" "${OUT_FILE}"
fi
