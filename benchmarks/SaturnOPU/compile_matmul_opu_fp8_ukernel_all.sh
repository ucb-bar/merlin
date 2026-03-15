#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_MLIR="${1:-${REPO_ROOT}/third_party/iree_bar/tests/e2e/SpacemiT/matmul_fp8_2048.mlir}"

TARGET="saturn_opu"
HW="OPU"
BASENAME="$(basename "${INPUT_MLIR%.*}")"
MODEL_NAME="$(basename "$(dirname "${INPUT_MLIR}")")"
OUT_DIR="${REPO_ROOT}/build/compiled_models/${MODEL_NAME}/${TARGET}_${HW}_${BASENAME}"

cd "${REPO_ROOT}"

conda run -n merlin-dev uv run tools/compile.py "${INPUT_MLIR}" \
  --build-dir host-merlin-release \
  --target "${TARGET}" \
  --hw "${HW}" \
  --dump-artifacts \
  --dump-phases \
  --iree-compile-arg=--iree-llvmcpu-enable-ukernels=all \
  --iree-compile-arg=--iree-llvmcpu-link-ukernel-bitcode=true \
  --iree-compile-arg=--iree-llvmcpu-enable-vector-contract-custom-kernels=false \
  --iree-compile-arg=--iree-opt-data-tiling=true \
  --iree-compile-arg=--iree-dispatch-creation-data-tiling=true

ASM_FILE="$(find "${OUT_DIR}" -type f -name '*.s' | head -n 1)"
if [[ -z "${ASM_FILE}" ]]; then
  echo "[FAIL] no .s file found under ${OUT_DIR}" >&2
  exit 1
fi

CONFIG_FILE="$(find "${OUT_DIR}/configs" -type f -name 'configured_module_*dispatch_0.mlir' | head -n 1)"
if [[ -z "${CONFIG_FILE}" ]]; then
  echo "[FAIL] no configured dispatch MLIR found under ${OUT_DIR}/configs" >&2
  exit 1
fi

if ! grep -E "linalg\\.mmt4d" "${CONFIG_FILE}" >/dev/null; then
  echo "[FAIL] expected mmt4d lowering in ${CONFIG_FILE}" >&2
  exit 1
fi

echo "[OK] OPU fp8 artifacts validated at ${OUT_DIR}"
