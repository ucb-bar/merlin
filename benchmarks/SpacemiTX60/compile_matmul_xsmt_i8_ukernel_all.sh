#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_MLIR="${1:-${REPO_ROOT}/third_party/iree_bar/tests/e2e/SpacemiT/matmul_i8_2048.mlir}"

TARGET="spacemit_x60"
HW="RVV"
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
  --iree-compile-arg=--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b,+xsmtvdot \
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

python3 "${REPO_ROOT}/third_party/iree_bar/tests/e2e/SpacemiT/check_hotloop_asm.py" \
  --asm "${ASM_FILE}" \
  --require-opcode "smt\\.vmadot|\\.insn\\s+r\\s+(?:0x2b|43),\\s*(?:0x3|3),\\s*(?:0x71|113)"

echo "[OK] XSMT i8 ukernel artifacts validated at ${OUT_DIR}"
