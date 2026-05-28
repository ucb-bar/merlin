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

# Optional XPU-RT feedback overlay. The runner that orchestrates this
# script (tools/run_full_loop.py) sets MERLIN_DIR to the merlin output
# dir whose breakdowns/feedback.json should drive the next compile. If
# unset, the path stays standalone and compile.py's --with-feedback is
# omitted (additive-only invariant).
FEEDBACK_ARGS=()
if [[ -n "${MERLIN_DIR:-}" && -f "${MERLIN_DIR}/breakdowns/feedback.json" ]]; then
  FEEDBACK_ARGS+=(--with-feedback "${MERLIN_DIR}/breakdowns/feedback.json")
  echo "[feedback] using ${MERLIN_DIR}/breakdowns/feedback.json"
fi

conda run -n merlin-dev uv run tools/compile.py "${INPUT_MLIR}" \
  --build-dir host-merlin-release \
  --target "${TARGET}" \
  --hw "${HW}" \
  --dump-artifacts \
  --dump-phases \
  "${FEEDBACK_ARGS[@]}" \
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
