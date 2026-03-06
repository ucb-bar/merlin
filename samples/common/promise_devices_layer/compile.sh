#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================
# The input file containing the High-Level Schedule (flow.tensor.transfer + linalg)
INPUT_FILE="input.mlir"

# The output compiled binary
OUTPUT_FILE="static_schedule.vmfb"
DUMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/phases_input"

# ==============================================================================
# Compilation
# ==============================================================================
echo "Compiling ${INPUT_FILE}..."

iree-compile "${INPUT_FILE}" \
  --iree-execution-model=async-external \
  --iree-hal-target-device=device_a=local[0] \
  --iree-hal-target-device=device_b=local[1] \
  --iree-hal-target-device=device_ab=local[2] \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --dump-compilation-phases-to="${DUMP_DIR}" \
  -o "${OUTPUT_FILE}"
  #--compile-from=stream

echo "Build complete: ${OUTPUT_FILE}"
