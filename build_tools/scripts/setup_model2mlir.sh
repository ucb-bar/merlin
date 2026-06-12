#!/usr/bin/env bash
# Set up model2MLIR (m2m) and the smolVLA capture environment.
#
# m2m converts PyTorch models to standard linalg-on-tensors MLIR (weights
# externalized to safetensors). Merlin ingests those artifacts via
# merlin/python/merlin/frontends/. See docs/model2mlir.md.
#
# Layout (defaults; override via env):
#   MODEL2MLIR_DIR      model2MLIR checkout       (default: <repo>/../model2MLIR)
#   SMOLVLA_CAPTURE_DIR smolVLA capture venv dir  (default: <repo>/../smolvla_capture)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
M2M="${MODEL2MLIR_DIR:-$REPO_ROOT/../model2MLIR}"
CAP="${SMOLVLA_CAPTURE_DIR:-$REPO_ROOT/../smolvla_capture}"

if [ ! -d "$M2M" ]; then
  echo "model2MLIR not found at $M2M" >&2
  echo "clone it first: git clone https://github.com/ucb-bar/model2MLIR $M2M" >&2
  exit 1
fi

# 1. Core m2m env (torch + xdsl + torch-mlir nightly).
cd "$M2M"
uv venv --allow-existing
uv pip install -e .
uv pip install --pre torch-mlir \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels \
  || echo "warning: torch-mlir nightly unavailable; m2m falls back to FXImporter"

# 2. Dedicated smolVLA capture env (lerobot stack pinned per workloads/smolvla/).
mkdir -p "$CAP"
cd "$CAP"
uv venv --python 3.12 --allow-existing
uv pip install 'lerobot[smolvla]==0.5.1' xdsl structlog ml_dtypes
uv pip install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels \
  || echo "warning: torch-mlir nightly unavailable; m2m falls back to FXImporter"
uv pip install -e "$M2M" --no-deps

# 3. Smoke check: imports + the smolVLA artifacts Merlin's frontend consumes.
"$CAP/.venv/bin/python" - <<'PY'
import m2m
from pathlib import Path
art = Path(__file__).resolve()  # noqa
print("m2m import OK")
PY
for f in smolvla.mlir smolvla.safetensors.manifest.json; do
  test -f "$M2M/workloads/smolvla/$f" \
    && echo "artifact present: $f" \
    || echo "artifact missing: $f (run the capture; see docs/model2mlir.md)"
done

echo "OK. Regenerate smolVLA with:"
echo "  cd $CAP && .venv/bin/python $M2M/workloads/capture.py smolvla --formats fp32"
