#!/usr/bin/env bash
# Regenerate the workload-contract case-study package from the committed real captures.
# Deterministic: reads merlin/benchmarks/dse_guidance/recaptures/<model>/model.mlir (committed;
# weights not needed) + measured_*.yaml + accuracy_gate.yaml. No hardware, no network.
#
# Usage:  bash merlin/benchmarks/dse_guidance/reproduce_case_study.sh [OUT_DIR]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-out/artifacts/dse-guidance/case_study_regen}"
PY="${MERLIN_PY:-.venv/bin/python}"

echo "[1/2] regenerating case study -> ${OUT}"
"$PY" -m merlin.dse_guidance.cli --case-study --out "$OUT"

echo "[2/2] guidance tests"
"$PY" -m pytest merlin/tests/dse/test_dse_guidance.py -q

echo "done. Compare against the committed reference:"
echo "  diff -r out/artifacts/dse-guidance/case_study ${OUT}"
