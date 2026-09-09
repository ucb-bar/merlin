#!/usr/bin/env bash
set -euo pipefail

control=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project=$(git -C "$control" rev-parse --show-toplevel)
launcher="$project/merlin/experiments/gemmini_perf_bench/scripts/launch_global_agent_experiment.py"
suite="$project/out/artifacts/perf-bench/gemmini/phase2_biasabi_pr_pq_pk_v7_20260906/suite.a31d57b28dbba98b4ec021e5c42d0473aac3ee9922c39dafccb874301f667c28.json"
candidate="$project/out/artifacts/perf-bench/gemmini/development_phase2_integrated_macro_v1_20260908/compiler"
objectives="$project/out/artifacts/perf-bench/gemmini/phase2_portfolio_bootstrap_v1_20260907/objectives"
output="$project/out/artifacts/perf-bench/gemmini/global_phase2_full_macro_four_model_v2_20260908"
expected_candidate_sha256=f8992bc35be76e805dc8dce9bfe5864167d50f0276da685306a2fb7d9dddfa36

cd "$project"
if [[ -e "$control/DISARMED" ]]; then
  echo "checkpoint is disarmed; remove $control/DISARMED only when an operator intends to launch" >&2
  exit 75
fi
actual_candidate_sha256=$(PYTHONPATH=merlin/python .venv/bin/python -c \
  'import sys; from pathlib import Path; from merlin.benchharness import hash_tree; print(hash_tree(Path(sys.argv[1]))["sha256"])' \
  "$candidate")
if [[ "$actual_candidate_sha256" != "$expected_candidate_sha256" ]]; then
  echo "candidate tree drifted: expected $expected_candidate_sha256, observed $actual_candidate_sha256" >&2
  exit 64
fi
if [[ -e "$output" ]]; then
  echo "fresh output path already exists: $output" >&2
  exit 64
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=merlin/python \
  .venv/bin/python "$launcher" \
  --campaign-config "$suite" \
  --candidate "$candidate" \
  --output "$output" \
  --external-objective "$objectives/resnet50_w8a8/external_objective.json" \
  --external-objective-sha256 206602694e3308087e4ee24aeb6a095be99c455a0698addd563e5c7ef77c768f \
  --portfolio-capsule SY_model_tiny_llama \
  --portfolio-external-objective "$objectives/lstmnetvit_w8a8/external_objective.json" \
  --portfolio-external-objective-sha256 5d015ddd1763c0270519ecea313b68ba5296359439096c450f7e70b772e3b26a \
  --portfolio-external-objective "$objectives/smolvla_denoise_step_w8a8/external_objective.json" \
  --portfolio-external-objective-sha256 62a649a1ca9fbc3731f56aef0605ce93e2911d603cc509cf5ebae5face82f17a \
  --portfolio-analysis-workers 1 \
  --iteration-seconds 600 \
  --round-seconds 600 \
  --max-rounds 6 \
  --total-authoring-seconds 3600 \
  --on-round-failure resume-last-checkpoint \
  --min-memory-available-gib 48 \
  --max-swap-used-gib 2 \
  --resource-sample-seconds 5 \
  --resource-trip-samples 2
