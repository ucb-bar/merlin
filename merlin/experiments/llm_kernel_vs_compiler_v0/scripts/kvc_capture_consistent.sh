#!/usr/bin/env bash
# Emit a merlin recapture bundle (inputs + golden + MLIR + weights + extra + input_order)
# from ONE seeded model instance. The golden is only valid when captured in the same
# process as the MLIR it checks, so this must go through capture_consistent.py -- the
# plain capture.py output has no golden and cannot be paired with one after the fact.
#
# Same capture.toml trap as kvc_capture.sh: [env] beats the process env, and the
# tiny_llama pin means RANDOM INIT, so it is removed for the run and restored after.
set -uo pipefail
M=/scratch/agustin/projects/model2MLIR
model="$1"; fmt="${2:-fp32}"; out="$3"
toml="$M/workloads/$model/capture.toml"

restore() { cd "$M" && git checkout -- "workloads/$model/capture.toml" 2>/dev/null && echo "[restored $model capture.toml]"; }
trap restore EXIT INT TERM

if [ -f "$toml" ] && grep -q 'M2M_LLAMA_LAYERS' "$toml"; then
  grep -v 'M2M_LLAMA_LAYERS' "$toml" > "$toml.tmp" && mv "$toml.tmp" "$toml"
  echo "[$model] removed M2M_LLAMA_LAYERS pin -> full depth, pretrained weights"
fi

cd "$M"
export TMPDIR=/scratch/agustin/tmp HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
timeout 10800 "$M/.venv/bin/python" workloads/capture_consistent.py "$model" "$fmt" "$out" 2>&1 | tail -30
rc=$?
echo "[$model/$fmt] consistent-capture exit=$rc -> $out"
ls -la "$out" 2>/dev/null
