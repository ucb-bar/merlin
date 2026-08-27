#!/usr/bin/env bash
# Capture a model2MLIR workload at FULL depth with REAL weights.
# capture.toml's [env] overrides the process env, and tiny_llama pins
# M2M_LLAMA_LAYERS=2 -- which the loader documents as a smoke path with RANDOM
# INIT. So the pin must be removed for the duration, then restored.
set -uo pipefail
M=/scratch/agustin/projects/model2MLIR
model="$1"; shift
fmts="${1:-fp32}"
toml="$M/workloads/$model/capture.toml"

restore() { cd "$M" && git checkout -- "workloads/$model/capture.toml" 2>/dev/null && echo "[restored $model capture.toml]"; }
trap restore EXIT INT TERM

if [ -f "$toml" ] && grep -q 'M2M_LLAMA_LAYERS' "$toml"; then
  grep -v 'M2M_LLAMA_LAYERS' "$toml" > "$toml.tmp" && mv "$toml.tmp" "$toml"
  echo "[$model] removed M2M_LLAMA_LAYERS pin -> full depth, pretrained weights"
fi
echo "[$model] capture.toml now:"; cat "$toml" 2>/dev/null

cd "$M"
export TMPDIR=/scratch/agustin/tmp HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
timeout 7200 "$M/.venv/bin/python" workloads/capture.py "$model" --formats "$fmts" 2>&1 | tail -40
echo "[$model] exit=$?"
ls -la "$M/workloads/$model/"*.mlir 2>/dev/null
