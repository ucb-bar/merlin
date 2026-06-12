#!/usr/bin/env bash
# Capture the full {model x datatype} consistent-bundle matrix for the RVV bring-up.
# Each (model,fmt) is captured INSIDE that model's own venv (deps differ per upstream),
# with the model's shrunk [env] from capture.toml, into oscar-merlin/output/<bundle>.
# Skips bundles that already have a model.mlir. Prints one __DONE__ line per combo.
set -u
M2M=/scratch/agustin/projects/model2MLIR
OUT=/scratch/agustin/projects/oscar-merlin/output
WORKER="$M2M/workloads/capture_consistent.py"

declare -A VENV=(
  [tiny_llama]=$M2M/.venv
  [small_llama]=$M2M/.venv
  [rdt]=$M2M/.venv
  [xr0]=/scratch/agustin/projects/xr0_capture/.venv
  [rdt2]=/scratch/agustin/projects/RDT2/.venv
  [groot_n1d7]=/scratch/agustin/projects/Isaac-GR00T/.venv
  [bitvla]=/scratch/agustin/projects/bitvla_capture/.venv
  [smolvla]=/scratch/agustin/projects/smolvla_capture/.venv
  [molmoact]=$M2M/workloads/molmoact/.venv
  [openvla]=/scratch/agustin/projects/openvla_capture/.venv
  [pi05]=/scratch/agustin/projects/openpi/.venv
)
declare -A ENVV=(
  [tiny_llama]="M2M_LLAMA_LAYERS=2"
  [small_llama]=""
  [rdt]="M2M_RDT_DEPTH=2"
  [xr0]=""
  [rdt2]="M2M_RDT2_DEPTH=2"
  [groot_n1d7]="M2M_GROOT_LAYERS=16"
  [bitvla]="BITVLA_LLM_LAYERS=2"
  [smolvla]=""
  [molmoact]="M2M_MOLMOACT_LAYERS=4 M2M_MOLMOACT_VOCAB=4096 M2M_SEQ=8"
  [openvla]="M2M_OPENVLA_LLM_LAYERS=2 M2M_OPENVLA_VIT_LAYERS=2 M2M_OPENVLA_VOCAB=512"
  [pi05]=""
)

# combos passed as args: "model:fmt model:fmt ..."
for combo in "$@"; do
  model="${combo%%:*}"; fmt="${combo##*:}"
  bundle="$OUT/${model}_${fmt}_consistent"
  if [ -f "$bundle/model.mlir" ]; then echo "__DONE__ $combo SKIP(exists)"; continue; fi
  py="${VENV[$model]}/bin/python"
  if [ ! -x "$py" ]; then echo "__DONE__ $combo FAIL(no venv $py)"; continue; fi
  line=$(cd "$M2M" && env ${ENVV[$model]} "$py" "$WORKER" "$model" "$fmt" "$bundle" 2>&1 \
         | grep -E "__BUNDLE_OK__" | tail -1)
  if [ -n "$line" ]; then echo "__DONE__ $combo OK ${line#__BUNDLE_OK__ }"
  else echo "__DONE__ $combo FAIL(no bundle)"; fi
done
echo "__ALL_CAPTURES_FINISHED__"
