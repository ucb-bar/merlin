#!/usr/bin/env bash
# compile_audit.sh — compile every (model, hw) cell for the FireSim Shuttle
# bitstream with full artifact dumping, then run audit_compile.py to
# classify per-dispatch backend routing + opcode coverage.
#
# Cells generated:
#   mlp_wide        × {scalar, rvv, opu, gemmini, gemmini_mx}
#   dronet          × {scalar, rvv, opu, gemmini, gemmini_mx}
#   yolov8n         × {scalar, rvv, opu, gemmini, gemmini_mx}
#
# Total = 15 compile cells.
#
# im2col preprocessing for dronet/yolov8_nano is baked into the YAMLs
# (models/firesim_shuttle{,_gemmini}.yaml `models:` block), mirroring
# models/saturn_opu.yaml. No separate `_im2col` cells are emitted here;
# if the bare-metal verifier bug re-trips, the per-cell rc != 0 captures
# it cleanly.
#
# Output:
#   build/compile_audit/<model>_<hw>/
#     ├── <model>.q.int8.vmfb
#     ├── phases/    (12 phase MLIRs from --dump-compilation-phases-to)
#     ├── sources/   (per-dispatch source)
#     ├── configs/   (per-dispatch with translation_info)
#     ├── benchmarks/ (IREE benchmark wrappers)
#     ├── binaries/  (linked dispatch ELF .so)
#     └── files/     (auxiliary)
#   tmp/firesim_shuttle_compile_audit.csv  (per-cell rc + wall)
#   tmp/firesim_shuttle_compile_audit.md   (rendered table; written by python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTDIR_ROOT="${REPO_ROOT}/build/compile_audit"
RESULTS_CSV="${REPO_ROOT}/tmp/firesim_shuttle_compile_audit.csv"
mkdir -p "${OUTDIR_ROOT}" "${REPO_ROOT}/tmp"

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "ts,model,hw,im2col,output_dir,vmfb_bytes,rc,wallclock_s" > "${RESULTS_CSV}"
fi

MODELS=(mlp_wide dronet yolov8n)
# FireSimGemminiAndOPUShuttleConfig has vanilla Gemmini (NOT mxGemmini) on
# hart 0 and Saturn OPU on hart 1. We keep gemmini_mx out of the default
# cell set because the bitstream's Gemmini does not implement the MX
# extensions (k_CONFIG_SCALE_MEM funct7=26 etc.). To audit gemmini_mx
# anyway: run `BASE_HW="gemmini_mx" ./compile_audit.sh`.
BASE_HW=(scalar rvv opu gemmini)

# Map hw → (target, hw_row) since gemmini variants live in a separate YAML.
hw_target_yaml() {
  case "$1" in
    scalar|rvv|opu) echo "firesim_shuttle" ;;
    gemmini)        echo "firesim_shuttle_gemmini" ;;
    gemmini_mx)     echo "firesim_shuttle_gemmini" ;;
    *) echo "UNKNOWN" ;;
  esac
}
# hw_row maps the cell-tag ("opu", "gemmini", ...) to the yaml's
# `targets` key, optionally specialised per model. For CNN models
# (dronet, yolov8n) the OPU cell uses OPU_IM2COL — the saturn-proven
# variant that drops +xopu so the compile-time encoding resolver does
# not trip on im2col-produced matmul shapes. OPU custom opcodes still
# get emitted via the linked mmt4d ukernel bitcode. mlp_wide is
# matmul-only and uses plain OPU.
hw_row() {
  local hw="$1" model="${2:-}"
  case "$hw" in
    scalar) echo "scalar" ;;
    rvv)    echo "RVV" ;;
    opu)
      case "$model" in
        dronet|yolov8n) echo "OPU_IM2COL" ;;
        *)              echo "OPU" ;;
      esac
      ;;
    gemmini) echo "Gemmini" ;;
    gemmini_mx) echo "GemminiMX" ;;
  esac
}

model_mlir() {
  case "$1" in
    mlp_wide) echo "${REPO_ROOT}/models/mlp_wide/mlp_wide.q.int8.mlir" ;;
    dronet)   echo "${REPO_ROOT}/models/dronet/dronet.q.int8.mlir" ;;
    yolov8n)  echo "${REPO_ROOT}/models/yolov8_nano/yolov8n.q.int8.mlir" ;;
  esac
}

compile_cell() {
  local model="$1" hw="$2"
  local target; target="$(hw_target_yaml "$hw")"
  local row; row="$(hw_row "$hw" "$model")"
  local tag="${model}_${hw}"

  local outdir="${OUTDIR_ROOT}/${tag}"
  local mlir; mlir="$(model_mlir "$model")"
  if [[ ! -f "$mlir" ]]; then
    echo "[skip] $tag: missing MLIR $mlir"
    return 0
  fi

  echo
  echo "================================================================="
  echo "[$(date +%H:%M:%S)] $tag  ($target / $row)"
  echo "================================================================="

  rm -rf "$outdir"
  mkdir -p "$outdir"

  # Per-model preprocessing (e.g. conv→im2col for dronet/yolov8_nano) is
  # baked into the YAML's `models:` block; we don't add extra CLI flags here.
  local t0; t0=$(date +%s)
  local rc=0
  (./merlin compile "$mlir" \
      --target "$target" --hw "$row" --quantized \
      --output-dir "$outdir" \
      --dump-artifacts --dump-phases \
  ) >"${outdir}/compile.log" 2>&1 || rc=$?
  local wall=$(( $(date +%s) - t0 ))
  local vmfb_bytes=0
  local vmfb_path
  vmfb_path="$(find "$outdir" -maxdepth 1 -name '*.vmfb' | head -1)"
  if [[ -f "$vmfb_path" ]]; then vmfb_bytes="$(stat -c%s "$vmfb_path")"; fi

  printf '%s,%s,%s,%s,%s,%d,%d,%d\n' \
    "$(date -Iseconds)" "$model" "$hw" "0" "$outdir" \
    "$vmfb_bytes" "$rc" "$wall" >> "$RESULTS_CSV"

  if [[ $rc -ne 0 ]]; then
    echo "[fail] $tag rc=$rc wall=${wall}s (see $outdir/compile.log)"
  else
    echo "[ok]   $tag rc=0 wall=${wall}s vmfb=${vmfb_bytes}B"
  fi
}

# Filter args: optional --only=<model>[,<model>] or --skip-mx
ONLY_MODEL="${ONLY_MODEL:-}"
for arg in "$@"; do
  case "$arg" in
    --only=*) ONLY_MODEL="${arg#--only=}" ;;
    --help|-h)
      echo "Usage: $0 [--only=<model>[,<model>]] [model...]"
      echo "  models: mlp_wide dronet yolov8n"
      exit 0
      ;;
    *) ONLY_MODEL="${ONLY_MODEL}${ONLY_MODEL:+,}${arg}" ;;
  esac
done

for m in "${MODELS[@]}"; do
  if [[ -n "$ONLY_MODEL" && ",$ONLY_MODEL," != *",$m,"* ]]; then continue; fi
  for hw in "${BASE_HW[@]}"; do
    compile_cell "$m" "$hw"
  done
done

echo
echo "================================================================="
echo "compile audit DONE. CSV: $RESULTS_CSV"
echo "Now run: uv run tools/audit_compile.py --root $OUTDIR_ROOT"
echo "================================================================="
