#!/usr/bin/env bash
# Compile {dronet, yolov8n, mlp_wide} × {scalar, RVV, OPU, Gemmini} VMFBs for
# the FireSim Shuttle bitstream (FireSimGemminiAndOPUShuttleConfig).
#
# Output layout:
#   build/compiled_models/<model>/firesim/<backend>/<model>.vmfb
#
# Usage:
#   benchmarks/firesim_shuttle/compile_all.sh
#   benchmarks/firesim_shuttle/compile_all.sh dronet         # one model
#   benchmarks/firesim_shuttle/compile_all.sh dronet opu     # one cell
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Optional filters (positional).
ONLY_MODEL="${1:-}"
ONLY_BACKEND="${2:-}"

declare -A MODEL_MLIR=(
  [mlp_wide]="${REPO_ROOT}/models/mlp_wide/mlp_wide.q.int8.mlir"
  [dronet]="${REPO_ROOT}/models/dronet/dronet.q.int8.mlir"
  [yolov8n]="${REPO_ROOT}/models/yolov8_nano/yolov8n.q.int8.mlir"
)

# Maps backend name -> "target hw" (default; OPU is overridden per-model).
declare -A BACKEND_FLAGS=(
  [scalar]="firesim_shuttle scalar"
  [rvv]="firesim_shuttle RVV"
  [opu]="firesim_shuttle OPU"
  [gemmini]="firesim_shuttle_gemmini Gemmini"
)

# 2026-05-24: pick OPU_IM2COL for CNN models so conv layers route through
# im2col→matmul→mmt4d ukernel (full OPU coverage). The bare OPU row
# (+xopu) targets matmul-only nets like mlp_wide. The prior bug was
# yolov8n × opu compiling with plain OPU → only 384 `.insn` ops on OPU,
# 27,632 dispatches dropped to RVV fallback.
pick_opu_hw() {
  case "$1" in
    dronet|yolov8n) echo "OPU_IM2COL" ;;
    *) echo "OPU" ;;
  esac
}

cd "${REPO_ROOT}"

for model in mlp_wide dronet yolov8n; do
  if [[ -n "${ONLY_MODEL}" && "${ONLY_MODEL}" != "${model}" ]]; then
    continue
  fi
  mlir="${MODEL_MLIR[${model}]}"
  if [[ ! -f "${mlir}" ]]; then
    echo "[SKIP] ${model}: source MLIR not found at ${mlir}" >&2
    continue
  fi

  for backend in scalar rvv opu gemmini; do
    if [[ -n "${ONLY_BACKEND}" && "${ONLY_BACKEND}" != "${backend}" ]]; then
      continue
    fi
    read -r target hw <<<"${BACKEND_FLAGS[${backend}]}"
    # Override OPU hw row per-model: CNN nets need OPU_IM2COL.
    if [[ "${backend}" == "opu" ]]; then
      hw="$(pick_opu_hw "${model}")"
    fi

    out_dir="${REPO_ROOT}/build/compiled_models/${model}/firesim/${backend}"
    mkdir -p "${out_dir}"

    echo
    echo "===================================================================="
    echo "[compile] model=${model} backend=${backend} target=${target} hw=${hw}"
    echo "          src=${mlir}"
    echo "          out=${out_dir}"
    echo "===================================================================="

    # The merlin compile wrapper picks build/host-merlin-release automatically
    # when plugin_flags are set (Gemmini), otherwise build/host-vanilla-release.
    # 2026-05-24: LOOP_WS tried + reverted. ELF shrinks 60% (yolov8n) but
    # dronet/yolov8n × gemmini × loop_ws crash in Zephyr worker startup
    # (mcause=1 Instruction Access fault, mepc=0) — the ELF layout LOOP_WS
    # emits breaks something in Zephyr's loader before any IREE dispatch
    # fires. mlp_wide × loop_ws works fine. Out of scope to chase further;
    # default per-tile RoCC path remains in use (already 100% routed to
    # Gemmini per audit).
    ./merlin compile "${mlir}" \
      --target "${target}" \
      --hw "${hw}" \
      --quantized \
      --output-dir "${out_dir}" \
      || { echo "[FAIL] ${model}/${backend}" >&2; continue; }

    # compile.py names the VMFB after the input basename (sans .mlir).
    # mlp_wide.q.int8.mlir -> mlp_wide.q.int8.vmfb
    src_basename="$(basename "${mlir}" .mlir)"
    src_vmfb="${out_dir}/${src_basename}.vmfb"
    dst_vmfb="${out_dir}/${model}.vmfb"
    if [[ -f "${src_vmfb}" ]]; then
      cp -f "${src_vmfb}" "${dst_vmfb}"
      size="$(stat -c%s "${dst_vmfb}")"
      echo "[OK]   ${model}/${backend} -> ${dst_vmfb} (${size} bytes)"
    else
      echo "[WARN] expected VMFB not found: ${src_vmfb}" >&2
    fi
  done
done

echo
echo "All compiles attempted. Artifact tree:"
find "${REPO_ROOT}/build/compiled_models" -name '*.vmfb' -path '*/firesim/*' \
  -printf '  %p (%s bytes)\n' 2>/dev/null | sort
