#!/usr/bin/env bash
# run_hetero.sh — parallel async heterogeneous run on FireSim Shuttle.
#
# Builds the Zephyr `merlin_hetero_runner` sample (sibling to
# merlin_model_runner) which embeds two VMFBs of the same model (one
# compiled for hart 0's ISA, one for hart 1's) and dispatches N inferences
# round-robin across both pinned worker threads.
#
# Per-job init is amortized: each hart creates its iree_vm_context and
# hal_device ONCE at startup, then loops on a k_msgq of (job_id,
# completion_sem). Per-job steady-state cost ~= invoke + d2h + hash.
#
# Usage:
#   run_hetero.sh [model] [backend_h0] [backend_h1] [jobs]
#
# Defaults: model=mlp_wide, hart0=scalar, hart1=rvv, jobs=32

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:-mlp_wide}"
BACKEND_H0="${2:-scalar}"
BACKEND_H1="${3:-rvv}"
JOBS="${4:-32}"

: "${CHIPYARD_ROOT:?CHIPYARD_ROOT not set}"
: "${ZEPHYR_BASE:?ZEPHYR_BASE not set}"
: "${ZEPHYR_BUILDS_ROOT:=/scratch2/agustin/zephyr-builds}"

DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
WORKLOADS_DIR="${DEPLOY_DIR}/workloads"
RUNTIME_YAML="${DEPLOY_DIR}/config_runtime.yaml"
ZEPHYR_SAMPLE_DIR="/scratch2/agustin/zephyr-chipyard-sw/samples/merlin_hetero_runner"

if [[ ! -d "${ZEPHYR_SAMPLE_DIR}" ]]; then
  echo "[fatal] sample missing: ${ZEPHYR_SAMPLE_DIR}" >&2; exit 1
fi
if [[ ! -f "${RUNTIME_YAML}" ]]; then
  echo "[fatal] firesim runtime yaml missing: ${RUNTIME_YAML}" >&2; exit 1
fi
mkdir -p "${WORKLOADS_DIR}" "${ZEPHYR_BUILDS_ROOT}" "${REPO_ROOT}/tmp"

# ---- firesim env ----
if ! command -v firesim >/dev/null 2>&1; then
  : "${CHIPYARD_CONDA_ENV:=${CHIPYARD_ROOT}/.conda-env}"
  : "${CHIPYARD_CONDA_PROFILE:=/scratch2/agustin/miniforge3/etc/profile.d/conda.sh}"
  : "${FIRESIM_SOURCEME:=${CHIPYARD_ROOT}/sims/firesim/sourceme-manager.sh}"
  set +u
  # shellcheck disable=SC1090,SC1091
  source "${CHIPYARD_CONDA_PROFILE}"
  conda activate "${CHIPYARD_CONDA_ENV}"
  pushd "${CHIPYARD_ROOT}/sims/firesim" >/dev/null
  if [[ -z "${SSH_AUTH_SOCK:-}" || ! -S "${SSH_AUTH_SOCK}" ]]; then
    eval "$(ssh-agent -s)" >/dev/null
  fi
  set --
  # shellcheck disable=SC1090,SC1091
  source "${FIRESIM_SOURCEME}" --skip-ssh-setup
  popd >/dev/null
  set -u
fi

# ---- VMFB paths ----
VMFB_H0="${REPO_ROOT}/build/compiled_models/${MODEL}/firesim/${BACKEND_H0}/${MODEL}.vmfb"
VMFB_H1="${REPO_ROOT}/build/compiled_models/${MODEL}/firesim/${BACKEND_H1}/${MODEL}.vmfb"
for v in "${VMFB_H0}" "${VMFB_H1}"; do
  if [[ ! -f "$v" ]]; then
    echo "[fatal] missing VMFB: $v" >&2
    echo "        Compile with ./merlin compile models/${MODEL}/${MODEL}.q.int8.mlir \\"
    echo "          --target firesim_shuttle --hw <${BACKEND_H0}|${BACKEND_H1}> \\"
    echo "          --quantized --output-dir build/compiled_models/${MODEL}/firesim/<hw>"
    exit 1
  fi
done

tag="hetero_${MODEL}_${BACKEND_H0}_${BACKEND_H1}_n${JOBS}"
workload_name="merlin-shuttle-${tag}"
workload_dir="${WORKLOADS_DIR}/${workload_name}"
workload_json="${WORKLOADS_DIR}/${workload_name}.json"
build_dir="${ZEPHYR_BUILDS_ROOT}/${tag}"
cell_log="${REPO_ROOT}/tmp/firesim_shuttle_${tag}.log"

ORIG_WORKLOAD_LINE="$(grep -E '^[[:space:]]*workload_name:' "${RUNTIME_YAML}" | head -1)"
echo "[env] preserving original ${ORIG_WORKLOAD_LINE# }"
cleanup() {
  if [[ -n "${ORIG_WORKLOAD_LINE:-}" ]]; then
    sed -i -E "s|^[[:space:]]*workload_name:.*|${ORIG_WORKLOAD_LINE}|" "${RUNTIME_YAML}" || true
  fi
  ( cd "${DEPLOY_DIR}" && firesim kill ) >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo
echo "===================================================================="
echo "[$(date +%H:%M:%S)] HETERO ${MODEL} h0=${BACKEND_H0} h1=${BACKEND_H1} jobs=${JOBS}"
echo "===================================================================="
echo "[build] west build -> ${build_dir}"
rm -rf "${build_dir}"
mkdir -p "${build_dir}"

build_rc=0
( west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 \
    "${ZEPHYR_SAMPLE_DIR}" \
    --build-dir "${build_dir}" \
    -- -DZEPHYR_EXTRA_MODULES="${CHIPYARD_ROOT}/software/zephyrproject/modules/merlin-iree" \
       -DMERLIN_BUILD_DIR="${REPO_ROOT}/build/zephyr-vanilla-release" \
       -DMERLIN_IREE_HEADERS_DIR="${REPO_ROOT}/third_party/iree_bar/runtime/src" \
       -DMERLIN_VMFB_H0="${VMFB_H0}" \
       -DMERLIN_VMFB_H1="${VMFB_H1}" \
       -DMERLIN_MODEL="${MODEL}" \
       -DMERLIN_BACKEND_H0="${BACKEND_H0}" \
       -DMERLIN_BACKEND_H1="${BACKEND_H1}" \
       -DMERLIN_JOBS="${JOBS}" \
       -DMERLIN_CPU_FEATURES="0x11" \
) >>"${cell_log}" 2>&1 || build_rc=$?

if [[ ${build_rc} -ne 0 ]]; then
  echo "[fail] west build returned ${build_rc} (see ${cell_log})" >&2
  exit 1
fi
elf="${build_dir}/zephyr/zephyr.elf"
echo "[build] OK -> ${elf} ($(stat -c%s "${elf}") bytes)"

# ---- stage workload ----
rm -rf "${workload_dir}"
mkdir -p "${workload_dir}"
elf_name="${workload_name}.elf"
cp "${elf}" "${workload_dir}/${elf_name}"
cat > "${workload_json}" <<JSONEOF
{
  "benchmark_name": "${workload_name}",
  "common_bootbinary": "${elf_name}",
  "common_rootfs": "../../../../../software/firemarshal/boards/default/installers/firesim/dummy.rootfs",
  "common_simulation_outputs": ["uartlog"]
}
JSONEOF

# ---- toggle workload_name in config_runtime.yaml ----
sed -i -E "s|^[[:space:]]*workload_name:.*|  workload_name: ${workload_name}.json|" "${RUNTIME_YAML}"

# ---- firesim infrasetup + runworkload + kill ----
echo "[fsim]  firesim infrasetup"
( cd "${DEPLOY_DIR}" && firesim kill ) >/dev/null 2>&1 || true
( cd "${DEPLOY_DIR}" && firesim infrasetup ) >>"${cell_log}" 2>&1
echo "[fsim]  firesim runworkload"
( cd "${DEPLOY_DIR}" && firesim runworkload ) >>"${cell_log}" 2>&1 || true
echo "[fsim]  firesim kill"
( cd "${DEPLOY_DIR}" && firesim kill ) >>"${cell_log}" 2>&1 || true

# ---- locate uartlog ----
UART="$(ls -dt "${DEPLOY_DIR}/results-workload/"*"${workload_name}" 2>/dev/null \
        | head -1)/${workload_name}0/uartlog"
if [[ ! -f "${UART}" ]]; then
  echo "[fail] uartlog not found (expected ${UART})" >&2
  exit 1
fi
echo "[uart] ${UART}"
echo "[done] hetero run complete; parsing..."
echo
echo "=== summary lines from uartlog ==="
grep -E "\[hetero\] (jobs_ok|hart0|hart1|parallel_wall|init_h|both harts|submitting)" "${UART}" || true
echo
echo "=== full hetero log (uartlog ${UART}) ==="
