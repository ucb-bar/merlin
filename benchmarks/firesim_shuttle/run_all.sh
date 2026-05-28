#!/usr/bin/env bash
# End-to-end FireSim Shuttle driver: compile -> build Zephyr ELF -> stage as
# bare-metal-style FireSim workload -> firesim infrasetup/runworkload/kill,
# looping every (model, backend) cell.
#
# Patterned after build_tools/hardware/scripts/run_baremetal_benchmarks.sh
# (the SaturnOPU bare-metal FireSim flow):
#   - per-cell workload JSON written inline (no `merlin chipyard` indirection)
#   - per-cell `workload_name:` toggle via sed on config_runtime.yaml
#     (restored on exit so the deploy config is left as it was found)
#   - `firesim kill` after each runworkload, plus a trap EXIT kill so a
#     Ctrl-C or hard failure still releases the U250 lease
#   - results-workload/*<name>*/<name>0/uartlog discovery
#
# Usage:
#   benchmarks/firesim_shuttle/run_all.sh                    # all 12 cells
#   benchmarks/firesim_shuttle/run_all.sh mlp_wide           # one model
#   benchmarks/firesim_shuttle/run_all.sh mlp_wide scalar    # one cell
#
# Per-cell rows land in tmp/firesim_shuttle_results.csv. Per-cell uartlogs
# land in tmp/firesim_shuttle_<model>_<backend>.uartlog.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ONLY_MODEL="${1:-}"
ONLY_BACKEND="${2:-}"

# ------- env requirements ------------------------------------------------
: "${CHIPYARD_ROOT:?CHIPYARD_ROOT not set}"
: "${ZEPHYR_BASE:?ZEPHYR_BASE not set (Zephyr SDK env, see dev blog)}"
: "${ZEPHYR_BUILDS_ROOT:=/scratch2/agustin/zephyr-builds}"

DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
WORKLOADS_DIR="${DEPLOY_DIR}/workloads"
RUNTIME_YAML="${DEPLOY_DIR}/config_runtime.yaml"

if [[ ! -d "${DEPLOY_DIR}" ]]; then
  echo "[fatal] FireSim deploy dir missing: ${DEPLOY_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUNTIME_YAML}" ]]; then
  echo "[fatal] FireSim config_runtime.yaml missing: ${RUNTIME_YAML}" >&2
  exit 1
fi
mkdir -p "${WORKLOADS_DIR}" "${ZEPHYR_BUILDS_ROOT}" "${REPO_ROOT}/tmp"

# ------- firesim environment (conditional sourcing) ---------------------
# The SaturnOPU bare-metal flow expected callers to source the firesim env
# manually. We do the same when `firesim` is already on PATH; otherwise we
# source it ourselves so the script is still self-sufficient under cron /
# fresh shells. Overrideable via env:
#   CHIPYARD_CONDA_ENV       — chipyard's conda env  (default: $CHIPYARD_ROOT/.conda-env)
#   CHIPYARD_CONDA_PROFILE   — miniforge conda.sh   (default: /scratch2/agustin/miniforge3/etc/profile.d/conda.sh)
#   FIRESIM_SOURCEME         — sourceme-manager.sh  (default: $CHIPYARD_ROOT/sims/firesim/sourceme-manager.sh)
if ! command -v firesim >/dev/null 2>&1; then
  : "${CHIPYARD_CONDA_ENV:=${CHIPYARD_ROOT}/.conda-env}"
  : "${CHIPYARD_CONDA_PROFILE:=/scratch2/agustin/miniforge3/etc/profile.d/conda.sh}"
  : "${FIRESIM_SOURCEME:=${CHIPYARD_ROOT}/sims/firesim/sourceme-manager.sh}"
  echo "[env] firesim not on PATH; sourcing chipyard conda + sourceme-manager"
  # Chipyard's conda-activate scripts (`activate-riscv-tools.sh`) reference
  # unset env vars (e.g. `RISCV`), tripping `set -u`. Temporarily relax
  # nounset around the sourcing block.
  set +u
  # shellcheck disable=SC1090,SC1091
  source "${CHIPYARD_CONDA_PROFILE}"
  conda activate "${CHIPYARD_CONDA_ENV}"
  pushd "${CHIPYARD_ROOT}/sims/firesim" >/dev/null
  # Make sure ssh-agent is alive in this shell. Fabric (firesim's deploy
  # backend) requires it for the ssh-to-localhost it does even on local-FPGA
  # runs. If SSH_AUTH_SOCK isn't set or its socket is dead, start a fresh
  # agent; otherwise reuse it.
  if [[ -z "${SSH_AUTH_SOCK:-}" || ! -S "${SSH_AUTH_SOCK}" ]]; then
    echo "[env] starting fresh ssh-agent (SSH_AUTH_SOCK was unset/stale)"
    eval "$(ssh-agent -s)" >/dev/null
  fi
  # Ensure the firesim key is loaded so fabric can ssh-to-localhost for
  # instance_liveness/kill. Idempotent — ssh-add silently no-ops if already
  # added. Key location matches the user's ~/.ssh/firesim convention.
  if [[ -f "${HOME}/.ssh/firesim" ]] && ! ssh-add -l 2>/dev/null | grep -q firesim; then
    ssh-add "${HOME}/.ssh/firesim" 2>/dev/null || true
  fi
  # `set --` clears positional args so sourceme-manager.sh doesn't try to
  # interpret our script's $1=model $2=backend as its own flags.
  set --
  # shellcheck disable=SC1090,SC1091
  source "${FIRESIM_SOURCEME}" --skip-ssh-setup
  popd >/dev/null
  set -u
  command -v firesim >/dev/null 2>&1 || {
    echo "[fatal] sourced firesim env but 'firesim' still not on PATH" >&2
    exit 1
  }
fi
echo "[env] firesim: $(command -v firesim)"

# ------- restore-on-exit posture ----------------------------------------
# Save the original workload_name from config_runtime.yaml so we can
# restore it after the sweep finishes (or aborts). This mirrors the
# SaturnOPU bare-metal script's behavior.
ORIG_WORKLOAD_LINE="$(grep -E '^[[:space:]]*workload_name:' "${RUNTIME_YAML}" | head -1)"
if [[ -z "${ORIG_WORKLOAD_LINE}" ]]; then
  echo "[fatal] could not find workload_name: line in ${RUNTIME_YAML}" >&2
  exit 1
fi
echo "[env] preserving original ${ORIG_WORKLOAD_LINE# }"

cleanup() {
  # Restore original workload_name (full line including its indentation
  # is captured in ORIG_WORKLOAD_LINE; the regex matches and replaces any
  # workload_name line with that exact captured text).
  if [[ -n "${ORIG_WORKLOAD_LINE:-}" ]]; then
    # The replacement contains the original indent already; the pattern
    # matches any indentation but replaces with the captured full line.
    sed -i -E "s|^[[:space:]]*workload_name:.*|${ORIG_WORKLOAD_LINE}|" "${RUNTIME_YAML}" || true
  fi
  ( cd "${DEPLOY_DIR}" && firesim kill ) >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ------- per-cell tables ------------------------------------------------
declare -A BACKEND_HART=(
  [scalar]=0
  [rvv]=1
  [opu]=1
  [gemmini]=0
)
declare -A BACKEND_FEATURES=(
  [scalar]=0x00
  [rvv]=0x01
  [opu]=0x11
  [gemmini]=0x01
)
MODELS=(mlp_wide dronet yolov8n)
BACKENDS=(scalar rvv opu gemmini)

ZEPHYR_SAMPLE_DIR="/scratch2/agustin/zephyr-chipyard-sw/samples/merlin_model_runner"
if [[ ! -d "${ZEPHYR_SAMPLE_DIR}" ]]; then
  echo "[fatal] Zephyr sample missing: ${ZEPHYR_SAMPLE_DIR}" >&2
  exit 1
fi

RESULTS_CSV="${REPO_ROOT}/tmp/firesim_shuttle_results.csv"
if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "ts,model,backend,hart,vmfb_bytes,build_rc,run_rc,cycles,hash,wallclock_s,uartlog" > "${RESULTS_CSV}"
fi

# ------- per-cell driver -------------------------------------------------
run_cell() {
  local model="$1"
  local backend="$2"
  local hart="${BACKEND_HART[${backend}]}"
  local features="${BACKEND_FEATURES[${backend}]}"
  local tag="${model}_${backend}"
  local workload_name="merlin-shuttle-${tag}"
  local workload_dir="${WORKLOADS_DIR}/${workload_name}"
  local workload_json="${WORKLOADS_DIR}/${workload_name}.json"

  echo
  echo "===================================================================="
  echo "[$(date +%H:%M:%S)] cell ${tag}  hart=${hart} cpu_features=${features}"
  echo "===================================================================="

  local vmfb="${REPO_ROOT}/build/compiled_models/${model}/firesim/${backend}/${model}.vmfb"
  if [[ ! -f "${vmfb}" ]]; then
    echo "[skip] ${tag}: VMFB not found at ${vmfb}. Run compile_all.sh first." >&2
    return 1
  fi
  local vmfb_bytes
  vmfb_bytes="$(stat -c%s "${vmfb}")"

  local build_dir="${ZEPHYR_BUILDS_ROOT}/${tag}"
  local cell_log="${REPO_ROOT}/tmp/firesim_shuttle_${tag}.log"
  local t0; t0="$(date +%s)"

  # ---- west build ------------------------------------------------------
  echo "[build] west build -> ${build_dir}"
  rm -rf "${build_dir}"
  mkdir -p "${build_dir}"
  local build_rc=0
  ( west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 \
      "${ZEPHYR_SAMPLE_DIR}" \
      --build-dir "${build_dir}" \
      -- -DZEPHYR_EXTRA_MODULES="${CHIPYARD_ROOT}/software/zephyrproject/modules/merlin-iree" \
         -DMERLIN_BUILD_DIR="${REPO_ROOT}/build/zephyr-vanilla-release" \
         -DMERLIN_IREE_HEADERS_DIR="${REPO_ROOT}/third_party/iree_bar/runtime/src" \
         -DMERLIN_VMFB="${vmfb}" \
         -DMERLIN_MODEL="${model}" \
         -DMERLIN_BACKEND="${backend}" \
         -DMERLIN_HART="${hart}" \
         -DMERLIN_CPU_FEATURES="${features}" \
  ) >>"${cell_log}" 2>&1 || build_rc=$?

  if [[ ${build_rc} -ne 0 ]]; then
    echo "[fail] ${tag}: west build returned ${build_rc} (see ${cell_log})" >&2
    printf '%s,%s,%s,%d,%d,%d,,,,%d,\n' \
      "$(date -Iseconds)" "${model}" "${backend}" "${hart}" \
      "${vmfb_bytes}" "${build_rc}" "$(( $(date +%s) - t0 ))" >> "${RESULTS_CSV}"
    return 1
  fi

  local elf="${build_dir}/zephyr/zephyr.elf"
  if [[ ! -f "${elf}" ]]; then
    echo "[fail] ${tag}: ELF not found at ${elf}" >&2
    return 1
  fi
  echo "[build] OK -> ${elf} ($(stat -c%s "${elf}") bytes)"

  # ---- stage workload (inline, no merlin chipyard indirection) ---------
  rm -rf "${workload_dir}"
  mkdir -p "${workload_dir}"
  local elf_name="${workload_name}.elf"
  cp "${elf}" "${workload_dir}/${elf_name}"
  # bare-metal workload JSON: no rootfs (common_rootfs: null) so FireSim
  # doesn't invoke `sudo mount` on a dummy image. Matches the merlin-bench
  # kernel workloads in deploy/workloads/. chipyard_riscv64 has no block
  # device so null is exactly what the driver expects.
  cat > "${workload_json}" <<JSONEOF
{
  "benchmark_name": "${workload_name}",
  "common_bootbinary": "${elf_name}",
  "common_rootfs": null,
  "common_simulation_outputs": ["uartlog"]
}
JSONEOF

  # ---- toggle workload_name in config_runtime.yaml --------------------
  # Use \1 to preserve leading indentation (workload_name: is nested
  # under `workload:`; replacing with a flush-left line corrupts the YAML).
  sed -i -E "s|^([[:space:]]*)workload_name:.*|\1workload_name: ${workload_name}.json|" "${RUNTIME_YAML}"

  # ---- firesim infrasetup / runworkload / kill -------------------------
  # When FIRESIM_QUEUE=1 (and the firesim-queue CLI is available), route
  # the FPGA-touching block through the shared job queue. Each user blocks
  # waiting their turn instead of clobbering each other's `firesim
  # runworkload`. Daemon must be running (see
  # /scratch2/agustin/firesim_queue/README.md). When FIRESIM_QUEUE is
  # unset, behavior is byte-identical to the pre-queue script.
  local run_rc=0
  local fq_bin="${FIRESIM_QUEUE_BIN:-/scratch2/agustin/firesim_queue/bin/firesim-queue}"
  if [[ "${FIRESIM_QUEUE:-0}" == "1" ]] && [[ -x "${fq_bin}" ]]; then
    local fq_prio="${FIRESIM_QUEUE_PRIORITY:-5}"
    echo "[fsim]  firesim-queue submit (priority=${fq_prio}, cell=${tag}, project=merlin)"
    # The queue daemon runs in a clean env without chipyard's sourceme-manager
    # active. The wrapped command must therefore re-source the env. Mirror
    # ModelBlaster's pattern so the daemon's child has `firesim` on PATH plus
    # the right SSH agent / fabric setup.
    "${fq_bin}" submit --priority "${fq_prio}" --cwd "${DEPLOY_DIR}" \
      --project merlin -- \
      bash -c "set -e
        unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER \
          CONDA_PYTHON_EXE CONDA_SHLVL CONDA_EXE _CE_M _CE_CONDA
        source ${CHIPYARD_ROOT}/env.sh
        # firesim infrasetup uses fabric to ssh into localhost (per-host
        # task model) even when there's only one host. We need an active
        # ssh-agent with ~/.ssh/firesim loaded. The queue daemon was
        # detached via nohup so its SSH_AUTH_SOCK may be stale or absent
        # — start a fresh ssh-agent and load the key inside the wrapped
        # command. This is the same setup that run_all.sh does in its
        # non-queue path (see the conditional sourcing block earlier).
        eval \$(ssh-agent -s) >/dev/null
        if [ -f \$HOME/.ssh/firesim ]; then
          ssh-add \$HOME/.ssh/firesim 2>/dev/null || true
        fi
        # sourceme-manager.sh lives in sims/firesim/, NOT in deploy/.
        # Source from there but execute firesim from deploy/ (which is
        # where infrasetup expects to find config_runtime.yaml etc.).
        cd ${CHIPYARD_ROOT}/sims/firesim
        source ./sourceme-manager.sh --skip-ssh-setup
        cd ${DEPLOY_DIR}
        # firesim kill BEFORE infrasetup — clears any stale FPGA state
        # (XDMA descriptors, previous bitstream's reservation station)
        # left behind by a prior aborted run. Without this, infrasetup's
        # bitstream-flash can fail with 'fabric: nonzero return code 1'
        # because the FPGA driver sees a busy state.
        firesim kill >/dev/null 2>&1 || true
        firesim infrasetup
        rc=0
        firesim runworkload || rc=\$?
        firesim kill >/dev/null 2>&1 || true
        ssh-agent -k >/dev/null 2>&1 || true
        exit \$rc" \
      >>"${cell_log}" 2>&1 \
      || run_rc=$?
  else
    echo "[fsim]  firesim infrasetup"
    ( cd "${DEPLOY_DIR}" && firesim infrasetup ) >>"${cell_log}" 2>&1 \
      || run_rc=$?

    if [[ ${run_rc} -eq 0 ]]; then
      echo "[fsim]  firesim runworkload"
      ( cd "${DEPLOY_DIR}" && firesim runworkload ) >>"${cell_log}" 2>&1 \
        || run_rc=$?
    fi

    # ---- firesim kill (always, matches SaturnOPU bare-metal pattern) ----
    echo "[fsim]  firesim kill"
    ( cd "${DEPLOY_DIR}" && firesim kill ) >>"${cell_log}" 2>&1 || true
  fi

  # ---- result extraction ----------------------------------------------
  local latest_result uartlog="" cycles="" hash=""
  latest_result="$(find "${DEPLOY_DIR}/results-workload/" -maxdepth 1 \
                    -name "*${workload_name}*" -type d \
                    -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 \
                    | cut -d' ' -f2-)"
  if [[ -n "${latest_result}" ]]; then
    for candidate in \
        "${latest_result}/${workload_name}0/uartlog" \
        "${latest_result}/uartlog"; do
      if [[ -f "${candidate}" ]]; then
        uartlog="${candidate}"; break
      fi
    done
    if [[ -z "${uartlog}" ]]; then
      uartlog="$(find "${latest_result}" -name uartlog -print -quit 2>/dev/null || true)"
    fi
  fi

  if [[ -n "${uartlog}" && -f "${uartlog}" ]]; then
    cp "${uartlog}" "${REPO_ROOT}/tmp/firesim_shuttle_${tag}.uartlog"
    local line
    line="$(grep -E '^\[merlin\] result ' "${uartlog}" | tail -n1 || true)"
    if [[ -n "${line}" ]]; then
      cycles="$(echo "${line}" | sed -nE 's/.*cycles=([0-9]+).*/\1/p')"
      hash="$(echo "${line}" | sed -nE 's/.*hash=(0x[0-9a-fA-F]+).*/\1/p')"
    fi
  fi

  local t1; t1="$(date +%s)"
  local wall=$((t1 - t0))
  printf '%s,%s,%s,%d,%d,%d,%d,%s,%s,%d,%s\n' \
    "$(date -Iseconds)" "${model}" "${backend}" "${hart}" \
    "${vmfb_bytes}" "${build_rc}" "${run_rc}" "${cycles}" "${hash}" \
    "${wall}" "${uartlog}" >> "${RESULTS_CSV}"

  echo "[done] ${tag}: build_rc=${build_rc} run_rc=${run_rc} cycles=${cycles} hash=${hash} wall=${wall}s"
}

# ------- loop ------------------------------------------------------------
fail=0
for model in "${MODELS[@]}"; do
  [[ -n "${ONLY_MODEL}" && "${ONLY_MODEL}" != "${model}" ]] && continue
  for backend in "${BACKENDS[@]}"; do
    [[ -n "${ONLY_BACKEND}" && "${ONLY_BACKEND}" != "${backend}" ]] && continue
    run_cell "${model}" "${backend}" || fail=$((fail + 1))
  done
done

echo
echo "Results CSV: ${RESULTS_CSV}"
echo "Failures:    ${fail}"
echo
echo "Tail of CSV:"
tail -n 15 "${RESULTS_CSV}"

[[ ${fail} -eq 0 ]]
