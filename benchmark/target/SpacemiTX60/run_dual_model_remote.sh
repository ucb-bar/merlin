#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_SCRIPT="${SCRIPT_DIR}/parse_dual_model_log.py"

ENV_FILE="${SCRIPT_DIR}/spacemitx60.env"

BINARY_PATH=""
DRONET_VMFB=""
MLP_VMFB=""

MLP_HZ=20
DURATION_S=10
DRONET_FN="dronet.main_graph"
MLP_FN="mlp.main_graph"
DRIVER="local-task"
REPORT_HZ=1
DRONET_SENSOR_HZ=60
MLP_SENSOR_HZ=""
PROFILE_CMD=""

usage() {
  cat <<EOF
Usage:
  $0 --binary <path> --dronet-vmfb <path> --mlp-vmfb <path> [options]

Required:
  --binary <path>       Path to compiled runtime executable
  --dronet-vmfb <path>  Dronet VMFB artifact
  --mlp-vmfb <path>     MLP VMFB artifact

Optional:
  --env-file <path>     Board config env file (default: ${ENV_FILE})
  --mlp-hz <value>      MLP invocation frequency (default: ${MLP_HZ})
  --duration-s <value>  Run duration in seconds (default: ${DURATION_S})
  --dronet-fn <name>    Dronet function name (default: ${DRONET_FN})
  --mlp-fn <name>       MLP function name (default: ${MLP_FN})
  --driver <name>       IREE driver on board (default: ${DRIVER})
  --report-hz <value>   Stats print frequency (default: ${REPORT_HZ})
  --dronet-sensor-hz <value>  Dronet sensor generation frequency
                             (default: ${DRONET_SENSOR_HZ})
  --mlp-sensor-hz <value>     MLP sensor generation frequency
                             (default: same as --mlp-hz)
  --profile-cmd <cmd>   Prefix remote run command with profiler command
                        Example: --profile-cmd "perf stat -d"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY_PATH="$2"; shift 2 ;;
    --dronet-vmfb) DRONET_VMFB="$2"; shift 2 ;;
    --mlp-vmfb) MLP_VMFB="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --mlp-hz) MLP_HZ="$2"; shift 2 ;;
    --duration-s) DURATION_S="$2"; shift 2 ;;
    --dronet-fn) DRONET_FN="$2"; shift 2 ;;
    --mlp-fn) MLP_FN="$2"; shift 2 ;;
    --driver) DRIVER="$2"; shift 2 ;;
    --report-hz) REPORT_HZ="$2"; shift 2 ;;
    --dronet-sensor-hz) DRONET_SENSOR_HZ="$2"; shift 2 ;;
    --mlp-sensor-hz) MLP_SENSOR_HZ="$2"; shift 2 ;;
    --profile-cmd) PROFILE_CMD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${BINARY_PATH}" || -z "${DRONET_VMFB}" || -z "${MLP_VMFB}" ]]; then
  echo "Missing required args." >&2
  usage
  exit 1
fi
if [[ -z "${MLP_SENSOR_HZ}" ]]; then
  MLP_SENSOR_HZ="${MLP_HZ}"
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  echo "Create it from ${SCRIPT_DIR}/spacemitx60.env.example" >&2
  exit 1
fi

if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "Binary not found: ${BINARY_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DRONET_VMFB}" ]]; then
  echo "Dronet VMFB not found: ${DRONET_VMFB}" >&2
  exit 1
fi
if [[ ! -f "${MLP_VMFB}" ]]; then
  echo "MLP VMFB not found: ${MLP_VMFB}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${REMOTE_HOST:-}" || -z "${REMOTE_USER:-}" ]]; then
  echo "REMOTE_HOST and REMOTE_USER must be set in ${ENV_FILE}" >&2
  exit 1
fi

REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/merlin_dual_model_async}"
RUN_ID="dual_model_$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN_DIR="${REMOTE_DIR%/}/${RUN_ID}"
RESULTS_ROOT="${LOCAL_RESULTS_DIR:-${SCRIPT_DIR}/results}"
LOCAL_RUN_DIR="${RESULTS_ROOT%/}/${RUN_ID}"
mkdir -p "${LOCAL_RUN_DIR}"

SSH_BASE=(ssh -p "${REMOTE_PORT}")
SCP_BASE=(scp -P "${REMOTE_PORT}")
if [[ -n "${SSH_KEY:-}" ]]; then
  SSH_BASE+=(-i "${SSH_KEY}")
  SCP_BASE+=(-i "${SSH_KEY}")
fi
if [[ -n "${SSH_EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_OPTS=(${SSH_EXTRA_OPTS})
  SSH_BASE+=("${EXTRA_OPTS[@]}")
  SCP_BASE+=("${EXTRA_OPTS[@]}")
fi
REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

quote_join() {
  local out=""
  local q=""
  for arg in "$@"; do
    printf -v q "%q" "${arg}"
    out+="${q} "
  done
  echo "${out% }"
}

echo "[remote] creating directory: ${REMOTE_RUN_DIR}"
"${SSH_BASE[@]}" "${REMOTE_TARGET}" "mkdir -p $(printf '%q' "${REMOTE_RUN_DIR}")"

echo "[upload] binary + vmfb artifacts"
"${SCP_BASE[@]}" "${BINARY_PATH}" "${DRONET_VMFB}" "${MLP_VMFB}" \
  "${REMOTE_TARGET}:${REMOTE_RUN_DIR}/"

REMOTE_BINARY="$(basename "${BINARY_PATH}")"
REMOTE_DRONET_VMFB="$(basename "${DRONET_VMFB}")"
REMOTE_MLP_VMFB="$(basename "${MLP_VMFB}")"

RUN_ARGS=(
  "./${REMOTE_BINARY}"
  "${REMOTE_DRONET_VMFB}"
  "${REMOTE_MLP_VMFB}"
  "${MLP_HZ}"
  "${DURATION_S}"
  "${DRONET_FN}"
  "${MLP_FN}"
  "${DRIVER}"
  "${REPORT_HZ}"
  "${DRONET_SENSOR_HZ}"
  "${MLP_SENSOR_HZ}"
)

RUN_CMD="$(quote_join "${RUN_ARGS[@]}")"
if [[ -n "${PROFILE_CMD}" ]]; then
  REMOTE_EXEC_CMD="${PROFILE_CMD} ${RUN_CMD}"
else
  REMOTE_EXEC_CMD="${RUN_CMD}"
fi

LOG_FILE="${LOCAL_RUN_DIR}/runtime.log"
META_FILE="${LOCAL_RUN_DIR}/run_meta.txt"
SUMMARY_JSON="${LOCAL_RUN_DIR}/summary.json"

cat > "${META_FILE}" <<EOF
run_id=${RUN_ID}
remote_target=${REMOTE_TARGET}
remote_run_dir=${REMOTE_RUN_DIR}
binary=${BINARY_PATH}
dronet_vmfb=${DRONET_VMFB}
mlp_vmfb=${MLP_VMFB}
mlp_hz=${MLP_HZ}
duration_s=${DURATION_S}
dronet_fn=${DRONET_FN}
mlp_fn=${MLP_FN}
driver=${DRIVER}
report_hz=${REPORT_HZ}
dronet_sensor_hz=${DRONET_SENSOR_HZ}
mlp_sensor_hz=${MLP_SENSOR_HZ}
profile_cmd=${PROFILE_CMD}
EOF

echo "[run] remote execution started"
"${SSH_BASE[@]}" "${REMOTE_TARGET}" \
  "set -euo pipefail; cd $(printf '%q' "${REMOTE_RUN_DIR}"); chmod +x $(printf '%q' "${REMOTE_BINARY}"); ${REMOTE_EXEC_CMD}" \
  | tee "${LOG_FILE}"

if [[ -f "${PARSER_SCRIPT}" ]]; then
  python3 "${PARSER_SCRIPT}" --log "${LOG_FILE}" --out "${SUMMARY_JSON}"
  echo "[summary] ${SUMMARY_JSON}"
fi

echo "[done] Logs and metadata: ${LOCAL_RUN_DIR}"
