#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IREE_COMPILE="${IREE_COMPILE:-${REPO_ROOT}/build-host/install/bin/iree-compile}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/artifacts/vmfb}"
TARGET="${TARGET:-spacemit-riscv}"  # spacemit-riscv | host

DRONET_MLIR="${DRONET_MLIR:-${REPO_ROOT}/samples/models/dronet/dronet.mlir}"
MLP_MLIR="${MLP_MLIR:-${REPO_ROOT}/samples/models/mlp/mlp.mlir}"

usage() {
  cat <<EOF
Usage:
  $0 [--iree-compile PATH] [--out-dir DIR] [--target host|spacemit-riscv]
     [--dronet-mlir PATH] [--mlp-mlir PATH]

Notes:
  - The script rewrites module names to 'dronet' and 'mlp' so both VMFBs can
    be loaded in the same runtime session.
  - Output files:
      <out-dir>/dronet.vmfb
      <out-dir>/mlp.vmfb
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iree-compile) IREE_COMPILE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --dronet-mlir) DRONET_MLIR="$2"; shift 2 ;;
    --mlp-mlir) MLP_MLIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -x "${IREE_COMPILE}" ]]; then
  echo "iree-compile not found/executable: ${IREE_COMPILE}" >&2
  exit 1
fi
if [[ ! -f "${DRONET_MLIR}" ]]; then
  echo "Missing dronet MLIR: ${DRONET_MLIR}" >&2
  exit 1
fi
if [[ ! -f "${MLP_MLIR}" ]]; then
  echo "Missing mlp MLIR: ${MLP_MLIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

DRONET_NAMED_MLIR="${TMP_DIR}/dronet.named.mlir"
MLP_NAMED_MLIR="${TMP_DIR}/mlp.named.mlir"

sed '1s/^module {/builtin.module @dronet {/' "${DRONET_MLIR}" > "${DRONET_NAMED_MLIR}"
sed '1s/^module {/builtin.module @mlp {/' "${MLP_MLIR}" > "${MLP_NAMED_MLIR}"

COMMON_FLAGS=(
  --iree-hal-target-device=local
  --iree-hal-local-target-device-backends=llvm-cpu
)

if [[ "${TARGET}" == "spacemit-riscv" ]]; then
  TARGET_FLAGS=(
    --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu
    --iree-llvmcpu-target-abi=lp64d
    --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b,+zba,+zbb,+zbc,+zbs,+zicbom,+zicboz,+zicbop,+zihintpause
    --iree-llvmcpu-enable-ukernels=all
    --iree-opt-level=O3
  )
elif [[ "${TARGET}" == "host" ]]; then
  TARGET_FLAGS=(
    --iree-llvmcpu-target-cpu=host
    --iree-opt-level=O3
  )
else
  echo "Unsupported --target value: ${TARGET}" >&2
  exit 1
fi

echo "[compile] dronet -> ${OUT_DIR}/dronet.vmfb"
"${IREE_COMPILE}" "${DRONET_NAMED_MLIR}" -o "${OUT_DIR}/dronet.vmfb" \
  "${COMMON_FLAGS[@]}" "${TARGET_FLAGS[@]}"

echo "[compile] mlp -> ${OUT_DIR}/mlp.vmfb"
"${IREE_COMPILE}" "${MLP_NAMED_MLIR}" -o "${OUT_DIR}/mlp.vmfb" \
  "${COMMON_FLAGS[@]}" "${TARGET_FLAGS[@]}"

echo "[done] VMFB artifacts in ${OUT_DIR}"
