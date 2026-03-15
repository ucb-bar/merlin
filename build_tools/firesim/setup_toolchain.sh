#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREBUILT_DIR="${WORKSPACE_DIR}/build_tools/riscv-tools-iree"
BOOTSTRAP_WORK_DIR="${PREBUILT_DIR}/.bootstrap"
IREE_ARTIFACT_URL="https://sharkpublic.blob.core.windows.net/sharkpublic/GCP-Migration-Files"

RISCV_CLANG_TOOLCHAIN_FILE_NAME="toolchain_iree_manylinux_2_28_20231012.tar.gz"
RISCV_CLANG_TOOLCHAIN_FILE_SHA="3af56a58551ed5ae7441214822461a5368fee9403d7c883762fa902489bfbff0"

QEMU_FILE_NAME="qemu-riscv_8.1.2_manylinux_2.28_20231026.tar.gz"
QEMU_FILE_SHA="dd77b39820d45b80bafab9155581578b4c625cb92fd6db9e9adbb9798fde3597"

TOOLCHAIN_PATH_PREFIX="${PREBUILT_DIR}/toolchain/clang/linux/RISCV"
QEMU_PATH_PREFIX="${PREBUILT_DIR}/qemu/linux/RISCV"

WITH_QEMU=0
FORCE=0
OFFLINE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-qemu)
      WITH_QEMU=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --offline)
      OFFLINE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

cleanup() {
  if [[ -d "${BOOTSTRAP_WORK_DIR}" ]]; then
    rm -rf "${BOOTSTRAP_WORK_DIR}"
  fi
}
trap cleanup EXIT

mkdir -p "${BOOTSTRAP_WORK_DIR}"

download_and_extract() {
  local file_name="$1"
  local install_path="$2"
  local file_sha="$3"
  local archive_path="${BOOTSTRAP_WORK_DIR}/${file_name}"

  if [[ -d "${install_path}" ]] && [[ -n "$(ls -A "${install_path}" 2>/dev/null)" ]] && [[ "${FORCE}" -ne 1 ]]; then
    echo "Existing install found, reusing ${install_path}"
    return 0
  fi

  if [[ "${FORCE}" -eq 1 ]]; then
    rm -rf "${install_path}"
  fi
  mkdir -p "${install_path}"

  if [[ ! -f "${archive_path}" ]]; then
    if [[ "${OFFLINE}" -eq 1 ]]; then
      echo "Offline mode enabled and archive not present: ${archive_path}"
      return 1
    fi
    echo "Downloading ${file_name}..."
    wget --progress=bar:force:noscroll --directory-prefix="${BOOTSTRAP_WORK_DIR}" \
      "${IREE_ARTIFACT_URL}/${file_name}"
  else
    echo "Reusing existing archive ${archive_path}"
  fi

  echo "${file_sha} ${archive_path}" | sha256sum -c -
  echo "Extracting ${file_name} into ${install_path}..."
  tar -C "${install_path}" -xf "${archive_path}" --no-same-owner --strip-components=1
}

echo "Installing FireSim RISCV clang toolchain into ${TOOLCHAIN_PATH_PREFIX}..."
download_and_extract \
  "${RISCV_CLANG_TOOLCHAIN_FILE_NAME}" \
  "${TOOLCHAIN_PATH_PREFIX}" \
  "${RISCV_CLANG_TOOLCHAIN_FILE_SHA}"

if [[ "${WITH_QEMU}" -eq 1 ]]; then
  echo "Installing FireSim RISCV QEMU into ${QEMU_PATH_PREFIX}..."
  download_and_extract \
    "${QEMU_FILE_NAME}" \
    "${QEMU_PATH_PREFIX}" \
    "${QEMU_FILE_SHA}"
fi

echo "Bootstrap finished."
