#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLCHAIN_ROOT="${WORKSPACE_DIR}/build_tools/riscv-tools-spacemit"
TOOLCHAIN_DIRNAME="spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
TOOLCHAIN_DIR="${TOOLCHAIN_ROOT}/${TOOLCHAIN_DIRNAME}"
ARCHIVE_PATH="${TOOLCHAIN_ROOT}/${TOOLCHAIN_DIRNAME}.tar.xz"
URL="https://archive.spacemit.com/toolchain/${TOOLCHAIN_DIRNAME}.tar.xz"

FORCE=0
OFFLINE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
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

mkdir -p "${TOOLCHAIN_ROOT}"

if [[ -d "${TOOLCHAIN_DIR}" && "${FORCE}" -ne 1 ]]; then
  echo "SpacemiT toolchain already present at ${TOOLCHAIN_DIR}"
  exit 0
fi

if [[ "${FORCE}" -eq 1 ]]; then
  rm -rf "${TOOLCHAIN_DIR}"
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  if [[ "${OFFLINE}" -eq 1 ]]; then
    echo "Offline mode enabled and archive not present: ${ARCHIVE_PATH}"
    exit 1
  fi
  echo "Downloading SpacemiT toolchain from ${URL}..."
  wget -O "${ARCHIVE_PATH}" "${URL}"
else
  echo "Reusing existing archive ${ARCHIVE_PATH}"
fi

echo "Extracting into ${TOOLCHAIN_ROOT}..."
tar -xvf "${ARCHIVE_PATH}" -C "${TOOLCHAIN_ROOT}"

if [[ ! -d "${TOOLCHAIN_DIR}" ]]; then
  echo "Expected toolchain directory not found after extraction: ${TOOLCHAIN_DIR}"
  exit 1
fi

echo "Done. Toolchain installed at ${TOOLCHAIN_DIR}"
