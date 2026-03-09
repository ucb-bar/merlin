#!/bin/bash
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export WORKSPACE_DIR
export TOOLCHAIN_DIR="${WORKSPACE_DIR}/build_tools/riscv-tools-spacemit"

mkdir -p "${TOOLCHAIN_DIR}"

echo "Downloading SpacemiT Toolchain v1.1.2..."
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz -P "${WORKSPACE_DIR}"

echo "Extracting..."
tar -xvf "${WORKSPACE_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz" -C "${TOOLCHAIN_DIR}"

echo "Cleaning up archive..."
rm "${WORKSPACE_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz"

echo "Done. Toolchain installed at ${TOOLCHAIN_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
