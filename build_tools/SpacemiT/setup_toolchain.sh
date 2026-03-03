#!/bin/bash
set -e

export WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p ${WORKSPACE_DIR}/riscv-tools-spacemit

echo "Downloading SpacemiT Toolchain v1.1.2..."
wget https://archive.spacemit.com/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz -P ${WORKSPACE_DIR}

echo "Extracting..."
tar -xvf ${WORKSPACE_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz -C ${WORKSPACE_DIR}/riscv-tools-spacemit

echo "Done. Toolchain installed at ${WORKSPACE_DIR}/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
