#!/usr/bin/env bash
# build_firemarshal_base.sh — Builds the FireMarshal base Linux image.
#
# Usage: build_firemarshal_base.sh <chipyard_root>
#
# This builds the br-base image that all Merlin FireSim workloads use as
# their base rootfs. Only needs to be run once per Chipyard workspace.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <chipyard_root>"
    exit 1
fi

CHIPYARD_ROOT="$1"
FIREMARSHAL_DIR="${CHIPYARD_ROOT}/software/firemarshal"

if [ ! -d "${FIREMARSHAL_DIR}" ]; then
    echo "FireMarshal not found at: ${FIREMARSHAL_DIR}"
    echo "Ensure Chipyard submodules are initialized."
    exit 1
fi

echo "Building FireMarshal base image (br-base)..."
echo "  FireMarshal: ${FIREMARSHAL_DIR}"

cd "${FIREMARSHAL_DIR}"

# Initialize submodules if needed
if [ ! -f "wlutil/__init__.py" ] && [ -f "init-submodules.sh" ]; then
    echo "  Initializing FireMarshal submodules..."
    ./init-submodules.sh
fi

# Build the base image
echo "  Running: marshal build br-base.json"
./marshal build br-base.json

# Install into FireSim deploy directory
echo "  Running: marshal install br-base.json"
./marshal install br-base.json

echo "Done. Base image built and installed."
echo "  Boot binary: ${FIREMARSHAL_DIR}/images/firechip/br-base/br-base-bin"
echo "  Root FS:     ${FIREMARSHAL_DIR}/images/firechip/br-base/br-base.img"
