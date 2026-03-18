#!/usr/bin/env bash
# stage_firesim_workload.sh — Stages a Merlin workload into FireSim's deploy directory.
#
# Usage: stage_firesim_workload.sh <chipyard_root> <workload_name> <overlay_src_dir> <command>
#
# Creates:
#   $CHIPYARD_ROOT/sims/firesim/deploy/workloads/<workload_name>.json
#   $CHIPYARD_ROOT/sims/firesim/deploy/workloads/<workload_name>/overlay/...
#
# The overlay_src_dir contents are copied into the workload overlay so they
# appear at /opt/merlin/ inside the simulated Linux filesystem.

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <chipyard_root> <workload_name> <overlay_src_dir> <command>"
    exit 1
fi

CHIPYARD_ROOT="$1"
WORKLOAD_NAME="$2"
OVERLAY_SRC="$3"
COMMAND="$4"

DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
WORKLOAD_DIR="${DEPLOY_DIR}/workloads/${WORKLOAD_NAME}"
WORKLOAD_JSON="${DEPLOY_DIR}/workloads/${WORKLOAD_NAME}.json"
OVERLAY_DIR="${WORKLOAD_DIR}/overlay"

echo "Staging workload '${WORKLOAD_NAME}' into ${DEPLOY_DIR}/workloads/"

# Create overlay directory
mkdir -p "${OVERLAY_DIR}/opt/merlin"

# Copy Merlin runtime and artifacts into overlay
if [ -d "${OVERLAY_SRC}" ]; then
    echo "  Copying overlay from ${OVERLAY_SRC} -> ${OVERLAY_DIR}/opt/merlin/"
    cp -a "${OVERLAY_SRC}/." "${OVERLAY_DIR}/opt/merlin/"
else
    echo "  WARNING: overlay source not found: ${OVERLAY_SRC}"
fi

# Create a run script inside the overlay
cat > "${OVERLAY_DIR}/opt/merlin/run.sh" <<'RUNEOF'
#!/bin/bash
set -e
echo "=== Merlin IREE workload ==="
cd /opt/merlin
# Run all .vmfb files found
for vmfb in *.vmfb models/*.vmfb; do
    [ -f "$vmfb" ] || continue
    echo "Running: $vmfb"
    ./install/bin/iree-run-module --module="$vmfb" 2>&1 || true
done
echo "=== Done ==="
RUNEOF
chmod +x "${OVERLAY_DIR}/opt/merlin/run.sh"

# Build the list of overlay files for the workload JSON
OVERLAY_FILES=""
if [ -d "${OVERLAY_DIR}" ]; then
    # Collect relative paths from overlay root
    while IFS= read -r rel_path; do
        [ -z "${rel_path}" ] && continue
        if [ -n "${OVERLAY_FILES}" ]; then
            OVERLAY_FILES="${OVERLAY_FILES}, "
        fi
        OVERLAY_FILES="${OVERLAY_FILES}\"${rel_path}\""
    done < <(cd "${WORKLOAD_DIR}" && find overlay -type f -printf '%P\n' | head -50)
fi

# Generate workload JSON
cat > "${WORKLOAD_JSON}" <<JSONEOF
{
  "benchmark_name": "${WORKLOAD_NAME}",
  "common_bootbinary": "../../../../../software/firemarshal/images/firechip/br-base/br-base-bin",
  "common_rootfs": "../../../../../software/firemarshal/images/firechip/br-base/br-base.img",
  "common_outputs": ["/etc/os-release"],
  "common_simulation_outputs": ["uartlog", "memory_stats*.csv"],
  "deliver_dir": "/",
  "common_args": [],
  "common_files": [],
  "no_post_run_hook": "",
  "workloads": [
    {
      "name": "job0",
      "files": [],
      "command": "${COMMAND}",
      "simulation_outputs": [],
      "outputs": []
    }
  ]
}
JSONEOF

echo "  Created: ${WORKLOAD_JSON}"
echo "  Overlay: ${OVERLAY_DIR}"
echo "Done."
