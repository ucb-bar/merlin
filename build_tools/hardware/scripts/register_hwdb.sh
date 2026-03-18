#!/usr/bin/env bash
# register_hwdb.sh — Registers a built bitstream in FireSim's config_hwdb.yaml.
#
# Usage: register_hwdb.sh <chipyard_root> <recipe_name> <target_config>
#
# Searches results-build/ for the matching bitstream tarball and adds/updates
# the entry in config_hwdb.yaml.

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <chipyard_root> <recipe_name> <target_config>"
    exit 1
fi

CHIPYARD_ROOT="$1"
RECIPE_NAME="$2"
TARGET_CONFIG="$3"

DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
RESULTS_DIR="${DEPLOY_DIR}/results-build"
HWDB_FILE="${DEPLOY_DIR}/config_hwdb.yaml"

echo "Searching for bitstream for '${RECIPE_NAME}'..."

# Find the newest matching tarball
TARBALL=""
if [ -d "${RESULTS_DIR}" ]; then
    TARBALL=$(find "${RESULTS_DIR}" \
        -path "*${TARGET_CONFIG}*" \
        -name "firesim.tar.gz" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn \
        | head -1 \
        | cut -d' ' -f2-)
fi

if [ -z "${TARBALL}" ]; then
    echo "  No bitstream found for ${TARGET_CONFIG}"
    echo "  Build one with: cd ${DEPLOY_DIR} && firesim buildbitstream"
    exit 1
fi

echo "  Found: ${TARBALL}"

# Add/update entry in config_hwdb.yaml
python3 -c "
import yaml

hwdb_path = '${HWDB_FILE}'
with open(hwdb_path, 'r') as f:
    data = yaml.safe_load(f) or {}

data['${RECIPE_NAME}'] = {
    'bitstream_tar': 'file://${TARBALL}',
    'deploy_quintuplet_override': None,
    'custom_runtime_config': None,
}

with open(hwdb_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print('  Registered in: ${HWDB_FILE}')
print('  Entry: ${RECIPE_NAME}')
print('  Tarball: file://${TARBALL}')
"

echo "Done."
