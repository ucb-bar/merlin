#!/usr/bin/env bash
# configure_firesim.sh — Configures FireSim deploy YAMLs for a Merlin recipe.
#
# Usage: configure_firesim.sh <chipyard_root> <recipe_name> <hw_config> <workload_json>
#
# Modifies:
#   config_build.yaml    — sets builds_to_run
#   config_runtime.yaml  — sets default_hw_config and workload_name
#
# Does NOT modify config_build_recipes.yaml or config_hwdb.yaml (those are
# more static and managed separately).

set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <chipyard_root> <recipe_name> <hw_config> <workload_json>"
    exit 1
fi

CHIPYARD_ROOT="$1"
RECIPE_NAME="$2"
HW_CONFIG="$3"
WORKLOAD_JSON="$4"

DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"

echo "Configuring FireSim for recipe '${RECIPE_NAME}'..."

# --- config_build.yaml: set the build target ---
CONFIG_BUILD="${DEPLOY_DIR}/config_build.yaml"
if [ -f "${CONFIG_BUILD}" ]; then
    # Use python to safely modify YAML (sed is fragile for YAML)
    python3 -c "
import yaml, sys

with open('${CONFIG_BUILD}', 'r') as f:
    data = yaml.safe_load(f)

data['builds_to_run'] = ['${RECIPE_NAME}']
data.setdefault('agfis_to_share', [])
if '${RECIPE_NAME}' not in data['agfis_to_share']:
    data['agfis_to_share'].append('${RECIPE_NAME}')

with open('${CONFIG_BUILD}', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print('  Updated: ${CONFIG_BUILD}')
print('    builds_to_run: [${RECIPE_NAME}]')
"
else
    echo "  WARNING: ${CONFIG_BUILD} not found"
fi

# --- config_runtime.yaml: set hw config and workload ---
CONFIG_RUNTIME="${DEPLOY_DIR}/config_runtime.yaml"
if [ -f "${CONFIG_RUNTIME}" ]; then
    python3 -c "
import yaml, sys

with open('${CONFIG_RUNTIME}', 'r') as f:
    data = yaml.safe_load(f)

data.setdefault('target_config', {})
data['target_config']['default_hw_config'] = '${HW_CONFIG}'

data.setdefault('workload', {})
data['workload']['workload_name'] = '${WORKLOAD_JSON}'

with open('${CONFIG_RUNTIME}', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print('  Updated: ${CONFIG_RUNTIME}')
print('    default_hw_config: ${HW_CONFIG}')
print('    workload_name: ${WORKLOAD_JSON}')
"
else
    echo "  WARNING: ${CONFIG_RUNTIME} not found"
fi

echo "Done. FireSim is configured for '${RECIPE_NAME}'."
echo ""
echo "Next steps:"
echo "  1. Build bitstream:  cd ${DEPLOY_DIR} && firesim buildbitstream"
echo "  2. After build:      Register in config_hwdb.yaml"
echo "  3. Stage workload:   merlin chipyard stage-workload ${RECIPE_NAME} <overlay_dir>"
echo "  4. Run:              cd ${DEPLOY_DIR} && firesim infrasetup && firesim runworkload"
