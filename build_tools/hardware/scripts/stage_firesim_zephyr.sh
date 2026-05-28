#!/usr/bin/env bash
# stage_firesim_zephyr.sh — stages a bare-metal Zephyr ELF as a FireSim workload.
#
# Usage:
#   stage_firesim_zephyr.sh <chipyard_root> <workload_name> <zephyr_elf>
#
# Drops the Zephyr ELF into FireSim's deploy/workloads/<workload_name>/ and
# writes a workload JSON of the form
#   {"name":"<workload_name>","distro":{"name":"bare"},"bin":"<elf path>"}
# This mirrors FireMarshal's existing bare-base.json schema (see
# software/firemarshal/boards/chipyard/base-workloads/bare-base.json) but
# uses an explicit ELF instead of a Linux kernel image.
#
# The simulation driver loads the ELF directly via Chipyard's LOADMEM=1
# pathway (reused by .github/scripts/run-tests.sh:176-177 for Zephyr ELFs);
# no rootfs and no Linux are involved.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <chipyard_root> <workload_name> <zephyr_elf>" >&2
    exit 1
fi

CHIPYARD_ROOT="$1"
WORKLOAD_NAME="$2"
ZEPHYR_ELF="$3"

if [ ! -f "${ZEPHYR_ELF}" ]; then
    echo "stage_firesim_zephyr: ELF not found: ${ZEPHYR_ELF}" >&2
    exit 1
fi

DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
WORKLOAD_DIR="${DEPLOY_DIR}/workloads/${WORKLOAD_NAME}"
WORKLOAD_JSON="${DEPLOY_DIR}/workloads/${WORKLOAD_NAME}.json"

if [ ! -d "${DEPLOY_DIR}" ]; then
    echo "stage_firesim_zephyr: FireSim deploy dir missing: ${DEPLOY_DIR}" >&2
    exit 1
fi

echo "Staging Zephyr workload '${WORKLOAD_NAME}' into ${DEPLOY_DIR}/workloads/"

mkdir -p "${WORKLOAD_DIR}"

# Copy the ELF into the workload directory under a stable name. FireMarshal /
# FireSim drivers will pick this up via the workload JSON's `bin` field.
ELF_BASENAME="$(basename "${ZEPHYR_ELF}")"
cp -f "${ZEPHYR_ELF}" "${WORKLOAD_DIR}/${ELF_BASENAME}"

# Pre-strip a copy alongside for faster transfer to the FPGA host.
if command -v riscv64-unknown-elf-strip >/dev/null 2>&1; then
    riscv64-unknown-elf-strip \
        -o "${WORKLOAD_DIR}/${ELF_BASENAME%.elf}.stripped.elf" \
        "${WORKLOAD_DIR}/${ELF_BASENAME}" || true
fi

# Workload JSON. `distro.name = bare` tells FireMarshal to skip Linux; `bin`
# is the ELF that the FireSim driver loads via LOADMEM. `outputs` is what
# `firesim runworkload` collects after the run.
cat > "${WORKLOAD_JSON}" <<JSON
{
    "name": "${WORKLOAD_NAME}",
    "distro": { "name": "bare" },
    "bin": "${WORKLOAD_DIR}/${ELF_BASENAME}",
    "outputs": ["uartlog"],
    "post_run_hook": null,
    "command": null
}
JSON

echo "  ELF:      ${WORKLOAD_DIR}/${ELF_BASENAME}"
echo "  workload: ${WORKLOAD_JSON}"

# Echo the full path so callers can chain (e.g. `cd $(stage_firesim_zephyr.sh
# ...)`).
echo "${WORKLOAD_JSON}"
