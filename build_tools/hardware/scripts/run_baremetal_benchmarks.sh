#!/usr/bin/env bash
# run_baremetal_benchmarks.sh
#
# Runs bare-metal benchmark ELFs from the simple_embedding_ukernel sample
# on FireSim sequentially, collecting results into a summary CSV.
#
# Usage:
#   ./run_baremetal_benchmarks.sh <elf_dir> [chipyard_root]
#
# Example:
#   ./run_baremetal_benchmarks.sh \
#     /scratch2/agustin/merlin/build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel \
#     /scratch2/agustin/chipyard
#
# The script will:
#   1. Find all bench_mm_* ELFs in <elf_dir>
#   2. For each ELF: create workload JSON, stage it, update config_runtime.yaml,
#      run firesim infrasetup + runworkload, collect uartlog
#   3. Print a summary table comparing OPU vs RVV for each size
#
# Prerequisites:
#   - Bitstream registered in config_hwdb.yaml
#   - config_runtime.yaml default_hw_config set correctly
#   - FireSim environment sourced (firesim command available)
#   - FPGA accessible (xdma driver loaded)

set -euo pipefail

ELF_DIR="${1:?Usage: $0 <elf_dir> [chipyard_root]}"
CHIPYARD_ROOT="${2:-${CHIPYARD_ROOT:?Set CHIPYARD_ROOT or pass as second argument}}"
DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
WORKLOADS_DIR="${DEPLOY_DIR}/workloads"
RUNTIME_YAML="${DEPLOY_DIR}/config_runtime.yaml"

# Results go next to the ELFs
RESULTS_DIR="${ELF_DIR}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_CSV="${RESULTS_DIR}/results-${TIMESTAMP}.csv"

# Find all benchmark ELFs
mapfile -t ELFS < <(find "${ELF_DIR}" -maxdepth 1 -name 'bench_mm_*' -type f -executable | sort)

if [ ${#ELFS[@]} -eq 0 ]; then
    echo "ERROR: No bench_mm_* ELFs found in ${ELF_DIR}"
    exit 1
fi

echo "============================================"
echo " FireSim Bare-Metal Benchmark Runner"
echo "============================================"
echo "ELF directory: ${ELF_DIR}"
echo "Deploy directory: ${DEPLOY_DIR}"
echo "Found ${#ELFS[@]} benchmarks:"
printf '  %s\n' "$(basename -a "${ELFS[@]}")"
echo

mkdir -p "${RESULTS_DIR}"

# Save original workload_name
ORIG_WORKLOAD=$(grep 'workload_name:' "${RUNTIME_YAML}" | head -1 | sed 's/.*workload_name:[[:space:]]*//')

# CSV header
echo "size,ukernel,avg_cycles,ops_per_cycle,status" > "${RESULTS_CSV}"

PASS_COUNT=0
FAIL_COUNT=0

run_benchmark() {
    local elf_path="$1"
    local elf_name
    elf_name=$(basename "${elf_path}")
    local workload_name="merlin-bench-${elf_name}"
    local workload_dir="${WORKLOADS_DIR}/${workload_name}"
    local workload_json="${WORKLOADS_DIR}/${workload_name}.json"

    echo ""
    echo "========================================"
    echo "[$(date +%H:%M:%S)] Running: ${elf_name}"
    echo "========================================"

    # Stage the ELF
    mkdir -p "${workload_dir}"
    cp "${elf_path}" "${workload_dir}/${elf_name}"

    # Create workload JSON
    cat > "${workload_json}" <<JSONEOF
{
  "benchmark_name": "${workload_name}",
  "common_bootbinary": "${elf_name}",
  "common_rootfs": null,
  "common_outputs": [],
  "common_simulation_outputs": ["uartlog"]
}
JSONEOF

    # Update config_runtime.yaml workload_name
    sed -i "s|workload_name:.*|workload_name: ${workload_name}.json|" "${RUNTIME_YAML}"

    # Run FireSim infrasetup + runworkload
    echo "[$(date +%H:%M:%S)] firesim infrasetup..."
    if ! (cd "${DEPLOY_DIR}" && firesim infrasetup); then
        echo "ERROR: infrasetup failed for ${elf_name}"
        echo "${elf_name},,,INFRASETUP_FAILED" >> "${RESULTS_CSV}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    echo "[$(date +%H:%M:%S)] firesim runworkload..."
    if ! (cd "${DEPLOY_DIR}" && firesim runworkload); then
        echo "ERROR: runworkload failed for ${elf_name}"
        echo "${elf_name},,,RUNWORKLOAD_FAILED" >> "${RESULTS_CSV}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Find the uartlog from the most recent matching results directory
    local latest_result
    latest_result=$(find "${DEPLOY_DIR}/results-workload/" -maxdepth 1 -name "*${workload_name}*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "${latest_result}" ]; then
        echo "WARNING: No results directory found for ${workload_name}"
        echo "${elf_name},,,NO_RESULTS" >> "${RESULTS_CSV}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Find uartlog (try common path patterns)
    local uartlog=""
    for candidate in \
        "${latest_result}/${workload_name}0/uartlog" \
        "${latest_result}/uartlog"; do
        if [ -f "${candidate}" ]; then
            uartlog="${candidate}"
            break
        fi
    done

    if [ -z "${uartlog}" ]; then
        uartlog=$(find "${latest_result}" -name 'uartlog' -print -quit 2>/dev/null)
    fi

    if [ -z "${uartlog}" ] || [ ! -f "${uartlog}" ]; then
        echo "WARNING: uartlog not found in ${latest_result}"
        echo "${elf_name},,,NO_UARTLOG" >> "${RESULTS_CSV}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    # Save a copy
    cp "${uartlog}" "${RESULTS_DIR}/${elf_name}.uartlog"

    # Extract CSV line
    local csv_line
    csv_line=$(grep '^CSV' "${uartlog}" 2>/dev/null | head -1)

    if [ -n "${csv_line}" ]; then
        # Parse: CSV, <size>, <ukernel>, <cycles>, <ops_per_cycle>
        local size uk cycles ops status
        size=$(echo "${csv_line}" | cut -d',' -f2 | tr -d ' ')
        uk=$(echo "${csv_line}" | cut -d',' -f3 | tr -d ' ')
        cycles=$(echo "${csv_line}" | cut -d',' -f4 | tr -d ' ')
        ops=$(echo "${csv_line}" | cut -d',' -f5 | tr -d ' ')

        # Check if verification passed
        if grep -q 'PASSED' "${uartlog}"; then
            status="PASS"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            status="FAIL"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi

        echo "${size},${uk},${cycles},${ops},${status}" >> "${RESULTS_CSV}"
        echo "  Result: size=${size} ukernel=${uk} cycles=${cycles} ops/cyc=${ops} [${status}]"
    else
        echo "WARNING: No CSV output found in uartlog"
        echo "${elf_name},,,NO_CSV_OUTPUT" >> "${RESULTS_CSV}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# Run each benchmark
for elf in "${ELFS[@]}"; do
    run_benchmark "${elf}" || true
done

# Restore original workload_name
sed -i "s|workload_name:.*|workload_name: ${ORIG_WORKLOAD}|" "${RUNTIME_YAML}"

# Print summary
echo ""
echo "============================================"
echo " BENCHMARK RESULTS SUMMARY"
echo "============================================"
echo ""

# Print as a formatted table
printf "%-8s %-8s %15s %12s %8s\n" "Size" "Ukernel" "Avg Cycles" "Ops/Cycle" "Status"
printf "%-8s %-8s %15s %12s %8s\n" "----" "-------" "----------" "---------" "------"

tail -n +2 "${RESULTS_CSV}" | sort -t',' -k1,1n -k2,2 | while IFS=',' read -r size uk cycles ops status; do
    printf "%-8s %-8s %15s %12s %8s\n" "${size}" "${uk}" "${cycles}" "${ops}" "${status}"
done

echo ""

# Compute speedups if we have both ALL and NONE for any size
echo "--- Speedup (OPU vs RVV) ---"
tail -n +2 "${RESULTS_CSV}" | grep ',ALL,' | sort -t',' -k1,1n | while IFS=',' read -r size _ cycles_all _ _; do
    cycles_none=$(grep "^${size},NONE," "${RESULTS_CSV}" | head -1 | cut -d',' -f3)
    if [ -n "${cycles_none}" ] && [ -n "${cycles_all}" ] && [ "${cycles_all}" -gt 0 ] 2>/dev/null; then
        # Integer-only speedup calculation: (none * 100) / all
        speedup_x100=$(( (cycles_none * 100) / cycles_all ))
        speedup_whole=$((speedup_x100 / 100))
        speedup_frac=$((speedup_x100 % 100))
        printf "  Size %s: %s / %s = %d.%02dx\n" "${size}" "${cycles_none}" "${cycles_all}" "${speedup_whole}" "${speedup_frac}"
    fi
done

echo ""
echo "Passed: ${PASS_COUNT}  Failed: ${FAIL_COUNT}"
echo "CSV: ${RESULTS_CSV}"
echo "Logs: ${RESULTS_DIR}/"
