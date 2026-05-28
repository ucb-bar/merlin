#!/usr/bin/env bash
# run_radiance_muon.sh — runs a Muon kernel ELF on Chipyard's RadianceMuonConfig
# (or compatible) sim via `make run-binary`.
#
# Usage:
#   run_radiance_muon.sh <chipyard_root> <config> <binary> [<simulator>]
#
# Examples:
#   run_radiance_muon.sh /scratch2/agustin/chipyard \
#       RadianceMuonConfig \
#       /scratch2/agustin/merlin/build/radiance_muon-vanilla-release/vecadd.radiance.elf \
#       vcs
#
# Returns:
#   0 if simulation completes (exits cleanly with tohost==1)
#   1 otherwise (timeout, segfault, harness assertion, etc.)
#
# Captures the uartlog tail and prints it on completion so the caller
# (./merlin chipyard run-radiance-muon) sees the result inline.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <chipyard_root> <config> <binary> [<simulator>]" >&2
    exit 1
fi

CHIPYARD_ROOT="$1"
CONFIG="$2"
BINARY="$3"
SIMULATOR="${4:-vcs}"

if [ ! -f "$BINARY" ]; then
    echo "run_radiance_muon: binary not found: $BINARY" >&2
    exit 1
fi

SIM_DIR="${CHIPYARD_ROOT}/sims/${SIMULATOR}"
if [ ! -d "$SIM_DIR" ]; then
    echo "run_radiance_muon: sim dir not found: $SIM_DIR" >&2
    exit 1
fi

echo "=== run_radiance_muon ==="
echo "  chipyard:  ${CHIPYARD_ROOT}"
echo "  sim:       ${SIMULATOR}"
echo "  config:    ${CONFIG}"
echo "  binary:    ${BINARY}"
echo "========================="

cd "$SIM_DIR"

# Default to LOADMEM=1 (the explicit-loadmem path; matches Hansung's flow).
# Override with RADIANCE_LOADMEM=0 for the legacy default-flow.
LOADMEM_ARG=""
if [ "${RADIANCE_LOADMEM:-1}" = "1" ]; then
    LOADMEM_ARG="LOADMEM=1"
fi

# Forward optional `make run-binary` envs from the caller.
EXTRA_FLAGS="${RADIANCE_EXTRA_SIM_FLAGS:-}"

set +e
make run-binary CONFIG="${CONFIG}" BINARY="${BINARY}" ${LOADMEM_ARG} \
    EXTRA_SIM_FLAGS="${EXTRA_FLAGS}" 2>&1 | tee /tmp/run_radiance_muon.log
RC=${PIPESTATUS[0]}
set -e

# Extract uartlog tail if produced.
LOG=$(find "$SIM_DIR/output" -name "*.log" -newer "$BINARY" 2>/dev/null \
    | sort | tail -1 || true)
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
    echo
    echo "=== uartlog tail: $LOG ==="
    tail -30 "$LOG"
fi

if [ "$RC" -ne 0 ]; then
    echo "run_radiance_muon: simulation exited with status $RC" >&2
    exit "$RC"
fi
echo "run_radiance_muon: OK"
