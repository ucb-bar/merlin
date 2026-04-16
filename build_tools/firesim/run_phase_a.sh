#!/usr/bin/env bash
# One-shot Phase A opcode survey on FireSim:
#   4 pre-built rvvtest binaries × 4 FireSim runs × (skip mask 0x00, 0x08, 0x18, 0x38).
#   Each run: kill stale sim → stage pre-built binary → infrasetup → runworkload
#             (host timeout) → kill → parse uartlog → print verdict.
#
# Produces /tmp/phaseA_<mask>.log per run and a consolidated summary at the
# end. Requires that build_rvv_selftest.sh has already produced the 4
# binaries under build/firesim-rvvtest/skip_*/.
#
# Usage:
#   bash build_tools/firesim/run_phase_a.sh          # default 360s timeout per run
#   bash build_tools/firesim/run_phase_a.sh 480      # custom per-run timeout

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TIMEOUT="${1:-360}"
MASKS=(0x00 0x08 0x18 0x38)

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# Pre-flight: all 4 pre-built binaries must exist.
for mask in "${MASKS[@]}"; do
	pre="$MERLIN_ROOT/build/firesim-rvvtest/skip_$mask/bench_model_opu_bench_vit_small_rvvtest"
	if [ ! -f "$pre" ]; then
		echo "Missing pre-built binary: $pre"
		echo "Run:  bash $MERLIN_ROOT/build_tools/firesim/build_rvv_selftest.sh"
		exit 1
	fi
done

for mask in "${MASKS[@]}"; do
	echo
	echo "################################################################"
	echo "##  Phase A — skip=$mask"
	echo "################################################################"
	bash "$MERLIN_ROOT/build_tools/firesim/run_rvv_selftest.sh" "$mask" "$TIMEOUT" 2>&1 |
		tee "/tmp/phaseA_${mask}.log"
done

echo
echo
echo "================ Phase A Summary ================"
for mask in "${MASKS[@]}"; do
	echo
	echo "---- skip=$mask ----"
	grep -E '\[rvv\] cp=|verdict|Stopped after|ALL ENABLED' "/tmp/phaseA_${mask}.log" 2>/dev/null | tail -20
done
