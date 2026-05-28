#!/usr/bin/env bash
# radiance_muon_smoke.sh — single-command end-to-end Merlin × Radiance smoke.
#
# Pipeline:
#   1. compile  : ./merlin compile <descriptor>.yaml --target radiance_muon
#   2. (build is implicit inside compile.py for this target)
#   3. run      : ./merlin chipyard run-radiance-muon radiance-muon
#
# Required env:
#   $LLVM_MUON              path to radiance-kernels/llvm/llvm-muon
#   $RADIANCE_KERNELS_ROOT  radiance-kernels checkout
#   $CHIPYARD_ROOT          chipyard checkout
#
# Optional:
#   $MERLIN_RADIANCE_DESCRIPTOR  yaml descriptor (default: vecadd.yaml)
#   $STEPS                       comma-list to skip stages
#                                (e.g. STEPS=compile to only emit the ELF)

set -euo pipefail

: "${LLVM_MUON:?export LLVM_MUON=$RADIANCE_KERNELS_ROOT/llvm/llvm-muon}"
: "${RADIANCE_KERNELS_ROOT:?export RADIANCE_KERNELS_ROOT=/path/to/radiance-kernels}"
: "${CHIPYARD_ROOT:?export CHIPYARD_ROOT=/path/to/chipyard}"

MERLIN_ROOT="${MERLIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DESCRIPTOR="${MERLIN_RADIANCE_DESCRIPTOR:-${MERLIN_ROOT}/models/radiance_muon/vecadd.yaml}"
STEPS="${STEPS:-compile,run}"

step_enabled() {
    case ",${STEPS}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

log() { printf "\n=== %s ===\n" "$*"; }

# --- 1. compile (emits kernel.cpp + builds kernel.radiance.elf) ----------

if step_enabled compile; then
    log "1. ./merlin compile  -> kernel.radiance.elf"
    (
        cd "${MERLIN_ROOT}"
        ./merlin compile "${DESCRIPTOR}" --target radiance_muon
    )
fi

# --- 2. run (RadianceMuonConfig sim via chipyard.py) --------------------

if step_enabled run; then
    log "2. ./merlin chipyard run-radiance-muon"
    (
        cd "${MERLIN_ROOT}"
        ./merlin chipyard run-radiance-muon radiance-muon
    )
fi

log "OK"
