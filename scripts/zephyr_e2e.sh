#!/usr/bin/env bash
# zephyr_e2e.sh — end-to-end driver for the Merlin × Zephyr × FireSim flow.
#
# Pipeline:
#   1. ./merlin compile  (model.mlir -> mlp.vmfb for zephyr_rocket_rv64)
#   2. ./merlin build --profile zephyr  (cross-build IREE static archives)
#   3. west build  (Zephyr application linking merlin-iree module + .vmfb)
#   4. ./merlin chipyard stage-zephyr-workload  (drop ELF into FireSim deploy)
#   5. firesim infrasetup && firesim runworkload  (FPGA simulation)
#
# Required env (must be exported BEFORE invoking):
#   MERLIN_ROOT       absolute path to the merlin checkout
#   CHIPYARD_ROOT     absolute path to the chipyard checkout
#   ZEPHYR_BASE       Zephyr workspace root  (default:
#                     $CHIPYARD_ROOT/software/zephyrproject/zephyr)
#
# Optional env:
#   MERLIN_MODEL      MLIR / ONNX model to compile (default:
#                     $MERLIN_ROOT/models/mlp/mlp.q.int8.mlir)
#   MERLIN_TARGET     Target spec  (default: zephyr_rocket_rv64)
#   ZEPHYR_BOARD      west -b argument (default: chipyard_riscv64)
#   ZEPHYR_SAMPLE     Sample dir relative to $ZEPHYR_BASE (default:
#                     samples/merlin/model_benchmark)
#   FIRESIM_RECIPE    Merlin recipe name (default: zephyr_quad_rocket)
#   STEPS             Comma-list to skip steps. e.g. STEPS=compile,build runs
#                     ONLY those two. Default = all five.
#
# Exits non-zero on the first step that fails. The uartlog from the FireSim
# run is dumped on success; the FireSim deploy `results-workload/` tree is
# left in place for post-mortem.

set -euo pipefail

# ---------- env validation ----------

: "${MERLIN_ROOT:?export MERLIN_ROOT=/path/to/merlin}"
: "${CHIPYARD_ROOT:?export CHIPYARD_ROOT=/path/to/chipyard}"

ZEPHYR_BASE="${ZEPHYR_BASE:-${CHIPYARD_ROOT}/software/zephyrproject/zephyr}"
export ZEPHYR_BASE

MERLIN_MODEL="${MERLIN_MODEL:-${MERLIN_ROOT}/models/mlp/mlp.q.int8.mlir}"
MERLIN_TARGET="${MERLIN_TARGET:-zephyr_rocket_rv64}"
ZEPHYR_BOARD="${ZEPHYR_BOARD:-chipyard_riscv64}"
ZEPHYR_SAMPLE="${ZEPHYR_SAMPLE:-samples/merlin/model_benchmark}"
FIRESIM_RECIPE="${FIRESIM_RECIPE:-zephyr_quad_rocket}"
STEPS="${STEPS:-compile,build,west,stage,run}"

ZEPHYR_SAMPLE_DIR="${ZEPHYR_BASE}/${ZEPHYR_SAMPLE}"
ZEPHYR_BUILD_DIR="${MERLIN_ROOT}/build/zephyr-app"
export ZEPHYR_BUILD_DIR

# Where the sample expects the .vmfb (same path it falls back to in CMake).
VMFB_DST="${ZEPHYR_SAMPLE_DIR}/data/mlp.vmfb"

step_enabled() {
    case ",${STEPS}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

log() { printf "\n=== %s ===\n" "$*"; }

# ---------- 1. compile ----------

if step_enabled compile; then
    log "1. ./merlin compile  -> ${VMFB_DST}"
    mkdir -p "$(dirname "${VMFB_DST}")"
    (
        cd "${MERLIN_ROOT}"
        ./merlin compile "${MERLIN_MODEL}" \
            --target "${MERLIN_TARGET}" \
            --output "${VMFB_DST}"
    )
    if [ ! -s "${VMFB_DST}" ]; then
        echo "compile produced empty vmfb: ${VMFB_DST}" >&2
        exit 1
    fi
fi

# ---------- 2. build IREE for Zephyr ----------

if step_enabled build; then
    log "2. ./merlin build --profile zephyr"
    (
        cd "${MERLIN_ROOT}"
        ./merlin build --profile zephyr --config release
    )
    # IREE doesn't install runtime archives, so we check the build tree
    # rather than an install/ directory. The merlin-iree Zephyr module
    # picks them up via file(GLOB_RECURSE) over runtime/.
    if [ ! -f "${MERLIN_ROOT}/build/zephyr-vanilla-release/runtime/src/iree/base/libiree_base_base.a" ]; then
        echo "merlin build did not produce libiree_base_base.a" >&2
        echo "  expected under: ${MERLIN_ROOT}/build/zephyr-vanilla-release/" >&2
        exit 1
    fi
fi

# ---------- 3. west build (Zephyr application) ----------

if step_enabled west; then
    log "3. west build -b ${ZEPHYR_BOARD} ${ZEPHYR_SAMPLE}"
    rm -rf "${ZEPHYR_BUILD_DIR}"
    (
        cd "${ZEPHYR_BASE}"
        west build \
            -b "${ZEPHYR_BOARD}" \
            -d "${ZEPHYR_BUILD_DIR}" \
            "${ZEPHYR_SAMPLE_DIR}" \
            -- \
            -DZEPHYR_EXTRA_MODULES="${CHIPYARD_ROOT}/software/zephyrproject/modules/merlin-iree" \
            -DMERLIN_BUILD_DIR="${MERLIN_ROOT}/build/zephyr-vanilla-release" \
            -DMERLIN_IREE_HEADERS_DIR="${MERLIN_ROOT}/third_party/iree_bar/runtime/src" \
            -DMERLIN_VMFB="${VMFB_DST}"
    )
    if [ ! -f "${ZEPHYR_BUILD_DIR}/zephyr/zephyr.elf" ]; then
        echo "west build did not produce zephyr.elf" >&2
        exit 1
    fi
fi

# ---------- 4. stage workload for FireSim ----------

if step_enabled stage; then
    log "4. ./merlin chipyard stage-zephyr-workload ${FIRESIM_RECIPE}"
    (
        cd "${MERLIN_ROOT}"
        ./merlin chipyard stage-zephyr-workload "${FIRESIM_RECIPE}" \
            --elf "${ZEPHYR_BUILD_DIR}/zephyr/zephyr.elf"
    )
fi

# ---------- 5. FireSim run ----------

if step_enabled run; then
    log "5. firesim infrasetup && firesim runworkload"
    DEPLOY_DIR="${CHIPYARD_ROOT}/sims/firesim/deploy"
    (
        cd "${DEPLOY_DIR}"
        firesim infrasetup
        firesim runworkload
    )

    # Tail the most-recent uartlog so the user sees the result inline.
    UARTLOG=$(find "${DEPLOY_DIR}/results-workload" -name uartlog \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -n "${UARTLOG:-}" ] && [ -f "${UARTLOG}" ]; then
        log "uartlog tail: ${UARTLOG}"
        tail -40 "${UARTLOG}"
    else
        echo "  (no uartlog found under results-workload/)" >&2
    fi
fi

log "OK"
