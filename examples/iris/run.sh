#!/usr/bin/env bash
# Iris (chipyard iris-cy, IrisConfig): compile and gate models and single operators for its RTL simulation
# board. RVV on all three harts; int8 contractions routed to the VME matrix unit on harts 0-1.
#
#   ./run.sh setup [--llvm P] [--chipyard D] [--zephyr-sw D] [--zephyr-sdk D] [--m2m D]
#                                 one-step machine setup: model capture env + .env; prints what is left
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh capture <model> [--loader loader.py]
#                                 capture any model to an int8 bundle + its W8A8 golden
#   ./run.sh op gemm [M N K] [--harts N]
#                                 ONE operator (gemm|conv2d|encoder_block|loader.py): RVV vs RVV+VME. The
#                                 VME image's stand-in twin is gated on spike; with $MERLIN_SPIKE_ZVT the
#                                 SHIPPED VME image also runs on a spike that implements the unit
#   ./run.sh op gemm --rvv-only   the RVV leg alone (no VME route)
#
# The VME route derives the unit's encodings and geometry from $MERLIN_CHIPYARD_IRIS (a chipyard iris-cy
# checkout at pin chipyard_iris, with generators/saturn at saturn_vme and generators/rocket-chip at
# rocket_chip_iris); see README.md.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
# Iris's boards ship with its support package (examples/iris/support, vendored from iris-mlir).
export MERLIN_BOARD_CATALOG="${MERLIN_BOARD_CATALOG:-$REPO_ROOT/examples/board-catalog.yaml:$REPO_ROOT/examples/iris/support}"
BOARD=iris_rtl
# The VME route's executable support and the unit facts it derives from, named by the iris package
# (contracts/matrix_units.yaml, contracts/overlay.yaml), never inferred.
MATRIX_SUPPORT=iris
MATRIX_UNIT=saturn_vme
MATRIX_CONFIG=IrisConfig

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
LOADER=""; HARTS_OPT=""; RVV_ONLY=0; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --loader)   LOADER="$2"; shift ;;
    --harts)    HARTS_OPT="$2"; shift ;;
    --rvv-only) RVV_ONLY=1 ;;
    *)          REST+=("$1") ;;
  esac
  shift
done
set -- ${REST[@]+"${REST[@]}"}
STAGE="${1:-preflight}"
cd "$REPO_ROOT"

case "$STAGE" in
setup)
  if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    ex_run uv sync --extra dev --extra xdsl --extra targetgen
    PY="$REPO_ROOT/.venv/bin/python"
  fi
  shift
  ex_run "$PY" "$EX_LIB_DIR/chip_setup.py" "$@" || true
  ex_preflight "$BOARD"
  ;;
preflight)
  ex_preflight "$BOARD"
  ;;
capture)
  MODEL="${2:-}"
  if [ -z "$MODEL" ]; then echo "usage: $0 capture <model> [--loader loader.py]" >&2; exit 2; fi
  ex_run "$PY" "$EX_LIB_DIR/chip_flow.py" capture "$MODEL" ${LOADER:+--loader "$LOADER"}
  ;;
op)
  # One operator, end to end: capture (once per shape), build as RVV and as RVV+VME, gate both on spike.
  # `op gemm 96 80 200` = M N K; `op conv2d 16 64 16 3` = CIN COUT HW K; or `op path/to/loader.py`.
  shift
  if [ $# -lt 1 ]; then echo "usage: $0 op <gemm|conv2d|encoder_block|loader.py> [dims...] [--harts N]" >&2; exit 2; fi
  MATRIX_ARGS=()
  if [ "$RVV_ONLY" = "1" ]; then
    ex_say "single-operator test on $BOARD (RVV only)"
  else
    MATRIX_ARGS=(--matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG" --matrix-support-target "$MATRIX_SUPPORT")
    ex_say "single-operator test on $BOARD: RVV vs RVV+VME"
    if [ -z "${MERLIN_CHIPYARD_IRIS:-}" ]; then
      ex_note "MERLIN_CHIPYARD_IRIS is unset: the VME encodings and geometry are derived from"
      ex_note "\$MERLIN_CHIPYARD_IRIS/generators/{saturn,rocket-chip} (pins saturn_vme, rocket_chip_iris)"
    fi
    if [ -z "${MERLIN_SPIKE_ZVT:-}" ]; then
      ex_note "MERLIN_SPIKE_ZVT is unset: only the stand-in twin is gated; the shipped VME image runs only"
      ex_note "on a spike built with Zvt (pin spike_zvt, see README.md)"
    fi
  fi
  ex_run "$PY" "$EX_LIB_DIR/chip_flow.py" op "$@" --board "$BOARD" --harts "${HARTS_OPT:-1}" \
      ${MATRIX_ARGS[@]+"${MATRIX_ARGS[@]}"}
  ;;
*)
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
