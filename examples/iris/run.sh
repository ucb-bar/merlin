#!/usr/bin/env bash
# Iris (chipyard iris-cy, IrisConfig): compile and gate models and single operators for its RTL simulation
# board. RVV on all three harts today; the VME matrix unit is added to the iris support package next.
#
#   ./run.sh setup [--llvm P] [--chipyard D] [--zephyr-sw D] [--zephyr-sdk D] [--m2m D]
#                                 one-step machine setup: model capture env + .env; prints what is left
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh capture <model> [--loader loader.py]
#                                 capture any model to an int8 bundle + its W8A8 golden
#   ./run.sh op gemm [M N K] [--harts N]
#                                 ONE operator (gemm|conv2d|encoder_block|loader.py), gated on spike
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
# Iris's boards ship with its support package (examples/iris/support, vendored from iris-mlir).
export MERLIN_BOARD_CATALOG="${MERLIN_BOARD_CATALOG:-$REPO_ROOT/examples/board-catalog.yaml:$REPO_ROOT/examples/iris/support}"
BOARD=iris_rtl

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
LOADER=""; HARTS_OPT=""; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --loader) LOADER="$2"; shift ;;
    --harts)  HARTS_OPT="$2"; shift ;;
    *)        REST+=("$1") ;;
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
  shift
  if [ $# -lt 1 ]; then echo "usage: $0 op <gemm|conv2d|encoder_block|loader.py> [dims...] [--harts N]" >&2; exit 2; fi
  ex_say "single-operator test on $BOARD (RVV; the VME unit is not routed yet)"
  ex_run "$PY" "$EX_LIB_DIR/chip_flow.py" op "$@" --board "$BOARD" --harts "${HARTS_OPT:-1}"
  ;;
*)
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
