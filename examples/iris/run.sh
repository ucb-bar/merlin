#!/usr/bin/env bash
# Iris (chipyard iris-cy, IrisConfig): compile and gate models and single operators for its RTL simulation
# board. RVV on all three harts; int8 contractions routed to the VME matrix unit on harts 0-1.
#
#   ./run.sh setup [--llvm P] [--chipyard D] [--zephyr-sw D] [--zephyr-sdk D] [--m2m D]
#                                 one-step machine setup: the iris-mlir support package at its pinned
#                                 commit, capture venv, Zephyr workspace, the pinned
#                                 chipyard iris-cy and Zvt spike checkouts (verified), the Zvt spike
#                                 build, .env; prints how to get whatever is missing
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh capture <model> [--loader loader.py]
#                                 capture any model to an int8 bundle + its W8A8 golden
#   ./run.sh op gemm [M N K] [--harts N]
#                                 ONE operator (gemm|conv2d|encoder_block|loader.py): RVV vs RVV+VME. The
#                                 VME image's stand-in twin is gated on spike; with $MERLIN_SPIKE_ZVT the
#                                 SHIPPED VME image also runs on a spike that implements the unit
#   ./run.sh op gemm --rvv-only   the RVV leg alone (no VME route)
#   ./run.sh op gemm --matrix-kernel K
#                                 every routed contraction calls VME kernel K: a name in the package's
#                                 kernels/kernels.yaml or path/to/kernel.c[:function] (default:
#                                 $MERLIN_MATRIX_KERNEL, else the package's default kernel)
#   ./run.sh kernel-test [kernel] [--quick|--full] [--list]
#                                 check ONE VME kernel in seconds: a sweep of shapes (edges, K tails, tile
#                                 multiples, bias, back-to-back calls) on the Zvt spike, bit-exact
#                                 against Merlin's reference
#
# The VME route derives the unit's encodings and geometry from $MERLIN_CHIPYARD_IRIS (a chipyard iris-cy
# checkout at pin chipyard_iris, with generators/saturn at saturn_vme and generators/rocket-chip at
# rocket_chip_iris); see README.md.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
BOARD=iris_rtl
# The VME route's executable support and the unit facts it derives from, named by the iris package
# (contracts/matrix_units.yaml, contracts/overlay.yaml), never inferred.
MATRIX_SUPPORT=iris
MATRIX_UNIT=saturn_vme
MATRIX_CONFIG=IrisConfig

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
LOADER=""; HARTS_OPT=""; RVV_ONLY=0; REST=()
MATRIX_KERNEL="${MERLIN_MATRIX_KERNEL:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --loader)   LOADER="$(ex_abs "$2")"; shift ;;
    --matrix-kernel) MATRIX_KERNEL="$2"; shift ;;
    --harts)    HARTS_OPT="$2"; shift ;;
    --rvv-only) RVV_ONLY=1 ;;
    *)          REST+=("$1") ;;
  esac
  shift
done
set -- ${REST[@]+"${REST[@]}"}
STAGE="${1:-preflight}"
KERNEL_ARGS=()
if [ -n "$MATRIX_KERNEL" ]; then KERNEL_ARGS=(--matrix-kernel "$(ex_kernel_spec "$MATRIX_KERNEL")"); fi
cd "$REPO_ROOT"
# Iris's support package (and its boards) is iris-mlir at the commit Merlin pins, fetched by setup.
case "$STAGE" in setup|help|-h|--help) ;; *) ex_select_support "$MATRIX_SUPPORT" ;; esac

case "$STAGE" in
setup)
  # The common prerequisites, plus Iris's: the chipyard iris-cy checkout the VME facts are derived from
  # (pin chipyard_iris, with saturn_vme and rocket_chip_iris nested in it) and riscv-isa-sim with Zvt (pin
  # spike_zvt). Each is found through its variable (MERLIN_CHIPYARD_IRIS, MERLIN_SPIKE_ZVT_SRC) or under
  # out/build/ext/<pin>/, verified by content, or the commands to fetch it are printed. The Zvt spike is
  # then built from the verified source into out/build/spike_zvt/ and recorded as MERLIN_SPIKE_ZVT.
  shift
  ex_setup "$BOARD" --support "$MATRIX_SUPPORT" --checkout-pin chipyard_iris --checkout-pin spike_zvt --build-spike spike_zvt=MERLIN_SPIKE_ZVT "$@"
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
  case "$1" in *.py) set -- "$(ex_abs "$1")" "${@:2}" ;; esac
  MATRIX_ARGS=()
  if [ "$RVV_ONLY" = "1" ]; then
    ex_say "single-operator test on $BOARD (RVV only)"
  else
    MATRIX_ARGS=(--matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG" --matrix-support-target "$MATRIX_SUPPORT"
                 ${KERNEL_ARGS[@]+"${KERNEL_ARGS[@]}"})
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
kernel-test)
  # One VME kernel against Merlin's reference on the Zvt spike, in seconds: no model, no Zephyr. The kernel
  # is the first argument or --matrix-kernel (a name in the package's kernels/kernels.yaml, or
  # path/to/kernel.c[:function]); neither, the package default. --quick: small shapes only; --full: also the
  # large corpus shapes; --list: the registered kernels.
  shift
  KT=()
  if [ $# -ge 1 ] && [ "${1#--}" = "$1" ]; then KT=("$(ex_kernel_spec "$1")"); shift
  elif [ -n "$MATRIX_KERNEL" ]; then KT=("$(ex_kernel_spec "$MATRIX_KERNEL")"); fi
  if [ "$FULL" = "1" ]; then set -- "$@" --full; fi
  ex_say "kernel-test ${KT[0]:-<package default>} for $MATRIX_UNIT ($MATRIX_CONFIG) at $BOARD's VLEN"
  ex_run "$PY" -m merlin.kernels.matrix_kernel_test ${KT[@]+"${KT[@]}"} --support-target "$MATRIX_SUPPORT" \
      --unit "$MATRIX_UNIT" --config "$MATRIX_CONFIG" --board "$BOARD" "$@"
  ;;
*)
  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
