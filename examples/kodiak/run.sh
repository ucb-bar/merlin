#!/usr/bin/env bash
# Kodiak (ucb-bar/zephyr-chipyard-sw @ kodiak): compile int8 models to multicore RVV binaries for a
# tapeout, gate them in simulation, and package them for someone else to run.
#
# Stages are ordered by what they cost and what they need. Run `preflight` first; it says which of them
# this machine can do. See README.md for the provenance of every input.
#
#   ./run.sh setup [--llvm P] [--chipyard D] [--zephyr-sw D] [--zephyr-sdk D] [--m2m D]
#                                 one-step machine setup: model capture env + .env; prints what is left
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh probe                build vlen_probe.elf and self-check it on spike        (no board)
#   ./run.sh capture <model> [--loader loader.py]
#                                 capture any model to an int8 bundle + its W8A8 golden   (no board)
#   ./run.sh op gemm [M N K]      ONE operator (gemm|conv2d|encoder_block|loader.py), RVV vs RVV+OPU,
#                                 both gated on spike, with the routing report             (no board)
#                                 --board firesim_kodiak_opu: built for FireSim, and one FireSim job
#                                 per image is printed (--submit runs them on the shared FPGA)
#   ./run.sh build                build + gate ONE small model at 1 and 2 harts          (no board)
#   ./run.sh package              assemble a full delivery package + zip                 (no board)
#   ./run.sh grade <console.txt> [model]  score a console log a board owner sent back
#   ./run.sh firesim [--models m] [--submit]
#                                 whole model on FireSim (firesim_kodiak_opu): OPU device leg + RVV
#                                 control leg, audited; prints one queue job per leg
#   ./run.sh firesim <job-dir> [--submit]    print (or run) the jobs of an existing job directory
#   ./run.sh firesim-collect <job-dir> [--job device=ID --job control=ID | --uartlog LEG=PATH]
#                                 grade finished FireSim runs: bitstream, build hash, cycles, W8A8
#
# Flags: --dry-run       prints the commands without running them (useful with no toolchain at all)
#        --full          with `build`/`package`, the whole model x hart matrix instead of the cheap subset
#        --opu           with `build`/`package`, ALSO build images that route int8 contractions to the
#                        outer-product unit (OPU), each gated through its scalar stand-in twin
#        --models a,b    with `build`/`package`/`firesim`, these bundles instead of the default set
#                        (a name -> recaptures/<name>_int8_full, or label:/path/to/bundle)
#        --board B       with `op`/`firesim`, build for board B (op default: chipyard_kodiak)
#        --submit        with `op`/`firesim` on a FireSim board, actually run on the shared FPGA;
#                        without it the queue submission is only printed
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
# Kodiak's boards ship with its support package (examples/kodiak/support, vendored from kodiak-mlir).
export MERLIN_BOARD_CATALOG="${MERLIN_BOARD_CATALOG:-$REPO_ROOT/examples/board-catalog.yaml:$REPO_ROOT/examples/kodiak/support}"
# The OPU route's executable support and the unit facts it derives from. The unit and config are named
# by the kodiak package (contracts/matrix_units.yaml, contracts/overlay.yaml), never inferred.
MATRIX_SUPPORT=kodiak
MATRIX_UNIT=saturn_opu
MATRIX_CONFIG=KodiakOPUConfig

BOARD=chipyard_kodiak
# Kodiak's console is HTIF, served by its own loader's --fesvr, so no vendor SDK is needed here. That
# is the main practical difference from the gemmelos example next door.
OUT="$REPO_ROOT/out/artifacts/delivery/example-kodiak"
# FireSim: Kodiak's vector unit with the OPU on ONE Shuttle core (the two-core config does not route on
# the U250). The board descriptor names its bitstream; nothing here does.
FIRESIM_BOARD=firesim_kodiak_opu
# The RVV package FireSim legs are lowered with: the one make_delivery.py packages with by default.
RVV_PACKAGE=rvv/impr_tuned_wholemodel_vf_int8

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
OPU=0; MODELS_OVERRIDE=""; LOADER=""; HARTS_OPT=""; BOARD_OPT=""; SUBMIT=(); REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --opu)     OPU=1 ;;
    --board)   BOARD_OPT="$2"; shift ;;
    --submit)  SUBMIT=(--submit) ;;
    --models)  MODELS_OVERRIDE="$2"; shift ;;
    --loader)  LOADER="$2"; shift ;;
    --harts)   HARTS_OPT="$2"; shift ;;
    *)         REST+=("$1") ;;
  esac
  shift
done
set -- ${REST[@]+"${REST[@]}"}
STAGE="${1:-preflight}"

cd "$REPO_ROOT"

case "$STAGE" in
setup)
  # Merlin's own venv first (everything else runs in it), then the capture environment and .env.
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

probe)
  # The cheapest thing that touches real code: a few-hundred-byte image that reads the chip's own CSRs
  # and prints vlenb / mstatus_vs / misa. `--run` also executes it on spike at two widths, which is how
  # we know the probe reports the width it is given rather than the width it was built for.
  ex_say "build the vector probe and self-check it on spike"
  ex_preflight "$BOARD" --require probe
  ex_run "$PY" -m merlin.runtime.vector_probe --board "$BOARD" --run --spike-vlen 512
  ex_note "on the board this is the FIRST thing to run: it costs seconds and settles VLEN before"
  ex_note "anyone spends minutes uploading a multi-megabyte model image"
  ;;

capture)
  # Any model model2MLIR can trace. With --loader, a loader.py that defines get_model_and_inputs()
  # (see docs/guides/model2mlir.md); without it, a model in model2MLIR's own workloads/ roster.
  # Writes recaptures/<model>_int8_full and the W8A8 golden the images are graded against.
  MODEL="${2:-}"
  if [ -z "$MODEL" ]; then echo "usage: $0 capture <model> [--loader loader.py]" >&2; exit 2; fi
  ex_say "capture $MODEL (int8) and write its W8A8 reference"
  ex_run "$PY" "$EX_LIB_DIR/chip_flow.py" capture "$MODEL" ${LOADER:+--loader "$LOADER"}
  ex_note "the W8A8 reference is Merlin's own int8 datapath on the host (it checks the DEVICE reproduces"
  ex_note "the host). Independent arithmetic check: build_tools/scripts/make_w8a8_independent_golden.py"
  ;;

op)
  # One operator, end to end, to exercise the OPU kernel: capture (once per shape), build as RVV and as
  # RVV+OPU, gate both on spike, print a summary. `op gemm 96 80 200` = M N K; `op conv2d 16 64 16 3` =
  # CIN COUT HW K; or `op path/to/loader.py` for your own one-op model.
  shift
  if [ $# -lt 1 ]; then echo "usage: $0 op <gemm|conv2d|encoder_block|loader.py> [dims...] [--harts N]" >&2; exit 2; fi
  # On a FireSim board (--board firesim_kodiak_opu) the same two images are built for that board, and
  # one FireSim job per image is planned and printed; --submit runs them on the shared FPGA.
  OP_BOARD="${BOARD_OPT:-$BOARD}"
  ex_say "single-operator test on $OP_BOARD: RVV vs RVV+OPU"
  ex_run "$PY" "$EX_LIB_DIR/chip_flow.py" op "$@" --board "$OP_BOARD" --harts "${HARTS_OPT:-1}" \
      --matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG" --matrix-support-target "$MATRIX_SUPPORT" \
      ${SUBMIT[@]+--firesim-submit}
  ;;

firesim)
  # Whole models on FireSim, the way the single-core Kodiak OPU numbers were measured: a DEVICE image
  # (int8 contractions routed to the OPU) and its CONTROL twin (identical lowering, nothing routed),
  # each audited for the unit's instructions, then one queue job per leg. The FPGA is shared, so the
  # jobs are only PRINTED unless --submit is given. `firesim <job-dir>` prints (or submits) the jobs of
  # an existing directory, e.g. one `op --board firesim_kodiak_opu` wrote.
  shift
  if [ $# -ge 1 ] && [ -d "$1" ]; then
    ex_say "FireSim jobs of $1"
    ex_run "$PY" "$EX_LIB_DIR/firesim_flow.py" submit "$@" ${SUBMIT[@]+"${SUBMIT[@]}"}
    exit 0
  fi
  FS_BOARD="${BOARD_OPT:-$FIRESIM_BOARD}"
  if [ -z "${MERLIN_CHIPYARD:-}" ]; then
    ex_note "MERLIN_CHIPYARD is unset: the OPU's encodings come from \$MERLIN_CHIPYARD/generators/saturn"
    ex_note "(pin saturn_opu_int8), and the FireSim deploy tree is found under it"
  fi
  IFS=, read -r -a FS_MODELS <<< "${MODELS_OVERRIDE:-spectformer}"
  for m in "${FS_MODELS[@]}"; do
    ex_say "FireSim legs for $m on $FS_BOARD: OPU device + RVV control"
    ex_run "$PY" "$EX_LIB_DIR/firesim_flow.py" legs "${m}_int8_full" --board "$FS_BOARD" \
        --package "$RVV_PACKAGE" --harts "${HARTS_OPT:-1}" --matrix-unit "$MATRIX_UNIT" \
        --matrix-config "$MATRIX_CONFIG" --matrix-support-target "$MATRIX_SUPPORT" \
        "$@" ${SUBMIT[@]+"${SUBMIT[@]}"}
  done
  ex_note "the legs are not spike-gated here (spike cannot execute the OPU). Gate the model first with"
  ex_note "./run.sh build --opu --models <model>, which runs each OPU image's stand-in twin on spike."
  ;;

firesim-collect)
  # Grade finished runs. FireSim's own HW_CFG_SUMMARY must name the board's bitstream, the console's
  # build_hash must be the planned image's, and the output is graded against the W8A8 golden.
  shift
  if [ $# -lt 1 ]; then echo "usage: $0 firesim-collect <job-dir> [--job LEG=ID ...] [--uartlog LEG=PATH ...]" >&2; exit 2; fi
  ex_say "grade the FireSim runs of $1"
  ex_run "$PY" "$EX_LIB_DIR/firesim_flow.py" collect "$@"
  ;;

build|package)
  ex_preflight "$BOARD" --require package
  if [ "$FULL" = "1" ]; then
    MODELS=spectformer,deepjscc,whisper_tiny_375pos
    HARTS=1,2
    SCALAR=2,3
    ex_note "--full: 12 gated images. Each is one single-threaded functional simulation of a whole"
    ex_note "int8 inference; the scalar ones are 40-100 G cycles. Budget hours, not minutes."
  else
    # deepjscc is the smallest whole-model gate here (~460 M cycles, a few minutes), so the example
    # exercises the entire path without costing an afternoon.
    MODELS=deepjscc
    HARTS=1,2
    SCALAR=""
    ex_note "cheap subset: deepjscc at 1 and 2 harts. Add --full for the shipped matrix."
  fi
  if [ -n "$MODELS_OVERRIDE" ]; then MODELS="$MODELS_OVERRIDE"; fi
  MATRIX_ARGS=()
  if [ "$OPU" = "1" ]; then
    MATRIX_ARGS=(--matrix-harts "$HARTS" --matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG"
                 --matrix-support-target "$MATRIX_SUPPORT")
    ex_note "--opu: each OPU image is gated through its scalar stand-in twin (same lowering, no unit"
    ex_note "instructions), since no simulator here executes the OPU. Each build writes"
    ex_note "routing_report.json: which contractions went to the OPU, which did not and why."
    if [ -z "${MERLIN_CHIPYARD:-}" ]; then
      ex_note "MERLIN_CHIPYARD is unset: the OPU's encodings and tile geometry are derived from"
      ex_note "\$MERLIN_CHIPYARD/generators/saturn (pin saturn_opu_int8); set it first."
    fi
  fi
  ex_say "build for $BOARD, gate every image on spike, audit the ELFs, write $OUT"
  ex_run "$PY" build_tools/scripts/make_delivery.py \
      --board "$BOARD" --models "$MODELS" --harts "$HARTS" \
      ${SCALAR:+--scalar-harts "$SCALAR"} ${MATRIX_ARGS[@]+"${MATRIX_ARGS[@]}"} --debug --out "$OUT"
  ex_note "every binary ships twice: the plain one for a number, and a _debug twin that emits STAGE"
  ex_note "lines, an ALIVE heartbeat naming the op it is inside, and one FAIL line carrying the hash"
  ;;

grade)
  # `grade <console.txt> [model]`. The model cannot be inferred from a log — a package holds several —
  # so it defaults to the one the cheap `build` stage produces.
  LOG="${2:-}"; MODEL="${3:-deepjscc}"
  if [ -z "$LOG" ]; then echo "usage: $0 grade <console.txt> [model]" >&2; exit 2; fi
  # grade.py is vendored INTO each package (numpy only, no merlin checkout), so grading uses the
  # package's own copy — the references it scores against have to be the ones that shipped.
  # A grader can only score a model whose references sit beside it, so choose by model
  # rather than by "first directory on the glob" -- see lib/find_grader.py.
  GRADER="$("$PY" "$EX_LIB_DIR/find_grader.py" --model "${MODEL:-deepjscc}" --prefer "$OUT")" || exit 1
  ex_say "grade $LOG with $(basename "$(dirname "$GRADER")")/grade.py"
  ex_run "$PY" "$GRADER" "$LOG" --model "$MODEL"
  ;;

*)
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
