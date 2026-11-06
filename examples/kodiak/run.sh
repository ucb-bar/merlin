#!/usr/bin/env bash
# Kodiak (ucb-bar/zephyr-chipyard-sw @ kodiak): compile int8 models to multicore RVV binaries for a
# tapeout, gate them in simulation, and package them for someone else to run.
#
# Stages are ordered by what they cost and what they need. Run `preflight` first; it says which of them
# this machine can do. See README.md for the provenance of every input.
#
#   ./run.sh setup [--llvm P] [--chipyard D] [--zephyr-sw D] [--zephyr-sdk D] [--m2m D]
#                                 one-step machine setup: the kodiak-mlir support package at its pinned
#                                 commit, capture venv, Zephyr workspace, OPU spike model, .env; verifies
#                                 the rest and prints how to get what is missing
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh probe                build vlen_probe.elf and self-check it on spike        (no board)
#   ./run.sh capture <model> [--loader loader.py]
#                                 capture any model to an int8 bundle + its W8A8 golden   (no board)
#   ./run.sh op gemm [M N K]      ONE operator (gemm|gemm_nt|conv2d|attention|encoder_block|loader.py), RVV vs RVV+OPU,
#                                 both gated on spike, with the routing report             (no board)
#                                 --board firesim_kodiak_opu: built for FireSim, and one FireSim job
#                                 per image is printed (--submit runs them on the shared FPGA)
#   ./run.sh spike-model          build the OPU's spike functional model from the selected Kodiak
#                                 package (needs its spike/ dir) into out/build/spike-ext/  (no board)
#   ./run.sh build                build + gate ONE small model at 1 and 2 harts          (no board)
#   ./run.sh package              assemble a full delivery package + zip                 (no board)
#   ./run.sh matrix --models a,b [--harts 1,2]
#                                 model regression matrix: each model x harts x {RVV, RVV+OPU}, every
#                                 image gated on spike, h2 checked bit-identical to h1   (no board)
#   ./run.sh kernel-test [kernel] [--quick|--full] [--list]
#                                 check ONE OPU kernel in seconds: a sweep of shapes (edges, K tails, tile
#                                 multiples, bias, back-to-back calls) on spike's OPU model, bit-exact
#                                 against Merlin's reference. kernel: a name in the package's
#                                 kernels/kernels.yaml or path/to/kernel.c[:function]   (no board)
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
#        --matrix-kernel K  with `op`/`build`/`package`/`matrix`/`firesim`, every routed contraction calls
#                        OPU kernel K: a name in kernels/kernels.yaml or path/to/kernel.c[:function]
#                        (default: $MERLIN_MATRIX_KERNEL, else the package's default kernel)
#        --rvv-package P  with `build`/`package`/`firesim`, select an RVV schedule package, e.g.
#                        rvv/kodiak_prepack_w8a8_v1 (default: the K1-certified wholemodel package)
#        --no-spike-backends matrix  with `build`/`package`, build and audit matrix images without
#                        running the slow scalar twin; the delivery marks those images UNGATED
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"
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
RVV_PACKAGE="${MERLIN_RVV_PACKAGE:-rvv/impr_tuned_wholemodel_vf_int8}"

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
OPU=0; MODELS_OVERRIDE=""; LOADER=""; HARTS_OPT=""; BOARD_OPT=""; NO_SPIKE_BACKENDS=""; SUBMIT=(); REST=()
MATRIX_KERNEL="${MERLIN_MATRIX_KERNEL:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --opu)     OPU=1 ;;
    --matrix-kernel) MATRIX_KERNEL="$2"; shift ;;
    --rvv-package) RVV_PACKAGE="$2"; shift ;;
    --no-spike-backends) NO_SPIKE_BACKENDS="$2"; shift ;;
    --board)   BOARD_OPT="$2"; shift ;;
    --submit)  SUBMIT=(--submit) ;;
    --models)  MODELS_OVERRIDE="$2"; shift ;;
    --loader)  LOADER="$(ex_abs "$2")"; shift ;;
    --harts)   HARTS_OPT="$2"; shift ;;
    *)         REST+=("$1") ;;
  esac
  shift
done
set -- ${REST[@]+"${REST[@]}"}
STAGE="${1:-preflight}"
# The selected OPU kernel, passed to every stage that builds a routed image (empty: the package default).
KERNEL_ARGS=()
if [ -n "$MATRIX_KERNEL" ]; then KERNEL_ARGS=(--matrix-kernel "$(ex_kernel_spec "$MATRIX_KERNEL")"); fi

cd "$REPO_ROOT"
RVV_PACKAGE_PATH="$RVV_PACKAGE"
case "$RVV_PACKAGE" in
  /*) ;;
  *) RVV_PACKAGE_PATH="$REPO_ROOT/out/artifacts/targets/$RVV_PACKAGE" ;;
esac
# Kodiak's support package (and its boards) is kodiak-mlir at the commit Merlin pins, fetched by setup.
case "$STAGE" in setup|help|-h|--help) ;; *) ex_select_support "$MATRIX_SUPPORT" ;; esac

case "$STAGE" in
setup)
  # The common prerequisites, plus Kodiak's: the OPU's RTL in $MERLIN_CHIPYARD/generators/saturn (pin
  # saturn_opu_int8, verified and reported) and the package's spike model of the OPU (built here).
  shift
  ex_setup "$BOARD" --support "$MATRIX_SUPPORT" --verify-pin saturn_opu_int8 --spike-model "$MATRIX_SUPPORT" "$@"
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
  if [ $# -lt 1 ]; then echo "usage: $0 op <gemm|gemm_nt|conv2d|attention|encoder_block|loader.py> [dims...] [--harts N]" >&2; exit 2; fi
  case "$1" in *.py) set -- "$(ex_abs "$1")" "${@:2}" ;; esac
  # On a FireSim board (--board firesim_kodiak_opu) the same two images are built for that board, and
  # one FireSim job per image is planned and printed; --submit runs them on the shared FPGA.
  OP_BOARD="${BOARD_OPT:-$BOARD}"
  ex_say "single-operator test on $OP_BOARD: RVV vs RVV+OPU"
  ex_run "$PY" "$EX_LIB_DIR/chip_flow.py" op "$@" --board "$OP_BOARD" --harts "${HARTS_OPT:-1}" \
      --matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG" --matrix-support-target "$MATRIX_SUPPORT" \
      ${KERNEL_ARGS[@]+"${KERNEL_ARGS[@]}"} ${SUBMIT[@]+--firesim-submit}
  ;;

kernel-test)
  # One OPU kernel against Merlin's reference on spike's OPU model, in seconds: no model, no Zephyr. The
  # kernel is the first argument or --matrix-kernel (a name in the package's kernels/kernels.yaml, or
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

firesim)
  # Whole models on FireSim, the way the single-core Kodiak OPU numbers were measured: a DEVICE image
  # (int8 contractions routed to the OPU) and its plain-RVV CONTROL image, each gated on spike before
  # planning one queue job per leg. The FPGA is shared, so the
  # jobs are only PRINTED unless --submit is given. `firesim <job-dir>` prints (or submits) the jobs of
  # an existing directory, e.g. one `op --board firesim_kodiak_opu` wrote.
  shift
  if [ $# -ge 1 ]; then set -- "$(ex_abs "$1")" "${@:2}"; fi
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
        ${KERNEL_ARGS[@]+"${KERNEL_ARGS[@]}"} "$@" ${SUBMIT[@]+"${SUBMIT[@]}"}
  done
  ex_note "both exact FireSim ELFs passed on Spike before this plan: the OPU image used its package's"
  ex_note "functional model; the job plan records the selected twin or independent-RVV identity gate."
  ;;

firesim-collect)
  # Grade finished runs. FireSim's own HW_CFG_SUMMARY must name the board's bitstream, the console's
  # build_hash must be the planned image's, and the output is graded against the W8A8 golden.
  shift
  if [ $# -lt 1 ]; then echo "usage: $0 firesim-collect <job-dir> [--job LEG=ID ...] [--uartlog LEG=PATH ...]" >&2; exit 2; fi
  set -- "$(ex_abs "$1")" "${@:2}"
  ex_say "grade the FireSim runs of $1"
  ex_run "$PY" "$EX_LIB_DIR/firesim_flow.py" collect "$@"
  ;;

spike-model)
  # The OPU's functional model for spike is OWNED by the Kodiak package (its spike/ directory): the
  # instruction words and geometry are generated from the RTL at build time. Merlin only loads it when the
  # selected package's contract declares it (runner.spike_extension: name, extlib_env, sha256).
  PKG="$SUPPORT_ROOT"
  if [ ! -f "$PKG/spike/build.py" ]; then
    echo "the selected Kodiak package ($PKG) has no spike/ model; select one that does with" >&2
    echo "MERLIN_TARGET_PATH=<kodiak-mlir checkout>" >&2
    exit 1
  fi
  ex_say "build the OPU spike model from $PKG"
  ex_run "$PY" "$PKG/spike/build.py" --out "$REPO_ROOT/out/build/spike-ext/kodiak"
  ex_note "export MERLIN_KODIAK_OPU_SPIKE_EXTLIB=$REPO_ROOT/out/build/spike-ext/kodiak/libsaturn_opu_model.so"
  ex_note "then \`op\` also runs the SHIPPED OPU image on spike (leg matrix_on_spike). Any build is accepted"
  ex_note "whose manifest records the model inputs the package declares (built_from) and a passing self-test."
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
  if [ -n "$HARTS_OPT" ]; then HARTS="$HARTS_OPT"; fi
  MATRIX_ARGS=()
  if [ "$OPU" = "1" ]; then
    MATRIX_ARGS=(--matrix-harts "$HARTS" --matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG"
                 --matrix-support-target "$MATRIX_SUPPORT" ${KERNEL_ARGS[@]+"${KERNEL_ARGS[@]}"})
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
      ${SCALAR:+--scalar-harts "$SCALAR"} ${MATRIX_ARGS[@]+"${MATRIX_ARGS[@]}"} \
      ${NO_SPIKE_BACKENDS:+--no-spike-backends "$NO_SPIKE_BACKENDS"} \
      --rvv-package "$RVV_PACKAGE_PATH" --debug --out "$OUT"
  ex_note "every binary ships twice: the plain one for a number, and a _debug twin that emits STAGE"
  ex_note "lines, an ALIVE heartbeat naming the op it is inside, and one FAIL line carrying the hash"
  ;;

matrix)
  # The regression matrix over whole models: for each captured model (recaptures/<model>_int8_full) and
  # hart count, the plain RVV image and the OPU image (its stand-in twin, then the shipped image on the
  # OPU's spike model, bit-exact against the twin), each graded against the W8A8 golden, plus every hart
  # count's output against 1 hart's. Results kept per cell; a rerun builds only the missing ones.
  if [ -z "$MODELS_OVERRIDE" ]; then echo "usage: $0 matrix --models a,b [--harts 1,2] [--rerun]" >&2; exit 2; fi
  ex_preflight "$BOARD" --require package
  ex_say "model matrix on $BOARD: $MODELS_OVERRIDE"
  ex_run "$PY" "$EX_LIB_DIR/model_matrix.py" --board "$BOARD" --models "$MODELS_OVERRIDE" \
      --harts "${HARTS_OPT:-1,2}" --matrix-unit "$MATRIX_UNIT" --matrix-config "$MATRIX_CONFIG" \
      --matrix-support-target "$MATRIX_SUPPORT" ${KERNEL_ARGS[@]+"${KERNEL_ARGS[@]}"} "${@:2}"
  ex_note "table: out/artifacts/probes/model_matrix/$BOARD${MATRIX_KERNEL:++kernel-<name>}/MATRIX.md (one JSON per cell beside it)"
  ;;

grade)
  # `grade <console.txt> [model]`. The model cannot be inferred from a log — a package holds several —
  # so it defaults to the one the cheap `build` stage produces.
  LOG="$(ex_abs "${2:-}")"; MODEL="${3:-deepjscc}"
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
  sed -n '2,51p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
