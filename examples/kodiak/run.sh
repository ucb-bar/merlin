#!/usr/bin/env bash
# Kodiak (ucb-bar/zephyr-chipyard-sw @ kodiak): compile int8 models to multicore RVV binaries for a
# tapeout, gate them in simulation, and package them for someone else to run.
#
# Stages are ordered by what they cost and what they need. Run `preflight` first; it says which of them
# this machine can do. See README.md for the provenance of every input.
#
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh probe                build vlen_probe.elf and self-check it on spike        (no board)
#   ./run.sh capture <model> [--loader loader.py]
#                                 capture any model to an int8 bundle + its W8A8 golden   (no board)
#   ./run.sh build                build + gate ONE small model at 1 and 2 harts          (no board)
#   ./run.sh package              assemble a full delivery package + zip                 (no board)
#   ./run.sh grade <console.txt> [model]  score a console log a board owner sent back
#
# Flags: --dry-run       prints the commands without running them (useful with no toolchain at all)
#        --full          with `build`/`package`, the whole model x hart matrix instead of the cheap subset
#        --opu           with `build`/`package`, ALSO build images that route int8 contractions to the
#                        outer-product unit (OPU), each gated through its scalar stand-in twin
#        --models a,b    with `build`/`package`, these bundles instead of the default set
#                        (a name -> recaptures/<name>_int8_full, or label:/path/to/bundle)
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

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
OPU=0; MODELS_OVERRIDE=""; LOADER=""; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --opu)     OPU=1 ;;
    --models)  MODELS_OVERRIDE="$2"; shift ;;
    --loader)  LOADER="$2"; shift ;;
    *)         REST+=("$1") ;;
  esac
  shift
done
set -- ${REST[@]+"${REST[@]}"}
STAGE="${1:-preflight}"

cd "$REPO_ROOT"

case "$STAGE" in
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
  # Capture is weight-only int8; Merlin then computes W8A8 (activations quantized too), so the
  # reference a W8A8 image is graded against is written next to the bundle by make_w8a8_golden.py.
  MODEL="${2:-}"
  if [ -z "$MODEL" ]; then echo "usage: $0 capture <model> [--loader loader.py]" >&2; exit 2; fi
  # `make_delivery --models <name>` reads recaptures/<name>_int8_full, so that is where it goes.
  BUNDLE="$("$PY" -c "from merlin.common.artifacts import recaptures_dir; print(recaptures_dir())")/${MODEL}_int8_full"
  ex_say "capture $MODEL (int8) into $BUNDLE"
  # model2MLIR's checkout, as Merlin resolves it (process env, then .env).
  M2M_DIR="$("$PY" -c "from merlin.common.paths import env; print(env('MERLIN_M2M_DIR') or '')")"
  if [ -z "$M2M_DIR" ]; then echo "set MERLIN_M2M_DIR to a model2MLIR checkout" >&2; exit 2; fi
  if [ -n "$LOADER" ]; then
    M2M_PY="$("$PY" -c "from merlin.common.paths import env; print(env('MERLIN_M2M_PYTHON') or '')")"
    if [ -z "$M2M_PY" ]; then echo "set MERLIN_M2M_PYTHON (the model2MLIR venv's python)" >&2; exit 2; fi
    ex_run "$M2M_PY" src/merlin/targetgen/_m2m_capture_worker.py --m2m-dir "$M2M_DIR" \
        --loader "$LOADER" --dtype int8 --seed 0 --materialize-bundle --out "$BUNDLE"
  else
    # A model in model2MLIR's workloads/ roster, captured in that model's own venv (capture.toml).
    M2M_PY="$("$PY" -c "from pathlib import Path; from merlin.compile.bundles import _capture_python; print(_capture_python(Path('$M2M_DIR'), '$MODEL'))")"
    ex_run "$M2M_PY" "$M2M_DIR/workloads/capture_consistent.py" "$MODEL" int8 "$BUNDLE"
  fi
  ex_say "write the W8A8 reference the images are graded against"
  ex_run "$PY" build_tools/scripts/make_w8a8_golden.py "$BUNDLE"
  ex_note "that reference is Merlin's own int8 datapath on the host (it checks the DEVICE reproduces the"
  ex_note "host). For an independent arithmetic check: build_tools/scripts/make_w8a8_independent_golden.py"
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
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
