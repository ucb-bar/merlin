#!/usr/bin/env bash
# Kodiak (ucb-bar/zephyr-chipyard-sw @ kodiak): compile int8 models to multicore RVV binaries for a
# tapeout, gate them in simulation, and package them for someone else to run.
#
# Stages are ordered by what they cost and what they need. Run `preflight` first; it says which of them
# this machine can do. See README.md for the provenance of every input.
#
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh probe                build vlen_probe.elf and self-check it on spike        (no board)
#   ./run.sh build                build + gate ONE small model at 1 and 2 harts          (no board)
#   ./run.sh package              assemble a full delivery package + zip                 (no board)
#   ./run.sh grade <console.txt>  score a console log a board owner sent back
#
# Flags: --dry-run prints the commands without running them (useful with no toolchain at all)
#        --full    with `build`/`package`, the whole model x hart matrix instead of the cheap subset
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

BOARD=chipyard_kodiak
# Kodiak's console is HTIF, served by its own loader's --fesvr, so no vendor SDK is needed here. That
# is the main practical difference from the gemmelos example next door.
OUT="$REPO_ROOT/out/artifacts/delivery/example-kodiak"

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
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
  ex_say "build for $BOARD, gate every image on spike, audit the ELFs, write $OUT"
  ex_run "$PY" build_tools/scripts/make_delivery.py \
      --board "$BOARD" --models "$MODELS" --harts "$HARTS" \
      ${SCALAR:+--scalar-harts "$SCALAR"} --debug --out "$OUT"
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
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
