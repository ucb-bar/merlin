#!/usr/bin/env bash
# gemmelos bearly25: the same flow as the kodiak example, against a chip with NO Zephyr port of its own
# and no host-assisted console — so every console and clock fact is derived from that chip's own SDK
# headers at build time. That derivation is the interesting part, and it is why this example needs a
# --sdk-dir while the kodiak one does not.
#
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh probe                build vlen_probe.elf and self-check it on spike        (no board)
#   ./run.sh facts                print the console/clock facts derived from the SDK headers
#   ./run.sh build                build + gate ONE small model at 1 and 2 harts          (no board)
#   ./run.sh package              assemble a full delivery package + zip                 (no board)
#   ./run.sh grade <console.txt>  score a console log a board owner sent back
#
# The SDK checkout: export GEMMELOS_SDK=/path/to/gemmelos-bringup, or pass --sdk-dir <path>.
# Flags: --dry-run prints commands without running; --full builds the shipped matrix, not a subset.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

# The 50 MHz reset-clock variant is the one to run first: it is the directly gated set, and it needs no
# PLL bring-up. `BOARD=gemmelos_bearly25_zephyr_500mhz ./run.sh package` builds the raised-clock twin.
BOARD="${BOARD:-gemmelos_bearly25_zephyr}"
OUT="$REPO_ROOT/out/artifacts/delivery/example-${BOARD}"

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
STAGE="${1:-preflight}"

SDK="${GEMMELOS_SDK:-}"
if [ "${2:-}" = "--sdk-dir" ]; then SDK="${3:-}"; fi

cd "$REPO_ROOT"

need_sdk() {
  if [ -z "$SDK" ] || [ ! -d "$SDK" ]; then
    cat >&2 <<EOF
This board's console is its OWN UART, so its base address and the clock rates its baud divisor depends
on are derived from the chip's SDK headers rather than hardcoded. Point the example at a checkout:

  export GEMMELOS_SDK=/path/to/gemmelos-bringup
  $0 $STAGE

That SDK is a third-party bring-up repo and is not ours to redistribute; see README.md ("Inputs, and
how to get each one") for what it supplies and what to do if you cannot get it. Every stage above
\`facts\` runs without it.
EOF
    exit 1
  fi
}

case "$STAGE" in
preflight)
  ex_preflight "$BOARD" ${SDK:+--sdk-dir "$SDK"}
  ;;

probe)
  ex_say "build the vector probe and self-check it on spike"
  ex_preflight "$BOARD" --require probe
  ex_run "$PY" -m merlin.runtime.vector_probe --board "$BOARD" --run --spike-vlen 256
  ex_note "the shipped probe speaks this chip's UART; the self-checked one speaks HTIF, because that"
  ex_note "is the console spike provides. Same source, same CSR reads, different console object."
  ;;

facts)
  # Worth its own stage: this is the derivation that replaces a table of hardcoded addresses, and it is
  # readable in a second. If a console comes back as line noise, this is the first thing to look at.
  need_sdk
  ex_say "derive this chip's console + clock facts from its own SDK headers"
  ex_run "$PY" "$EX_LIB_DIR/show_console_facts.py" "$SDK" --chip bearly25
  ;;

build|package)
  need_sdk
  ex_preflight "$BOARD" --sdk-dir "$SDK" --require package
  if [ "$FULL" = "1" ]; then
    MODELS=spectformer,deepjscc,lstmnetvit,whisper_tiny_375pos
    ex_note "--full: 8 gated images plus 8 debug twins. whisper alone is ~5.8 G cycles on spike."
  else
    MODELS=deepjscc
    ex_note "cheap subset: deepjscc at 1 and 2 harts. Add --full for the shipped matrix."
  fi
  ex_say "build for $BOARD, gate on spike, audit, write $OUT"
  ex_run "$PY" build_tools/scripts/make_delivery.py \
      --board "$BOARD" --models "$MODELS" --harts 1,2 --debug \
      --sdk-dir "$SDK" --out "$OUT"
  ex_note "a UART console cannot be simulated on spike, so the gate runs an HTIF twin built from the"
  ex_note "same IR and the package says so rather than implying it ran the shipped ELF"
  ;;

grade)
  # `grade [log] [model]`. With no arguments it grades the bundled real failure, whose model the
  # grader cannot infer from the log itself (a package holds several), so name it.
  LOG="${2:-}"; MODEL="${3:-}"
  if [ -z "$LOG" ]; then
    LOG="$(dirname "${BASH_SOURCE[0]}")/returned/whisper_h1_debug_500mhz_fault.txt"
    MODEL=whisper_tiny_375pos
  fi
  # A grader can only score a model whose references sit beside it, so choose by model
  # rather than by "first directory on the glob" -- see lib/find_grader.py.
  GRADER="$("$PY" "$EX_LIB_DIR/find_grader.py" --model "${MODEL:-deepjscc}" --prefer "$OUT")" || exit 1
  ex_say "grade $LOG"
  ex_note "the default log is a REAL failure returned from the chip — see README.md, 'Read a log that"
  ex_note "came back broken', for how its four lines localise the bug to one wrong descriptor field"
  # `|| true`: a log that records a FAILED run makes the grader exit non-zero, which is the correct
  # behaviour and not an error in the example.
  ex_run "$PY" "$GRADER" "$LOG" ${MODEL:+--model "$MODEL"} || true
  ;;

*)
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
