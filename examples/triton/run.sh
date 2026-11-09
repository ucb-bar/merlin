#!/usr/bin/env bash
# Triton -> Merlin -> an accelerator: write an ordinary Triton kernel, compile it for hardware that is
# not a GPU. Historical convergence/RTL regression actions require their original Git checkout.
#
# Stages are ordered by what they cost and what they need. Run `preflight` first; it says which of them
# this machine can do. Current compile and route actions use the repo and its venv.
#
#   ./run.sh preflight            what this machine can run, and what would fix the rest
#   ./run.sh walk                 the whole pipeline, stage by stage, explained     (seconds)
#   ./run.sh compile              what you would actually run: the CLI, end to end   (seconds)
#   ./run.sh route                why an elementwise kernel takes a DIFFERENT route  (seconds)
#   ./run.sh compare              the same kernels against two accelerators          (seconds)
#   ./run.sh converge             retired: replay in its original immutable Git checkout
#   ./run.sh certify              retired: replay in its original immutable Git checkout
#
# Flags: --dry-run prints every command without running it, so this file reads as documentation on a
#                  machine with no toolchain at all
#        --pause   with `walk`, stop after each stage and wait for Enter
#        --package <dir>  compile for a DIFFERENT accelerator (default: the tracked gemmini package)
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/common.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE="out/artifacts/targets/gemmini/hand_v0"
PAUSE=""

# --package and --pause are specific to this example, so parse them before the shared flag pass.
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --package) PACKAGE="$2"; shift ;;
    --pause)   PAUSE="--pause" ;;
    *)         ARGS+=("$1") ;;
  esac
  shift
done
set -- ${ARGS[@]+"${ARGS[@]}"}

ex_parse_flags "$@"
set -- ${EX_ARGS[@]+"${EX_ARGS[@]}"}
STAGE="${1:-preflight}"

cd "$REPO_ROOT"

# The kernel and its declarations. A Triton kernel is not self-describing -- untyped parameters,
# shapeless pointers, the grid at the call site -- so these state the facts Merlin refuses to guess.
MATMUL="examples/triton/matmul_simple.py:repeated_rhs_matmul"
MATMUL_ARGS=(--arg 'a0_ptr=*i8:16x32:read' --arg 'a1_ptr=*i8:16x32:read' --arg 'w_ptr=*i8:32x16:read'
             --arg 'c0_ptr=*i32:16x16:write' --arg 'c1_ptr=*i32:16x16:write'
             --constexpr BM=16 --constexpr BN=16 --constexpr BK=32 --grid 1)

VECTOR_ADD="examples/triton/vector_add.py:vector_add"
VECTOR_ADD_ARGS=(--arg 'x_ptr=*fp32:1025:read' --arg 'y_ptr=*fp32:1025:read'
                 --arg 'out_ptr=*fp32:1025:write' --arg 'n_elements=i32'
                 --assume n_elements=1025 --constexpr BLOCK_SIZE=256 --grid 5)

case "$STAGE" in
preflight)
  ex_run "$PY" "$HERE/preflight.py" --package "$PACKAGE"
  ;;

walk)
  # The inside view: every stage between the kernel and the command buffer, with the reason each one
  # exists. Writes nothing. Add --pause to step through it.
  ex_say "the whole pipeline, stage by stage"
  ex_run "$PY" "$HERE/preflight.py" --package "$PACKAGE" --require walk
  ex_run "$PY" "$HERE/walkthrough.py" --package "$PACKAGE" ${PAUSE:+$PAUSE}
  ;;

compile)
  # The outside view: the one command a user actually runs. Artifacts land in a versioned product
  # directory carrying the route, the triton version and the TTIR digest.
  ex_say "compile the matmul kernel for this accelerator"
  ex_run "$PY" "$HERE/preflight.py" --package "$PACKAGE" --require compile
  ex_run "$PY" -m merlin.triton.cli "$MATMUL" --target-package "$PACKAGE" \
    "${MATMUL_ARGS[@]}" --emit all --verify
  ex_note "--verify is MLIR structural verification of each stage module, NOT a numerical check."
  ex_note "For numbers, use 'walk' (checks against numpy) or 'certify' (checks against RTL)."
  ;;

route)
  # The instructive refusal. Same target, same CLI, different PAYLOAD -- and the route changes,
  # because routing reads what the target's dialect plan covers rather than the target's name.
  ex_say "an elementwise kernel on the SAME accelerator takes a different route"
  ex_run "$PY" "$HERE/preflight.py" --package "$PACKAGE" --require route
  ex_run "$PY" -m merlin.triton.cli "$VECTOR_ADD" --target-package "$PACKAGE" \
    "${VECTOR_ADD_ARGS[@]}" --route-only
  ex_note "This is the design working, not failing: a vector add has no matmul, so it compiles as"
  ex_note "generic computation even on an accelerator. 'accelerator => staged path' is the wrong model."
  ;;

compare)
  # The whole architecture in one output: ONE kernel, ONE set of declarations, two accelerators, two
  # different answers -- and the reason printed each time, derived from what each package declares.
  ex_say "the same two kernels against two different accelerators"
  ex_run "$PY" "$HERE/preflight.py" --package "$PACKAGE" --require route
  for pkg in out/artifacts/targets/gemmini/hand_v0 out/artifacts/targets/radiance/hand_v0; do
    printf '\n  --- %s ---\n' "$pkg"
    ex_run "$PY" -m merlin.triton.cli "$MATMUL" --target-package "$pkg" \
      "${MATMUL_ARGS[@]}" --route-only
    ex_run "$PY" -m merlin.triton.cli "$VECTOR_ADD" --target-package "$pkg" \
      "${VECTOR_ADD_ARGS[@]}" --route-only
  done
  ex_note "The matmul is staged on both. The vector add is staged only on radiance, because radiance"
  ex_note "declares interface.elementwise and gemmini does not. Coverage decided that, not the name."
  ex_note ""
  ex_note "What is compared here is ROUTING, which is settled before any hardware is involved. Only"
  ex_note "gemmini is RTL-certified; the radiance package is status: prototype, certification: none,"
  ex_note "so nothing above is a hardware claim about it. 'certify' runs on gemmini for that reason."
  ;;

converge|certify)
  echo "'$STAGE' is an archived regression action; its pytest inputs are not part of the current example." >&2
  echo "Replay it from the same immutable historical Git revision as those inputs and simulator/RTL dependencies:" >&2
  echo "  git worktree add --detach <replay-directory> <historical-full-commit>" >&2
  echo "  cd <replay-directory> && ./examples/triton/run.sh $STAGE" >&2
  echo "The current compile/route actions do not claim historical convergence or RTL qualification." >&2
  exit 2
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  sed -n '2,20p' "${BASH_SOURCE[0]}" >&2
  exit 2
  ;;
esac
