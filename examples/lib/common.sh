# Shared plumbing for the examples. Sourced, not executed.
#
# Three jobs, all of them about making an example honest rather than merely runnable:
#   * locate the repo and its interpreter from the SCRIPT's own path, so an example works from any
#     working directory and needs no `cd` in the instructions;
#   * echo every command before running it, and support --dry-run, so the example doubles as
#     documentation of the exact invocations — a reader with no toolchain still learns the flow;
#   * keep the long stages opt-in, because the full delivery matrix is hours of functional simulation
#     and an example that silently costs an afternoon is a trap.

set -euo pipefail

EX_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EX_LIB_DIR/../.." && pwd)"

# Prefer the repo's own venv (uv, py3.13) — `python` is frequently not on PATH here, and a system
# interpreter would not have merlin importable.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "no interpreter: expected $REPO_ROOT/.venv/bin/python or python3 on PATH" >&2
  exit 1
fi

DRY_RUN=0
FULL=0

ex_parse_flags() {
  EX_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --full)    FULL=1 ;;
      *)         EX_ARGS+=("$1") ;;
    esac
    shift
  done
}

ex_say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# Echo then run. The echo is not decoration: these commands are the deliverable half of the example,
# and a reader who cannot run them should still be able to copy them.
ex_run() {
  printf '$ %s\n' "$*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

ex_preflight() {   # ex_preflight <board> [--require <stage>] [--sdk-dir <dir>]
  ex_run "$PY" "$EX_LIB_DIR/preflight.py" --board "$@"
}

ex_note() { printf '   note: %s\n' "$*"; }
