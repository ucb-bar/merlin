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
# Where the user ran the example from. Every run.sh then `cd`s to the repo root, so a relative path the
# user typed (a loader, a console log, a job directory) has to be made absolute against this first, or it
# is silently looked up under the repo root instead.
EX_CALLER_PWD="$PWD"

# ex_abs <path>: <path> made absolute against the caller's directory when it names something there;
# otherwise unchanged, so the tool's own "not found" names exactly what was typed.
ex_abs() {
  case "$1" in
    ''|/*) printf '%s' "$1" ;;
    *) if [ -e "$EX_CALLER_PWD/$1" ]; then printf '%s/%s' "$EX_CALLER_PWD" "$1"; else printf '%s' "$1"; fi ;;
  esac
}

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

# The repo's .env (written by `run.sh setup`) into this shell's environment, for every key the
# environment does not already set: the process environment wins, exactly as merlin.common.paths.env
# resolves it. Tools that read only os.environ (a chip package's own scripts, a spike extension's
# variable) then see the same configuration as the library does. The keys taken from .env are listed in
# MERLIN_DOTENV_KEYS, so a later reload (after setup rewrote .env) refreshes them, and setup and preflight
# can tell a value from .env from one the user's shell exports.
ex_load_dotenv() {
  local file="$REPO_ROOT/.env" line key value
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"; value="${line#*=}"
    key="${key// /}"
    value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"
    case "$value" in \"*\"|\'*\') value="${value:1:${#value}-2}" ;; esac
    case "$key" in ''|*[!A-Za-z0-9_]*|[0-9]*|MERLIN_DOTENV_KEYS) continue ;; esac
    case " ${MERLIN_DOTENV_KEYS:-} " in
      *" $key "*) export "$key=$value" ;;
      *) if [ -z "${!key+x}" ]; then
           export "$key=$value"; MERLIN_DOTENV_KEYS="${MERLIN_DOTENV_KEYS:+$MERLIN_DOTENV_KEYS }$key"
         fi ;;
    esac
  done < "$file"
  export MERLIN_DOTENV_KEYS="${MERLIN_DOTENV_KEYS:-}"
}
ex_load_dotenv

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

# ex_kernel_spec <spec>: a matrix kernel as merlin.kernels.matrix_kernels takes it. A registered name stays as
# it is; a C file (file.c, or file.c:function) is made absolute against the caller's directory.
ex_kernel_spec() {
  local spec="$1" file func=""
  case "$spec" in *.c:*) file="${spec%:*}"; func=":${spec##*:}" ;; *) file="$spec" ;; esac
  case "$file" in *.c) printf '%s%s' "$(ex_abs "$file")" "$func" ;; *) printf '%s' "$spec" ;; esac
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

# A chip's support package is not copied into Merlin: `run.sh setup` fetches it at the exact commit
# Merlin pins (build_tools/upstreams/target_support.json, pinned_support), and the default selection
# finds it there. `ex_select_support <target>` sets SUPPORT_ROOT to the selected package (an explicit
# MERLIN_TARGET_PATH still wins) and, unless already set, selects the chip's boards beside the example
# catalog. Not fetched, it stops with the setup command, except under --dry-run, where it notes it.
ex_select_support() {   # ex_select_support <target>
  local target="$1"
  if ! SUPPORT_ROOT="$("$PY" -m merlin.targetgen.oot_fetch "$target" --root)"; then
    if [ "$DRY_RUN" = "1" ]; then
      SUPPORT_ROOT="<$target support: run setup>"
      ex_note "the $target support package is not fetched; ./run.sh setup fetches it"
    else
      exit 1
    fi
  fi
  export MERLIN_BOARD_CATALOG="${MERLIN_BOARD_CATALOG:-$REPO_ROOT/examples/board-catalog.yaml:$SUPPORT_ROOT}"
}

# `run.sh setup`: Merlin's own venv first (everything else runs in it), then examples/lib/chip_setup.py
# with what this chip needs on top of the common prerequisites, then the preflight. Returns
# chip_setup's status, so a setup that still lists something to do does not read as a success.
ex_setup() {   # ex_setup <board> [chip_setup args...]
  local board="$1" status=0
  shift
  if ! command -v uv >/dev/null 2>&1; then
    echo "setup needs uv (https://docs.astral.sh/uv/): curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
  fi
  if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    ex_run uv sync --extra dev --extra xdsl --extra targetgen
  fi
  PY="$REPO_ROOT/.venv/bin/python"
  ex_run "$PY" "$EX_LIB_DIR/chip_setup.py" "$@" || status=$?
  ex_load_dotenv   # what setup just recorded, for the preflight below
  ex_preflight "$board" || true
  return "$status"
}
