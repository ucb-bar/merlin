#!/usr/bin/env bash
# firesim_shell.sh — one-stop wrapper to drop into a shell (or run an
# arbitrary command) with chipyard's conda env, firesim on PATH, and
# ssh-agent available.
#
# Use this any time you want to interact with chipyard/firesim without
# worrying about `source sourceme-manager.sh` cwd games or `conda activate`
# chains.
#
# Usage:
#   bash build_tools/firesim/firesim_shell.sh                # drop into an interactive subshell
#   bash build_tools/firesim/firesim_shell.sh firesim kill   # run one command
#   bash build_tools/firesim/firesim_shell.sh -- firesim runworkload
#
# `--` is optional; anything after it (or all positional args if none uses --)
# is executed. With no args, an interactive bash replaces this process.

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# shellcheck disable=SC1091
source "$MERLIN_ROOT/build_tools/firesim/env.sh"

# Strip a leading `--` if present.
if [ $# -gt 0 ] && [ "$1" = "--" ]; then
	shift
fi

if [ $# -eq 0 ]; then
	echo "[firesim_shell] dropping into interactive bash with firesim env loaded."
	echo "                CHIPYARD_ROOT=$CHIPYARD_ROOT"
	echo "                type 'exit' to return."
	exec bash --noprofile --norc -i
else
	exec "$@"
fi
