#!/usr/bin/env bash
# build_tools/firesim/env.sh — sourceable chipyard/FireSim environment setup.
# Idempotent: safe to source multiple times. Designed so *any* downstream
# script that needs `firesim` can just `source` this and be done with env
# gymnastics.
#
# Usage (interactive shell):
#     source build_tools/firesim/env.sh
# Usage (inside a script):
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# What it does, in order:
#   1. Activates chipyard's conda env (.conda-env) so firesim's python deps
#      resolve. Skips if already active.
#   2. Puts firesim's deploy/ dir on PATH (that's where the `firesim` python
#      entry-point script lives).
#   3. Starts an ssh-agent and runs ssh-add for the current user if no agent
#      is available. Without this, firesim's fabric-based remote-command
#      layer prompts for an ssh password on localhost and aborts in
#      non-interactive parallel mode.
#
# Env vars the caller may set to override defaults:
#   CHIPYARD_ROOT — default /scratch2/agustin/chipyard
#   CONDA_BASE    — default /scratch2/agustin/miniforge3

CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
export CHIPYARD_ROOT

# --- 1. chipyard conda env ---
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != ".conda-env" ]; then
	CONDA_BASE="${CONDA_BASE:-/scratch2/agustin/miniforge3}"
	if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
		# conda activate reads many variables internally; relax `set -u`
		# if the caller enabled it so we don't explode here.
		_had_nounset=0
		case $- in *u*) _had_nounset=1; set +u ;; esac
		# shellcheck disable=SC1091
		source "$CONDA_BASE/etc/profile.d/conda.sh"
		conda activate "$CHIPYARD_ROOT/.conda-env" >/dev/null 2>&1 || true
		[ "$_had_nounset" = "1" ] && set -u
		unset _had_nounset
	fi
fi

# --- 2. firesim on PATH ---
_FS_DEP="$CHIPYARD_ROOT/sims/firesim/deploy"
case ":$PATH:" in
*":$_FS_DEP:"*) ;;
*) export PATH="$_FS_DEP:$PATH" ;;
esac
export FIRESIM_SOURCED=1
unset _FS_DEP

# --- 3. ssh-agent ---
if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
	eval "$(ssh-agent -s)" >/dev/null 2>&1 || true
	# ssh-add uses default keys (~/.ssh/id_*) — non-interactive if keys are
	# unencrypted or already unlocked. If a key needs a passphrase, the
	# caller must run `ssh-add` interactively in their shell first.
	ssh-add >/dev/null 2>&1 || true
fi

# Sanity check.
if ! command -v firesim >/dev/null 2>&1; then
	echo "[env.sh] WARNING: firesim not found on PATH after setup." >&2
	echo "         CHIPYARD_ROOT=$CHIPYARD_ROOT" >&2
fi
