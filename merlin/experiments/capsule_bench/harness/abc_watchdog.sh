#!/bin/bash
# Thin, target-agnostic shim over watchdog.py (the real, descriptor-driven auto-resume watchdog).
# Kept for the historical `abc_watchdog.sh <tag>` entry point; all logic now lives in watchdog.py so a
# single implementation guards every target (gemmini/atlas/radiance/…). The target comes from
# MERLIN_TARGET_EXPERIMENT, exactly like launch_ab_batch.
#
# Usage:
#   MERLIN_TARGET_EXPERIMENT=.../targets/<t>/target_experiment.yaml abc_watchdog.sh <tag> [extra args]
# Any extra args pass straight through to watchdog.py (e.g. --arms merlin_rtlchecks --login-kick).
set -u
TAG="${1:?usage: abc_watchdog.sh <tag> [watchdog.py args]}"; shift || true
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"           # resolves through the per-target symlink
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
exec "$REPO/.venv/bin/python" "$HERE/watchdog.py" --tag "$TAG" "$@"
