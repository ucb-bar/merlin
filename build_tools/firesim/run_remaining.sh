#!/usr/bin/env bash
# Resume the final sweep — runs only the (model, variant) pairs that
# don't already have a successful row in /tmp/sweep_iters.csv.
#
# Use this instead of `run_final_sweep.sh --phase=both` when you want
# to pick up after a partial run without re-running completed entries.
#
# Usage:
#   bash build_tools/firesim/run_remaining.sh
#       (auto-skip everything with a numeric iter1)
#   bash build_tools/firesim/run_remaining.sh --force=yolov8_nano:rvv,tinyllama:opu
#       (force-include even if marked done — useful after a recompile)
#   bash build_tools/firesim/run_remaining.sh --phase=both
#       (also run the profile-phase _opu_prof binaries afterwards)

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"
ITERS_CSV="${ITERS_CSV:-/tmp/sweep_iters.csv}"

ALL_CLEAN=(
	"mlp_wide:opu"
	"mlp_wide:rvv"
	"opu_bench_large_mlp:opu"
	"opu_bench_large_mlp:rvv"
	"opu_bench_vit_small:opu"
	"opu_bench_vit_small:rvv"
	"opu_bench_vit:opu"
	"opu_bench_vit:rvv"
	"dronet:opu"
	"dronet:rvv"
	"yolov8_nano:opu"
	"yolov8_nano:rvv"
	"mlp:opu"
	"mlp:rvv"
	"opu_bench_hybrid:opu"
	"opu_bench_hybrid:rvv"
	"opu_bench_convnet:opu"
	"opu_bench_convnet:rvv"
	"tinyllama:opu"
	"tinyllama:rvv"
	"mlp_fast:opu"
	"mlp_fast:rvv"
)
ALL_PROF=(
	"mlp_wide:opu_prof"
	"opu_bench_large_mlp:opu_prof"
	"opu_bench_vit_small:opu_prof"
	"opu_bench_vit:opu_prof"
	"dronet:opu_prof"
	"mlp:opu_prof"
	"opu_bench_hybrid:opu_prof"
	"opu_bench_convnet:opu_prof"
)

# Pairs known to hang on FireSim and not yet fixable in a single
# iteration — skipped by default to avoid burning the per-run timeout.
# Pass --include-known-hangs to force them through anyway.
KNOWN_HANGS=(
	"mlp:opu"   # narrow-M (K=10, not multiple of 16) OPU encoding-resolver path
	# opu_bench_vit:opu — fixed in v2 (dim=256, seq=256, all dims ≥16)
	# mlp:rvv fixed by switching to FLAGS_MODEL_RVV_NO_TILING (pure vec, no ukernels)
	# hybrid:rvv fixed by same — attention shape breaks batch_mmt4d under data-tiling
)

PHASE="clean"
FORCE=""
INCLUDE_HANGS=0
for arg in "$@"; do
	case "$arg" in
	--phase=clean | --phase=profile | --phase=both) PHASE="${arg#--phase=}" ;;
	--force=*) FORCE="${arg#--force=}" ;;
	--include-known-hangs) INCLUDE_HANGS=1 ;;
	*) echo "Unknown arg: $arg"; exit 1 ;;
	esac
done

is_known_hang() {
	[ "$INCLUDE_HANGS" -eq 1 ] && return 1
	local pair="$1"
	for h in "${KNOWN_HANGS[@]}"; do
		[ "$h" = "$pair" ] && return 0
	done
	return 1
}

is_done() {
	local pair="$1" m="${1%:*}" v="${1##*:}"
	# Force list overrides "done" status.
	[[ ",$FORCE," == *",$pair,"* ]] && return 1
	awk -F, -v m="$m" -v v="$v" '
		$1==m && $2==v && $3 ~ /^[0-9]+$/ { found=1 }
		END { exit !found }' "$ITERS_CSV" 2>/dev/null
}

remaining_clean=()
for pair in "${ALL_CLEAN[@]}"; do
	if is_done "$pair"; then
		echo "  done (skip):       $pair"
	elif is_known_hang "$pair"; then
		echo "  known hang (skip): $pair  (use --include-known-hangs to force)"
	else
		echo "  PENDING:           $pair"
		remaining_clean+=("$pair")
	fi
done

# For --phase=both also surface the profile entries (always re-run).
if [ "$PHASE" = "both" ]; then
	echo
	echo "Profile-phase entries (will run after clean):"
	for pair in "${ALL_PROF[@]}"; do
		echo "  PROF:        $pair"
	done
fi

echo
echo "${#remaining_clean[@]} clean-phase pairs to run."
if [ ${#remaining_clean[@]} -eq 0 ] && [ "$PHASE" != "profile" ] && [ "$PHASE" != "both" ]; then
	echo "Nothing to run."
	exit 0
fi

# Hand the exact-match list to run_final_sweep.sh via --include
# (comma-separated to survive single-arg shell parsing).
INCLUDE_ARG=$(IFS=,; echo "${remaining_clean[*]}")

echo
if [ -n "$INCLUDE_ARG" ]; then
	echo "===== invoking run_final_sweep.sh --phase=$PHASE --include=$INCLUDE_ARG ====="
	exec bash "$MERLIN_ROOT/build_tools/firesim/run_final_sweep.sh" \
		"--phase=$PHASE" "--include=$INCLUDE_ARG"
else
	echo "===== invoking run_final_sweep.sh --phase=$PHASE ====="
	exec bash "$MERLIN_ROOT/build_tools/firesim/run_final_sweep.sh" "--phase=$PHASE"
fi
