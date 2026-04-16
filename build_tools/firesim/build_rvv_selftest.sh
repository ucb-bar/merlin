#!/usr/bin/env bash
# Ahead-of-time build of every FireSim binary we care about for the
# Saturn OPU vfredusum investigation:
#
#   1. Phase A — 4 rvvtest selftest binaries, one per
#      SATURN_RVV_SELFTEST_SKIP bitmask, cached into
#      build/firesim-rvvtest/skip_${MASK}/bench_model_opu_bench_vit_small_rvvtest
#
#   2. Phase D — post-fix production benchmarks (regular vit_small_opu and
#      large_mlp_opu, built with the +xopu-gated scalarization pattern active
#      in iree-compile). These live at their standard build path:
#      build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/
#          simple_embedding_ukernel/bench_model_opu_bench_{vit_small,large_mlp}_opu
#
# Run this from *any* shell. It uses the merlin-dev miniforge env directly
# so it works even from a chipyard-activated shell (chipyard's conda
# shadows `conda run -n merlin-dev`).
#
# Usage:
#   bash build_tools/firesim/build_rvv_selftest.sh                    # everything
#   bash build_tools/firesim/build_rvv_selftest.sh 0x00 0x08          # subset of skip masks
#   bash build_tools/firesim/build_rvv_selftest.sh --phase-a-only     # only Phase A
#   bash build_tools/firesim/build_rvv_selftest.sh --phase-d-only     # only Phase D
#
# Prerequisites: the host iree-compile must already have been built with
# the +xopu scalarization pattern (i.e. merlin `--profile vanilla --config
# release`) — this script only builds the bare-metal RISC-V binaries.

set -euo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

DO_PHASE_A=1
DO_PHASE_D=1
MASKS=()
for arg in "$@"; do
	case "$arg" in
	--phase-a-only)
		DO_PHASE_D=0
		;;
	--phase-d-only)
		DO_PHASE_A=0
		;;
	0x*)
		MASKS+=("$arg")
		;;
	*)
		echo "Unknown arg: $arg"
		exit 1
		;;
	esac
done
if [ ${#MASKS[@]} -eq 0 ]; then
	MASKS=(0x00 0x08 0x18 0x38)
fi

# Use miniforge merlin-dev env *directly* (no `conda run -n`), so chipyard
# being activated doesn't break the lookup. Falls back to any `uv` on PATH
# if the miniforge layout changes.
MERLIN_UV="/scratch2/agustin/miniforge3/envs/merlin-dev/bin/uv"
if [ ! -x "$MERLIN_UV" ]; then
	MERLIN_UV="$(command -v uv)"
fi
if [ -z "$MERLIN_UV" ] || [ ! -x "$MERLIN_UV" ]; then
	echo "Could not locate uv (tried /scratch2/agustin/miniforge3/envs/merlin-dev/bin/uv and PATH)"
	exit 1
fi

# The firesim toolchain file reads these from the env, not just cmake cache.
: "${RISCV_TOOLCHAIN_ROOT:=$MERLIN_ROOT/build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV}"
: "${RISCV:=$RISCV_TOOLCHAIN_ROOT}"
: "${CHIPYARD_ROOT:=/scratch2/agustin/chipyard}"
export RISCV_TOOLCHAIN_ROOT RISCV CHIPYARD_ROOT

OUT_ROOT="$MERLIN_ROOT/build/firesim-rvvtest"
BUILD_DIR="$MERLIN_ROOT/build/firesim-merlin-release"
ELF_REL="runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/bench_model_opu_bench_vit_small_rvvtest"
CMAKE_TARGET="iree_.._.._samples_SaturnOPU_simple_embedding_ukernel_bench_model_opu_bench_vit_small_rvvtest"

echo "MERLIN_ROOT=$MERLIN_ROOT"
echo "uv=$MERLIN_UV"
echo "masks=${MASKS[*]}"

if [ "$DO_PHASE_A" = "1" ]; then
	for mask in "${MASKS[@]}"; do
		echo
		echo "========================================================================"
		echo "  Phase A — rvvtest binary, SATURN_RVV_SELFTEST_SKIP=$mask"
		echo "========================================================================"
		(cd "$MERLIN_ROOT" && "$MERLIN_UV" run tools/merlin.py build \
			--profile firesim \
			--cmake-target "$CMAKE_TARGET" \
			"--cmake-arg=-DSATURN_RVV_SELFTEST_SKIP=$mask")

		src="$BUILD_DIR/$ELF_REL"
		if [ ! -f "$src" ]; then
			echo "Build produced no ELF at $src — aborting."
			exit 1
		fi
		dest="$OUT_ROOT/skip_$mask"
		mkdir -p "$dest"
		cp "$src" "$dest/bench_model_opu_bench_vit_small_rvvtest"
		echo "  → $dest/bench_model_opu_bench_vit_small_rvvtest  ($(stat -c%s "$dest/bench_model_opu_bench_vit_small_rvvtest") bytes)"
	done
fi

if [ "$DO_PHASE_D" = "1" ]; then
	echo
	echo "========================================================================"
	echo "  Phase D — post-fix production benchmarks (vit_small_opu, large_mlp_opu)"
	echo "========================================================================"
	# Delete stale vmfb/c files so the bytecode-module custom command re-runs
	# iree-compile (which now has the +xopu scalarization pattern). Otherwise
	# CMake may re-use a pre-fix vmfb keyed off the .mlir timestamp alone.
	STALE_DIR="$BUILD_DIR/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
	rm -f \
		"$STALE_DIR/model_opu_bench_vit_small_opu.vmfb" \
		"$STALE_DIR/model_opu_bench_vit_small_opu_c.c" \
		"$STALE_DIR/model_opu_bench_vit_small_opu_c.h" \
		"$STALE_DIR/model_opu_bench_large_mlp_opu.vmfb" \
		"$STALE_DIR/model_opu_bench_large_mlp_opu_c.c" \
		"$STALE_DIR/model_opu_bench_large_mlp_opu_c.h"
	(cd "$MERLIN_ROOT" && "$MERLIN_UV" run tools/merlin.py build \
		--profile firesim \
		--cmake-target iree_.._.._samples_SaturnOPU_simple_embedding_ukernel_bench_model_opu_bench_vit_small_opu)
	(cd "$MERLIN_ROOT" && "$MERLIN_UV" run tools/merlin.py build \
		--profile firesim \
		--cmake-target iree_.._.._samples_SaturnOPU_simple_embedding_ukernel_bench_model_opu_bench_large_mlp_opu)
	for name in vit_small_opu large_mlp_opu; do
		bin="$STALE_DIR/bench_model_opu_bench_$name"
		if [ -f "$bin" ]; then
			echo "  → $bin  ($(stat -c%s "$bin") bytes)"
		fi
	done
fi

echo
echo "All builds complete."
if [ "$DO_PHASE_A" = "1" ]; then
	echo "Phase A artifacts (rvvtest binaries):"
	ls -la "$OUT_ROOT"/*/bench_model_opu_bench_vit_small_rvvtest 2>/dev/null
fi
if [ "$DO_PHASE_D" = "1" ]; then
	echo "Phase D artifacts (post-fix benchmarks):"
	ls -la "$BUILD_DIR/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/bench_model_opu_bench_vit_small_opu" \
		"$BUILD_DIR/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/bench_model_opu_bench_large_mlp_opu" 2>/dev/null
fi
