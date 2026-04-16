#!/usr/bin/env bash
# Scan a freshly-compiled model's linked ELF .s for the Saturn vector-
# unit hang signatures. Run this BEFORE launching FireSim so we don't
# burn 10 min watching it freeze.
#
# Two hang classes that have bitten us:
#   1. `vfredusum.vs` ANYWHERE (the OG vector-reduction hang).
#   2. `vfmacc.vf` + `vslidedown.vi` paired in a NON-reduction function
#      — LLVM's tree-reduction codegen for f32 reductions, which hangs
#      Saturn even when not formally a vfredusum.
#
# Either pattern in a freshly-compiled binary means Option E didn't
# fire on those reductions. Fix: rebuild iree-compile from current
# ConvertToLLVM.cpp (which broadens Option E to any RISC-V +v target).
#
# Usage:
#   bash build_tools/firesim/verify_no_hang.sh path/to/linked.s [more...]
#   bash build_tools/firesim/verify_no_hang.sh --model=yolov8_nano [--variant=rvv]
#       (auto-resolve the linked .s under build/compiled_models/)

set -uo pipefail

MERLIN_ROOT="${MERLIN_ROOT:-/scratch2/agustin/merlin}"

usage() {
	echo "Usage: $0 path/to/linked.s [more...]"
	echo "       $0 --model=NAME [--variant=opu|rvv|opu_im2col|opu_llm]"
	exit 1
}

asm_files=()
MODEL=""
VARIANT=""
for arg in "$@"; do
	case "$arg" in
	--model=*) MODEL="${arg#--model=}" ;;
	--variant=*) VARIANT="${arg#--variant=}" ;;
	-h|--help) usage ;;
	*) asm_files+=("$arg") ;;
	esac
done

# Auto-resolve linked .s when --model is given.
if [ -n "$MODEL" ]; then
	# Compiler intermediates are under build/compiled_models/<model>/<artifact_dir>/files/
	mapfile -t found < <(find "$MERLIN_ROOT/build/compiled_models" \
		-path "*${MODEL}*" -name "*linked*riscv_64.s" 2>/dev/null)
	if [ ${#found[@]} -eq 0 ]; then
		echo "ERROR: no linked .s under build/compiled_models/ matching '*${MODEL}*'"
		echo "  Did the model get compiled with --iree-hal-dump-executable-intermediates-to=...?"
		exit 2
	fi
	if [ -n "$VARIANT" ]; then
		filtered=()
		for f in "${found[@]}"; do
			[[ "$f" == *"${VARIANT}"* ]] && filtered+=("$f")
		done
		[ ${#filtered[@]} -gt 0 ] && asm_files=("${filtered[@]}") || asm_files=("${found[@]}")
	else
		asm_files=("${found[@]}")
	fi
fi

[ ${#asm_files[@]} -eq 0 ] && usage

overall_fail=0

for s in "${asm_files[@]}"; do
	echo "================================================================"
	echo " $s"
	echo "================================================================"
	if [ ! -f "$s" ]; then
		echo "  MISSING — skip"
		overall_fail=1
		continue
	fi

	# 1) vfredusum.vs anywhere is fatal.
	vfredusum=$(grep -c "vfredusum" "$s" 2>/dev/null)
	vfredusum=${vfredusum:-0}
	if [ "$vfredusum" -gt 0 ] 2>/dev/null; then
		echo "  ✗ HANG RISK: $vfredusum vfredusum.* opcode(s) — Saturn vector unit hangs on these"
		grep -n "vfredusum" "$s" | head -5 | sed 's/^/      /'
		overall_fail=1
	else
		echo "  ✓ no vfredusum opcodes"
	fi

	# 2) vfmacc.vf + vslidedown.vi paired in a non-reduction function.
	# Walk function-by-function. A function is "reduction" iff its label
	# matches *_reduction_*_f32* or *_softmax_*xf32* (the Option-E set).
	bad_fns=$(awk '
		BEGIN { fn=""; has_macc=0; has_slide=0; bad=0; bad_list="" }
		/^[A-Za-z_][A-Za-z_0-9$.]*:$/ {
			if (fn != "" && has_macc && has_slide) {
				is_reduction = (fn ~ /reduction.*_f32/ || fn ~ /softmax.*xf32/);
				if (!is_reduction) {
					bad++; bad_list = bad_list fn "\n"
				}
			}
			fn=$0; sub(":","",fn); has_macc=0; has_slide=0
			next
		}
		/vfmacc\.vf/  { has_macc=1 }
		/vslidedown\.vi/ { has_slide=1 }
		END {
			if (fn != "" && has_macc && has_slide) {
				is_reduction = (fn ~ /reduction.*_f32/ || fn ~ /softmax.*xf32/);
				if (!is_reduction) bad++
			}
			print bad
		}' "$s")

	if [ "$bad_fns" -gt 0 ]; then
		echo "  ✗ HANG RISK: $bad_fns non-reduction function(s) contain the vfmacc.vf+vslidedown.vi tree-reduction pattern"
		echo "      (Option E didn't fire — most likely the iree-compile binary is older than"
		echo "       compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp)"
		overall_fail=1
	else
		echo "  ✓ no non-reduction functions with the hang pattern"
	fi

	# Informational: count Option-E devectorized functions.
	devec=$(grep -c 'target-features.*-v' "$s" 2>/dev/null)
	devec=${devec:-0}
	echo "  i  $devec function(s) carry per-function -v override (Option E)"
done

echo
if [ "$overall_fail" -eq 0 ]; then
	echo "ALL CLEAR — safe to run on FireSim."
	exit 0
else
	echo "VERIFICATION FAILED — recompile after rebuilding iree-compile."
	exit 1
fi
