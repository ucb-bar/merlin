#include "compiler/plugins/target/Radiance/RadianceOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::RadianceOptions);

namespace mlir::iree_compiler {

void RadianceOptions::bindOptions(OptionsBinder &binder) {
	static llvm::cl::OptionCategory category("IREE Radiance plugin options");

	binder.opt<bool>("iree-radiance-enable", enable,
		llvm::cl::desc(
			"Enables the Radiance/Muon GPU compile-time pipeline. "
			"Phase 2.6 only registers the options; lowering passes are "
			"added incrementally."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-radiance-num-warps", numWarps,
		llvm::cl::desc(
			"Warps per Muon threadblock for mu_schedule. Must match the "
			"NUM_WARPS macro emitted into kernel.cpp."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-radiance-emit-llvm-ir", emitLLVMIR,
		llvm::cl::desc("Emit the lowered kernel body as an LLVM IR text (.ll) "
					   "file. Consumed by kernels/core/precompile.py with "
					   "source_lang=ll."),
		llvm::cl::cat(category));

	binder.opt<std::string>("iree-radiance-emit-llvm-ir-path", emitLLVMIRPath,
		llvm::cl::desc(
			"Output path for --iree-radiance-emit-llvm-ir. When empty, "
			"the file is written next to the input MLIR with .ll suffix."),
		llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
