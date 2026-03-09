#include "iree-gemmini/compiler/plugin/GemminiOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::GemminiOptions);

namespace mlir::iree_compiler {

void GemminiOptions::bindOptions(OptionsBinder &binder) {
	static llvm::cl::OptionCategory category("IREE Gemmini plugin options");

	binder.opt<bool>("iree-gemmini-enable", enable,
		llvm::cl::desc(
			"Enables the Gemmini post-global-optimization pipeline."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-lower-back-to-iree", lowerBackToIREE,
		llvm::cl::desc(
			"Lower recovered gemmini.* ops back to ordinary IREE/MLIR IR "
			"before dispatch creation."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-matmul", enableMatmul,
		llvm::cl::desc("Recover Gemmini matmul patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-conv2d", enableConv2D,
		llvm::cl::desc("Recover Gemmini conv2d patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-requantize", enableRequantize,
		llvm::cl::desc("Recover Gemmini requantize patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-clamp", enableClamp,
		llvm::cl::desc("Recover Gemmini clamp patterns."),
		llvm::cl::cat(category));

	binder.opt<std::string>("iree-gemmini-dataflow", dataflow,
		llvm::cl::desc("Default Gemmini dataflow mode: os or ws."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-tile-m", tileM,
		llvm::cl::desc("Default Gemmini tile size M."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-tile-n", tileN,
		llvm::cl::desc("Default Gemmini tile size N."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-tile-k", tileK,
		llvm::cl::desc("Default Gemmini tile size K."),
		llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
