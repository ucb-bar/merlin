#include "compiler/plugins/target/NPU/NPUOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::NPUOptions);

namespace mlir::iree_compiler {

void NPUOptions::bindOptions(OptionsBinder &binder) {
	static llvm::cl::OptionCategory category("IREE NPU plugin options");

	binder.opt<bool>("iree-npu-enable", enable,
		llvm::cl::desc(
			"Enables the NPU post-global-optimization lowering pipeline."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-npu-enable-ukernel-verify", enableUkernelVerify,
		llvm::cl::desc("Enable verification of NPU ukernel symbols/shapes."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-npu-strict-ukernel-verify", strictUkernelVerify,
		llvm::cl::desc("Treat unknown ukernel symbol families as hard errors."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-npu-allow-unknown-ukernel-fallback",
		allowUnknownUkernelFallback,
		llvm::cl::desc("Allow unknown ukernel symbols to lower via generic "
					   "matmul fallback."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-npu-matmul-use-mxu1-weights", matmulUseMxu1Weights,
		llvm::cl::desc("Use dma.load.mxu1 (instead of dma.load.mxu0) for "
					   "matmul.mxu0 weights."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-npu-enable-memory-planner", enableMemoryPlanner,
		llvm::cl::desc(
			"Enable deterministic NPU ISA memory/flag assignment pass."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-npu-dma-flag-modulo", dmaFlagModulo,
		llvm::cl::desc("Modulo used for planned DMA flag assignment."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-npu-load-base", loadBase,
		llvm::cl::desc("Base address for planned dma.load activations."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-npu-weight-base", weightBase,
		llvm::cl::desc("Base address for planned dma.load.mxu* weights."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-npu-store-base", storeBase,
		llvm::cl::desc("Base address for planned dma.store outputs."),
		llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
