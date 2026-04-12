#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

namespace mlir::iree_compiler::NPU {

void registerNPUPasses() {
	registerFoldFP8ScalesAroundContractionsPass();
	registerFuseF32IntermediateConversionsPass();
	registerDemoteF32ToBF16Pass();
	registerConvertLinalgToNPUKernelPass();
	registerConvertNPUKernelToSchedulePass();
	registerVerifyNPUUkernelSymbolsPass();
	registerConvertNPUScheduleToISAPass();
	registerPlanNPUISAMemoryPass();
}

} // namespace mlir::iree_compiler::NPU
