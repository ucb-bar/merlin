#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

namespace mlir::iree_compiler::NPU {

void registerNPUPasses() {
	registerConvertLinalgToNPUKernelPass();
	registerConvertNPUKernelToSchedulePass();
	registerVerifyNPUUkernelSymbolsPass();
	registerConvertNPUScheduleToISAPass();
	registerPlanNPUISAMemoryPass();
}

} // namespace mlir::iree_compiler::NPU
