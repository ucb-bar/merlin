#ifndef IREE_NPU_COMPILER_DIALECT_NPU_TRANSFORMS_PASSES_H_
#define IREE_NPU_COMPILER_DIALECT_NPU_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::NPU {

struct NPUUkernelVerifyOptions {
	bool strict = true;
};

struct NPULoweringOptions {
	// If true, emit dma.load.mxu1 for matmul.mxu0 source weights.
	// If false, emit dma.load.mxu0.
	bool matmulUseMxu1Weights = false;
	// If false, unknown ukernel symbols are rejected during lowering.
	bool allowUnknownUkernelFallback = true;
	// If true, lower supported schedule ops by splicing native manifest-backed
	// npu_model instruction streams instead of older abstract npu_isa
	// skeletons.
	bool nativeKernelLowering = false;
	// JSON manifest containing native SaturnNPU kernel instruction streams.
	std::string kernelManifestPath;
	// If true, manifest mode rejects missing kernel family coverage instead of
	// falling back to abstract skeleton lowering.
	bool strictNativeKernelCoverage = true;
};

struct NPUMemoryPlannerOptions {
	int64_t loadBase = 0;
	int64_t weightBase = 0x2000;
	int64_t storeBase = 0x5000;
	int64_t dmaFlagModulo = 3;
};

std::unique_ptr<Pass> createConvertLinalgToNPUKernelPass();
std::unique_ptr<Pass> createConvertNPUKernelToSchedulePass();
std::unique_ptr<Pass> createTileNPUKernelToSchedulePass();
std::unique_ptr<Pass> createConvertNPUScheduleToISAPass();
std::unique_ptr<Pass> createConvertNPUScheduleToISAPass(
	const NPULoweringOptions &options);
std::unique_ptr<Pass> createVerifyNPUUkernelSymbolsPass(
	const NPUUkernelVerifyOptions &options = {});
std::unique_ptr<Pass> createPlanNPUISAMemoryPass(
	const NPUMemoryPlannerOptions &options = {});

void registerConvertLinalgToNPUKernelPass();
void registerConvertNPUKernelToSchedulePass();
void registerTileNPUKernelToSchedulePass();
void registerConvertNPUScheduleToISAPass();
void registerVerifyNPUUkernelSymbolsPass();
void registerPlanNPUISAMemoryPass();

void registerNPUPasses();

} // namespace mlir::iree_compiler::NPU

#endif // IREE_NPU_COMPILER_DIALECT_NPU_TRANSFORMS_PASSES_H_
