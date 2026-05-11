#include "compiler/plugins/target/NPU/NPUOptions.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {
namespace {

struct NPUSession : public PluginSession<NPUSession, NPUOptions,
						PluginActivationPolicy::Explicit> {
	static void registerPasses() {
		NPU::registerNPUPasses();
	}

	void onRegisterDialects(DialectRegistry &registry) override {
		registry.insert<NPUKernel::NPUKernelDialect>();
		registry.insert<NPUSchedule::NPUScheduleDialect>();
		registry.insert<NPUISA::NPUISADialect>();
	}

	void extendPostGlobalOptimizationPassPipeline(
		OpPassManager &passManager) override {
		if (!options.enable)
			return;

		NPU::NPUUkernelVerifyOptions verifyOptions;
		verifyOptions.strict = options.strictUkernelVerify;

		NPU::NPULoweringOptions loweringOptions;
		loweringOptions.matmulUseMxu1Weights = options.matmulUseMxu1Weights;
		loweringOptions.allowUnknownUkernelFallback =
			options.allowUnknownUkernelFallback;
		loweringOptions.nativeKernelLowering = options.nativeKernelLowering;
		loweringOptions.kernelManifestPath = options.kernelManifestPath;
		loweringOptions.strictNativeKernelCoverage =
			options.strictNativeKernelCoverage;

		NPU::NPUMemoryPlannerOptions plannerOptions;
		plannerOptions.loadBase = options.loadBase;
		plannerOptions.weightBase = options.weightBase;
		plannerOptions.storeBase = options.storeBase;
		plannerOptions.dmaFlagModulo = options.dmaFlagModulo;

		passManager.addPass(NPU::createConvertLinalgToNPUKernelPass());
		if (options.enableUkernelVerify) {
			passManager.addPass(
				NPU::createVerifyNPUUkernelSymbolsPass(verifyOptions));
		}
		passManager.addPass(createCanonicalizerPass());
		passManager.addPass(createCSEPass());

		passManager.addPass(NPU::createConvertNPUKernelToSchedulePass());
		if (options.enableUkernelVerify) {
			passManager.addPass(
				NPU::createVerifyNPUUkernelSymbolsPass(verifyOptions));
		}
		passManager.addPass(createCanonicalizerPass());
		passManager.addPass(createCSEPass());

		passManager.addPass(
			NPU::createConvertNPUScheduleToISAPass(loweringOptions));
		if (options.enableMemoryPlanner) {
			passManager.addPass(
				NPU::createPlanNPUISAMemoryPass(plannerOptions));
		}
		passManager.addPass(createCanonicalizerPass());
		passManager.addPass(createCSEPass());
	}
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_npu(
	mlir::iree_compiler::PluginRegistrar *registrar) {
	registrar->registerPlugin<::mlir::iree_compiler::NPUSession>("npu");
	return true;
}
