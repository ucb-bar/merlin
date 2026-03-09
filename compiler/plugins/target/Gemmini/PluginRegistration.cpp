#include "compiler/plugins/target/Gemmini/GemminiOptions.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h"
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.h"
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {
namespace {

static Gemmini::Dataflow parseDataflowMode(llvm::StringRef value) {
	if (value.equals_insensitive("ws")) {
		return Gemmini::Dataflow::WeightStationary;
	}
	return Gemmini::Dataflow::OutputStationary;
}

struct GemminiSession : public PluginSession<GemminiSession, GemminiOptions,
							PluginActivationPolicy::Explicit> {
	static void registerPasses() {
		Gemmini::registerGemminiPasses();
	}

	void onRegisterDialects(DialectRegistry &registry) override {
		registry.insert<Gemmini::GemminiDialect>();
	}

	void extendPostGlobalOptimizationPassPipeline(
		OpPassManager &passManager) override {
		if (!options.enable)
			return;

		Gemmini::GemminiTransformOptions transformOptions;
		transformOptions.enableMatmul = options.enableMatmul;
		transformOptions.enableConv2D = options.enableConv2D;
		transformOptions.enableRequantize = options.enableRequantize;
		transformOptions.enableClamp = options.enableClamp;
		transformOptions.defaultDataflow = parseDataflowMode(options.dataflow);
		transformOptions.tileM = options.tileM;
		transformOptions.tileN = options.tileN;
		transformOptions.tileK = options.tileK;

		passManager.addNestedPass<func::FuncOp>(
			Gemmini::createConvertToGemminiPass(transformOptions));
		passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
		passManager.addNestedPass<func::FuncOp>(createCSEPass());

		passManager.addNestedPass<func::FuncOp>(
			Gemmini::createLowerToISAPass(transformOptions));
		passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
		passManager.addNestedPass<func::FuncOp>(createCSEPass());

		passManager.addNestedPass<func::FuncOp>(
			Gemmini::createGemminiCanonicalizeFuncPass());
		passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
		passManager.addNestedPass<func::FuncOp>(createCSEPass());

		if (options.lowerBackToIREE) {
			passManager.addNestedPass<func::FuncOp>(
				Gemmini::createLowerGemminiToIREEPass());
			passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
			passManager.addNestedPass<func::FuncOp>(createCSEPass());
		}

		passManager.addNestedPass<IREE::Util::FuncOp>(
			Gemmini::createConvertToGemminiPass(transformOptions));
		passManager.addNestedPass<IREE::Util::FuncOp>(
			createCanonicalizerPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(createCSEPass());

		passManager.addNestedPass<IREE::Util::FuncOp>(
			Gemmini::createLowerToISAPass(transformOptions));
		passManager.addNestedPass<IREE::Util::FuncOp>(
			createCanonicalizerPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(createCSEPass());

		passManager.addNestedPass<IREE::Util::FuncOp>(
			Gemmini::createGemminiCanonicalizeFuncPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(
			createCanonicalizerPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(createCSEPass());

		if (options.lowerBackToIREE) {
			passManager.addNestedPass<IREE::Util::FuncOp>(
				Gemmini::createLowerGemminiToIREEPass());
			passManager.addNestedPass<IREE::Util::FuncOp>(
				createCanonicalizerPass());
			passManager.addNestedPass<IREE::Util::FuncOp>(createCSEPass());
		}
	}
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_gemmini(
	mlir::iree_compiler::PluginRegistrar *registrar) {
	registrar->registerPlugin<::mlir::iree_compiler::GemminiSession>("gemmini");
	return true;
}
