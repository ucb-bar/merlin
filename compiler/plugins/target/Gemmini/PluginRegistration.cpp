#include "compiler/plugins/target/Gemmini/GemminiOptions.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h"
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.h"
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"
#include "compiler/src/merlin/Target/LLVMIR/Dialect/Gemmini/GemminiToLLVMIRTranslation.h"
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

static Gemmini::MxFormat parseMxFormat(llvm::StringRef value) {
	if (value.equals_insensitive("fp4"))
		return Gemmini::MxFormat::Fp4;
	if (value.equals_insensitive("fp6_0"))
		return Gemmini::MxFormat::Fp6_0;
	if (value.equals_insensitive("fp6_1"))
		return Gemmini::MxFormat::Fp6_1;
	if (value.equals_insensitive("fp8_0"))
		return Gemmini::MxFormat::Fp8_0;
	if (value.equals_insensitive("fp8_1"))
		return Gemmini::MxFormat::Fp8_1;
	// "disabled", or any unrecognized value, falls back to vanilla Gemmini.
	return Gemmini::MxFormat::Disabled;
}

static Gemmini::CommandIssue parseCommandIssue(llvm::StringRef value) {
	if (value.equals_insensitive("mmio"))
		return Gemmini::CommandIssue::MMIO;
	return Gemmini::CommandIssue::RoCC;
}

struct GemminiSession : public PluginSession<GemminiSession, GemminiOptions,
							PluginActivationPolicy::Explicit> {
	static void registerPasses() {
		Gemmini::registerGemminiPasses();
	}

	void onRegisterDialects(DialectRegistry &registry) override {
		registry.insert<Gemmini::GemminiDialect>();
		merlin::registerGemminiDialectTranslation(registry);
	}

	void extendPostGlobalOptimizationPassPipeline(
		OpPassManager &passManager) override {
		if (!options.enable)
			return;

		Gemmini::GemminiTransformOptions transformOptions;
		transformOptions.enableMatmul = options.enableMatmul;
		transformOptions.enableFP8Matmul = options.enableFP8Matmul;
		transformOptions.enableConv2D = options.enableConv2D;
		transformOptions.enableRequantize = options.enableRequantize;
		transformOptions.enableClamp = options.enableClamp;
		transformOptions.defaultDataflow = parseDataflowMode(options.dataflow);
		transformOptions.tileM = options.tileM;
		transformOptions.tileN = options.tileN;
		transformOptions.tileK = options.tileK;
		// Phase-5 GemminiTargetConfig: MX format / command-issue path.
		transformOptions.target.mxFormat = parseMxFormat(options.mxFormat);
		transformOptions.target.commandIssue =
			parseCommandIssue(options.commandIssue);
		transformOptions.target.mmioBase = options.mmioBase;
		// Phase-8 LOOP_WS toggle. "auto" follows commandIssue=mmio; "true"
		// / "false" force the explicit setting. Anything unrecognized is
		// treated as "auto" (preserves Phase 1-7 RoCC behavior).
		if (llvm::StringRef(options.useLoopWs).equals_insensitive("true")) {
			transformOptions.target.useLoopWs = true;
		} else if (llvm::StringRef(options.useLoopWs)
					   .equals_insensitive("false")) {
			transformOptions.target.useLoopWs = false;
		} else {
			transformOptions.target.useLoopWs =
				transformOptions.target.commandIssue ==
				Gemmini::CommandIssue::MMIO;
		}
		// Phase-4 GemminiTargetConfig: hardware descriptor.
		transformOptions.target.dim = options.dim;
		transformOptions.target.addrLen = options.addrLen;
		transformOptions.target.accRows = options.accRows;
		transformOptions.target.bankRows = options.bankRows;
		transformOptions.target.bankNum = options.bankNum;
		transformOptions.target.elemBits = options.elemBits;
		transformOptions.target.accBits = options.accBits;

		// Two end-to-end paths share this hook:
		//
		// 1) `lowerBackToIREE=true` — recover gemmini ops at host scope
		//    and immediately lower them back to vanilla linalg/tensor IR.
		//    IREE's standard codegen handles the rest. Validates the
		//    recovery patterns and produces a runnable .vmfb whose matmul
		//    executes as scalar/vector RISC-V (no RoCC instructions).
		//
		// 2) `lowerBackToIREE=false` — leave `linalg.matmul` untouched at
		//    host scope, attach an `iree_codegen.compilation_info` on it
		//    pointing at a textual pass_pipeline that runs the gemmini
		//    recovery and ISA-tier lowering INSIDE the dispatch executable
		//    (via PipelineAttrInterface). IREE outlines the matmul as
		//    usual, then `MaterializeUserConfigsPass` propagates our
		//    translation_info to the dispatch func, and
		//    `LLVMCPULowerExecutableTargetPass` invokes our textual
		//    pipeline. The result: gemmini ops appear inside the dispatch
		//    body, lower to `gemmini.intr.*`, translate to
		//    `llvm.intr.riscv.*`, and the RISC-V backend emits custom-3
		//    RoCC opcodes in the dispatch ELF.
		if (options.lowerBackToIREE) {
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

			passManager.addNestedPass<func::FuncOp>(
				Gemmini::createLowerGemminiToIREEPass());
			passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
			passManager.addNestedPass<func::FuncOp>(createCSEPass());

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

			passManager.addNestedPass<IREE::Util::FuncOp>(
				Gemmini::createLowerGemminiToIREEPass());
			passManager.addNestedPass<IREE::Util::FuncOp>(
				createCanonicalizerPass());
			passManager.addNestedPass<IREE::Util::FuncOp>(createCSEPass());
			return;
		}

		// Native gemmini codegen path. We do NOT recover at host scope;
		// instead, we attach the codegen pipeline as an attribute on each
		// linalg.matmul and let IREE's dispatch outliner + executable
		// lowering machinery drive recovery inside the dispatch.
		// Pass the GemminiTargetConfig through so the textual pipeline
		// string formats `merlin-gemmini-legalize-for-llvm-export{dim=N
		// addr-len=N ...}` with the user-configured values.

		// Linalg-level rewrites that benefit Gemmini regardless of branch.
		// Run BEFORE AttachCompilationInfo so the rewritten linalg ops are
		// what AttachCompilationInfo sees.  Need both func.func and
		// util.func variants to cover the IR at this stage.
		passManager.addNestedPass<func::FuncOp>(
			Gemmini::createGemminiPreprocessPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(
			Gemmini::createGemminiPreprocessPass());
		passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
		passManager.addNestedPass<func::FuncOp>(createCSEPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(
			createCanonicalizerPass());
		passManager.addNestedPass<IREE::Util::FuncOp>(createCSEPass());

		passManager.addNestedPass<func::FuncOp>(
			Gemmini::createGemminiAttachCompilationInfoPassWithOptions(
				transformOptions));
		passManager.addNestedPass<IREE::Util::FuncOp>(
			Gemmini::createGemminiAttachCompilationInfoPassWithOptions(
				transformOptions));
	}
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_gemmini(
	mlir::iree_compiler::PluginRegistrar *registrar) {
	registrar->registerPlugin<::mlir::iree_compiler::GemminiSession>("gemmini");
	return true;
}
