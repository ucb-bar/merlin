#include "compiler/plugins/target/Radiance/RadianceOptions.h"

#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceDialect.h"
#include "compiler/src/merlin/Dialect/Radiance/Transforms/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {
namespace {

// Phase 2.6 plugin: registers the radiance dialect, exposes the option
// surface, and (when --iree-radiance-enable=true) inserts the lowering
// pipeline into the post-global-optimization pass manager.
//
// Pipeline:
//   1. ConvertRadianceAddrSpaces — rewrite #radiance.global memref attrs
//      to integer addrspaces consumable by the standard MemRef-to-LLVM
//      conversion. Pre-pass before the actual lowering.
//   2. LowerRadianceToLLVM — runs the standard MLIR-to-LLVM conversion
//      pipeline on the module, then translates to LLVM IR text and writes
//      to disk if --iree-radiance-emit-llvm-ir=true.
//
// Output (when emit-llvm-ir is set): a kernel_body.ll file consumable by
// kernels/core/precompile.py with source_lang=ll.
struct RadianceSession : public PluginSession<RadianceSession, RadianceOptions,
							 PluginActivationPolicy::Explicit> {
	static void registerPasses() {
		Radiance::registerRadiancePasses();
	}

	void onRegisterDialects(DialectRegistry &registry) override {
		registry.insert<Radiance::RadianceDialect>();
	}

	void extendInputConversionPreprocessingPassPipeline(
		OpPassManager &passManager,
		InputDialectOptions::Type inputType) override {
		if (!options.enable)
			return;

		// Run BEFORE iree's input conversion, while functions are still
		// `func.func` with their original attributes (radiance.kernel,
		// radiance.entry_symbol, radiance.num_warps). After IREE's input
		// conversion these get rewritten to `util.func` with stripped
		// attrs.
		passManager.addPass(Radiance::createConvertRadianceAddrSpacesPass());

		Radiance::LowerRadianceToLLVMOptions lowerOpts;
		lowerOpts.emitLLVMIR = options.emitLLVMIR;
		lowerOpts.emitLLVMIRPath = options.emitLLVMIRPath;
		lowerOpts.numWarps = options.numWarps;
		passManager.addPass(Radiance::createLowerRadianceToLLVMPass(lowerOpts));
	}
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_radiance(
	mlir::iree_compiler::PluginRegistrar *registrar) {
	registrar->registerPlugin<::mlir::iree_compiler::RadianceSession>(
		"radiance");
	return true;
}
