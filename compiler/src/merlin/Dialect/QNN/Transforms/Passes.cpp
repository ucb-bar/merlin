#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::QNN {

void registerQNNPasses() {
	// Explicit registration. The previous .cpp-local `static
	// PassRegistration<>` initializers don't fire when those .o files live in a
	// static lib that hasn't pulled their symbols. Calling
	// registerPass(create*) from this central entry point — invoked by the QNN
	// plugin session at iree-compile / iree-opt load — guarantees both passes
	// show up by name.
	::mlir::registerPass([]() { return createConvertLinalgToQNNPass(); });
	::mlir::registerPass([]() { return createLegalizeLayoutToNHWCPass(); });
	::mlir::registerPass([]() { return createRewriteQDQToQuantUniformPass(); });
	::mlir::registerPass(
		[]() { return createApplyPlacementRequantizationPass(); });
	::mlir::registerPass([]() { return createFoldBodyQDQRoundtripPass(); });
	::mlir::registerPass([]() { return createRewriteToNHWCBindingsPass(); });
	::mlir::registerPass(
		[]() { return createLowerNHWCCastsToTransposesPass(); });
	::mlir::registerPass(
		[]() { return createInlineConstantUtilGlobalsPass(); });
}

} // namespace mlir::iree_compiler::QNN
