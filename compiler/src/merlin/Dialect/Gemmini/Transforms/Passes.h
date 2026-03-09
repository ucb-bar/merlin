#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_H_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_H_

#include <memory>

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Gemmini {

struct GemminiTransformOptions {
	bool enableMatmul = true;
	bool enableConv2D = true;
	bool enableRequantize = true;
	bool enableClamp = true;

	Dataflow defaultDataflow = Dataflow::OutputStationary;

	int64_t tileM = 16;
	int64_t tileN = 16;
	int64_t tileK = 16;
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createConvertToGemminiPass(
	const GemminiTransformOptions &options = {});
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGemminiCanonicalizeFuncPass();
std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerToISAPass(
	const GemminiTransformOptions &options = {});
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLowerGemminiToIREEPass();

void registerGemminiPasses();

#define GEN_PASS_DECL
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

} // namespace mlir::iree_compiler::Gemmini

#endif // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_H_
