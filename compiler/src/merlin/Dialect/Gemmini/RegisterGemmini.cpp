// compiler/src/merlin/Dialect/Gemmini/RegisterGemmini.cpp

#include "merlin/Dialect/Gemmini/RegisterGemmini.h"
#include "merlin/Dialect/Gemmini/IR/GemminiDialect.h"

// Include the translation header we moved to Target/
#include "merlin/Target/LLVMIR/Dialect/Gemmini/GemminiToLLVMIRTranslation.h"

using namespace mlir;

namespace merlin {
namespace gemmini {

void registerGemminiDialect(DialectRegistry &registry) {
	// 1. Insert the Dialect itself (this calls GemminiDialect::initialize via
	// hook)
	registry.insert<GemminiDialect>();

	// 2. Add the interface that allows Gemmini ops to be translated to LLVM IR
	//    (This calls the function from GemminiToLLVMIRTranslation.cpp)
	merlin::registerGemminiDialectTranslation(registry);
}

} // namespace gemmini
} // namespace merlin
