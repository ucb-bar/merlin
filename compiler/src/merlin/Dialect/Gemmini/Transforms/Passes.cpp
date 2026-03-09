#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

namespace mlir::iree_compiler::Gemmini {

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"
} // namespace

void registerGemminiPasses() {
	registerPasses();
}

} // namespace mlir::iree_compiler::Gemmini
