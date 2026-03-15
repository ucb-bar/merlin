#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSDETAIL_H_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Gemmini {

class GemminiDialect;

} // namespace mlir::iree_compiler::Gemmini

#define GEN_PASS_CLASSES
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

#endif // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSDETAIL_H_
