// compiler/src/merlin/Dialect/Gemmini/RegisterGemmini.h

#ifndef MERLIN_DIALECT_GEMMINI_REGISTERGEMMINI_H
#define MERLIN_DIALECT_GEMMINI_REGISTERGEMMINI_H

#include "mlir/IR/DialectRegistry.h"

namespace merlin {
namespace gemmini {

// Registers the Gemmini dialect and optionally the LLVM IR translation
// interfaces.
void registerGemminiDialect(mlir::DialectRegistry &registry);

} // namespace gemmini
} // namespace merlin

#endif // MERLIN_DIALECT_GEMMINI_REGISTERGEMMINI_H
