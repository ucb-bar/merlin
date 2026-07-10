#ifndef GEMMINI_OOT_BACKEND_H
#define GEMMINI_OOT_BACKEND_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace backend {

// Clone the merlin_iface ops in `m` into mirror gemmini ops (in place).
mlir::LogicalResult convertIfaceToGemmini(mlir::ModuleOp m);

// Walk the program (iface or gemmini ops) and write a command_buffer.json to `path`.
mlir::LogicalResult emitCommandBuffer(mlir::ModuleOp m, llvm::StringRef path);

// Replace the module body with an llvm.func @gemmini_kernel driving Gemmini via RoCC inline-asm.
mlir::LogicalResult lowerToLlvmRocc(mlir::ModuleOp m);

} // namespace backend

#endif // GEMMINI_OOT_BACKEND_H
