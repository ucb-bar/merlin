#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_IR_GEMMINIOPS_H_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_IR_GEMMINIOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h"
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.h"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h.inc"

#endif // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_IR_GEMMINIOPS_H_
