#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELDIALECT_H_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelOps.h.inc"
#undef GET_OP_CLASSES

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELDIALECT_H_
