#ifndef IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNDIALECT_H_
#define IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "compiler/src/merlin/Dialect/QNN/IR/QNNDialect.h.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/QNN/IR/QNNOps.h.inc"
#undef GET_OP_CLASSES

#endif // IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNDIALECT_H_
