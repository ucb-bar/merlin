#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEDIALECT_H_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleOps.h.inc"
#undef GET_OP_CLASSES

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEDIALECT_H_
