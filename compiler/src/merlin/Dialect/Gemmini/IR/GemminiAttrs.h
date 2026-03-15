#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_IR_GEMMINIATTRS_H_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_IR_GEMMINIATTRS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h.inc"

#endif // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_IR_GEMMINIATTRS_H_
