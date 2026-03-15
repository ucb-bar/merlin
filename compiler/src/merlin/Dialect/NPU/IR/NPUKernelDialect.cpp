#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::NPUKernel;

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.cpp.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelOps.cpp.inc"

void NPUKernelDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelOps.cpp.inc"
		>();
}
