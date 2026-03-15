#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::NPUSchedule;

#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleOps.cpp.inc"

void NPUScheduleDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleOps.cpp.inc"
		>();
}
