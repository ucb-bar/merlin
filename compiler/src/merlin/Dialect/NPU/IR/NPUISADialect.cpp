#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::NPUISA;

#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.cpp.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/NPU/IR/NPUISAOps.cpp.inc"

void NPUISADialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "compiler/src/merlin/Dialect/NPU/IR/NPUISAOps.cpp.inc"
		>();
}
