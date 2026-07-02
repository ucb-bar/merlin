#include "Dialects.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

// ============================ merlin_iface ============================
#include "IfaceDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "IfaceTypes.cpp.inc"

#define GET_OP_CLASSES
#include "IfaceOps.cpp.inc"

void iface::IfaceDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "IfaceTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IfaceOps.cpp.inc"
      >();
}

// ============================ gemmini ============================
#include "GemminiDialect.cpp.inc"

#define GET_OP_CLASSES
#include "GemminiOps.cpp.inc"

void gem::GemminiDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GemminiOps.cpp.inc"
      >();
}
