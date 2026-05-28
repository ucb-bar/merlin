#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iree_compiler::Radiance;

// Generated enum definitions.
#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceEnums.cpp.inc"

// Note: GET_ATTRDEF_CLASSES is defined in RadianceDialect.cpp (where the
// storage type and definitions need to be visible for addAttributes<>).
// Defining them here too would cause double-definition.
