#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceDialect.h"

#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace mlir::iree_compiler::Radiance;

#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceDialect.cpp.inc"

// Include the AttrDef class definitions (storage classes etc.) before the
// addAttributes<>() call so the templates see complete types.
#define GET_ATTRDEF_CLASSES
#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.cpp.inc"

void RadianceDialect::initialize() {
	addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.cpp.inc"
		>();
}

// Note: parseAttribute / printAttribute are auto-generated from
// `useDefaultAttributePrinterParser = 1` in RadianceDialect.td. The
// generated dispatcher delegates to each AttrDef's own assemblyFormat-driven
// parse/print. The short forms `#radiance.global` / `#radiance.shared` are
// expressed via `assemblyFormat = "$value"` on Radiance_AddrSpaceAttr (the
// $value parameter is parsed as the enum keyword "global"/"shared").
