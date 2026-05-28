#ifndef IREE_RADIANCE_COMPILER_DIALECT_RADIANCE_IR_RADIANCEATTRS_H_
#define IREE_RADIANCE_COMPILER_DIALECT_RADIANCE_IR_RADIANCEATTRS_H_

#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

// Generated enum + attr declarations.
#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.h.inc"

namespace mlir::iree_compiler::Radiance {

// String-keyed unit attribute name we attach to `func.func` ops to mark
// them as Radiance/Muon kernel entry points. The lowering pass picks up
// any `func.func` bearing this attribute.
inline llvm::StringRef getKernelAttrName() {
	return "radiance.kernel";
}

// Companion integer attribute carrying the warp count for mu_schedule.
inline llvm::StringRef getNumWarpsAttrName() {
	return "radiance.num_warps";
}

// Companion string attribute carrying the entry symbol name (must match
// the manifest's entry_symbol field). Optional; when missing, the
// function's own name is used.
inline llvm::StringRef getEntrySymbolAttrName() {
	return "radiance.entry_symbol";
}

} // namespace mlir::iree_compiler::Radiance

#endif // IREE_RADIANCE_COMPILER_DIALECT_RADIANCE_IR_RADIANCEATTRS_H_
