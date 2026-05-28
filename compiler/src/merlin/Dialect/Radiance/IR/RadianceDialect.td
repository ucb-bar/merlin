#ifndef IREE_RADIANCE_DIALECT_RADIANCE_IR_RADIANCEDIALECT_TD_
#define IREE_RADIANCE_DIALECT_RADIANCE_IR_RADIANCEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Radiance_Dialect : Dialect {
  let name = "radiance";
  let cppNamespace = "::mlir::iree_compiler::Radiance";
  let summary = "Radiance/Muon GPU compile-time annotations dialect";
  let description = [{
    A compiler-internal dialect used to tag MLIR programs destined for the
    UCB-BAR Radiance/Muon GPU. The dialect is intentionally minimal: it
    provides address-space attributes (matching mu_intrinsics.h's
    `__global` / `__shared` qualifiers) and recognizes `func.func`
    operations bearing the `radiance.kernel` unit attribute, which the
    LowerRadianceToLLVM pass picks up and lowers to LLVM IR text consumable
    by kernels/core/precompile.py with source_lang=ll.

    No ops are defined here; the dialect is attribute-only by design (per
    the 2026-04-30 dev-blog: lowering a custom op set provides no codegen
    win when the actual ISA exposure is inline asm in mu_intrinsics.h).
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif  // IREE_RADIANCE_DIALECT_RADIANCE_IR_RADIANCEDIALECT_TD_
