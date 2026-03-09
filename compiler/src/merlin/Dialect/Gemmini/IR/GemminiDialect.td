#ifndef IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIDIALECT_TD_
#define IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Gemmini_Dialect : Dialect {
  let name = "gemmini";
  let cppNamespace = "::mlir::iree_compiler::Gemmini";
  let summary = "Gemmini accelerator recovery dialect";
  let description = [{
    A compiler-internal dialect used to recover Gemmini-friendly semantics
    from normalized IREE/MLIR IR after global optimization.
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif  // IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIDIALECT_TD_
