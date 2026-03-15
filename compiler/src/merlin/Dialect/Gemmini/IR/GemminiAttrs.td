#ifndef IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIATTRS_TD_
#define IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIATTRS_TD_

include "GemminiDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Gemmini_Dataflow : I32EnumAttr<
    "Dataflow",
    "Gemmini dataflow mode",
    [
      I32EnumAttrCase<"OutputStationary", 0, "os">,
      I32EnumAttrCase<"WeightStationary", 1, "ws">
    ]> {
  let cppNamespace = "::mlir::iree_compiler::Gemmini";
}

#endif  // IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIATTRS_TD_
