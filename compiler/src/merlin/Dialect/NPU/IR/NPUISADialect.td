#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUISADIALECT_TD_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUISADIALECT_TD_

include "mlir/IR/DialectBase.td"

def NPUISA_Dialect : Dialect {
  let name = "npu_isa";
  let cppNamespace = "::mlir::iree_compiler::NPUISA";
  let summary = "NPU internal near-ISA dialect";
  let description = [{
    Internal layer that is close to hardware instruction forms and is directly
    translated to simulator-compatible textual ISA.
  }];
}

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUISADIALECT_TD_
