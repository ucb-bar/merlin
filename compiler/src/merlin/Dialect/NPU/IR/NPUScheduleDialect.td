#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEDIALECT_TD_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEDIALECT_TD_

include "mlir/IR/DialectBase.td"

def NPUSchedule_Dialect : Dialect {
  let name = "npu_schedule";
  let cppNamespace = "::mlir::iree_compiler::NPUSchedule";
  let summary = "NPU scheduled pipeline dialect";
  let description = [{
    Public NPU layer that makes scheduling decisions explicit:
    tile boundaries, engine assignment, and async dependencies.
  }];
}

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEDIALECT_TD_
