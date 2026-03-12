#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELDIALECT_TD_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELDIALECT_TD_

include "mlir/IR/DialectBase.td"

def NPUKernel_Dialect : Dialect {
  let name = "npu_kernel";
  let cppNamespace = "::mlir::iree_compiler::NPUKernel";
  let summary = "NPU kernel semantic dialect";
  let description = [{
    Public NPU layer for kernel-level semantics that cannot be represented
    precisely in existing standard dialects.
  }];
}

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELDIALECT_TD_
