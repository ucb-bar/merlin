#ifndef IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNDIALECT_TD_
#define IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNDIALECT_TD_

include "mlir/IR/DialectBase.td"

def QNN_Dialect : Dialect {
  let name = "qnn";
  let cppNamespace = "::mlir::iree_compiler::QNN";
  let summary = "QNN op set dialect — mirrors Qualcomm's QnnOpDef.h";
  let description = [{
    A finite, op-set dialect that represents the subset of QNN's standard ops
    we currently codegen for (Conv2d, DepthwiseConv2d, FullyConnected, MatMul,
    ElementWiseNeuron, ElementWise{Add,Sub,Mul,Div}, Pool{Max,Avg}2d, Concat,
    Reshape, Transpose, Quantize, Dequantize). Every op mirrors the
    parameter set of its `QNN_OP_*` counterpart byte-for-byte so the
    serialize step can mechanically project (op, attrs) -> Qnn_OpConfig_t.

    Quant params live on the tensor type via `quant.uniform` (builtin) for
    per-tensor schemes. Per-channel weights use a custom `qnn.quant_params`
    attribute. Spatial ops (Conv2d / DepthwiseConv2d / Pool*) require NHWC
    inputs; the verifier rejects NCHW shapes — a layout-legalization pass
    inserts transposes upstream.

    See plan: ~/.claude/plans/i-want-to-enable-rosy-sundae.md (Ph1).
  }];
}

#endif // IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNDIALECT_TD_
