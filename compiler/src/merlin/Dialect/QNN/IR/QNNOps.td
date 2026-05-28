#ifndef IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNOPS_TD_
#define IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "QNNDialect.td"
include "QNNAttrs.td"

class QNN_Op<string mnemonic, list<Trait> traits = []> :
    Op<QNN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Convolution ops
//===----------------------------------------------------------------------===//

// QNN_OP_CONV_2D — Qualcomm's NHWC Conv2d op. See QnnOpDef.h:120-125.
//
// Per-backend dtype constraints (QAIRT 2.45):
//   HTA: in/out UFIXED_POINT_8 or UFIXED_POINT_16; weight UFIXED_POINT_8;
//        bias UFIXED_POINT_8 or SFIXED_POINT_32.
//   GPU (Adreno): all FLOAT_32 or FLOAT_16 — int8 path rejected.
//   HTP: like HTA + UFIXED_POINT_16 paths.
// Verifier enforces NHWC (input rank=4) and weight HWIO (rank=4).
def QNN_Conv2dOp : QNN_Op<"conv2d", []> {
  let summary = "Qualcomm QNN Conv2d (NHWC, weight HWIO).";
  let description = [{
    Mirrors `QNN_OP_CONV_2D` from QnnOpDef.h. `stride`, `pad_amount`,
    `dilation`, and `group` are op-level attrs (matching the `params`
    array Qualcomm's API takes). Weight layout is HWIO; bias is 1D over
    output channels.

    Layout invariant: `input` and `output` are NHWC (rank 4, channel last).
    Weight is HWIO (kh, kw, ic, oc). Bias is shape [oc]. Input quant
    params live on the tensor's `quant.uniform` element type or via the
    `input_quant` attribute when type-encoding isn't expressive enough
    (e.g., per-channel).
  }];
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$weight,
      Optional<AnyRankedTensor>:$bias,
      I32ArrayAttr:$stride,        // [stride_h, stride_w]
      I32ArrayAttr:$pad_amount,    // [pad_top, pad_bottom, pad_left, pad_right]
      I32ArrayAttr:$dilation,      // [dilation_h, dilation_w]
      I32Attr:$group);
  let results = (outs AnyRankedTensor:$output);
  // Custom builders are easier than a complex assemblyFormat with optional
  // operand. Use the default one-line format; canonical printing only shows
  // attrs + operand types. Pretty-printing is a nice-to-have, not load-bearing.
  let hasVerifier = 1;
}

// QNN_OP_DEPTH_WISE_CONV_2D — depthwise variant. See QnnOpDef.h:163-170.
def QNN_DepthwiseConv2dOp : QNN_Op<"depthwise_conv2d", []> {
  let summary = "Qualcomm QNN DepthwiseConv2d (NHWC, weight HWCM).";
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$weight,
      Optional<AnyRankedTensor>:$bias,
      I32ArrayAttr:$stride,
      I32ArrayAttr:$pad_amount,
      I32ArrayAttr:$dilation);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Linear / matmul
//===----------------------------------------------------------------------===//

// QNN_OP_FULLY_CONNECTED. Weight is [out_features, in_features]; QNN
// internally transposes for the gemm.
def QNN_FullyConnectedOp : QNN_Op<"fully_connected", []> {
  let summary = "Qualcomm QNN FullyConnected (dense matmul + optional bias).";
  let arguments = (ins
      AnyRankedTensor:$input,
      AnyRankedTensor:$weight,
      Optional<AnyRankedTensor>:$bias,
      DefaultValuedAttr<BoolAttr, "false">:$keep_dims);
  let results = (outs AnyRankedTensor:$output);
}

// QNN_OP_MAT_MUL. Generic matmul; useful for non-FC matmul patterns.
def QNN_MatMulOp : QNN_Op<"matmul", []> {
  let summary = "Qualcomm QNN MatMul.";
  let arguments = (ins
      AnyRankedTensor:$lhs,
      AnyRankedTensor:$rhs,
      DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
      DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Element-wise ops
//===----------------------------------------------------------------------===//

// QNN_OP_ELEMENT_WISE_NEURON — single-input activation (Relu, Relu6, Sigmoid,
// Tanh, ...). The `operation` attribute selects the activation kind via
// QNN_OP_ELEMENT_WISE_NEURON_OPERATION_* constants from QnnOpDef.h.
def QNN_ElementWiseNeuronOp : QNN_Op<"element_wise_neuron"> {
  let summary = "Qualcomm QNN ElementWiseNeuron activation.";
  let arguments = (ins
      AnyRankedTensor:$input,
      // Maps to QNN's `operation` parameter (Relu, Relu6, Sigmoid, Tanh,
      // ...) via QNN_OP_ELEMENT_WISE_NEURON_OPERATION_* enum. Renamed
      // here from "operation" to avoid shadowing Op::getOperation()
      // which returns Operation*. Codegen translates op_kind back to
      // QNN's "operation" parameter name on emit.
      I32Attr:$op_kind);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

// QNN_OP_ELEMENT_WISE_BINARY — binary ops (Add, Sub, Mul, Div, Maximum, ...).
def QNN_ElementWiseBinaryOp : QNN_Op<"element_wise_binary", []> {
  let summary = "Qualcomm QNN ElementWiseBinary.";
  let arguments = (ins
      AnyRankedTensor:$lhs,
      AnyRankedTensor:$rhs,
      I32Attr:$op_kind);  // QNN_OP_ELEMENT_WISE_BINARY_OPERATION_*
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Pooling
//===----------------------------------------------------------------------===//

def QNN_PoolMax2dOp : QNN_Op<"pool_max2d", []> {
  let summary = "Qualcomm QNN PoolMax2d (NHWC).";
  let arguments = (ins
      AnyRankedTensor:$input,
      I32ArrayAttr:$filter_size,    // [kh, kw]
      I32ArrayAttr:$stride,
      I32ArrayAttr:$pad_amount,     // [pad_top, pad_bottom, pad_left, pad_right]
      DefaultValuedAttr<I32Attr, "0">:$rounding_mode);  // 0=Floor, 1=Ceil
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def QNN_PoolAvg2dOp : QNN_Op<"pool_avg2d", []> {
  let summary = "Qualcomm QNN PoolAvg2d (NHWC).";
  let arguments = (ins
      AnyRankedTensor:$input,
      I32ArrayAttr:$filter_size,
      I32ArrayAttr:$stride,
      I32ArrayAttr:$pad_amount,
      DefaultValuedAttr<I32Attr, "0">:$rounding_mode,
      DefaultValuedAttr<BoolAttr, "true">:$count_pad_for_edges);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tensor-manipulation ops
//===----------------------------------------------------------------------===//

def QNN_ConcatOp : QNN_Op<"concat", []> {
  let summary = "Qualcomm QNN Concat along a given axis.";
  let arguments = (ins
      Variadic<AnyRankedTensor>:$inputs,
      I32Attr:$axis);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $inputs `)` attr-dict `:` type($inputs) `->` type($output)";
}

def QNN_ReshapeOp : QNN_Op<"reshape", []> {
  let summary = "Qualcomm QNN Reshape.";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

def QNN_TransposeOp : QNN_Op<"transpose", []> {
  let summary = "Qualcomm QNN Transpose by explicit perm.";
  let arguments = (ins AnyRankedTensor:$input, I32ArrayAttr:$perm);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Quantization boundaries
//===----------------------------------------------------------------------===//

// QNN_OP_QUANTIZE — fp -> fixed-point. Output's quant params live on its type.
def QNN_QuantizeOp : QNN_Op<"quantize", []> {
  let summary = "Qualcomm QNN Quantize (fp -> fixed-point).";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

// QNN_OP_DEQUANTIZE — fixed-point -> fp.
def QNN_DequantizeOp : QNN_Op<"dequantize", []> {
  let summary = "Qualcomm QNN Dequantize (fixed-point -> fp).";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// Phase 4b: yolov8-coverage ops
//===----------------------------------------------------------------------===//

// QNN_OP_PAD — constant/edge/reflect padding. pad_amount is [rank, 2]
// shape encoding (low, high) per dim. We model it as I32ArrayAttr with
// 2*rank entries: [low_0, high_0, low_1, high_1, ...].
def QNN_PadOp : QNN_Op<"pad", []> {
  let summary = "Qualcomm QNN Pad.";
  let arguments = (ins
      AnyRankedTensor:$input,
      I32ArrayAttr:$pad_amount,
      DefaultValuedAttr<I32Attr, "0">:$scheme,           // 0=CONSTANT, 1=MIRROR_SYMMETRIC, 2=MIRROR_REFLECT, 3=EDGE
      DefaultValuedAttr<F32Attr, "0.0">:$pad_constant);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

// QNN_OP_SOFTMAX — softmax along an axis with optional beta.
def QNN_SoftmaxOp : QNN_Op<"softmax", []> {
  let summary = "Qualcomm QNN Softmax.";
  let arguments = (ins
      AnyRankedTensor:$input,
      I32Attr:$axis,
      DefaultValuedAttr<F32Attr, "1.0">:$beta);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

// QNN_OP_REDUCE_{SUM,MEAN,MAX} — reduce over axes with optional keep_dims.
// op_kind: 0=ReduceSum, 1=ReduceMean, 2=ReduceMax.
def QNN_ReduceOp : QNN_Op<"reduce", []> {
  let summary = "Qualcomm QNN Reduce (sum/mean/max).";
  let arguments = (ins
      AnyRankedTensor:$input,
      I32ArrayAttr:$axes,
      I32Attr:$op_kind,
      DefaultValuedAttr<BoolAttr, "false">:$keep_dims);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "`ins` `(` $input `)` attr-dict `:` type($input) `->` type($output)";
}

#endif // IREE_MERLIN_COMPILER_DIALECT_QNN_IR_QNNOPS_TD_
