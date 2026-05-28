// Helpers for matching post-global-opt linalg.generic shapes against the
// QNN op set. Wraps upstream MLIR helpers (isaConvolutionOpInterface,
// inferConvolutionDims, isContractionBody, isElementwise) and adds our
// own body-shape matchers for the quant/dequant chains and the producer-
// pad recovery used by the conv/pool patterns.
//
// Designed so that each linalg→qnn rewrite pattern stays in the ~30-LoC
// range: classify the generic with the upstream helper, walk the body
// with one of the matchers below, recover any producer attributes
// (pad, etc.), and emit the qnn op.

#ifndef IREE_MERLIN_COMPILER_DIALECT_QNN_TRANSFORMS_GENERICMATCHUTILS_H_
#define IREE_MERLIN_COMPILER_DIALECT_QNN_TRANSFORMS_GENERICMATCHUTILS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::iree_compiler::QNN {

// Activation kinds matched in a fused conv-act tail (or a stand-alone
// activation generic). Mirrors QNN's ElementWiseNeuron `operation`
// values byte-for-byte; the conversion pattern can pass the matched
// kind directly to `qnn.element_wise_neuron`.
enum class ActivationKind : int32_t {
	Relu = 0,
	Relu6 = 1,
	Sigmoid = 2,
	Tanh = 3,
};

// Body shapes:
//
// Quant conv (zero-zp folded): %ax = extsi %in : i8 to i32 ; %wx = extsi
// %w : i8 to i32 ; %p = muli %ax, %wx ; %acc = addi %out, %p ; yield.
// Non-zero zp adds a `subi(extsi, zp_const)` between each `extsi` and
// the `muli`. Returns true and writes the recovered zero-points (0 if
// the body is the folded shape).
bool matchQuantConvBody(Block &block, int64_t *inZpOut, int64_t *wZpOut);

// Dequant body: extsi(i8→i32) ; subi(i32, zp) ; sitofp(i32→f32) ;
// mulf(f32, scale). Returns true and writes scale + zero-point. Pre-fold
// IR may also have the chain in a different order.
bool matchDequantBody(Block &block, double *scaleOut, int64_t *zpOut);

// Quantize body: divf(f32, scale) ; <round> ; addf(zp) ; fptosi(f32→i8).
// Returns true and writes scale + zero-point.
bool matchQuantizeBody(Block &block, double *scaleOut, int64_t *zpOut);

// Conv-rescale body (the trailing generic that follows a quant-conv):
//   sitofp(i32→f32) ; mulf(scale_acc) ; divf(scale_out) ; <round> ;
//   addf(zp) ; [maximumf, minimumf] ; fptosi(f32→i8).
//
// scale_acc is the accumulator-side scale; scale_out is the output
// quantization step (the i8 output's scale). Combined output scale is
// scale_out (HTA computes the rescale factor scale_acc/scale_out
// internally from the input/weight quant params).
bool matchConvRescaleBody(
	Block &block, double *scaleAccOut, double *scaleOutOut, int64_t *zpOut);

// Activation body: a single `arith.maximumf`/`arith.minimumf`/etc that
// implements relu / relu6 / sigmoid / tanh. Returns true and writes the
// matched kind.
bool matchActivationBody(Block &block, ActivationKind *kindOut);

// FP32 residual + bias + SiLU body — yolov8 fp32 residual blocks:
//   added = addf(conv_in, bias_in)        // conv + bias (broadcasted)
//   silu  = mulf(added, sigmoid(added))   // SiLU(conv + bias)
//   yield = addf(residual_in, silu)       // residual connection
// 3 inputs total: residual, conv_result, bias. Returns true on match.
bool matchFp32ResidualBiasSiLUBody(Block &block);

// FP32 bias+SiLU body — yolov8 fp32 conv-tail. Body shape:
//   y = addf(in, bias_in)           // in = conv output (broadcasted), bias_in
//   = 1D bias yield = mulf(y, divf(1, addf(exp(negf(y)), 1)))  // y *
//   sigmoid(y)
// Returns true on match. Caller identifies which DPS input is bias by
// indexing-map rank-1 broadcast; both block-args of the addf must be
// distinct BlockArgument operands.
bool matchFp32BiasSiLUBody(Block &block);

// SiLU-rescale body (yolov8 conv-tail with quantized SiLU = x*sigmoid(x)):
//   q_y = quantize(sitofp(in) * sAcc / sInt)
//   y_dq = sitofp(q_y) * sInt
//   q_sig = quantize(sigmoid(y_dq) / sSig)   // sigmoid = 1/(1+exp(-y_dq))
//   sig_dq = sitofp(q_sig) * sSig
//   yield = quantize(y_dq * sig_dq / sFinal)
// Returns true and writes the four scales + zero-point. Lowered to:
//   qnn.conv2d (i8 output, sInt) -> qnn.element_wise_neuron(SIGMOID) (i8, sSig)
//   -> qnn.element_wise_binary(MUL) (i8, sFinal).
bool matchSiLURescaleBody(Block &block, double *scaleAccOut,
	double *scaleIntermediateOut, double *scaleSigmoidOut,
	double *scaleFinalOut, int64_t *zpOut);

// Detect a constant f32 value; sets `*out` and returns true on match.
bool isF32Const(Value v, double *out);

// Walk the producer chain of `value` for a `tensor.pad`. If found, set
// the pad amount in [pad_top, pad_bottom, pad_left, pad_right] order
// (matching QNN's pad_amount layout) and return the pre-pad source
// value; otherwise return `value` unchanged with pads set to zero.
//
// Handles both rank-3 (NHC after IREE strips the batch dim of the conv
// input slice) and rank-4 (NHWC) producers.
Value recoverPadFromProducer(
	Value value, llvm::SmallVectorImpl<int32_t> &padAmountOut);

} // namespace mlir::iree_compiler::QNN

#endif // IREE_MERLIN_COMPILER_DIALECT_QNN_TRANSFORMS_GENERICMATCHUTILS_H_
