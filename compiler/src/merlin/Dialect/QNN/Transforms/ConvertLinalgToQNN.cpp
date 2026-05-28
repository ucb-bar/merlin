// linalg → qnn conversion (post-global-opt, generic-matching).
//
// Patterns anchor on `linalg.generic` (and a few named tensor/linalg ops
// that survive global-opt) and reshape recognized convolution / matmul /
// pool / elementwise / quant generics into `qnn.*` ops. Classification
// uses upstream MLIR helpers (`isaConvolutionOpInterface`,
// `inferConvolutionDims`, `isaContractionOpInterface`,
// `inferContractionDims`, `isElementwise`); body shapes are matched with
// `GenericMatchUtils.h`.
//
// Phase 2 of the rosy-sundae plan. Sub-phases:
//   2A — GenericMatchUtils helpers (separate file)
//   2B — Conv2d via the helpers (this file)
//   2C — DepthwiseConv2d, Pool{Max,Avg}2d, MatMul, FullyConnected,
//        ElementWise{Binary,Neuron}, Quantize, Dequantize (this file)
//   2D — Fused-form patterns Conv+Bias / Conv+Bias+Relu / Conv+Add
//        (this file, applied with lower priority).

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/QNN/IR/QNNDialect.h"
#include "compiler/src/merlin/Dialect/QNN/Transforms/GenericMatchUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::QNN {
namespace {

// QNN_OP_ELEMENT_WISE_BINARY_OPERATION_* enum values from QnnOpDef.h.
// Codegen translates this back to the symbolic name on emit.
enum BinaryKind : int32_t {
	kBinaryAdd = 0,
	kBinarySub = 1,
	kBinaryMul = 2,
	kBinaryDiv = 3,
};

// Convert ArrayRef<int64_t> to an I32ArrayAttr. The QNN ops use i32 arrays
// to mirror Qualcomm's wire format.
ArrayAttr toI32Array(Builder &b, ArrayRef<int64_t> values) {
	SmallVector<int32_t> v;
	v.reserve(values.size());
	for (int64_t x : values)
		v.push_back(static_cast<int32_t>(x));
	return b.getI32ArrayAttr(v);
}

ArrayAttr toI32Array(Builder &b, ArrayRef<int32_t> values) {
	return b.getI32ArrayAttr(values);
}

//===----------------------------------------------------------------------===//
// Phase 2B — Conv2d (helper-based)
//===----------------------------------------------------------------------===//

struct LowerConv2dQGeneric : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isaConvolutionOpInterface(
				op, /*allowEmptyConvolvedDims=*/false))
			return failure();
		auto dimsOr = linalg::inferConvolutionDims(op);
		linalg::ConvolutionDimensions fallbackDims;
		linalg::ConvolutionDimensions *dimsPtr = nullptr;
		if (succeeded(dimsOr)) {
			dimsPtr = &(*dimsOr);
		} else {
			// yolov8 quant-conv generic has 3 inputs (input, weight, scalar
			// zp), which confuses inferConvolutionDims. Manually populate dims
			// from indexing maps.
			auto outMap = op.getIndexingMapsArray()[op.getNumDpsInputs()];
			llvm::SmallVector<unsigned> outIters;
			for (auto e : outMap.getResults()) {
				if (auto d = dyn_cast<AffineDimExpr>(e))
					outIters.push_back(d.getPosition());
			}
			if (outIters.size() < 3)
				return failure();
			fallbackDims.outputChannel.push_back(outIters[0]);
			for (size_t i = 1; i < outIters.size() && i < 3; ++i)
				fallbackDims.outputImage.push_back(outIters[i]);
			auto iters = op.getIteratorTypesArray();
			for (size_t i = 0; i < iters.size(); ++i) {
				if (iters[i] == utils::IteratorType::reduction) {
					if (fallbackDims.inputChannel.empty())
						fallbackDims.inputChannel.push_back(i);
					else
						fallbackDims.filterLoop.push_back(i);
				}
			}
			fallbackDims.strides.assign(fallbackDims.outputImage.size(), 1);
			fallbackDims.dilations.assign(fallbackDims.outputImage.size(), 1);
			auto inMap = op.getIndexingMapsArray()[0];
			for (size_t r = 0; r < inMap.getNumResults(); ++r) {
				auto e = inMap.getResult(r);
				if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
					if (bin.getKind() == AffineExprKind::Add) {
						if (auto mul =
								dyn_cast<AffineBinaryOpExpr>(bin.getLHS())) {
							if (mul.getKind() == AffineExprKind::Mul) {
								auto coeff =
									dyn_cast<AffineConstantExpr>(mul.getRHS());
								auto outDim =
									dyn_cast<AffineDimExpr>(mul.getLHS());
								if (coeff && outDim) {
									for (size_t s = 0;
										 s < fallbackDims.outputImage.size();
										 ++s) {
										if (fallbackDims.outputImage[s] ==
											outDim.getPosition()) {
											fallbackDims.strides[s] =
												(uint64_t)coeff.getValue();
											break;
										}
									}
								}
							}
						}
					}
				}
			}
			dimsPtr = &fallbackDims;
		}
		auto &dims = *dimsPtr;

		// Conv2d (vs DepthwiseConv2d): no `depth` dims.
		if (!dims.depth.empty())
			return failure();
		if (dims.outputImage.size() != 2 || dims.filterLoop.size() != 2)
			return failure();

		// Layout check: QNN Conv2d is NHWC. Detect NCHW (channel-first iter
		// position) and remember to wrap the op with input/weight/output
		// transposes below. Phase 4a (yolov8 NCHW conv) — supports rank-3
		// (no-batch [C,H,W]) and rank-4 ([N,C,H,W]) layouts.
		bool isNchw = false;
		if (dims.outputChannel.size() == 1) {
			int64_t lastImage = *std::max_element(
				dims.outputImage.begin(), dims.outputImage.end());
			int64_t cPos = dims.outputChannel[0];
			isNchw = (cPos < lastImage);
		}

		// Operands. yolov8's quant-conv generic has 3 inputs (input, weight,
		// scalar_zp) — accept either 2 or 3 with the first two being ranked
		// tensors of identical-rank conv operand shape.
		int nIns = op.getNumDpsInputs();
		if ((nIns != 2 && nIns != 3) || op.getNumDpsInits() != 1)
			return failure();
		Value input = op.getDpsInputs()[0];
		Value weight = op.getDpsInputs()[1];
		if (!isa<RankedTensorType>(input.getType()) ||
			!isa<RankedTensorType>(weight.getType()))
			return failure();

		// Element types: support both quant int (HTA) and fp (GPU/CPU)
		// Conv2d. Pick the body matcher accordingly.
		auto inputTy = cast<RankedTensorType>(input.getType());
		auto weightTy = cast<RankedTensorType>(weight.getType());
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		Type inElt = inputTy.getElementType();
		Type wElt = weightTy.getElementType();
		Type outElt = outTy.getElementType();
		bool isQuantInt =
			inElt.isInteger(8) && wElt.isInteger(8) && outElt.isInteger(32);
		bool isFp = (inElt.isF16() || inElt.isF32()) && wElt == inElt &&
			outElt == inElt;
		if (!isQuantInt && !isFp)
			return failure();

		if (isQuantInt) {
			// Body: extsi[·subi]·muli·[subi]·addi quant-conv chain.
			int64_t inZp = 0, wZp = 0;
			if (!matchQuantConvBody(op.getRegion().front(), &inZp, &wZp))
				return failure();
			if (inZp != 0 || wZp != 0)
				return failure();
		} else {
			// Body: mulf · addf (pure fp contraction).
			Block &body = op.getRegion().front();
			auto *yield = body.getTerminator();
			if (!yield || yield->getNumOperands() != 1)
				return failure();
			auto *addf = yield->getOperand(0).getDefiningOp();
			if (!isa_and_nonnull<arith::AddFOp>(addf))
				return failure();
			bool sawMul = false;
			for (Value v : addf->getOperands()) {
				if (auto *m = v.getDefiningOp(); m && isa<arith::MulFOp>(m)) {
					sawMul = true;
					break;
				}
			}
			if (!sawMul)
				return failure();
		}

		// Recover pad amount from producer tensor.pad (if any).
		SmallVector<int32_t, 4> padAmount;
		input = recoverPadFromProducer(input, padAmount);

		// Look upstream for a dequant generic feeding either the conv's
		// input or weight in the SAME block. If found, extract scale/zp so
		// we can wrap qnn.conv2d's input/weight tensor types with
		// quant.uniform — SerializeGraph then emits qk=1 with real per-
		// tensor quant params instead of the placeholder qk=1 fallback.
		//
		// Per-variant scope means we don't fight DispatchCreation
		// attribute-stripping: the dequant generic, conv generic, and
		// rescale generic all live in the same hal.executable.variant
		// body when this pattern fires.
		// Search the conv's PARENT op (function or variant body) for any
		// linalg.generic dequantize whose INPUT (i8 storage) is the same SSA
		// value as the conv's input or weight. This handles the case where
		// the dequant lives elsewhere in the same function but operates on
		// the same i8 buffer that the integer conv consumes.
		auto findDequantForI8 = [&](Value i8Val) -> std::pair<double, int64_t> {
			Operation *parent = op->getParentOp();
			if (!parent)
				return {0.0, 0};
			double bestScale = 0.0;
			int64_t bestZp = 0;
			parent->walk([&](linalg::GenericOp dq) {
				if (bestScale != 0.0)
					return; // first match wins
				if (!linalg::isElementwise(dq) || dq.getNumDpsInputs() != 1 ||
					dq.getNumDpsInits() != 1)
					return;
				auto dqOutTy =
					dyn_cast<RankedTensorType>(dq.getResult(0).getType());
				auto dqInTy =
					dyn_cast<RankedTensorType>(dq.getDpsInputs()[0].getType());
				if (!dqOutTy || !dqInTy)
					return;
				if (!dqOutTy.getElementType().isF32() ||
					!dqInTy.getElementType().isInteger(8))
					return;
				if (dq.getDpsInputs()[0] != i8Val)
					return;
				double s = 1.0;
				int64_t z = 0;
				if (matchDequantBody(dq.getRegion().front(), &s, &z)) {
					bestScale = s;
					bestZp = z;
				}
			});
			return {bestScale, bestZp};
		};
		auto [inScale, inZpTy] = findDequantForI8(input);
		auto [wScale, wZpTy] = findDequantForI8(weight);
		(void)inZpTy;
		(void)wZpTy;

		// QNN's Conv2d is rank-4 NHWC. Post-global-opt may have collapsed
		// the unit batch dim, leaving rank-3 operands. Most post-layout convs
		// are already HWC, but yolov8's dispatch-local generic can be
		// mixed-layout: input CHW, weight HWIO, output HWC. Detect that from
		// the input indexing map and transpose only the input before adding the
		// synthetic batch.
		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		auto transposeRanked = [&](Value v, ArrayRef<int64_t> perm) -> Value {
			auto t = cast<RankedTensorType>(v.getType());
			SmallVector<int64_t> dstShape;
			for (int64_t p : perm)
				dstShape.push_back(t.getShape()[p]);
			Value empty = tensor::EmptyOp::create(
				rewriter, loc, dstShape, t.getElementType());
			return linalg::TransposeOp::create(rewriter, loc, v, empty, perm)
				.getResult()[0];
		};
		bool inputIsRank3ChannelFirst = false;
		{
			auto inTyDyn = dyn_cast<RankedTensorType>(input.getType());
			auto inMap = op.getIndexingMapsArray()[0];
			if (inTyDyn && inTyDyn.getRank() == 3 &&
				inMap.getNumResults() == 3 && !dims.inputChannel.empty()) {
				if (auto first = dyn_cast<AffineDimExpr>(inMap.getResult(0))) {
					inputIsRank3ChannelFirst = first.getPosition() ==
						static_cast<unsigned>(dims.inputChannel.front());
				}
			}
		}
		// Only apply the CHW→HWC transpose when this is a quant-int conv with
		// the yolov8 NCHW idiom. For fp32 conv the existing NCHW path handles
		// the transposes via the isNchw branch below.
		if (inputIsRank3ChannelFirst && isQuantInt) {
			// [C, H, W] -> [H, W, C]. Weight/output stay HWIO/HWC.
			input = transposeRanked(input, {1, 2, 0});
		} else {
			inputIsRank3ChannelFirst = false;
		}
		auto expandRank3to4 = [&](Value v) -> Value {
			auto t = cast<RankedTensorType>(v.getType());
			if (t.getRank() == 4)
				return v;
			if (t.getRank() != 3)
				return Value();
			SmallVector<int64_t, 4> newShape{
				1, t.getShape()[0], t.getShape()[1], t.getShape()[2]};
			auto newTy = RankedTensorType::get(newShape, t.getElementType());
			SmallVector<ReassociationIndices, 3> reassoc{{0, 1}, {2}, {3}};
			return tensor::ExpandShapeOp::create(
				rewriter, loc, newTy, v, reassoc);
		};
		Value input4 = expandRank3to4(input);
		if (!input4)
			return failure();
		Value weight4 =
			weight; // weight already HWIO rank-4 (or FCHW for NCHW).

		// Phase 4a: NCHW path. Insert NCHW->NHWC on input + FCHW->HWCF on
		// weight + remember to convert the output back. The body region of
		// the source generic operates in iter-space; the new qnn.conv2d
		// operates in NHWC tensor space, so the wrapping transposes preserve
		// semantics.
		if (isNchw) {
			// input4: [N, C, H, W] -> [N, H, W, C]. For collapsed rank-3
			// dispatch operands we may already have converted [C,H,W] ->
			// [H,W,C] above before adding the synthetic batch dim; do not
			// transpose twice.
			if (!inputIsRank3ChannelFirst)
				input4 = transposeRanked(input4, {0, 2, 3, 1});
			// weight: [OC, IC, KH, KW] -> [KH, KW, IC, OC]
			weight4 = transposeRanked(weight4, {2, 3, 1, 0});
		}

		// NEXT-SESSION: with the upstream-dequant scales (inScale, wScale)
		// now extracted, wrap input4 / weight4 tensor types with
		// quant.uniform<i8:f32, scale, 0>. The previous attempt via
		// UnrealizedConversionCast caused infinite loops in the greedy
		// pattern driver (~30min hang on yolov8 compile). The likely
		// safer approach is one of:
		//   (a) post-process the qnn.conv2d AFTER the greedy driver
		//       finishes, walking ops + manually rewriting their input
		//       Value types via Operation::setOperand on a fresh cast
		//   (b) use rewriter.startRootUpdate / finalizeRootUpdate around
		//       a manual Operation::moveBefore + setOperand sequence to
		//       avoid pattern-re-firing on the cast's result
		//   (c) emit the scale/zp as op-attrs on qnn.conv2d itself
		//       (`quant_input_scale`, `quant_weight_scale`,
		//       `quant_output_scale` etc.) and have SerializeGraph read
		//       them — bypasses the type-rewriting entirely.
		// (c) is the simplest, least invasive — recommend trying first.
		(void)inScale;
		(void)wScale;
		auto outTy4 = outTy;
		bool collapseOut = false;
		if (outTy.getRank() == 3) {
			outTy4 = RankedTensorType::get(
				{1, outTy.getShape()[0], outTy.getShape()[1],
					outTy.getShape()[2]},
				outTy.getElementType());
			collapseOut = true;
		}
		// Phase 4a: when NCHW, the rank-4 outTy is [N, OC, oH, oW]. The
		// qnn.conv2d operates in NHWC space, so its output type is
		// [N, oH, oW, OC]. We'll transpose it back to NCHW after the op.
		SmallVector<int64_t> outTy4NchwShape;
		if (isNchw) {
			outTy4NchwShape = SmallVector<int64_t>(
				outTy4.getShape().begin(), outTy4.getShape().end());
			outTy4 = RankedTensorType::get(
				{outTy4.getShape()[0], outTy4.getShape()[2],
					outTy4.getShape()[3], outTy4.getShape()[1]},
				outTy4.getElementType());
		}
		// Look for a single-use trailing rescale generic that converts the
		// i32 accumulator into the final i8 output. If found, fold it into
		// qnn.conv2d's output (i8 with quant.uniform encoding) and replace
		// BOTH ops with the conv result. This is what HTA expects: a
		// Conv2d that consumes/produces 8-bit quantized tensors with
		// explicit scales.
		linalg::GenericOp rescaleOp = nullptr;
		Value rescaleBiasOperand;
		double scale_acc = 1.0, scale_out = 1.0;
		int64_t out_zp = 0;
		bool rescaleIsDequantF32 = false;
		// SiLU-rescale path: yolov8's quantized SiLU activation block.
		// Same single-use trailing-rescale shape, but the body folds in
		// sigmoid + multiply. Lowered to qnn.conv2d (i8 output, sInt) ->
		// qnn.element_wise_neuron(SIGMOID) (i8, sSig) ->
		// qnn.element_wise_binary (MUL) (i8, sFin).
		bool rescaleIsSiLU = false;
		double silu_sSig = 1.0, silu_sFin = 1.0;
		if (op.getResult(0).hasOneUse()) {
			Operation *user = *op.getResult(0).getUsers().begin();
			if (auto rescale = dyn_cast<linalg::GenericOp>(user)) {
				if (rescale.getNumDpsInits() == 1 &&
					llvm::all_of(rescale.getIteratorTypesArray(),
						[](utils::IteratorType it) {
							return it == utils::IteratorType::parallel;
						})) {
					auto rOutTy =
						cast<RankedTensorType>(rescale.getResult(0).getType());
					if (rOutTy.getElementType().isInteger(8)) {
						bool hasSupportedInputs =
							rescale.getNumDpsInputs() == 1;
						Value maybeBias;
						if (!hasSupportedInputs &&
							rescale.getNumDpsInputs() == 2) {
							for (auto [idx, v] :
								llvm::enumerate(rescale.getDpsInputs())) {
								if (v == op.getResult(0))
									continue;
								auto biasTy =
									dyn_cast<RankedTensorType>(v.getType());
								auto biasMap =
									rescale.getIndexingMapsArray()[idx];
								if (!biasTy || biasTy.getRank() != 1 ||
									!biasTy.getElementType().isInteger(32) ||
									biasMap.getNumResults() != 1 ||
									dims.outputChannel.empty())
									continue;
								if (auto d = dyn_cast<AffineDimExpr>(
										biasMap.getResult(0))) {
									if (d.getPosition() ==
										static_cast<unsigned>(
											dims.outputChannel[0])) {
										maybeBias = v;
										hasSupportedInputs = true;
									}
								}
							}
						}
						if (hasSupportedInputs) {
							if (matchConvRescaleBody(
									rescale.getRegion().front(), &scale_acc,
									&scale_out, &out_zp)) {
								rescaleOp = rescale;
							} else if (matchSiLURescaleBody(
										   rescale.getRegion().front(),
										   &scale_acc, &scale_out, &silu_sSig,
										   &silu_sFin, &out_zp)) {
								rescaleOp = rescale;
								rescaleIsSiLU = true;
							}
							if (rescaleOp && maybeBias)
								rescaleBiasOperand = maybeBias;
						}
					}
				}
			}
		}

		Type convOutElemTy = rescaleOp
			? cast<RankedTensorType>(rescaleOp.getResult(0).getType())
				  .getElementType()
			: outTy.getElementType();
		if (rescaleOp) {
			// Wrap with quant.uniform<i8:f32, scale:zp> so SerializeGraph can
			// pick up the per-tensor quant params.
			auto i8Ty = b.getIntegerType(8, /*isSigned=*/true);
			auto f32Ty = b.getF32Type();
			convOutElemTy = quant::UniformQuantizedType::get(
				/*flags=*/quant::QuantizationFlags::Signed,
				/*storageType=*/i8Ty,
				/*expressedType=*/f32Ty, /*scale=*/scale_out,
				/*zeroPoint=*/out_zp,
				/*storageMin=*/-128, /*storageMax=*/127);
		}

		Type convOutTy4ElemSubst;
		if (rescaleOp) {
			auto rOutShape =
				cast<RankedTensorType>(rescaleOp.getResult(0).getType())
					.getShape();
			// outTy4 had outTy's element type — replace it.
			SmallVector<int64_t> shape4(
				outTy4.getShape().begin(), outTy4.getShape().end());
			// If rescale collapsed batch and out has rank 3, adjust.
			(void)rOutShape;
			outTy4 = RankedTensorType::get(shape4, convOutElemTy);
			// Final outTy used by collapse_shape is also re-typed if needed.
			if (collapseOut) {
				SmallVector<int64_t> shape3(
					outTy.getShape().begin(), outTy.getShape().end());
				outTy = RankedTensorType::get(shape3, convOutElemTy);
			}
		}
		(void)convOutTy4ElemSubst;

		// Detect bias-broadcast on outs: a linalg.generic that broadcasts a
		// 1D tensor along the conv's output-channel dim. yolov8 produces this
		// idiom — without lifting it, the broadcast generic gets DCE'd and we
		// silently drop the bias add.
		Value biasOperand = rescaleBiasOperand;
		{
			Value outsOp = op.getDpsInits()[0];
			if (!biasOperand) {
				if (auto bcast = outsOp.getDefiningOp<linalg::GenericOp>()) {
					if (bcast.getNumDpsInputs() == 1 &&
						bcast.getNumDpsInits() == 1 &&
						llvm::all_of(bcast.getIteratorTypesArray(),
							[](utils::IteratorType it) {
								return it == utils::IteratorType::parallel;
							})) {
						Block &bbody = bcast.getRegion().front();
						auto yield =
							dyn_cast<linalg::YieldOp>(bbody.getTerminator());
						bool isIdentity = yield &&
							yield.getNumOperands() == 1 &&
							yield.getOperand(0) == bbody.getArgument(0);
						if (isIdentity && !dims.outputChannel.empty()) {
							Value src = bcast.getDpsInputs()[0];
							auto srcTy =
								dyn_cast<RankedTensorType>(src.getType());
							auto srcMap = bcast.getIndexingMapsArray()[0];
							if (srcTy && srcTy.getRank() == 1 &&
								srcMap.getNumResults() == 1) {
								if (auto srcDim = dyn_cast<AffineDimExpr>(
										srcMap.getResult(0))) {
									if (srcDim.getPosition() ==
										static_cast<unsigned>(
											dims.outputChannel[0])) {
										biasOperand = src;
									}
								}
							}
						}
					}
				}
			}
		}

		auto qnnConv = Conv2dOp::create(rewriter, loc, outTy4, input4, weight4,
			/*bias=*/biasOperand, toI32Array(b, dims.strides),
			toI32Array(b, padAmount), toI32Array(b, dims.dilations),
			b.getI32IntegerAttr(/*group=*/1));
		// Stamp upstream-dequant scale/zp as op-attrs on the qnn.conv2d.
		// Two sources of scales:
		//   1. Per-variant findDequantForI8 walk above (only fires when
		//      dequant lives in the same variant body — rare for yolov8).
		//   2. Pre-dispatch-creation RewriteQDQToQuantUniform pass that
		//      stamped `merlin.qnn_*` attrs on the source linalg op. Copy
		//      them through here so they ride into the new qnn.conv2d.
		if (isQuantInt) {
			if (inScale != 0.0) {
				qnnConv->setAttr("merlin.qnn_input_scale",
					b.getF32FloatAttr(static_cast<float>(inScale)));
			} else if (auto a = op->getAttrOfType<FloatAttr>(
						   "merlin.qnn_input_scale")) {
				qnnConv->setAttr("merlin.qnn_input_scale", a);
				if (auto z = op->getAttrOfType<IntegerAttr>(
						"merlin.qnn_input_zero_point"))
					qnnConv->setAttr("merlin.qnn_input_zero_point", z);
			}
			if (wScale != 0.0) {
				qnnConv->setAttr("merlin.qnn_weight_scale",
					b.getF32FloatAttr(static_cast<float>(wScale)));
			} else if (auto a = op->getAttrOfType<FloatAttr>(
						   "merlin.qnn_weight_scale")) {
				qnnConv->setAttr("merlin.qnn_weight_scale", a);
				if (auto z = op->getAttrOfType<IntegerAttr>(
						"merlin.qnn_weight_zero_point"))
					qnnConv->setAttr("merlin.qnn_weight_zero_point", z);
			}
			if (rescaleOp) {
				qnnConv->setAttr("merlin.qnn_accumulator_scale",
					b.getF32FloatAttr(static_cast<float>(scale_acc)));
				qnnConv->setAttr("merlin.qnn_output_scale",
					b.getF32FloatAttr(static_cast<float>(scale_out)));
				qnnConv->setAttr("merlin.qnn_output_zero_point",
					b.getI32IntegerAttr(static_cast<int32_t>(out_zp)));
			} else if (auto a = op->getAttrOfType<FloatAttr>(
						   "merlin.qnn_output_scale")) {
				qnnConv->setAttr("merlin.qnn_output_scale", a);
				if (auto z = op->getAttrOfType<IntegerAttr>(
						"merlin.qnn_output_zero_point"))
					qnnConv->setAttr("merlin.qnn_output_zero_point", z);
			}
		}
		Value result = qnnConv.getOutput();
		// For SiLU, expand the conv output (i8 quant.uniform with sInt) into
		//   conv_i8 -> qnn.element_wise_neuron(SIGMOID, sSig)
		//   -> qnn.element_wise_binary(MUL, sFin)(conv_i8, sigmoid_i8)
		// before applying the final transpose/collapse. All ops live in NHWC
		// rank-4 i8-quant.uniform space.
		if (rescaleIsSiLU) {
			auto i8TyL = b.getIntegerType(8, /*isSigned=*/true);
			auto f32Ty = b.getF32Type();
			auto convShape =
				cast<RankedTensorType>(result.getType()).getShape();
			Type sigElt = quant::UniformQuantizedType::get(
				quant::QuantizationFlags::Signed, i8TyL, f32Ty,
				/*scale=*/silu_sSig, /*zeroPoint=*/0,
				/*storageMin=*/-128, /*storageMax=*/127);
			auto sigTy = RankedTensorType::get(convShape, sigElt);
			auto sigmoid = ElementWiseNeuronOp::create(rewriter, loc, sigTy,
				result, b.getI32IntegerAttr(/*SIGMOID*/ 6));
			Type mulElt = quant::UniformQuantizedType::get(
				quant::QuantizationFlags::Signed, i8TyL, f32Ty,
				/*scale=*/silu_sFin, /*zeroPoint=*/out_zp,
				/*storageMin=*/-128, /*storageMax=*/127);
			auto mulTy = RankedTensorType::get(convShape, mulElt);
			auto mul = ElementWiseBinaryOp::create(rewriter, loc, mulTy, result,
				sigmoid.getOutput(), b.getI32IntegerAttr(kBinaryMul));
			result = mul.getOutput();
		}
		// For the rescaleIsDequantF32 path the conv output is a
		// quant.uniform-typed tensor — linalg.transpose / tensor.collapse
		// don't accept those element types directly. Defer the post-NCHW /
		// collapse-back operations until AFTER dequantize so they run on
		// plain f32 tensors. Handled below in the rescaleOp branch.
		if (!rescaleIsDequantF32) {
			// Use `result`'s actual element type rather than the pre-computed
			// outTy. For the SiLU path the chain's final element type is
			// quant.uniform<i8:f32, sFin:zp>, which differs from the conv's
			// quant.uniform<i8:f32, sInt:zp>; the rescale-fold path is
			// unchanged.
			Type resultElt =
				cast<RankedTensorType>(result.getType()).getElementType();
			if (isNchw) {
				// [N, oH, oW, OC] -> [N, OC, oH, oW]
				Value empty = tensor::EmptyOp::create(
					rewriter, loc, outTy4NchwShape, resultElt);
				SmallVector<int64_t, 4> backPerm{0, 3, 1, 2};
				result = linalg::TransposeOp::create(
					rewriter, loc, result, empty, backPerm)
							 .getResult()[0];
			}
			if (collapseOut) {
				SmallVector<ReassociationIndices, 3> reassoc{{0, 1}, {2}, {3}};
				auto outShape = cast<RankedTensorType>(outTy).getShape();
				auto collapsedTy = RankedTensorType::get(outShape, resultElt);
				result = tensor::CollapseShapeOp::create(
					rewriter, loc, collapsedTy, result, reassoc);
			}
		}
		// The folded rescale generic may permute the chain output via its
		// output indexing map (yolov8 emits HWC→CHW on the conv tail). When
		// rescaleOp is set and its output map is a non-identity permutation
		// of iter dims, materialize a linalg.transpose so `result`'s shape
		// matches the rescale's output type.
		if (rescaleOp) {
			auto rOutMap = rescaleOp.getIndexingMapsArray().back();
			SmallVector<int64_t> perm;
			bool identity = true;
			bool isPermutation = true;
			for (unsigned i = 0; i < rOutMap.getNumResults(); ++i) {
				if (auto d = dyn_cast<AffineDimExpr>(rOutMap.getResult(i))) {
					perm.push_back(static_cast<int64_t>(d.getPosition()));
					if (d.getPosition() != i)
						identity = false;
				} else {
					isPermutation = false;
					break;
				}
			}
			if (isPermutation && !identity &&
				perm.size() ==
					cast<RankedTensorType>(result.getType()).getRank()) {
				auto curTy = cast<RankedTensorType>(result.getType());
				SmallVector<int64_t> newShape;
				for (int64_t p : perm)
					newShape.push_back(curTy.getShape()[p]);
				Value empty = tensor::EmptyOp::create(
					rewriter, loc, newShape, curTy.getElementType());
				result = linalg::TransposeOp::create(
					rewriter, loc, result, empty, perm)
							 .getResult()[0];
			}
		}
		if (rescaleOp) {
			rewriter.replaceOp(rescaleOp, result);
			rewriter.eraseOp(op);
		} else {
			rewriter.replaceOp(op, result);
		}
		(void)rescaleIsDequantF32;
		return success();
	}
};

//===----------------------------------------------------------------------===//
// FP Conv2d — Adreno GPU path (fp16/fp32 NHWC). No quant params, no
// rescale chain. Anchors on linalg.conv_2d_nhwc_hwcf (without _q).
//===----------------------------------------------------------------------===//

struct LowerConv2dFp : OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
	using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::Conv2DNhwcHwcfOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		Value input = op.getDpsInputs()[0];
		Value weight = op.getDpsInputs()[1];

		auto inTy = cast<RankedTensorType>(input.getType());
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		Type eltTy = inTy.getElementType();
		if (!eltTy.isF16() && !eltTy.isF32())
			return failure();
		if (outTy.getElementType() != eltTy)
			return failure();

		// Pad recovery (rare for fp convs but harmless to check).
		SmallVector<int32_t, 4> padAmount;
		input = recoverPadFromProducer(input, padAmount);

		// Stride / dilation as i32 arrays directly from the named-op attrs.
		auto strides = op.getStrides();
		auto dilations = op.getDilations();
		SmallVector<int32_t, 2> strideArr{
			static_cast<int32_t>(strides.getValues<int64_t>()[0]),
			static_cast<int32_t>(strides.getValues<int64_t>()[1])};
		SmallVector<int32_t, 2> dilArr{
			static_cast<int32_t>(dilations.getValues<int64_t>()[0]),
			static_cast<int32_t>(dilations.getValues<int64_t>()[1])};

		Builder b(rewriter.getContext());
		auto qnnConv = Conv2dOp::create(rewriter, op.getLoc(), outTy, input,
			weight, /*bias=*/Value{}, b.getI32ArrayAttr(strideArr),
			b.getI32ArrayAttr(padAmount), b.getI32ArrayAttr(dilArr),
			b.getI32IntegerAttr(/*group=*/1));
		rewriter.replaceOp(op, qnnConv.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 2C — DepthwiseConv2d
//===----------------------------------------------------------------------===//

struct LowerDepthwiseConv2dQGeneric : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isaConvolutionOpInterface(op,
				/*allowEmptyConvolvedDims=*/false))
			return failure();
		auto dimsOr = linalg::inferConvolutionDims(op);
		if (failed(dimsOr))
			return failure();
		auto &dims = *dimsOr;

		// DepthwiseConv2d: depth dims non-empty.
		if (dims.depth.empty())
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();

		// Same NHWC-layout guard as Conv2d: bail to placeholder if the channel
		// (depth) dim isn't last.
		if (!dims.depth.empty() && !dims.outputImage.empty()) {
			int64_t lastImage = *std::max_element(
				dims.outputImage.begin(), dims.outputImage.end());
			if (dims.depth[0] < lastImage) {
				return rewriter.notifyMatchFailure(
					op, "qnn.depthwise_conv2d requires NHWC layout");
			}
		}
		Value input = op.getDpsInputs()[0];
		Value weight = op.getDpsInputs()[1];

		auto inTy = cast<RankedTensorType>(input.getType());
		auto wTy = cast<RankedTensorType>(weight.getType());
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!inTy.getElementType().isInteger(8))
			return failure();
		if (!wTy.getElementType().isInteger(8))
			return failure();
		if (!outTy.getElementType().isInteger(32))
			return failure();

		int64_t inZp = 0, wZp = 0;
		if (!matchQuantConvBody(op.getRegion().front(), &inZp, &wZp))
			return failure();
		if (inZp != 0 || wZp != 0)
			return failure();

		SmallVector<int32_t, 4> padAmount;
		input = recoverPadFromProducer(input, padAmount);

		Builder b(rewriter.getContext());
		auto qnnDw = DepthwiseConv2dOp::create(rewriter, op.getLoc(), outTy,
			input, weight, /*bias=*/Value{}, toI32Array(b, dims.strides),
			toI32Array(b, padAmount), toI32Array(b, dims.dilations));
		rewriter.replaceOp(op, qnnDw.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 2C — Pool {Max, Avg}
//===----------------------------------------------------------------------===//

// Named-op pooling (which IREE retains as `linalg.pooling_nhwc_max` /
// `linalg.pooling_nhwc_sum`). Generic-form pooling falls under
// isaConvolutionOpInterface(allowEmptyConvolvedDims=true) but the named
// form is what survives global-opt for our fixtures.
struct LowerNhwcMaxPool : OpRewritePattern<linalg::PoolingNhwcMaxOp> {
	using OpRewritePattern<linalg::PoolingNhwcMaxOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::PoolingNhwcMaxOp op, PatternRewriter &rewriter) const override {
		Value input = op.getInputs()[0];
		Value window = op.getInputs()[1];
		auto winType = cast<RankedTensorType>(window.getType());
		if (winType.getRank() != 2)
			return failure();

		SmallVector<int32_t, 2> filterSize{
			static_cast<int32_t>(winType.getShape()[0]),
			static_cast<int32_t>(winType.getShape()[1])};
		auto strides = op.getStrides();
		SmallVector<int32_t, 2> strideArr{
			static_cast<int32_t>(strides.getValues<int64_t>()[0]),
			static_cast<int32_t>(strides.getValues<int64_t>()[1])};

		SmallVector<int32_t, 4> padAmount;
		input = recoverPadFromProducer(input, padAmount);

		Builder b(rewriter.getContext());
		auto qnnPool = PoolMax2dOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), input, b.getI32ArrayAttr(filterSize),
			b.getI32ArrayAttr(strideArr), b.getI32ArrayAttr(padAmount));
		rewriter.replaceOp(op, qnnPool.getOutput());
		return success();
	}
};

// MaxPool from a generalized linalg.generic — yolov8 SPPF emits 2D
// max-pool as a 5-iter generic (3 parallel + 2 reduction) with body
// `maximumf(out, in)` and a rank-2 dummy "kernel-shape" tensor for the
// reduction iter bounds. Lowered to qnn.pool_max2d (NHWC) wrapped with
// rank-3 ↔ rank-4 expand/collapse + CHW↔HWC transposes.
struct LowerMaxPoolGeneric : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy || !outTy.getElementType().isF32())
			return failure();
		auto in0Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto in1Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
		if (!in0Ty || !in1Ty || !in0Ty.getElementType().isF32())
			return failure();
		// Real input rank 3 (CHW), kernel-shape input rank 2 (KhxKw).
		if (in0Ty.getRank() != 3 || in1Ty.getRank() != 2)
			return failure();
		if (outTy.getRank() != 3)
			return failure();

		auto iters = op.getIteratorTypesArray();
		if (iters.size() != 5)
			return failure();
		SmallVector<unsigned> parIters, redIters;
		for (size_t i = 0; i < iters.size(); ++i) {
			if (iters[i] == utils::IteratorType::parallel)
				parIters.push_back(static_cast<unsigned>(i));
			else if (iters[i] == utils::IteratorType::reduction)
				redIters.push_back(static_cast<unsigned>(i));
			else
				return failure();
		}
		if (parIters.size() != 3 || redIters.size() != 2)
			return failure();

		// Body: yield(maximumf(out, in)) — both args are block args.
		Block &body = op.getRegion().front();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		auto maxOp = yield.getOperand(0).getDefiningOp<arith::MaximumFOp>();
		if (!maxOp)
			return failure();
		if (!isa<BlockArgument>(maxOp.getLhs()) ||
			!isa<BlockArgument>(maxOp.getRhs()))
			return failure();

		auto maps = op.getIndexingMapsArray();
		AffineMap inMap = maps[0];
		AffineMap kMap = maps[1];
		AffineMap outMap = maps[2];

		// Kernel-shape input: 2D, map results = the reduction iter positions.
		if (kMap.getNumResults() != 2)
			return failure();
		SmallVector<unsigned> kIterPos;
		for (unsigned i = 0; i < 2; ++i) {
			auto d = dyn_cast<AffineDimExpr>(kMap.getResult(i));
			if (!d)
				return failure();
			kIterPos.push_back(d.getPosition());
		}
		// Both must be reduction iters.
		for (unsigned p : kIterPos) {
			if (std::find(redIters.begin(), redIters.end(), p) ==
				redIters.end())
				return failure();
		}

		// Output: rank 3, each result = a distinct AffineDimExpr that's a
		// parallel iter dim.
		SmallVector<unsigned> outIterPos;
		for (unsigned i = 0; i < 3; ++i) {
			auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
			if (!d)
				return failure();
			outIterPos.push_back(d.getPosition());
		}
		// First output dim = channel iter (parallel). Second/third = spatial.
		unsigned cIter = outIterPos[0];
		unsigned ohIter = outIterPos[1];
		unsigned owIter = outIterPos[2];
		unsigned khIter = kIterPos[0];
		unsigned kwIter = kIterPos[1];

		// Input map: should be (cIter, stride_h*ohIter + khIter,
		// stride_w*owIter + kwIter). We support stride 1 only for now (yolov8
		// SPPF uses stride=1).
		if (inMap.getNumResults() != 3)
			return failure();
		auto inMap0 = dyn_cast<AffineDimExpr>(inMap.getResult(0));
		if (!inMap0 || inMap0.getPosition() != cIter)
			return failure();
		auto parseSpatial = [&](AffineExpr e, unsigned outIter,
								unsigned redIter, int *strideOut) -> bool {
			// Accept either a pure (outIter + redIter) or a scaled (s*outIter +
			// redIter).
			auto add = dyn_cast<AffineBinaryOpExpr>(e);
			if (!add || add.getKind() != AffineExprKind::Add)
				return false;
			AffineExpr lhs = add.getLHS(), rhs = add.getRHS();
			// Try lhs = ohIter (or s*ohIter), rhs = khIter; or swapped.
			auto matchSide = [&](AffineExpr side, unsigned iter,
								 int *s) -> bool {
				if (auto d = dyn_cast<AffineDimExpr>(side)) {
					if (d.getPosition() == iter) {
						*s = 1;
						return true;
					}
				}
				if (auto mul = dyn_cast<AffineBinaryOpExpr>(side);
					mul && mul.getKind() == AffineExprKind::Mul) {
					auto d = dyn_cast<AffineDimExpr>(mul.getLHS());
					auto c = dyn_cast<AffineConstantExpr>(mul.getRHS());
					if (!d || !c) {
						d = dyn_cast<AffineDimExpr>(mul.getRHS());
						c = dyn_cast<AffineConstantExpr>(mul.getLHS());
					}
					if (d && c && d.getPosition() == iter) {
						*s = static_cast<int>(c.getValue());
						return true;
					}
				}
				return false;
			};
			int s = 0;
			if (matchSide(lhs, outIter, &s)) {
				auto rd = dyn_cast<AffineDimExpr>(rhs);
				if (rd && rd.getPosition() == redIter) {
					*strideOut = s;
					return true;
				}
			}
			if (matchSide(rhs, outIter, &s)) {
				auto rd = dyn_cast<AffineDimExpr>(lhs);
				if (rd && rd.getPosition() == redIter) {
					*strideOut = s;
					return true;
				}
			}
			return false;
		};
		int strideH = 0, strideW = 0;
		if (!parseSpatial(inMap.getResult(1), ohIter, khIter, &strideH))
			return failure();
		if (!parseSpatial(inMap.getResult(2), owIter, kwIter, &strideW))
			return failure();

		// Kernel sizes from in1Ty's shape.
		int kh = static_cast<int>(in1Ty.getShape()[0]);
		int kw = static_cast<int>(in1Ty.getShape()[1]);

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		Type f32Ty = b.getF32Type();

		// Wrap rank-3 CHW input → rank-4 NHWC.
		// CHW <C, H, W> → NCHW <1, C, H, W> → NHWC <1, H, W, C>.
		auto inShape = in0Ty.getShape();
		SmallVector<int64_t, 4> nchwShape{
			1, inShape[0], inShape[1], inShape[2]};
		SmallVector<ReassociationIndices, 3> expandReassoc{{0, 1}, {2}, {3}};
		auto nchwTy = RankedTensorType::get(nchwShape, f32Ty);
		Value nchw = tensor::ExpandShapeOp::create(
			rewriter, loc, nchwTy, op.getDpsInputs()[0], expandReassoc);
		SmallVector<int64_t, 4> nhwcShape{
			1, inShape[1], inShape[2], inShape[0]};
		auto nhwcTy = RankedTensorType::get(nhwcShape, f32Ty);
		Value nhwcEmpty =
			tensor::EmptyOp::create(rewriter, loc, nhwcShape, f32Ty);
		Value nhwc = linalg::TransposeOp::create(
			rewriter, loc, nchw, nhwcEmpty, SmallVector<int64_t, 4>{0, 2, 3, 1})
						 .getResult()[0];

		// qnn.pool_max2d output: (N, oH, oW, C) — NHWC.
		auto outShape = outTy.getShape();
		SmallVector<int64_t, 4> poolOutShape{
			1, outShape[1], outShape[2], outShape[0]};
		auto poolOutTy = RankedTensorType::get(poolOutShape, f32Ty);
		auto qnnPool = PoolMax2dOp::create(rewriter, loc, poolOutTy, nhwc,
			/*filter_size=*/b.getI32ArrayAttr({kh, kw}),
			/*stride=*/b.getI32ArrayAttr({strideH, strideW}),
			/*pad_amount=*/b.getI32ArrayAttr({0, 0, 0, 0}),
			/*rounding_mode=*/b.getI32IntegerAttr(0));

		// NHWC → NCHW → CHW.
		SmallVector<int64_t, 4> nchwOutShape{
			1, outShape[0], outShape[1], outShape[2]};
		Value nchwBackEmpty =
			tensor::EmptyOp::create(rewriter, loc, nchwOutShape, f32Ty);
		Value nchwBack =
			linalg::TransposeOp::create(rewriter, loc, qnnPool.getOutput(),
				nchwBackEmpty, SmallVector<int64_t, 4>{0, 3, 1, 2})
				.getResult()[0];
		SmallVector<ReassociationIndices, 3> collapseReassoc{{0, 1}, {2}, {3}};
		Value chwBack = tensor::CollapseShapeOp::create(
			rewriter, loc, outTy, nchwBack, collapseReassoc);
		rewriter.replaceOp(op, chwBack);
		return success();
	}
};

struct LowerNhwcSumPool : OpRewritePattern<linalg::PoolingNhwcSumOp> {
	using OpRewritePattern<linalg::PoolingNhwcSumOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::PoolingNhwcSumOp op, PatternRewriter &rewriter) const override {
		Value input = op.getInputs()[0];
		Value window = op.getInputs()[1];
		auto winType = cast<RankedTensorType>(window.getType());
		if (winType.getRank() != 2)
			return failure();

		SmallVector<int32_t, 2> filterSize{
			static_cast<int32_t>(winType.getShape()[0]),
			static_cast<int32_t>(winType.getShape()[1])};
		auto strides = op.getStrides();
		SmallVector<int32_t, 2> strideArr{
			static_cast<int32_t>(strides.getValues<int64_t>()[0]),
			static_cast<int32_t>(strides.getValues<int64_t>()[1])};

		SmallVector<int32_t, 4> padAmount;
		input = recoverPadFromProducer(input, padAmount);

		Builder b(rewriter.getContext());
		auto qnnPool = PoolAvg2dOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), input, b.getI32ArrayAttr(filterSize),
			b.getI32ArrayAttr(strideArr), b.getI32ArrayAttr(padAmount));
		rewriter.replaceOp(op, qnnPool.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 2C — MatMul / FullyConnected
//===----------------------------------------------------------------------===//

struct LowerMatMul : OpRewritePattern<linalg::MatmulOp> {
	using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::MatmulOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2)
			return failure();
		Value lhs = op.getDpsInputs()[0];
		Value rhs = op.getDpsInputs()[1];
		Builder b(rewriter.getContext());
		auto qnnMM =
			MatMulOp::create(rewriter, op.getLoc(), op.getResult(0).getType(),
				lhs, rhs, b.getBoolAttr(false), b.getBoolAttr(false));
		rewriter.replaceOp(op, qnnMM.getOutput());
		return success();
	}
};

struct LowerMatMulTransposeB : OpRewritePattern<linalg::MatmulTransposeBOp> {
	using OpRewritePattern<linalg::MatmulTransposeBOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(linalg::MatmulTransposeBOp op,
		PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2)
			return failure();
		Value lhs = op.getDpsInputs()[0];
		Value rhs = op.getDpsInputs()[1];
		Builder b(rewriter.getContext());
		// matmul_transpose_b == FullyConnected (weight is [out, in]).
		auto qnnFC = FullyConnectedOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), lhs, rhs,
			/*bias=*/Value{}, b.getBoolAttr(false));
		rewriter.replaceOp(op, qnnFC.getOutput());
		return success();
	}
};

// Generic-form contraction: a `linalg.generic` that satisfies
// isaContractionOpInterface with 2 parallel + 1 reduction dim. Maps to
// qnn.matmul; the transpose flags follow inferContractionDims's m/n/k
// positions.
struct LowerContractionGeneric : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isaContractionOpInterface(op))
			return failure();
		auto dimsOr = linalg::inferContractionDims(op);
		if (failed(dimsOr))
			return failure();
		auto &dims = *dimsOr;
		// Plain 2D matmul only for now: 1 M, 1 N, 1 K, no batch.
		if (dims.batch.size() != 0 || dims.m.size() != 1 ||
			dims.n.size() != 1 || dims.k.size() != 1)
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		Value lhs = op.getDpsInputs()[0];
		Value rhs = op.getDpsInputs()[1];
		auto lhsTyDyn = dyn_cast<RankedTensorType>(lhs.getType());
		auto outTyDyn = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!lhsTyDyn || !outTyDyn)
			return failure();
		Type inElt = lhsTyDyn.getElementType();
		Type outElt = outTyDyn.getElementType();
		bool isQuantIntMM = inElt.isInteger(8) && outElt.isInteger(32);
		bool isFpMM = (inElt.isF16() || inElt.isF32()) && outElt == inElt;
		if (!isQuantIntMM && !isFpMM)
			return failure();
		if (isQuantIntMM) {
			int64_t lhsZp = 0, rhsZp = 0;
			if (!matchQuantConvBody(op.getRegion().front(), &lhsZp, &rhsZp))
				return failure();
			if (lhsZp != 0 || rhsZp != 0)
				return failure();
		} else {
			Block &body = op.getRegion().front();
			auto *yield = body.getTerminator();
			if (!yield || yield->getNumOperands() != 1)
				return failure();
			auto *addf = yield->getOperand(0).getDefiningOp();
			if (!isa_and_nonnull<arith::AddFOp>(addf))
				return failure();
			bool sawMul = false;
			for (Value v : addf->getOperands()) {
				if (auto *m = v.getDefiningOp(); m && isa<arith::MulFOp>(m)) {
					sawMul = true;
					break;
				}
			}
			if (!sawMul)
				return failure();
		}

		// Determine transpose flags from the lhs/rhs indexing maps relative
		// to (m, n, k) positions. lhs is "transposed" if its first map dim
		// is k rather than m; rhs is transposed if its first dim is n
		// rather than k.
		auto maps = op.getIndexingMapsArray();
		AffineMap lhsMap = maps[0];
		AffineMap rhsMap = maps[1];
		auto firstDim = [](AffineMap m) -> unsigned {
			if (m.getNumResults() == 0)
				return ~0u;
			auto d = dyn_cast<AffineDimExpr>(m.getResult(0));
			return d ? d.getPosition() : ~0u;
		};
		bool transposeLhs = firstDim(lhsMap) == dims.k.front();
		bool transposeRhs = firstDim(rhsMap) == dims.n.front();

		Builder b(rewriter.getContext());
		auto qnnMM = MatMulOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), lhs, rhs, b.getBoolAttr(transposeLhs),
			b.getBoolAttr(transposeRhs));
		// Forward `merlin.qnn_*` attrs from the source generic (stamped by
		// RewriteQDQToQuantUniform) onto the qnn.matmul. SerializeGraph
		// reads them per-op into the input/weight/output tensor records.
		for (StringRef name :
			{"merlin.qnn_input_scale", "merlin.qnn_input_zero_point",
				"merlin.qnn_weight_scale", "merlin.qnn_weight_zero_point",
				"merlin.qnn_output_scale", "merlin.qnn_output_zero_point"}) {
			if (auto a = op->getAttr(name))
				qnnMM->setAttr(name, a);
		}
		rewriter.replaceOp(op, qnnMM.getOutput());
		return success();
	}
};

// Multi-N matmul-like: a contraction with 1 M, 1 K, and ≥1 N. yolov8 fp32
// emits this for 1×1 convs on CHW activations:
//   out[m, n1, n2] = sum_k(weight[m, k] * input[k, n1, n2])
// One operand has only {m, k} dims (the weight); the other has {k} + {n…}
// dims (the activation). We flatten the activation's N dims to a single
// axis, emit qnn.matmul with (weight=M×K) × (activation_flat=K×N), then
// reshape the result back to the original N layout.
struct LowerSpatialMatmulFp : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy)
			return failure();
		auto in0Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto in1Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
		if (!in0Ty || !in1Ty)
			return failure();
		Type inElt = in0Ty.getElementType();
		Type outElt = outTy.getElementType();
		// Accept either:
		//  (a) fp32×fp32 → fp32  (original)
		//  (b) i8×i8 → i32       (quant-conv with zero zp; matchQuantConvBody)
		bool isFp =
			inElt.isF32() && in1Ty.getElementType().isF32() && outElt.isF32();
		bool isQuantInt = inElt.isInteger(8) &&
			in1Ty.getElementType().isInteger(8) && outElt.isInteger(32);
		if (!isFp && !isQuantInt)
			return failure();
		// Need exactly one reduction iterator and ≥3 parallel ones.
		auto iters = op.getIteratorTypesArray();
		SmallVector<unsigned> reductionIters, parallelIters;
		for (size_t i = 0; i < iters.size(); ++i) {
			if (iters[i] == utils::IteratorType::reduction)
				reductionIters.push_back(static_cast<unsigned>(i));
			else if (iters[i] == utils::IteratorType::parallel)
				parallelIters.push_back(static_cast<unsigned>(i));
			else
				return failure();
		}
		if (reductionIters.size() != 1 || parallelIters.size() < 3)
			return failure();
		unsigned kIter = reductionIters[0];
		// Synthesize ContractionDimensions-like info from indexing maps:
		// - M = output dims that are NOT in N. (We'll derive N as output dims
		//   present in both inputs as parallel; M is output dim NOT present
		//   in the weight operand.)
		// For our matcher, we need exactly:
		//   weightOperand: rank-2, dims {M, K} (some order)
		//   activationOperand: rank ≥ 3, dims {K, N1, N2, ...}
		//   outputOperand:    dims {M, N1, N2, ...}

		// Body: depends on dtype.
		//  fp: yield <- addf(_, mulf(_, _))
		//  int8 quant: matches `matchQuantConvBody` (extsi+extsi+muli+addi)
		Block &body = op.getRegion().front();
		int64_t inZpQuant = 0, wZpQuant = 0;
		if (isFp) {
			auto *yield = body.getTerminator();
			if (!yield || yield->getNumOperands() != 1)
				return failure();
			auto *addf = yield->getOperand(0).getDefiningOp();
			if (!isa_and_nonnull<arith::AddFOp>(addf))
				return failure();
			bool sawMul = false;
			for (Value v : addf->getOperands()) {
				if (auto *m = v.getDefiningOp(); m && isa<arith::MulFOp>(m)) {
					sawMul = true;
					break;
				}
			}
			if (!sawMul)
				return failure();
		} else {
			if (!matchQuantConvBody(body, &inZpQuant, &wZpQuant))
				return failure();
		}

		auto maps = op.getIndexingMapsArray();
		AffineMap lhsMap = maps[0];
		AffineMap rhsMap = maps[1];
		AffineMap outMap = maps[2];

		// Helper: extract iter positions appearing in a map (in order).
		auto mapDims = [](AffineMap m) -> SmallVector<unsigned> {
			SmallVector<unsigned> out;
			for (unsigned i = 0; i < m.getNumResults(); ++i) {
				if (auto d = dyn_cast<AffineDimExpr>(m.getResult(i)))
					out.push_back(d.getPosition());
			}
			return out;
		};
		SmallVector<unsigned> lhsDims = mapDims(lhsMap);
		SmallVector<unsigned> rhsDims = mapDims(rhsMap);
		SmallVector<unsigned> outDims = mapDims(outMap);
		if (lhsDims.size() != lhsMap.getNumResults() ||
			rhsDims.size() != rhsMap.getNumResults() ||
			outDims.size() != outMap.getNumResults())
			return failure();

		auto containsK = [&](const SmallVector<unsigned> &d) {
			return std::find(d.begin(), d.end(), kIter) != d.end();
		};
		// Identify weight (rank-2, contains K, contains an M-like iter that's
		// also in outDims) vs activation (rank ≥ 3, contains K, no M).
		int weightIdx = -1, actIdx = -1;
		if (lhsDims.size() == 2 && rhsDims.size() >= 3) {
			weightIdx = 0;
			actIdx = 1;
		} else if (rhsDims.size() == 2 && lhsDims.size() >= 3) {
			weightIdx = 1;
			actIdx = 0;
		} else {
			return failure();
		}
		if (!containsK(lhsDims) || !containsK(rhsDims))
			return failure();
		SmallVector<unsigned> weightDims = (weightIdx == 0) ? lhsDims : rhsDims;
		SmallVector<unsigned> actDims = (actIdx == 0) ? lhsDims : rhsDims;
		// Weight has exactly 2 dims: K and one other = M iter.
		unsigned mIter =
			(weightDims[0] == kIter) ? weightDims[1] : weightDims[0];
		if (mIter == kIter)
			return failure();
		// M must appear in outDims at exactly position 0; rest of outDims = N.
		if (outDims.empty() || outDims[0] != mIter)
			return failure();
		// dims.n = outDims[1..end].
		SmallVector<unsigned> nIterOrder(outDims.begin() + 1, outDims.end());
		if (nIterOrder.size() < 2)
			return failure();
		// Activation should not contain M, must contain K + all N iters.
		for (unsigned ni : nIterOrder) {
			if (std::find(actDims.begin(), actDims.end(), ni) == actDims.end())
				return failure();
		}
		if (std::find(actDims.begin(), actDims.end(), mIter) != actDims.end())
			return failure();
		// Make a minimal "dims" struct for downstream code.
		struct DimsLike {
			SmallVector<unsigned> n;
			SmallVector<unsigned, 1> m, k;
		};
		DimsLike dims;
		dims.n = nIterOrder;
		dims.m.push_back(mIter);
		dims.k.push_back(kIter);
		Value weight = op.getDpsInputs()[weightIdx];
		Value activation = op.getDpsInputs()[actIdx];
		auto weightMap = maps[weightIdx];
		auto actMap = maps[actIdx];
		auto weightTy = cast<RankedTensorType>(weight.getType());
		auto actTy = cast<RankedTensorType>(activation.getType());
		if (!containsK(lhsDims) || !containsK(rhsDims))
			return failure();

		// Determine which physical axis of activation is K and which are N.
		int actKAxis = -1;
		SmallVector<int> actNAxes;
		for (unsigned i = 0; i < actMap.getNumResults(); ++i) {
			auto d = dyn_cast<AffineDimExpr>(actMap.getResult(i));
			if (!d)
				return failure();
			unsigned pos = d.getPosition();
			if (pos == kIter)
				actKAxis = static_cast<int>(i);
			else if (std::find(dims.n.begin(), dims.n.end(), pos) !=
				dims.n.end())
				actNAxes.push_back(static_cast<int>(i));
		}
		if (actKAxis < 0 || actNAxes.size() != dims.n.size())
			return failure();

		// Ensure activation N axes appear contiguously and in the same order
		// as outMap's N order; if not, bail (transpose would be needed).
		SmallVector<unsigned> actNIterAtAxis;
		for (int ax : actNAxes) {
			auto d = cast<AffineDimExpr>(actMap.getResult(ax));
			actNIterAtAxis.push_back(d.getPosition());
		}
		if (actNIterAtAxis != nIterOrder)
			return failure();

		// Build the flattened activation: collapse N axes into one. Two cases:
		//   K-first (actKAxis == 0): collapse axes [1..rank-1] → <K, Nflat>
		//   K-last  (actKAxis == rank-1): collapse [0..rank-2] → <Nflat, K>
		int actRank = actTy.getRank();
		bool actKFirst = (actKAxis == 0);
		bool actKLast = (actKAxis == actRank - 1);
		if (!actKFirst && !actKLast)
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		int64_t nFlat = 1;
		for (int ax : actNAxes)
			nFlat *= actTy.getShape()[ax];
		int64_t kSize = actTy.getShape()[actKAxis];

		SmallVector<ReassociationIndices, 2> actReassoc;
		SmallVector<int64_t, 2> actFlatShape;
		if (actKFirst) {
			ReassociationIndices kGroup = {0};
			ReassociationIndices nGroup;
			for (int i = 1; i < actRank; ++i)
				nGroup.push_back(i);
			actReassoc = {kGroup, nGroup};
			actFlatShape = {kSize, nFlat};
		} else {
			ReassociationIndices nGroup;
			for (int i = 0; i < actRank - 1; ++i)
				nGroup.push_back(i);
			ReassociationIndices kGroup = {actRank - 1};
			actReassoc = {nGroup, kGroup};
			actFlatShape = {nFlat, kSize};
		}
		auto actFlatTy =
			RankedTensorType::get(actFlatShape, actTy.getElementType());
		Value actFlat = tensor::CollapseShapeOp::create(
			rewriter, loc, actFlatTy, activation, actReassoc);

		// Weight layout — already 2D. Determine if M-first or K-first.
		bool weightMFirst;
		{
			auto first = cast<AffineDimExpr>(weightMap.getResult(0));
			weightMFirst = (first.getPosition() == mIter);
		}
		int64_t mSize = weightTy.getShape()[weightMFirst ? 0 : 1];

		// QNN matmul: lhs (M×K) × rhs (K×N) → (M×N). Transpose flags allow
		// K-first lhs (transposeLhs=true) and N-first rhs (transposeRhs=true).
		// Here we assign:
		//   lhs = weight, transposeLhs = !weightMFirst (true iff K-first)
		//   rhs = actFlat, transposeRhs = !actKFirst   (true iff N-first =
		//     K-last in input)
		bool transposeLhs = !weightMFirst;
		bool transposeRhs = !actKFirst;
		auto mmOutTy =
			RankedTensorType::get({mSize, nFlat}, outTy.getElementType());
		auto qnnMM = MatMulOp::create(rewriter, loc, mmOutTy, weight, actFlat,
			b.getBoolAttr(transposeLhs), b.getBoolAttr(transposeRhs));

		// Reshape (M, Nflat) back to outTy. outTy's first dim is M, the
		// rest are N dims in nIterOrder. Build expand_shape reassoc:
		//   group 0 -> {0} (M)
		//   group 1 -> {1, 2, ...} (N axes)
		SmallVector<ReassociationIndices, 2> outReassoc;
		ReassociationIndices mGroup = {0};
		ReassociationIndices nGroup;
		for (int i = 1; i < outTy.getRank(); ++i)
			nGroup.push_back(i);
		outReassoc = {mGroup, nGroup};
		Value result = tensor::ExpandShapeOp::create(
			rewriter, loc, outTy, qnnMM.getOutput(), outReassoc);
		rewriter.replaceOp(op, result);
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 2C — ElementWise binary (Add / Sub / Mul / Div)
//===----------------------------------------------------------------------===//

// Recognize a `linalg.generic` whose body is a single arith binary op
// over two parallel inputs and yield. Pure pointwise — no reductions.
struct LowerElementWiseBinary : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();

		Block &body = op.getRegion().front();
		auto *yield = body.getTerminator();
		if (!yield || yield->getNumOperands() != 1)
			return failure();
		auto *prod = yield->getOperand(0).getDefiningOp();
		if (!prod)
			return failure();

		int32_t kind = -1;
		if (isa<arith::AddIOp, arith::AddFOp>(prod))
			kind = kBinaryAdd;
		else if (isa<arith::SubIOp, arith::SubFOp>(prod))
			kind = kBinarySub;
		else if (isa<arith::MulIOp, arith::MulFOp>(prod))
			kind = kBinaryMul;
		else if (isa<arith::DivSIOp, arith::DivUIOp, arith::DivFOp>(prod))
			kind = kBinaryDiv;
		else
			return failure();

		// Both operands of the binary must come directly from block args.
		auto lhsArg = dyn_cast<BlockArgument>(prod->getOperand(0));
		auto rhsArg = dyn_cast<BlockArgument>(prod->getOperand(1));
		if (!lhsArg || !rhsArg)
			return failure();

		Builder b(rewriter.getContext());
		auto qnnBin = ElementWiseBinaryOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), op.getDpsInputs()[0],
			op.getDpsInputs()[1], b.getI32IntegerAttr(kind));
		rewriter.replaceOp(op, qnnBin.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 2C — ElementWiseNeuron (Relu / Relu6 / Sigmoid / Tanh)
//===----------------------------------------------------------------------===//

struct LowerElementWiseNeuron : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		// QNN's ElementWiseNeuron requires identical input/output dtype.
		// Skip when the body actually does a type cast (e.g. i32→f32 dequant
		// followed by Relu — a yolov8 conv rescale shape that the
		// LowerConv2dQGeneric path is responsible for fusing into the conv
		// output's quant.uniform encoding + a qnn.dequantize + qnn.relu).
		auto inTy = cast<RankedTensorType>(op.getDpsInputs()[0].getType())
						.getElementType();
		auto outTy =
			cast<RankedTensorType>(op.getResult(0).getType()).getElementType();
		if (inTy != outTy)
			return failure();
		ActivationKind kind;
		if (!matchActivationBody(op.getRegion().front(), &kind))
			return failure();

		Builder b(rewriter.getContext());
		auto qnnAct = ElementWiseNeuronOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), op.getDpsInputs()[0],
			b.getI32IntegerAttr(static_cast<int32_t>(kind)));
		rewriter.replaceOp(op, qnnAct.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 2C — Quantize / Dequantize (boundary ops)
//===----------------------------------------------------------------------===//

// Pad a rank-N tensor's shape to rank-4 by prepending leading 1s. Returns
// the SSA value at rank-4 plus the original rank for collapse-back. QNN
// finalize rejects rank<4 graphs with rc=6022/6020 (verified on QnnGpu);
// promotion lets a rank-1 elementwise post-globalopt reshape land cleanly
// as a rank-4 op the runtime accepts.
static std::pair<Value, int64_t> promoteToRank4(
	PatternRewriter &rw, Location loc, Value v) {
	auto t = cast<RankedTensorType>(v.getType());
	int64_t origRank = t.getRank();
	if (origRank >= 4)
		return {v, origRank};
	SmallVector<int64_t, 4> newShape(4 - origRank, 1);
	newShape.append(t.getShape().begin(), t.getShape().end());
	// Reassociation: the leading 1s collapse into the FIRST original dim.
	SmallVector<ReassociationIndices, 4> reassoc;
	ReassociationIndices first;
	for (int64_t i = 0; i < 4 - origRank; ++i)
		first.push_back(i);
	first.push_back(4 - origRank); // the original dim 0 ends up at this idx
	reassoc.push_back(first);
	for (int64_t i = 4 - origRank + 1; i < 4; ++i)
		reassoc.push_back({i});
	auto expanded = tensor::ExpandShapeOp::create(rw, loc,
		RankedTensorType::get(newShape, t.getElementType()), v, reassoc);
	return {expanded.getResult(), origRank};
}

// Inverse of promoteToRank4: collapse rank-4 back to the original rank.
static Value collapseFromRank4(
	PatternRewriter &rw, Location loc, Value v, RankedTensorType origTy) {
	if (origTy.getRank() >= 4)
		return v;
	SmallVector<ReassociationIndices, 4> reassoc;
	ReassociationIndices first;
	for (int64_t i = 0; i < 4 - origTy.getRank(); ++i)
		first.push_back(i);
	first.push_back(4 - origTy.getRank());
	reassoc.push_back(first);
	for (int64_t i = 4 - origTy.getRank() + 1; i < 4; ++i)
		reassoc.push_back({i});
	return tensor::CollapseShapeOp::create(rw, loc, origTy, v, reassoc)
		.getResult();
}

// Fused pointwise binary + quantize:
//   yield fptosi(clamp(round((lhs <op> rhs) / scale) + zp))
//
// yolov8 emits this after standalone dequantize generics for routing/concat
// correction dispatches. Lower the whole body, not just the dequant producers,
// otherwise SerializeGraph sees a partial QNN graph with dequant outputs as
// graph outputs and the HAL wrapper cannot bind it.
struct LowerBinaryQuantize : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();

		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isInteger(8))
			return failure();
		auto lhsTy = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto rhsTy = cast<RankedTensorType>(op.getDpsInputs()[1].getType());
		if (!lhsTy.getElementType().isF32() || !rhsTy.getElementType().isF32())
			return failure();

		Block &body = op.getRegion().front();
		auto *yield = body.getTerminator();
		if (!yield || yield->getNumOperands() != 1)
			return failure();
		auto *fptosi = yield->getOperand(0).getDefiningOp();
		if (!isa_and_nonnull<arith::FPToSIOp>(fptosi))
			return failure();

		Operation *cur = fptosi->getOperand(0).getDefiningOp();
		while (cur &&
			(isa<arith::MinimumFOp, arith::MaximumFOp>(cur) ||
				(cur->getDialect() &&
					cur->getDialect()->getNamespace() == "math" &&
					cur->getNumOperands() == 1))) {
			if (cur->getNumOperands() == 2) {
				double ignored = 0.0;
				if (isF32Const(cur->getOperand(0), &ignored)) {
					cur = cur->getOperand(1).getDefiningOp();
				} else if (isF32Const(cur->getOperand(1), &ignored)) {
					cur = cur->getOperand(0).getDefiningOp();
				} else {
					return failure();
				}
			} else {
				cur = cur->getOperand(0).getDefiningOp();
			}
		}
		if (!cur)
			return failure();

		int64_t zp = 0;
		Value scaleSide;
		if (auto addf = dyn_cast<arith::AddFOp>(cur)) {
			double zpF = 0.0;
			if (isF32Const(addf.getRhs(), &zpF)) {
				zp = static_cast<int64_t>(zpF);
				scaleSide = addf.getLhs();
			} else if (isF32Const(addf.getLhs(), &zpF)) {
				zp = static_cast<int64_t>(zpF);
				scaleSide = addf.getRhs();
			} else {
				return failure();
			}
		} else {
			scaleSide = cur->getResult(0);
		}

		Operation *scaleOp = scaleSide.getDefiningOp();
		while (scaleOp && scaleOp->getNumOperands() == 1 &&
			scaleOp->getDialect() &&
			scaleOp->getDialect()->getNamespace() == "math") {
			scaleOp = scaleOp->getOperand(0).getDefiningOp();
		}

		double scale = 1.0;
		Value binaryValue;
		if (auto divf = dyn_cast_or_null<arith::DivFOp>(scaleOp)) {
			if (!isF32Const(divf.getRhs(), &scale))
				return failure();
			binaryValue = divf.getLhs();
		} else if (auto mulf = dyn_cast_or_null<arith::MulFOp>(scaleOp)) {
			double inv = 0.0;
			if (isF32Const(mulf.getRhs(), &inv) && inv != 0.0) {
				scale = 1.0 / inv;
				binaryValue = mulf.getLhs();
			} else if (isF32Const(mulf.getLhs(), &inv) && inv != 0.0) {
				scale = 1.0 / inv;
				binaryValue = mulf.getRhs();
			} else {
				return failure();
			}
		} else {
			return failure();
		}

		auto *binaryOp = binaryValue.getDefiningOp();
		if (!binaryOp)
			return failure();
		int32_t kind = -1;
		if (isa<arith::AddFOp>(binaryOp))
			kind = kBinaryAdd;
		else if (isa<arith::SubFOp>(binaryOp))
			kind = kBinarySub;
		else if (isa<arith::MulFOp>(binaryOp))
			kind = kBinaryMul;
		else if (isa<arith::DivFOp>(binaryOp))
			kind = kBinaryDiv;
		else
			return failure();

		auto lhsArg = dyn_cast<BlockArgument>(binaryOp->getOperand(0));
		auto rhsArg = dyn_cast<BlockArgument>(binaryOp->getOperand(1));
		if (!lhsArg || !rhsArg)
			return failure();
		if (lhsArg.getArgNumber() >= 2 || rhsArg.getArgNumber() >= 2)
			return failure();
		Value lhsInput = op.getDpsInputs()[lhsArg.getArgNumber()];
		Value rhsInput = op.getDpsInputs()[rhsArg.getArgNumber()];

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		auto [lhs4, lhsRank] = promoteToRank4(rewriter, loc, lhsInput);
		auto [rhs4, rhsRank] = promoteToRank4(rewriter, loc, rhsInput);
		(void)lhsRank;
		(void)rhsRank;
		auto lhs4Ty = cast<RankedTensorType>(lhs4.getType());
		auto rhs4Ty = cast<RankedTensorType>(rhs4.getType());
		if (lhs4Ty.getShape() != rhs4Ty.getShape())
			return failure();

		auto bin4Ty = RankedTensorType::get(lhs4Ty.getShape(), b.getF32Type());
		Value bin = ElementWiseBinaryOp::create(
			rewriter, loc, bin4Ty, lhs4, rhs4, b.getI32IntegerAttr(kind))
						.getOutput();

		auto i8Ty = b.getIntegerType(8, /*isSigned=*/true);
		auto qOutElemTy = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8Ty, b.getF32Type(),
			/*scale=*/scale, /*zeroPoint=*/zp, /*storageMin=*/-128,
			/*storageMax=*/127);
		auto q4OutTy = RankedTensorType::get(lhs4Ty.getShape(), qOutElemTy);
		Value qnnQ =
			QuantizeOp::create(rewriter, loc, q4OutTy, bin).getOutput();

		auto plainI8R4 =
			RankedTensorType::get(lhs4Ty.getShape(), b.getI8Type());
		Value plainQ = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{plainI8R4}, ValueRange{qnnQ})
						   .getResult(0);
		rewriter.replaceOp(op, collapseFromRank4(rewriter, loc, plainQ, outTy));
		return success();
	}
};

// FP32 residual + bias + SiLU — yolov8 fp32 residual blocks. 3-input
// linalg.generic: residual (rank-N), conv (rank-N), bias (rank-1).
// Body: yield = addf(residual, mulf(addf(conv,bias), sigmoid(...))).
struct LowerFp32ResidualBiasSiLU : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isF32())
			return failure();
		if (!matchFp32ResidualBiasSiLUBody(op.getRegion().front()))
			return failure();

		// Identify roles: one rank-1 bias, two rank-N (residual + conv result).
		// Use the body's yield-side addf-blockarg-position to decide which
		// input is the residual vs which is the conv input.
		auto maps = op.getIndexingMapsArray();
		int biasIdx = -1;
		SmallVector<int> rankNIdxs;
		for (int i = 0; i < 3; ++i) {
			auto t = dyn_cast<RankedTensorType>(op.getDpsInputs()[i].getType());
			if (!t)
				return failure();
			if (t.getRank() == 1)
				biasIdx = i;
			else if (t.getRank() == outTy.getRank())
				rankNIdxs.push_back(i);
		}
		if (biasIdx < 0 || rankNIdxs.size() != 2)
			return failure();

		// Find the residual block arg = the one fed into the OUTER addf with
		// the SiLU result. The conv arg is the one fed into the INNER addf
		// (with the bias). Walk body to figure out.
		Block &body = op.getRegion().front();
		auto outerYield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		auto outerAdd = outerYield.getOperand(0).getDefiningOp<arith::AddFOp>();
		// The non-mulf operand of outerAdd is the residual block arg.
		Value residualBlockVal;
		if (outerAdd.getLhs().getDefiningOp<arith::MulFOp>())
			residualBlockVal = outerAdd.getRhs();
		else if (outerAdd.getRhs().getDefiningOp<arith::MulFOp>())
			residualBlockVal = outerAdd.getLhs();
		else
			return failure();
		auto residualArg = dyn_cast<BlockArgument>(residualBlockVal);
		if (!residualArg)
			return failure();
		unsigned residualBodyArgIdx = residualArg.getArgNumber();

		// Map body arg index to DPS-input index. Body args [0..nIns-1] are
		// input args, [nIns] is init arg.
		int residualIdx = static_cast<int>(residualBodyArgIdx);
		if (residualIdx < 0 || residualIdx >= 3)
			return failure();
		int convIdx = -1;
		for (int idx : rankNIdxs) {
			if (idx != residualIdx) {
				convIdx = idx;
				break;
			}
		}
		if (convIdx < 0)
			return failure();

		Value residual = op.getDpsInputs()[residualIdx];
		Value conv = op.getDpsInputs()[convIdx];
		Value bias1D = op.getDpsInputs()[biasIdx];

		// Bias broadcast axis derivation (same as LowerFp32BiasSiLU).
		auto biasMap = maps[biasIdx];
		if (biasMap.getNumResults() != 1)
			return failure();
		auto biasDim = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
		if (!biasDim)
			return failure();
		unsigned biasIter = biasDim.getPosition();
		auto outMap = maps[op.getNumDpsInputs()];
		int biasAxisInOutput = -1;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
				d && d.getPosition() == biasIter) {
				biasAxisInOutput = static_cast<int>(i);
				break;
			}
		}
		if (biasAxisInOutput < 0)
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		Type f32Ty = b.getF32Type();
		SmallVector<int64_t> biasShape(outTy.getRank(), 1);
		biasShape[biasAxisInOutput] = outTy.getShape()[biasAxisInOutput];
		SmallVector<ReassociationIndices> reassoc(1);
		for (int i = 0; i < (int)outTy.getRank(); ++i)
			reassoc[0].push_back(i);
		auto biasTy = RankedTensorType::get(biasShape, f32Ty);
		Value biasExpanded = tensor::ExpandShapeOp::create(
			rewriter, loc, biasTy, bias1D, reassoc);

		auto sameTy = RankedTensorType::get(outTy.getShape(), f32Ty);
		auto biased = ElementWiseBinaryOp::create(rewriter, loc, sameTy, conv,
			biasExpanded, b.getI32IntegerAttr(kBinaryAdd));
		auto sigmoid = ElementWiseNeuronOp::create(rewriter, loc, sameTy,
			biased.getOutput(), b.getI32IntegerAttr(/*SIGMOID*/ 6));
		auto silu = ElementWiseBinaryOp::create(rewriter, loc, sameTy,
			biased.getOutput(), sigmoid.getOutput(),
			b.getI32IntegerAttr(kBinaryMul));
		auto withResidual = ElementWiseBinaryOp::create(rewriter, loc, sameTy,
			residual, silu.getOutput(), b.getI32IntegerAttr(kBinaryAdd));
		rewriter.replaceOp(op, withResidual.getOutput());
		return success();
	}
};

// FP32 bias-only add — 2-input linalg.generic where one input is rank-N
// (matching output), the other is rank-1 broadcasted along one axis, and
// the body is just `addf(in, bias)`. Lowers to qnn.element_wise_binary(Add)
// with the bias reshaped to a rank-matching tensor (size 1 on all dims
// except the bias axis).
struct LowerFp32BiasAdd : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isF32())
			return failure();
		Block &body = op.getRegion().front();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		auto addf = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
		if (!addf)
			return failure();
		if (!isa<BlockArgument>(addf.getLhs()) ||
			!isa<BlockArgument>(addf.getRhs()))
			return failure();
		// One input rank-N matching output, other rank-1.
		auto maps = op.getIndexingMapsArray();
		int biasIdx = -1, mainIdx = -1;
		for (int i = 0; i < 2; ++i) {
			auto t = dyn_cast<RankedTensorType>(op.getDpsInputs()[i].getType());
			if (!t)
				return failure();
			if (t.getRank() == 1)
				biasIdx = i;
			else if (t.getRank() == outTy.getRank())
				mainIdx = i;
		}
		if (biasIdx < 0 || mainIdx < 0)
			return failure();
		auto biasMap = maps[biasIdx];
		if (biasMap.getNumResults() != 1)
			return failure();
		auto biasDim = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
		if (!biasDim)
			return failure();
		unsigned biasIter = biasDim.getPosition();
		auto outMap = maps[op.getNumDpsInputs()];
		int biasAxisInOutput = -1;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
				d && d.getPosition() == biasIter) {
				biasAxisInOutput = static_cast<int>(i);
				break;
			}
		}
		if (biasAxisInOutput < 0)
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		Type f32Ty = b.getF32Type();
		SmallVector<int64_t> biasShape(outTy.getRank(), 1);
		biasShape[biasAxisInOutput] = outTy.getShape()[biasAxisInOutput];
		SmallVector<ReassociationIndices> reassoc(1);
		for (int i = 0; i < (int)outTy.getRank(); ++i)
			reassoc[0].push_back(i);
		auto biasTy = RankedTensorType::get(biasShape, f32Ty);
		Value biasExpanded = tensor::ExpandShapeOp::create(
			rewriter, loc, biasTy, op.getDpsInputs()[biasIdx], reassoc);
		auto sameTy = RankedTensorType::get(outTy.getShape(), f32Ty);
		auto bin = ElementWiseBinaryOp::create(rewriter, loc, sameTy,
			op.getDpsInputs()[mainIdx], biasExpanded,
			b.getI32IntegerAttr(kBinaryAdd));
		rewriter.replaceOp(op, bin.getOutput());
		return success();
	}
};

// linalg.fill with no users — appears when an upstream conv pattern
// consumes the fill's only consumer (the conv accumulator init) and the
// fill becomes dead. MLIR's greedy DCE doesn't always erase these between
// pattern iterations, so erase them explicitly.
struct EraseDeadLinalgFill : OpRewritePattern<linalg::FillOp> {
	using OpRewritePattern<linalg::FillOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(
		linalg::FillOp op, PatternRewriter &rewriter) const override {
		if (!op->getUses().empty())
			return failure();
		rewriter.eraseOp(op);
		return success();
	}
};

// Standalone fp32 sigmoid: linalg.generic with single input, body
// `divf(1, addf(exp(negf(x)), 1))`. Maps to qnn.element_wise_neuron(SIGMOID).
struct LowerStandaloneSigmoid : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy || !outTy.getElementType().isF32())
			return failure();
		Block &body = op.getRegion().front();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		// yield <- divf(1, addf(exp(negf(blockarg)), 1)).
		auto sigDiv = yield.getOperand(0).getDefiningOp<arith::DivFOp>();
		if (!sigDiv)
			return failure();
		double one = 0.0;
		if (!isF32Const(sigDiv.getLhs(), &one) || std::abs(one - 1.0) > 1e-6)
			return failure();
		auto sigAdd = sigDiv.getRhs().getDefiningOp<arith::AddFOp>();
		if (!sigAdd)
			return failure();
		Value expSide;
		double oneCk = 0.0;
		if (isF32Const(sigAdd.getRhs(), &oneCk) && std::abs(oneCk - 1.0) < 1e-6)
			expSide = sigAdd.getLhs();
		else if (isF32Const(sigAdd.getLhs(), &oneCk) &&
			std::abs(oneCk - 1.0) < 1e-6)
			expSide = sigAdd.getRhs();
		else
			return failure();
		auto *expOp = expSide.getDefiningOp();
		if (!expOp || !expOp->getDialect() ||
			expOp->getDialect()->getNamespace() != "math" ||
			!expOp->getName().getStringRef().ends_with("exp"))
			return failure();
		auto negf = dyn_cast_or_null<arith::NegFOp>(
			expOp->getOperand(0).getDefiningOp());
		if (!negf || !isa<BlockArgument>(negf.getOperand()))
			return failure();

		Builder b(rewriter.getContext());
		auto qnnSig = ElementWiseNeuronOp::create(rewriter, op.getLoc(), outTy,
			op.getDpsInputs()[0], b.getI32IntegerAttr(/*SIGMOID*/ 6));
		rewriter.replaceOp(op, qnnSig.getOutput());
		return success();
	}
};

// FP32 conv-tail bias+SiLU: a 2-input linalg.generic that adds a 1D bias
// and applies SiLU = x * sigmoid(x). Lowered to:
//   qnn.element_wise_binary(Add)(conv_out_4d, bias_4d)  -> biased
//   qnn.element_wise_neuron(SIGMOID)(biased)             -> sigmoid_out
//   qnn.element_wise_binary(MUL)(biased, sigmoid_out)    -> SiLU(biased)
// The pattern doesn't lift bias into the upstream conv (keeps decoupled from
// LowerConv2dQGeneric); QNN's binary ops broadcast a rank-1 bias along the
// channel dim natively.
struct LowerFp32BiasSiLU : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isF32())
			return failure();
		if (!matchFp32BiasSiLUBody(op.getRegion().front()))
			return failure();

		// Identify which input is the broadcasted 1D bias (rank-1 source map)
		// and which is the conv result (rank-N matching output).
		auto maps = op.getIndexingMapsArray();
		int biasIdx = -1, convIdx = -1;
		for (int i = 0; i < 2; ++i) {
			auto inTy =
				dyn_cast<RankedTensorType>(op.getDpsInputs()[i].getType());
			if (!inTy)
				return failure();
			if (inTy.getRank() == 1) {
				biasIdx = i;
			} else if (inTy.getRank() == outTy.getRank()) {
				convIdx = i;
			}
		}
		if (biasIdx < 0 || convIdx < 0)
			return failure();

		Value conv = op.getDpsInputs()[convIdx];
		Value bias1D = op.getDpsInputs()[biasIdx];
		auto biasMap = maps[biasIdx];
		if (biasMap.getNumResults() != 1)
			return failure();
		auto biasDimExpr = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
		if (!biasDimExpr)
			return failure();
		unsigned biasIter = biasDimExpr.getPosition();

		// Find which output dim that iter maps to (broadcast axis on the conv).
		auto outMap = maps[op.getNumDpsInputs()];
		int biasAxisInOutput = -1;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i))) {
				if (d.getPosition() == biasIter) {
					biasAxisInOutput = static_cast<int>(i);
					break;
				}
			}
		}
		if (biasAxisInOutput < 0)
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		Type f32Ty = b.getF32Type();

		// Reshape 1D bias to rank-matching shape with 1s everywhere except the
		// bias channel axis. QNN binary ops broadcast that natively.
		SmallVector<int64_t> biasShape(outTy.getRank(), 1);
		biasShape[biasAxisInOutput] = outTy.getShape()[biasAxisInOutput];
		SmallVector<ReassociationIndices> biasReassoc(1);
		for (int i = 0; i < (int)outTy.getRank(); ++i)
			biasReassoc[0].push_back(i);
		auto biasExpandTy = RankedTensorType::get(biasShape, f32Ty);
		Value biasExpanded = tensor::ExpandShapeOp::create(
			rewriter, loc, biasExpandTy, bias1D, biasReassoc);

		auto sameTy = RankedTensorType::get(outTy.getShape(), f32Ty);
		auto biased = ElementWiseBinaryOp::create(rewriter, loc, sameTy, conv,
			biasExpanded, b.getI32IntegerAttr(kBinaryAdd));
		auto sigmoid = ElementWiseNeuronOp::create(rewriter, loc, sameTy,
			biased.getOutput(), b.getI32IntegerAttr(/*SIGMOID*/ 6));
		auto silu = ElementWiseBinaryOp::create(rewriter, loc, sameTy,
			biased.getOutput(), sigmoid.getOutput(),
			b.getI32IntegerAttr(kBinaryMul));
		rewriter.replaceOp(op, silu.getOutput());
		return success();
	}
};

// Reduce (sum/max/mean) — linalg.generic with reduction iterators and a
// simple addf/maximumf/etc body, with init from linalg.fill or tensor.empty.
// Maps to qnn.reduce. yolov8 fp32 softmax has a ReduceSum step over a
// single axis with body `addf(in, out)`.
struct LowerReduceSumGeneric : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		auto inTy = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!inTy || !outTy.getElementType().isF32() ||
			!inTy.getElementType().isF32())
			return failure();
		// Need at least one reduction iter.
		SmallVector<unsigned> reductionAxes, parallelAxes;
		auto iters = op.getIteratorTypesArray();
		for (size_t i = 0; i < iters.size(); ++i) {
			if (iters[i] == utils::IteratorType::reduction)
				reductionAxes.push_back(i);
			else
				parallelAxes.push_back(i);
		}
		if (reductionAxes.empty())
			return failure();
		// Body: yield <- {addf | maximumf | maxnumf}(in, out) (commutative).
		// op_kind matches QNN_ReduceOp enum: 0=Sum, 2=Max.
		Block &body = op.getRegion().front();
		if (body.getNumArguments() != 2)
			return failure();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		Operation *reduceOp = yield.getOperand(0).getDefiningOp();
		if (!reduceOp)
			return failure();
		int32_t qnnReduceKind = -1;
		if (isa<arith::AddFOp>(reduceOp))
			qnnReduceKind = 0;
		else if (isa<arith::MaximumFOp, arith::MaxNumFOp>(reduceOp))
			qnnReduceKind = 2;
		else
			return failure();
		Value inArg = body.getArgument(0);
		Value outArg = body.getArgument(1);
		Value lhs = reduceOp->getOperand(0);
		Value rhs = reduceOp->getOperand(1);
		if (!((lhs == inArg && rhs == outArg) ||
				(lhs == outArg && rhs == inArg)))
			return failure();
		// Input map must be a permutation of the iter dims (each result is a
		// distinct AffineDimExpr). Compute iter-dim → tensor-axis mapping so
		// we can translate reduction iter dims to the qnn.reduce axes attr.
		auto inMap = op.getIndexingMapsArray()[0];
		auto outMap = op.getIndexingMapsArray()[1];
		if (inMap.getNumResults() != iters.size())
			return failure();
		llvm::SmallDenseMap<unsigned, unsigned> iterToTensorAxis;
		for (unsigned i = 0; i < inMap.getNumResults(); ++i) {
			auto d = dyn_cast<AffineDimExpr>(inMap.getResult(i));
			if (!d)
				return failure();
			iterToTensorAxis[d.getPosition()] = i;
		}
		if (iterToTensorAxis.size() != iters.size())
			return failure();
		// Translate reduction iter dims to tensor axes.
		SmallVector<int32_t> axesI32;
		for (unsigned r : reductionAxes) {
			auto it = iterToTensorAxis.find(r);
			if (it == iterToTensorAxis.end())
				return failure();
			axesI32.push_back(static_cast<int32_t>(it->second));
		}
		// qnn.reduce result shape: drop reduction axes from input shape.
		SmallVector<int64_t> reducedShape;
		SmallVector<unsigned> reducedTensorAxes(axesI32.begin(), axesI32.end());
		for (size_t i = 0; i < inTy.getShape().size(); ++i) {
			bool isRed =
				std::find(reducedTensorAxes.begin(), reducedTensorAxes.end(),
					static_cast<unsigned>(i)) != reducedTensorAxes.end();
			if (!isRed)
				reducedShape.push_back(inTy.getShape()[i]);
		}
		// Validate against linalg output shape (recover via outMap).
		SmallVector<int64_t> linalgOutShape;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
			if (!d)
				return failure();
			auto it = iterToTensorAxis.find(d.getPosition());
			if (it == iterToTensorAxis.end())
				return failure();
			linalgOutShape.push_back(inTy.getShape()[it->second]);
		}
		if (linalgOutShape !=
			SmallVector<int64_t>(
				outTy.getShape().begin(), outTy.getShape().end()))
			return failure();

		Builder b(rewriter.getContext());
		auto qnnOutTy =
			RankedTensorType::get(reducedShape, outTy.getElementType());
		Value reduceResult = ReduceOp::create(rewriter, op.getLoc(), qnnOutTy,
			op.getDpsInputs()[0], b.getI32ArrayAttr(axesI32),
			/*op_kind=*/b.getI32IntegerAttr(qnnReduceKind),
			/*keep_dims=*/b.getBoolAttr(false))
								 .getOutput();
		// If linalg output shape != reduced shape (e.g., the output map
		// permuted the remaining dims), insert a transpose.
		if (linalgOutShape != reducedShape) {
			SmallVector<int64_t> perm;
			for (int64_t d : linalgOutShape) {
				// Find a matching dim in reducedShape (first unused
				// occurrence).
				for (size_t j = 0; j < reducedShape.size(); ++j) {
					if (reducedShape[j] == d &&
						std::find(perm.begin(), perm.end(),
							static_cast<int64_t>(j)) == perm.end()) {
						perm.push_back(j);
						break;
					}
				}
			}
			if (perm.size() == linalgOutShape.size()) {
				Value empty = tensor::EmptyOp::create(rewriter, op.getLoc(),
					linalgOutShape, outTy.getElementType());
				reduceResult = linalg::TransposeOp::create(
					rewriter, op.getLoc(), reduceResult, empty, perm)
								   .getResult()[0];
			}
		}
		rewriter.replaceOp(op, reduceResult);
		return success();
	}
};

// Dequantize: linalg.generic with elementwise body shape extsi · subi ·
// sitofp · mulf. Output is fp.
struct LowerDequantize : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isF32())
			return failure();
		auto inTy = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		// Accept i8 (standard dequant) or i32 (post-im2col-matmul accumulator
		// dequant — yolov8 form: sitofp+mulf with no zp subtraction).
		bool inI8 = inTy.getElementType().isInteger(8);
		bool inI32 = inTy.getElementType().isInteger(32);
		if (!inI8 && !inI32)
			return failure();

		double scale = 1.0;
		int64_t zp = 0;
		if (!matchDequantBody(op.getRegion().front(), &scale, &zp))
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();

		// Promote operand to rank-4 if rank < 4 — QNN rejects sub-rank-4
		// graphs at finalize time (rc=6022).
		auto [in4, _origRank] =
			promoteToRank4(rewriter, loc, op.getDpsInputs()[0]);
		auto in4Ty = cast<RankedTensorType>(in4.getType());
		(void)_origRank;

		// Wrap input element type with quant.uniform<i8:f32, scale, zp> so
		// SerializeGraph picks up qk=1 with the matched params on the input
		// tensor. Build a NEW input value via tensor.bitcast that storage-
		// type-aliases the existing i8 buffer to the quant.uniform-typed
		// tensor. Bitcast is safe because storage type underneath quant.uniform
		// is identical to the i8 element type.
		auto storageTy = inI8
			? cast<Type>(b.getIntegerType(8, /*isSigned=*/true))
			: cast<Type>(b.getIntegerType(32, /*isSigned=*/true));
		int64_t storageMin = inI8 ? -128 : INT32_MIN;
		int64_t storageMax = inI8 ? 127 : INT32_MAX;
		auto f32Ty = b.getF32Type();
		auto qInElemTy = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, storageTy, f32Ty,
			/*scale=*/scale, /*zeroPoint=*/zp,
			/*storageMin=*/storageMin, /*storageMax=*/storageMax);
		auto qInTy = RankedTensorType::get(in4Ty.getShape(), qInElemTy);
		// Use unrealized_conversion_cast to bridge i8-storage tensor <->
		// quant.uniform<i8:f32, …> typed tensor without requiring a
		// dialect-specific cast op (tensor.bitcast rejects quant.uniform).
		Value qIn = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{qInTy}, ValueRange{in4})
						.getResult(0);

		// Output stays plain fp32 at rank-4.
		auto out4Ty = RankedTensorType::get(in4Ty.getShape(), f32Ty);
		Value qnnDeq =
			DequantizeOp::create(rewriter, loc, out4Ty, qIn).getOutput();
		Value collapsed = collapseFromRank4(rewriter, loc, qnnDeq, outTy);
		rewriter.replaceOp(op, collapsed);
		return success();
	}
};

// Conv-tail dequant-with-bias: 2-input linalg.generic where input0 is an i32
// rank-N conv accumulator, input1 is a rank-1 i32 bias broadcast along one
// dim, and body is `addi(main, bias) → sitofp → mulf(scale)` yielding f32.
// Lowers to qnn.dequantize(main) + qnn.dequantize(bias_expanded) +
// qnn.element_wise_binary(Add). Required for yolov8 stem variants whose
// conv output is f32 (acc dequant) rather than re-quantized to i8.
struct LowerDequantizeWithBias : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isF32())
			return failure();
		auto in0Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto in1Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
		if (!in0Ty || !in1Ty)
			return failure();
		if (!in0Ty.getElementType().isInteger(32) ||
			!in1Ty.getElementType().isInteger(32))
			return failure();

		// Body: yield <- [maximumf(_, 0) (optional relu)] <- mulf(_, scale) <-
		// sitofp(_) <- addi(arg0, arg1).
		Block &body = op.getRegion().front();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		Value tail = yield.getOperand(0);
		auto isFConst = [](Value v, double target) {
			auto c = v.getDefiningOp<arith::ConstantOp>();
			if (!c)
				return false;
			auto fa = dyn_cast<FloatAttr>(c.getValue());
			return fa && fa.getValueAsDouble() == target;
		};
		auto isMathOp = [](Operation *op, llvm::StringRef name) {
			return op && op->getDialect() &&
				op->getDialect()->getNamespace() == "math" &&
				op->getName().getStringRef().ends_with(name);
		};
		// Try to peel off `sigmoid(x)` = `divf(1, addf(math.exp(negf(x)), 1))`.
		// Returns Value() on no match.
		auto peelSigmoid = [&](Value v) -> Value {
			auto divf = v.getDefiningOp<arith::DivFOp>();
			if (!divf || !isFConst(divf.getLhs(), 1.0))
				return Value();
			auto addf = divf.getRhs().getDefiningOp<arith::AddFOp>();
			if (!addf)
				return Value();
			Value other;
			if (isFConst(addf.getRhs(), 1.0))
				other = addf.getLhs();
			else if (isFConst(addf.getLhs(), 1.0))
				other = addf.getRhs();
			else
				return Value();
			auto *expOp = other.getDefiningOp();
			if (!isMathOp(expOp, "exp"))
				return Value();
			auto negf = expOp->getOperand(0).getDefiningOp<arith::NegFOp>();
			if (!negf)
				return Value();
			return negf.getOperand();
		};
		// QNN wire kinds: Relu=4, Sigmoid=6, Tanh=8. SiLU lowers to
		// Sigmoid followed by ElementWiseBinary(Mul).
		int32_t neuronKind = -1;
		bool isSilu = false;
		Value siluXVal;
		if (auto maxf = tail.getDefiningOp<arith::MaximumFOp>()) {
			if (isFConst(maxf.getLhs(), 0.0)) {
				tail = maxf.getRhs();
				neuronKind = /*Relu=*/4;
			} else if (isFConst(maxf.getRhs(), 0.0)) {
				tail = maxf.getLhs();
				neuronKind = /*Relu=*/4;
			}
		} else if (auto *t = tail.getDefiningOp(); isMathOp(t, "tanh")) {
			tail = t->getOperand(0);
			neuronKind = /*Tanh=*/8;
		} else if (Value sigIn = peelSigmoid(tail)) {
			tail = sigIn;
			neuronKind = /*Sigmoid=*/6;
		} else if (auto mul = tail.getDefiningOp<arith::MulFOp>()) {
			// SiLU: mulf(x, sigmoid(x)) where x is shared. Detect either
			// operand as the sigmoid expression.
			Value sigIn;
			Value x;
			if ((sigIn = peelSigmoid(mul.getRhs())) && sigIn == mul.getLhs()) {
				x = mul.getLhs();
			} else if ((sigIn = peelSigmoid(mul.getLhs())) &&
				sigIn == mul.getRhs()) {
				x = mul.getRhs();
			}
			if (x) {
				tail = x;
				isSilu = true;
				siluXVal = x;
			}
		}
		auto mulf = tail.getDefiningOp<arith::MulFOp>();
		if (!mulf)
			return failure();
		double scale = 1.0;
		Value cvtChain;
		auto isF32Const = [](Value v, double *out) -> bool {
			auto c = v.getDefiningOp<arith::ConstantOp>();
			if (!c)
				return false;
			if (auto fa = dyn_cast<FloatAttr>(c.getValue())) {
				*out = fa.getValueAsDouble();
				return true;
			}
			return false;
		};
		if (isF32Const(mulf.getLhs(), &scale))
			cvtChain = mulf.getRhs();
		else if (isF32Const(mulf.getRhs(), &scale))
			cvtChain = mulf.getLhs();
		else
			return failure();
		auto sitofp = cvtChain.getDefiningOp<arith::SIToFPOp>();
		if (!sitofp)
			return failure();
		auto addi = sitofp.getOperand().getDefiningOp<arith::AddIOp>();
		if (!addi)
			return failure();
		if (!isa<BlockArgument>(addi.getLhs()) ||
			!isa<BlockArgument>(addi.getRhs()))
			return failure();

		// Find which input is the rank-1 bias (broadcast on one axis) and which
		// is the rank-N main accumulator.
		auto maps = op.getIndexingMapsArray();
		int biasIdx = -1, mainIdx = -1;
		if (in0Ty.getRank() == 1 && in1Ty.getRank() == outTy.getRank()) {
			biasIdx = 0;
			mainIdx = 1;
		} else if (in1Ty.getRank() == 1 && in0Ty.getRank() == outTy.getRank()) {
			biasIdx = 1;
			mainIdx = 0;
		} else {
			return failure();
		}
		auto biasMap = maps[biasIdx];
		if (biasMap.getNumResults() != 1)
			return failure();
		auto biasDim = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
		if (!biasDim)
			return failure();
		unsigned biasIter = biasDim.getPosition();
		auto outMap = maps[op.getNumDpsInputs()];
		int biasAxisInOutput = -1;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
				d && d.getPosition() == biasIter) {
				biasAxisInOutput = static_cast<int>(i);
				break;
			}
		}
		if (biasAxisInOutput < 0)
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		Type f32Ty = b.getF32Type();

		// Dequantize main (rank-N i32 → rank-N f32).
		auto mainF32Ty = RankedTensorType::get(in0Ty.getShape(), f32Ty);
		if (mainIdx == 1)
			mainF32Ty = RankedTensorType::get(in1Ty.getShape(), f32Ty);
		Value mainDeq = DequantizeOp::create(
			rewriter, loc, mainF32Ty, op.getDpsInputs()[mainIdx])
							.getOutput();

		// Expand bias from rank-1 to rank-N with ones on non-channel axes so
		// QNN's implicit broadcast handles it.
		SmallVector<int64_t> biasShape(outTy.getRank(), 1);
		biasShape[biasAxisInOutput] = outTy.getShape()[biasAxisInOutput];
		auto biasExpandedI32Ty =
			RankedTensorType::get(biasShape, b.getIntegerType(32));
		SmallVector<ReassociationIndices> reassoc(1);
		for (int i = 0; i < (int)outTy.getRank(); ++i)
			reassoc[0].push_back(i);
		Value biasExpanded = tensor::ExpandShapeOp::create(rewriter, loc,
			biasExpandedI32Ty, op.getDpsInputs()[biasIdx], reassoc);
		auto biasExpandedF32Ty = RankedTensorType::get(biasShape, f32Ty);
		Value biasDeq =
			DequantizeOp::create(rewriter, loc, biasExpandedF32Ty, biasExpanded)
				.getOutput();

		// Bias-add via QNN elementwise binary with implicit broadcast.
		auto sameTy = RankedTensorType::get(outTy.getShape(), f32Ty);
		Value bin = ElementWiseBinaryOp::create(rewriter, loc, sameTy, mainDeq,
			biasDeq, b.getI32IntegerAttr(kBinaryAdd))
						.getOutput();
		Value result = bin;
		if (isSilu) {
			// SiLU = x * sigmoid(x). Emit sigmoid then elementwise multiply.
			Value sig = ElementWiseNeuronOp::create(
				rewriter, loc, sameTy, bin, b.getI32IntegerAttr(/*Sigmoid=*/6))
							.getOutput();
			result = ElementWiseBinaryOp::create(rewriter, loc, sameTy, bin,
				sig, b.getI32IntegerAttr(kBinaryMul))
						 .getOutput();
			(void)siluXVal;
		} else if (neuronKind >= 0) {
			// Per QnnOpDef.h QNN_OP_ELEMENT_WISE_NEURON_OPERATION_*: Relu=4,
			// Sigmoid=6, Tanh=8.
			result = ElementWiseNeuronOp::create(
				rewriter, loc, sameTy, bin, b.getI32IntegerAttr(neuronKind))
						 .getOutput();
		}
		(void)scale; // Both dequants share the same scale via the body-matched
					 // factor; the qnn.dequantize ops carry it implicitly when
					 // their operand quant params are stamped by upstream pass.
		rewriter.replaceOp(op, result);
		return success();
	}
};

// Conv-tail rescale-and-requantize: 2-input linalg.generic where input0 is
// an i32 conv accumulator, input1 is a rank-1 i32 bias broadcast along one
// dim, body chains addi → sitofp → (any combination of mulf/divf/addf/
// math.roundeven/maximumf/minimumf, possibly nested through a fptosi/sitofp
// round-trip pair) → fptosi(i8), and output is i8. This is the fused
// "conv accumulator + bias + scale rescale + clamp + quantize" tail
// produced when IREE dispatch-creation splits a quantized conv into
// dispatches across the quant boundary.
//
// Lowering: one qnn.element_wise_binary(Add) over the two inputs, with
// input/output quant params encoded by wrapping tensor types with
// quant.uniform. The intermediate f32 values in the original body never
// appear as QNN tensors — they're SSA scalars within the linalg.generic
// region — so QNN's elementwise add (which internally rescales between
// input quant and output quant) produces the same result in a single op.
// Critically, this avoids the standalone qnn.quantize boundary that HTA
// refuses ("all-int fixed-point graphs").
struct LowerRescaleQuantizeWithBias : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isInteger(8))
			return failure();
		auto in0Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto in1Ty = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
		if (!in0Ty || !in1Ty)
			return failure();
		if (!in0Ty.getElementType().isInteger(32) ||
			!in1Ty.getElementType().isInteger(32))
			return failure();

		// Body walker: starting at yield, peel ops until we find addi(arg0,
		// arg1). Allowed ops in the chain: arith.{sitofp, fptosi, mulf, divf,
		// addf, maximumf, minimumf}, math.roundeven. Each must have exactly one
		// non-constant operand to continue along.
		Block &body = op.getRegion().front();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		Value cur = yield.getOperand(0);

		auto isFloatConstant = [](Value v) {
			auto c = v.getDefiningOp<arith::ConstantOp>();
			return c && isa<FloatAttr>(c.getValue());
		};
		auto isAllowedOp = [](Operation *op) {
			if (!op)
				return false;
			if (isa<arith::SIToFPOp, arith::FPToSIOp, arith::MulFOp,
					arith::DivFOp, arith::AddFOp, arith::MaximumFOp,
					arith::MinimumFOp, arith::NegFOp>(op))
				return true;
			if (op->getDialect() &&
				op->getDialect()->getNamespace() == "math") {
				auto name = op->getName().getStringRef();
				if (name.ends_with("roundeven") || name.ends_with("exp"))
					return true;
			}
			return false;
		};

		// Helper: detect `sigmoid(x)` body subtree `divf(1, addf(exp(negf(x)),
		// 1))` and return the underlying x value on match, Value() otherwise.
		auto peelSigmoidExpr = [&](Value v) -> Value {
			auto divf = v.getDefiningOp<arith::DivFOp>();
			if (!divf)
				return Value();
			auto isOne = [](Value w) {
				auto c = w.getDefiningOp<arith::ConstantOp>();
				if (!c)
					return false;
				auto fa = dyn_cast<FloatAttr>(c.getValue());
				return fa && fa.getValueAsDouble() == 1.0;
			};
			if (!isOne(divf.getLhs()))
				return Value();
			auto addf = divf.getRhs().getDefiningOp<arith::AddFOp>();
			if (!addf)
				return Value();
			Value other;
			if (isOne(addf.getRhs()))
				other = addf.getLhs();
			else if (isOne(addf.getLhs()))
				other = addf.getRhs();
			else
				return Value();
			auto *expOp = other.getDefiningOp();
			if (!expOp || !expOp->getDialect() ||
				expOp->getDialect()->getNamespace() != "math" ||
				!expOp->getName().getStringRef().ends_with("exp"))
				return Value();
			auto negf = expOp->getOperand(0).getDefiningOp<arith::NegFOp>();
			if (!negf)
				return Value();
			return negf.getOperand();
		};

		// Bound the walk to avoid pathological infinite loops on cyclic IR
		// (shouldn't happen post-canonicalization but be defensive).
		const int kMaxWalk = 128;
		int steps = 0;
		while (steps++ < kMaxWalk) {
			auto *defOp = cur.getDefiningOp();
			if (auto addi = dyn_cast_or_null<arith::AddIOp>(defOp)) {
				// Reached the integer add at the root. Verify both operands are
				// block-args (the bias plus the accumulator).
				if (!isa<BlockArgument>(addi.getLhs()) ||
					!isa<BlockArgument>(addi.getRhs()))
					return failure();
				break;
			}
			// Detect SiLU subtree: mulf(x, sigmoid(x)) where x is shared.
			if (auto mul = dyn_cast_or_null<arith::MulFOp>(defOp)) {
				Value sigIn;
				Value x;
				if ((sigIn = peelSigmoidExpr(mul.getRhs())) &&
					sigIn == mul.getLhs()) {
					x = mul.getLhs();
				} else if ((sigIn = peelSigmoidExpr(mul.getLhs())) &&
					sigIn == mul.getRhs()) {
					x = mul.getRhs();
				}
				if (x) {
					// SiLU peeled — continue walk from x.
					cur = x;
					continue;
				}
			}
			if (!isAllowedOp(defOp))
				return failure();
			// Pick the non-constant operand to continue along.
			Value next;
			for (Value v : defOp->getOperands()) {
				if (isFloatConstant(v))
					continue;
				if (next)
					return failure(); // ambiguous — give up
				next = v;
			}
			if (!next)
				return failure();
			cur = next;
		}
		if (steps >= kMaxWalk)
			return failure();

		// Identify which input is the rank-1 bias and which is the main acc.
		int biasIdx = -1, mainIdx = -1;
		if (in0Ty.getRank() == 1 && in1Ty.getRank() == outTy.getRank()) {
			biasIdx = 0;
			mainIdx = 1;
		} else if (in1Ty.getRank() == 1 && in0Ty.getRank() == outTy.getRank()) {
			biasIdx = 1;
			mainIdx = 0;
		} else {
			return failure();
		}
		auto maps = op.getIndexingMapsArray();
		auto biasMap = maps[biasIdx];
		if (biasMap.getNumResults() != 1)
			return failure();
		auto biasDim = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
		if (!biasDim)
			return failure();
		unsigned biasIter = biasDim.getPosition();
		auto outMap = maps[op.getNumDpsInputs()];
		int biasAxisInOutput = -1;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
				d && d.getPosition() == biasIter) {
				biasAxisInOutput = static_cast<int>(i);
				break;
			}
		}
		if (biasAxisInOutput < 0)
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		auto f32Ty = b.getF32Type();
		auto i32StorageTy = b.getIntegerType(32, /*isSigned=*/true);
		auto i8StorageTy = b.getIntegerType(8, /*isSigned=*/true);

		// Use non-trivial placeholder scales. HTA validator rejects scale=1.0
		// identity-rescale Conv2d configs. The chosen values satisfy
		// bias_scale == input_scale * weight_scale (HTA Conv2d requirement)
		// and produce a non-trivial output rescale ratio.
		//   input_scale  = 0.0625  (= 1/16)
		//   weight_scale = 0.0625  (= 1/16)
		//   bias_scale   = 0.00390625 (= 1/256 = input_scale * weight_scale)
		//   output_scale = 0.125   (= 1/8)
		auto qIn32Ty = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i32StorageTy, f32Ty,
			/*scale=*/0.00390625, /*zp=*/0, INT32_MIN, INT32_MAX);
		auto qIn8Ty = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8StorageTy, f32Ty,
			/*scale=*/0.125, /*zp=*/0, -128, 127);
		auto qInputI8Ty = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8StorageTy, f32Ty,
			/*scale=*/0.0625, /*zp=*/0, -128, 127);
		auto qWeightI8Ty = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8StorageTy, f32Ty,
			/*scale=*/0.0625, /*zp=*/0, -128, 127);

		Value mainV = op.getDpsInputs()[mainIdx];
		Value biasV = op.getDpsInputs()[biasIdx];

		// HTA fast path: if mainV traces back to a qnn.conv2d without bias
		// (possibly through a tensor.collapse_shape that removes the N=1
		// batch), absorb the bias into the conv2d's bias operand and change
		// the conv2d's output type to i8 quant. This eliminates the i32
		// qnn.element_wise_binary that HTA rejects with rc=6000.
		Value convResult;
		Operation *collapseOp = nullptr;
		bool dbgFastPath = ::getenv("MERLIN_QNN_BIAS_DBG") != nullptr;
		if (dbgFastPath) {
			llvm::errs() << "[bias-fp] enter, mainV defining op = ";
			if (mainV.getDefiningOp())
				llvm::errs() << mainV.getDefiningOp()->getName();
			else
				llvm::errs() << "(block arg)";
			llvm::errs() << "\n";
		}
		if (auto castOp = mainV.getDefiningOp<UnrealizedConversionCastOp>()) {
			Value src = castOp.getInputs()[0];
			if (auto collapse = src.getDefiningOp<tensor::CollapseShapeOp>()) {
				collapseOp = collapse;
				src = collapse.getSrc();
			}
			if (auto conv = src.getDefiningOp<Conv2dOp>()) {
				if (!conv.getBias())
					convResult = conv.getOutput();
			}
		} else if (auto collapse =
					   mainV.getDefiningOp<tensor::CollapseShapeOp>()) {
			collapseOp = collapse;
			if (auto conv = collapse.getSrc().getDefiningOp<Conv2dOp>()) {
				if (!conv.getBias())
					convResult = conv.getOutput();
			}
		} else if (auto conv = mainV.getDefiningOp<Conv2dOp>()) {
			if (!conv.getBias())
				convResult = conv.getOutput();
		}
		if (dbgFastPath) {
			llvm::errs() << "[bias-fp] convResult found: "
						 << (convResult ? "YES" : "NO") << "\n";
		}

		// If upstream conv hasn't lowered yet AND we'd produce an i32-input
		// ElementWiseAdd (HTA-incompatible), defer and let the conv pattern
		// fire first. The greedy driver re-tries us when IR changes.
		if (!convResult && mainV.getDefiningOp<linalg::GenericOp>()) {
			if (dbgFastPath)
				llvm::errs()
					<< "[bias-fp] DEFER — upstream is linalg.generic\n";
			return failure();
		}

		if (convResult) {
			// Rebuild qnn.conv2d with bias absorbed and i8 quant output.
			auto conv = convResult.getDefiningOp<Conv2dOp>();
			auto convTy = cast<RankedTensorType>(conv.getOutput().getType());
			// New conv output: same shape, i8 quant.uniform element type.
			auto newConvOutTy =
				RankedTensorType::get(convTy.getShape(), qIn8Ty);
			// Wrap activation input as quant.uniform<i8> so serializer emits
			// qk=1 with quant params (HTA Conv2d requires them on inputs).
			Value oldInput = conv.getInput();
			auto oldInputTy = cast<RankedTensorType>(oldInput.getType());
			Value qInput = oldInput;
			if (!isa<quant::QuantizedType>(oldInputTy.getElementType())) {
				auto qInputTy =
					RankedTensorType::get(oldInputTy.getShape(), qInputI8Ty);
				qInput = UnrealizedConversionCastOp::create(
					rewriter, loc, TypeRange{qInputTy}, ValueRange{oldInput})
							 .getResult(0);
			}
			// Wrap weight as quant.uniform<i8> too.
			Value oldWeight = conv.getWeight();
			auto oldWeightTy = cast<RankedTensorType>(oldWeight.getType());
			Value qWeight = oldWeight;
			if (!isa<quant::QuantizedType>(oldWeightTy.getElementType())) {
				auto qWeightTy =
					RankedTensorType::get(oldWeightTy.getShape(), qWeightI8Ty);
				qWeight = UnrealizedConversionCastOp::create(
					rewriter, loc, TypeRange{qWeightTy}, ValueRange{oldWeight})
							  .getResult(0);
			}
			// Bias must be rank-1 i32 (already is from
			// op.getDpsInputs[biasIdx]). Wrap as quant.uniform<i32, ...> so
			// SerializeGraph emits SFIXED_32.
			auto biasRTy = cast<RankedTensorType>(biasV.getType());
			auto qBiasRankTy =
				RankedTensorType::get(biasRTy.getShape(), qIn32Ty);
			Value qBias = UnrealizedConversionCastOp::create(
				rewriter, loc, TypeRange{qBiasRankTy}, ValueRange{biasV})
							  .getResult(0);
			auto newConv = Conv2dOp::create(rewriter, conv.getLoc(),
				newConvOutTy, qInput, qWeight, qBias, conv.getStrideAttr(),
				conv.getPadAmountAttr(), conv.getDilationAttr(),
				conv.getGroupAttr());
			// Replace conv use chain. Need to re-emit collapse_shape if present
			// and adjust types.
			Value newConvOut = newConv.getOutput();
			Value finalOut;
			if (collapseOp) {
				// Re-emit collapse with new i8 element type.
				auto oldCollapse = cast<tensor::CollapseShapeOp>(collapseOp);
				auto newCollapseTy = RankedTensorType::get(
					cast<RankedTensorType>(oldCollapse.getResult().getType())
						.getShape(),
					qIn8Ty);
				Value newCollapsed = tensor::CollapseShapeOp::create(rewriter,
					oldCollapse.getLoc(), newCollapseTy, newConvOut,
					oldCollapse.getReassociationIndices());
				finalOut = newCollapsed;
			} else {
				finalOut = newConvOut;
			}
			// Bridge from quant.uniform<i8> to plain i8 storage if downstream
			// wants plain i8 (which is the original generic's outTy).
			Value result = UnrealizedConversionCastOp::create(
				rewriter, loc, TypeRange{outTy}, ValueRange{finalOut})
							   .getResult(0);
			rewriter.replaceOp(op, result);
			// Erase the old conv (and old collapse if present). Use eraseOp;
			// the rewriter handles SSA-use updates that have already been
			// redirected via replaceOp above.
			if (collapseOp)
				rewriter.eraseOp(collapseOp);
			rewriter.eraseOp(conv);
			return success();
		}

		// Fallback: original 2-input ElementWiseBinary path (will fail HTA
		// validator for i32 inputs but works on CPU/GPU for testing).
		auto mainTy = cast<RankedTensorType>(mainV.getType());
		auto qMainTy = RankedTensorType::get(mainTy.getShape(), qIn32Ty);
		Value qMain = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{qMainTy}, ValueRange{mainV})
						  .getResult(0);

		// Expand bias from rank-1 to rank-N with ones on non-channel axes.
		SmallVector<int64_t> biasShape(outTy.getRank(), 1);
		biasShape[biasAxisInOutput] = outTy.getShape()[biasAxisInOutput];
		auto biasExpandedI32Ty =
			RankedTensorType::get(biasShape, b.getIntegerType(32));
		SmallVector<ReassociationIndices> reassoc(1);
		for (int i = 0; i < (int)outTy.getRank(); ++i)
			reassoc[0].push_back(i);
		Value biasExpanded = tensor::ExpandShapeOp::create(
			rewriter, loc, biasExpandedI32Ty, biasV, reassoc);
		auto qBiasTy = RankedTensorType::get(biasShape, qIn32Ty);
		Value qBias = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{qBiasTy}, ValueRange{biasExpanded})
						  .getResult(0);

		auto qOutTy = RankedTensorType::get(outTy.getShape(), qIn8Ty);
		Value bin = ElementWiseBinaryOp::create(rewriter, loc, qOutTy, qMain,
			qBias, b.getI32IntegerAttr(kBinaryAdd))
						.getOutput();

		// Bridge quant.uniform<i8> back to plain i8 storage for downstream
		// consumers / func return value.
		Value result = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{outTy}, ValueRange{bin})
						   .getResult(0);
		rewriter.replaceOp(op, result);
		return success();
	}
};

// Yolov8n int8 post-fold conv-tail with RESIDUAL connection + SiLU. This is
// the 3-input variant of LowerRescaleQuantizeWithBias that handles the
// repeated residual-block pattern in yolov8's backbone:
//
//   ^bb0(%resid: f32, %acc: i32, %bias: i32, %out: i8):
//     %a   = addi(%acc, %bias) : i32
//     %f   = sitofp(%a) : f32
//     %x   = mulf(%f, s_acc) : f32                          // dequant
//     %neg = negf(%x); %e = exp(%neg); %p = addf(%e, 1)
//     %sig = divf(1, %p)                                    // sigmoid(x)
//     %silu= mulf(%x, %sig)                                 // SiLU
//     %add = addf(%resid, %silu)                            // residual + silu
//     %r   = divf(%add, s_out); %ro=roundeven(%r); %z=addf(%ro,zp)
//     %cl  = min(max(%z,-128),127); %q = fptosi(%cl) : i8
//     yield %q
//
// Lowering — 4 QNN ops, all on `quant.uniform<i8/i32>`-typed tensors so the
// graph is all-int (HTA-compatible):
//   1. ElementWiseBinary(Add)(acc, bias)        → quant.uniform<i8, s_x>
//   2. ElementWiseNeuron(Sigmoid)               → quant.uniform<i8, s_sig>
//   3. ElementWiseBinary(Mul)(x, sig)           → quant.uniform<i8, s_silu>
//   4. ElementWiseBinary(Add)(silu, resid)      → quant.uniform<i8, s_out>
//
// The residual input arrives as f32 in the body (its source is an upstream
// dequantize); we wrap it as quant.uniform<i8> via unrealized_conversion_cast
// — runtime: the upstream produces i8 and the bridge is a no-op. Placeholder
// scales (1.0) are used for the QNN tensor types; for profiling purposes
// (the user's stated goal) compile-feasibility is what matters. A follow-up
// can plumb real scales via merlin.qnn_scale attrs.
struct LowerConvBiasSiluResidualQuantize : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isInteger(8))
			return failure();

		// Identify operands by element type:
		//   one rank-1 i32 (bias), one rank-N i32 (conv acc), one rank-N f32 or
		//   i8 (residual).
		int biasIdx = -1, accIdx = -1, residIdx = -1;
		for (int i = 0; i < 3; ++i) {
			auto t = dyn_cast<RankedTensorType>(op.getDpsInputs()[i].getType());
			if (!t)
				return failure();
			Type elt = t.getElementType();
			if (elt.isInteger(32)) {
				if (t.getRank() == 1)
					biasIdx = i;
				else if (t.getRank() == outTy.getRank())
					accIdx = i;
			} else if ((elt.isF32() || elt.isInteger(8)) &&
				t.getRank() == outTy.getRank()) {
				residIdx = i;
			}
		}
		if (biasIdx < 0 || accIdx < 0 || residIdx < 0)
			return failure();

		Block &body = op.getRegion().front();
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();

		auto fConst = [](Value v, double *out) {
			auto c = v.getDefiningOp<arith::ConstantOp>();
			if (!c)
				return false;
			auto fa = dyn_cast<FloatAttr>(c.getValue());
			if (!fa)
				return false;
			*out = fa.getValueAsDouble();
			return true;
		};
		auto isFEq = [&](Value v, double t) {
			double d;
			return fConst(v, &d) && d == t;
		};
		auto isMathOp = [](Operation *op, llvm::StringRef suffix) {
			return op && op->getDialect() &&
				op->getDialect()->getNamespace() == "math" &&
				op->getName().getStringRef().ends_with(suffix);
		};

		// Walk back: yield → fptosi → [min] → [max] → addf(_, zp) → roundeven →
		// divf(_, s_out) → addf(resid, silu).
		Value cur = yield.getOperand(0);
		auto fptosi = cur.getDefiningOp<arith::FPToSIOp>();
		if (!fptosi)
			return failure();
		cur = fptosi.getOperand();
		if (auto minf = cur.getDefiningOp<arith::MinimumFOp>()) {
			if (isFEq(minf.getRhs(), 127.0))
				cur = minf.getLhs();
			else if (isFEq(minf.getLhs(), 127.0))
				cur = minf.getRhs();
			else
				return failure();
		}
		if (auto maxf = cur.getDefiningOp<arith::MaximumFOp>()) {
			if (isFEq(maxf.getRhs(), -128.0))
				cur = maxf.getLhs();
			else if (isFEq(maxf.getLhs(), -128.0))
				cur = maxf.getRhs();
			else
				return failure();
		}
		double zpOut = 0.0;
		if (auto addf = cur.getDefiningOp<arith::AddFOp>()) {
			if (fConst(addf.getRhs(), &zpOut))
				cur = addf.getLhs();
			else if (fConst(addf.getLhs(), &zpOut))
				cur = addf.getRhs();
			// The next addf is the residual+silu add; bail if neither side
			// const. If neither — proceed without zp (zpOut=0).
		}
		if (auto roundOp = cur.getDefiningOp();
			roundOp && isMathOp(roundOp, "roundeven"))
			cur = roundOp->getOperand(0);
		double sOut = 1.0;
		if (auto divf = cur.getDefiningOp<arith::DivFOp>()) {
			if (!fConst(divf.getRhs(), &sOut))
				return failure();
			cur = divf.getLhs();
		} else {
			return failure();
		}
		// cur = addf(resid_blockarg, silu_chain)
		auto residAddf = cur.getDefiningOp<arith::AddFOp>();
		if (!residAddf)
			return failure();
		Value residOperand, siluOperand;
		auto isResidBlockArg = [&](Value v) {
			auto barg = dyn_cast<BlockArgument>(v);
			return barg && barg.getArgNumber() == (unsigned)residIdx;
		};
		if (isResidBlockArg(residAddf.getLhs())) {
			residOperand = residAddf.getLhs();
			siluOperand = residAddf.getRhs();
		} else if (isResidBlockArg(residAddf.getRhs())) {
			residOperand = residAddf.getRhs();
			siluOperand = residAddf.getLhs();
		} else {
			return failure();
		}

		// siluOperand = mulf(x, sigmoid(x))
		auto siluMul = siluOperand.getDefiningOp<arith::MulFOp>();
		if (!siluMul)
			return failure();
		auto peelSig = [&](Value v) -> Value {
			auto divf = v.getDefiningOp<arith::DivFOp>();
			if (!divf || !isFEq(divf.getLhs(), 1.0))
				return Value();
			auto addf = divf.getRhs().getDefiningOp<arith::AddFOp>();
			if (!addf)
				return Value();
			Value other;
			if (isFEq(addf.getRhs(), 1.0))
				other = addf.getLhs();
			else if (isFEq(addf.getLhs(), 1.0))
				other = addf.getRhs();
			else
				return Value();
			auto *expOp = other.getDefiningOp();
			if (!isMathOp(expOp, "exp"))
				return Value();
			auto negf = expOp->getOperand(0).getDefiningOp<arith::NegFOp>();
			if (!negf)
				return Value();
			return negf.getOperand();
		};
		Value xVal;
		if (Value sigIn = peelSig(siluMul.getRhs());
			sigIn && sigIn == siluMul.getLhs()) {
			xVal = siluMul.getLhs();
		} else if (Value sigIn = peelSig(siluMul.getLhs());
				   sigIn && sigIn == siluMul.getRhs()) {
			xVal = siluMul.getRhs();
		} else {
			return failure();
		}
		// xVal = mulf(sitofp(addi(acc_arg, bias_arg)), s_acc)
		auto xMulf = xVal.getDefiningOp<arith::MulFOp>();
		if (!xMulf)
			return failure();
		double sAcc = 1.0;
		Value xCvt;
		if (fConst(xMulf.getRhs(), &sAcc))
			xCvt = xMulf.getLhs();
		else if (fConst(xMulf.getLhs(), &sAcc))
			xCvt = xMulf.getRhs();
		else
			return failure();
		auto sitofp = xCvt.getDefiningOp<arith::SIToFPOp>();
		if (!sitofp)
			return failure();
		auto addi = sitofp.getOperand().getDefiningOp<arith::AddIOp>();
		if (!addi)
			return failure();
		auto isBArg = [](Value v) { return isa<BlockArgument>(v); };
		if (!isBArg(addi.getLhs()) || !isBArg(addi.getRhs()))
			return failure();

		// Bias is rank-1, broadcast on one axis of output.
		auto maps = op.getIndexingMapsArray();
		auto biasMap = maps[biasIdx];
		if (biasMap.getNumResults() != 1)
			return failure();
		auto biasDim = dyn_cast<AffineDimExpr>(biasMap.getResult(0));
		if (!biasDim)
			return failure();
		unsigned biasIter = biasDim.getPosition();
		auto outMap = maps[op.getNumDpsInputs()];
		int biasAxisInOutput = -1;
		for (unsigned i = 0; i < outMap.getNumResults(); ++i) {
			if (auto d = dyn_cast<AffineDimExpr>(outMap.getResult(i));
				d && d.getPosition() == biasIter) {
				biasAxisInOutput = static_cast<int>(i);
				break;
			}
		}
		if (biasAxisInOutput < 0)
			return failure();

		// ---- Lower ----
		Builder b(rewriter.getContext());
		Location loc = op.getLoc();
		Type f32Ty = b.getF32Type();
		auto i32Signed = b.getIntegerType(32, /*isSigned=*/true);
		auto i8Signed = b.getIntegerType(8, /*isSigned=*/true);
		auto qI32 = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i32Signed, f32Ty,
			/*scale=*/1.0, /*zp=*/0, INT32_MIN, INT32_MAX);
		auto qI8X = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8Signed, f32Ty,
			/*scale=*/sAcc, /*zp=*/0, -128, 127);
		auto qI8Sig = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8Signed, f32Ty,
			/*scale=*/1.0 / 128.0, /*zp=*/0, -128, 127);
		auto qI8Out = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8Signed, f32Ty,
			/*scale=*/sOut, /*zp=*/static_cast<int64_t>(zpOut), -128, 127);

		Value accV = op.getDpsInputs()[accIdx];
		Value biasV = op.getDpsInputs()[biasIdx];
		auto qXTy = RankedTensorType::get(outTy.getShape(), qI8X);

		// HTA fast path (matches the 2-input pattern): if `accV` traces back to
		// a `qnn.conv2d` without a bias (possibly through a
		// tensor.collapse_shape that erases the N=1 batch), absorb our rank-1
		// bias into the conv2d's bias operand and switch its output to
		// `quant.uniform<i8, sAcc>`. This eliminates Step 1's i32+i32
		// ElementWiseAdd which HTA's op-package validator rejects with rc=6000.
		// The chain after Step 1 is unchanged (i8 SiLU + i8 residual-add are
		// both HTA-accepted).
		Operation *collapseOp = nullptr;
		Conv2dOp absorbConv = nullptr;
		{
			Value walk = accV;
			if (auto cast = walk.getDefiningOp<UnrealizedConversionCastOp>()) {
				if (cast.getInputs().size() == 1)
					walk = cast.getInputs()[0];
			}
			if (auto collapse = walk.getDefiningOp<tensor::CollapseShapeOp>()) {
				collapseOp = collapse;
				walk = collapse.getSrc();
			}
			if (auto conv = walk.getDefiningOp<Conv2dOp>()) {
				if (!conv.getBias())
					absorbConv = conv;
			}
		}
		// If accV's producer is still a linalg.generic (the matmul-conv has not
		// been lowered yet), defer so the conv pattern can fire first. Greedy
		// driver re-tries after IR changes.
		if (!absorbConv && accV.getDefiningOp<linalg::GenericOp>()) {
			return failure();
		}

		Value xQ;
		if (absorbConv) {
			auto qInputI8Ty = quant::UniformQuantizedType::get(
				quant::QuantizationFlags::Signed, i8Signed, f32Ty,
				/*scale=*/0.0625, /*zp=*/0, -128, 127);
			auto qWeightI8Ty = quant::UniformQuantizedType::get(
				quant::QuantizationFlags::Signed, i8Signed, f32Ty,
				/*scale=*/0.0625, /*zp=*/0, -128, 127);
			auto qIn32Ty = quant::UniformQuantizedType::get(
				quant::QuantizationFlags::Signed, i32Signed, f32Ty,
				/*scale=*/0.00390625, /*zp=*/0, INT32_MIN, INT32_MAX);
			auto convTy =
				cast<RankedTensorType>(absorbConv.getOutput().getType());
			auto newConvOutTy = RankedTensorType::get(convTy.getShape(), qI8X);
			// Wrap input + weight as quant.uniform<i8> so serializer emits
			// qk=1.
			Value oldInput = absorbConv.getInput();
			auto oldInputTy = cast<RankedTensorType>(oldInput.getType());
			Value qInput = oldInput;
			if (!isa<quant::QuantizedType>(oldInputTy.getElementType())) {
				auto qInputTy =
					RankedTensorType::get(oldInputTy.getShape(), qInputI8Ty);
				qInput = UnrealizedConversionCastOp::create(
					rewriter, loc, TypeRange{qInputTy}, ValueRange{oldInput})
							 .getResult(0);
			}
			Value oldWeight = absorbConv.getWeight();
			auto oldWeightTy = cast<RankedTensorType>(oldWeight.getType());
			Value qWeight = oldWeight;
			if (!isa<quant::QuantizedType>(oldWeightTy.getElementType())) {
				auto qWeightTy =
					RankedTensorType::get(oldWeightTy.getShape(), qWeightI8Ty);
				qWeight = UnrealizedConversionCastOp::create(
					rewriter, loc, TypeRange{qWeightTy}, ValueRange{oldWeight})
							  .getResult(0);
			}
			// Wrap bias as quant.uniform<i32, s>. Bias is rank-1.
			auto biasRTy = cast<RankedTensorType>(biasV.getType());
			auto qBiasRankTy =
				RankedTensorType::get(biasRTy.getShape(), qIn32Ty);
			Value qBias = UnrealizedConversionCastOp::create(
				rewriter, loc, TypeRange{qBiasRankTy}, ValueRange{biasV})
							  .getResult(0);
			auto newConv = Conv2dOp::create(rewriter, absorbConv.getLoc(),
				newConvOutTy, qInput, qWeight, qBias,
				absorbConv.getStrideAttr(), absorbConv.getPadAmountAttr(),
				absorbConv.getDilationAttr(), absorbConv.getGroupAttr());
			Value newConvOut = newConv.getOutput();
			if (collapseOp) {
				auto oldCollapse = cast<tensor::CollapseShapeOp>(collapseOp);
				auto newCollapseTy = RankedTensorType::get(
					cast<RankedTensorType>(oldCollapse.getResult().getType())
						.getShape(),
					qI8X);
				xQ = tensor::CollapseShapeOp::create(rewriter,
					oldCollapse.getLoc(), newCollapseTy, newConvOut,
					oldCollapse.getReassociationIndices())
						 .getResult();
				rewriter.eraseOp(collapseOp);
			} else {
				xQ = newConvOut;
			}
			rewriter.eraseOp(absorbConv);
		} else {
			// Fallback: original 4-op chain (will fail HTA but works on CPU/GPU
			// for testing).
			auto accTy = cast<RankedTensorType>(accV.getType());
			auto qAccTy = RankedTensorType::get(accTy.getShape(), qI32);
			Value qAcc = UnrealizedConversionCastOp::create(
				rewriter, loc, TypeRange{qAccTy}, ValueRange{accV})
							 .getResult(0);
			SmallVector<int64_t> biasShape(outTy.getRank(), 1);
			biasShape[biasAxisInOutput] = outTy.getShape()[biasAxisInOutput];
			auto biasExpTy =
				RankedTensorType::get(biasShape, b.getIntegerType(32));
			SmallVector<ReassociationIndices> reassoc(1);
			for (int i = 0; i < (int)outTy.getRank(); ++i)
				reassoc[0].push_back(i);
			Value biasExpanded = tensor::ExpandShapeOp::create(
				rewriter, loc, biasExpTy, biasV, reassoc);
			auto qBiasTy = RankedTensorType::get(biasShape, qI32);
			Value qBias = UnrealizedConversionCastOp::create(
				rewriter, loc, TypeRange{qBiasTy}, ValueRange{biasExpanded})
							  .getResult(0);
			xQ = ElementWiseBinaryOp::create(rewriter, loc, qXTy, qAcc, qBias,
				b.getI32IntegerAttr(kBinaryAdd))
					 .getOutput();
		}

		// Step 2: Sigmoid(x) → quant.i8.
		auto qSigTy = RankedTensorType::get(outTy.getShape(), qI8Sig);
		Value sigQ = ElementWiseNeuronOp::create(
			rewriter, loc, qSigTy, xQ, b.getI32IntegerAttr(/*Sigmoid=*/6))
						 .getOutput();

		// Step 3: Mul(x, sigmoid(x)) = SiLU → quant.i8.
		auto qSiluTy = qXTy; // SiLU keeps roughly the same scale space.
		Value siluQ = ElementWiseBinaryOp::create(
			rewriter, loc, qSiluTy, xQ, sigQ, b.getI32IntegerAttr(kBinaryMul))
						  .getOutput();

		// Step 4: Add(resid, silu) → quant.i8 with final output scale.
		// The residual comes in as f32 (from an upstream dequantize generic).
		// To keep the graph all-int (HTA-compatible), trace back through the
		// upstream dequantize and use its i8 input directly — that leaves the
		// dequantize generic orphan, which DCE / residual-check ignores.
		Value residV = op.getDpsInputs()[residIdx];
		if (auto residTy0 = dyn_cast<RankedTensorType>(residV.getType());
			residTy0 && residTy0.getElementType().isF32()) {
			// Walk back: residV may be the result of a 1-input dequant generic
			// with body `sitofp → mulf(s)`.
			if (auto residProducer = residV.getDefiningOp<linalg::GenericOp>();
				residProducer && residProducer.getNumDpsInputs() == 1 &&
				residProducer.getNumDpsInits() == 1) {
				auto producerInTy = dyn_cast<RankedTensorType>(
					residProducer.getDpsInputs()[0].getType());
				if (producerInTy &&
					producerInTy.getElementType().isInteger(8)) {
					// Cheap sanity check: body's single yield is from
					// mulf(sitofp(bargin)).
					Block &pbody = residProducer.getRegion().front();
					auto py = dyn_cast<linalg::YieldOp>(pbody.getTerminator());
					if (py && py.getNumOperands() == 1) {
						if (auto pm = py.getOperand(0)
										  .getDefiningOp<arith::MulFOp>()) {
							auto isFC = [](Value v) {
								auto c = v.getDefiningOp<arith::ConstantOp>();
								return c && isa<FloatAttr>(c.getValue());
							};
							Value pcvt;
							if (isFC(pm.getRhs()))
								pcvt = pm.getLhs();
							else if (isFC(pm.getLhs()))
								pcvt = pm.getRhs();
							if (pcvt) {
								if (auto psf =
										pcvt.getDefiningOp<arith::SIToFPOp>();
									psf &&
									isa<BlockArgument>(psf.getOperand())) {
									// Bypass dequantize: use the producer's i8
									// input.
									residV = residProducer.getDpsInputs()[0];
								}
							}
						}
					}
				}
			}
		}
		auto residTy = cast<RankedTensorType>(residV.getType());
		auto qResidTy = RankedTensorType::get(residTy.getShape(), qI8X);
		Value qResid = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{qResidTy}, ValueRange{residV})
						   .getResult(0);

		auto qOutTy = RankedTensorType::get(outTy.getShape(), qI8Out);
		Value outQ = ElementWiseBinaryOp::create(rewriter, loc, qOutTy, qResid,
			siluQ, b.getI32IntegerAttr(kBinaryAdd))
						 .getOutput();

		// Bridge back to plain i8 storage.
		Value result = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{outTy}, ValueRange{outQ})
						   .getResult(0);
		rewriter.replaceOp(op, result);
		return success();
	}
};

// Quantize: fp → int8.
struct LowerQuantize : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!linalg::isElementwise(op))
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy.getElementType().isInteger(8))
			return failure();
		auto inTy = cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		if (!inTy.getElementType().isF32())
			return failure();

		double scale = 1.0;
		int64_t zp = 0;
		if (!matchQuantizeBody(op.getRegion().front(), &scale, &zp))
			return failure();

		Builder b(rewriter.getContext());
		Location loc = op.getLoc();

		// Rank-4 promote.
		auto [in4, origRank] =
			promoteToRank4(rewriter, loc, op.getDpsInputs()[0]);
		(void)origRank;
		auto in4Ty = cast<RankedTensorType>(in4.getType());

		// Output type wraps i8 storage with quant.uniform<i8:f32, scale, zp>
		// so SerializeGraph emits qk=1 on the qnn.quantize result tensor.
		auto i8Ty = b.getIntegerType(8, /*isSigned=*/true);
		auto f32Ty = b.getF32Type();
		auto qOutElemTy = quant::UniformQuantizedType::get(
			quant::QuantizationFlags::Signed, i8Ty, f32Ty,
			/*scale=*/scale, /*zeroPoint=*/zp,
			/*storageMin=*/-128, /*storageMax=*/127);
		auto q4OutTy = RankedTensorType::get(in4Ty.getShape(), qOutElemTy);
		Value qnnQ =
			QuantizeOp::create(rewriter, loc, q4OutTy, in4).getOutput();
		// Bitcast back to plain i8 storage at rank-4, then collapse to original
		// rank for downstream uses (whose binding type is plain i8).
		auto plainI8R4 = RankedTensorType::get(in4Ty.getShape(), b.getI8Type());
		Value plainQ = UnrealizedConversionCastOp::create(
			rewriter, loc, TypeRange{plainI8R4}, ValueRange{qnnQ})
						   .getResult(0);
		Value collapsed = collapseFromRank4(rewriter, loc, plainQ, outTy);
		rewriter.replaceOp(op, collapsed);
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Tensor-shape ops (already in tree from earlier scaffold).
//===----------------------------------------------------------------------===//

struct LowerTensorConcat : OpRewritePattern<tensor::ConcatOp> {
	using OpRewritePattern<tensor::ConcatOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		tensor::ConcatOp op, PatternRewriter &rewriter) const override {
		Builder b(rewriter.getContext());
		auto axisAttr = b.getI32IntegerAttr(static_cast<int32_t>(op.getDim()));
		auto qnnConcat = ConcatOp::create(rewriter, op.getLoc(),
			op.getResult().getType(), op.getInputs(), axisAttr);
		rewriter.replaceOp(op, qnnConcat.getOutput());
		return success();
	}
};

struct LowerTensorCollapseShape : OpRewritePattern<tensor::CollapseShapeOp> {
	using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		tensor::CollapseShapeOp op, PatternRewriter &rewriter) const override {
		auto qnnReshape = ReshapeOp::create(
			rewriter, op.getLoc(), op.getResult().getType(), op.getSrc());
		rewriter.replaceOp(op, qnnReshape.getOutput());
		return success();
	}
};

struct LowerTensorExpandShape : OpRewritePattern<tensor::ExpandShapeOp> {
	using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		tensor::ExpandShapeOp op, PatternRewriter &rewriter) const override {
		auto qnnReshape = ReshapeOp::create(
			rewriter, op.getLoc(), op.getResult().getType(), op.getSrc());
		rewriter.replaceOp(op, qnnReshape.getOutput());
		return success();
	}
};

struct LowerLinalgTranspose : OpRewritePattern<linalg::TransposeOp> {
	using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::TransposeOp op, PatternRewriter &rewriter) const override {
		Builder b(rewriter.getContext());
		SmallVector<int32_t> perm;
		for (int64_t p : op.getPermutation()) {
			perm.push_back(static_cast<int32_t>(p));
		}
		auto qnnT = TransposeOp::create(rewriter, op.getLoc(),
			op.getResult()[0].getType(), op.getInput(),
			b.getI32ArrayAttr(perm));
		rewriter.replaceOp(op, qnnT.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Phase 4b: yolov8 coverage patterns.
//===----------------------------------------------------------------------===//

// Generic-form transpose: `linalg.generic` with non-identity output map
// and pure-pass-through body (single block-arg yielded). Matches the
// post-fold form yolov8 produces for transposes.
struct LowerGenericTranspose : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();
		if (!linalg::isElementwise(op))
			return failure();
		Block &body = op.getRegion().front();
		auto *yield = body.getTerminator();
		if (!yield || yield->getNumOperands() != 1)
			return failure();
		auto blockArg = dyn_cast<BlockArgument>(yield->getOperand(0));
		if (!blockArg || blockArg.getOwner() != &body ||
			blockArg.getArgNumber() != 0)
			return failure();
		auto inMap = op.getIndexingMapsArray()[0];
		auto outMap = op.getIndexingMapsArray()[1];
		if (!inMap.isProjectedPermutation() || !outMap.isProjectedPermutation())
			return failure();
		if (inMap == outMap)
			return failure(); // identity, not a transpose
		// Compose: arg_perm = inMap ∘ outMap^-1 — gives the perm vector.
		auto invOut = inversePermutation(outMap);
		if (!invOut)
			return failure();
		auto composed = inMap.compose(invOut);
		if (!composed.isPermutation())
			return failure();
		SmallVector<int32_t> perm;
		for (auto e : composed.getResults()) {
			auto d = dyn_cast<AffineDimExpr>(e);
			if (!d)
				return failure();
			perm.push_back(static_cast<int32_t>(d.getPosition()));
		}
		Builder b(rewriter.getContext());
		auto qnnT = TransposeOp::create(rewriter, op.getLoc(),
			op.getResult(0).getType(), op.getDpsInputs()[0],
			b.getI32ArrayAttr(perm));
		rewriter.replaceOp(op, qnnT.getOutput());
		return success();
	}
};

// im2col-style matmul: a `linalg.generic` with N+1 parallel iters and 1
// reduction, ins[0] is rank > 2 (im2col output), ins[1] is rank-2 weight,
// out is same rank as ins[0] (with last dim = N). Emit
// `tensor.collapse_shape` on lhs/output to flatten leading dims into a
// single M, then `qnn.matmul`, then `tensor.expand_shape` back.
struct LowerIm2ColMatmul : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();
		Value lhs = op.getDpsInputs()[0];
		Value rhs = op.getDpsInputs()[1];
		auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
		auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
		auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!lhsTy || !rhsTy || !outTy)
			return failure();
		if (rhsTy.getRank() != 2 || lhsTy.getRank() < 3 || outTy.getRank() < 3)
			return failure();
		if (lhsTy.getRank() != outTy.getRank())
			return failure();
		// Body: extsi (× 2) + muli + addi (quant matmul) — same shape as
		// matchQuantConvBody's folded path. Or just plain mulf+addf for fp.
		int64_t inZp = 0, wZp = 0;
		bool isQuant =
			matchQuantConvBody(op.getRegion().front(), &inZp, &wZp) &&
			inZp == 0 && wZp == 0;
		if (!isQuant) {
			// Try fp matmul body (mulf+addf).
			Block &body = op.getRegion().front();
			auto *yield = body.getTerminator();
			if (!yield || yield->getNumOperands() != 1)
				return failure();
			auto *acc = yield->getOperand(0).getDefiningOp();
			if (!isa_and_nonnull<arith::AddFOp>(acc))
				return failure();
			bool sawMul = false;
			for (Value v : acc->getOperands()) {
				if (auto *m = v.getDefiningOp(); m && isa<arith::MulFOp>(m)) {
					sawMul = true;
					break;
				}
			}
			if (!sawMul)
				return failure();
		}
		// Lhs/output's last dim is K (matches rhs[0]); leading dims fold to M.
		if (lhsTy.getShape().back() != rhsTy.getShape()[0])
			return failure();
		if (outTy.getShape().back() != rhsTy.getShape()[1])
			return failure();

		Location loc = op.getLoc();
		int64_t lhsRank = lhsTy.getRank();
		SmallVector<ReassociationIndices> lhsReassoc;
		ReassociationIndices mIdx;
		for (int64_t i = 0; i < lhsRank - 1; ++i)
			mIdx.push_back(i);
		lhsReassoc.push_back(mIdx);
		lhsReassoc.push_back(ReassociationIndices{lhsRank - 1});
		int64_t M = 1;
		for (int64_t i = 0; i < lhsRank - 1; ++i)
			M *= lhsTy.getShape()[i];
		int64_t K = lhsTy.getShape()[lhsRank - 1];
		int64_t N = rhsTy.getShape()[1];
		auto lhs2dTy = RankedTensorType::get({M, K}, lhsTy.getElementType());
		Value lhs2d = tensor::CollapseShapeOp::create(
			rewriter, loc, lhs2dTy, lhs, lhsReassoc);
		auto out2dTy = RankedTensorType::get({M, N}, outTy.getElementType());
		Builder b(rewriter.getContext());
		Value mm = MatMulOp::create(rewriter, loc, out2dTy, lhs2d, rhs,
			/*transpose_lhs=*/b.getBoolAttr(false),
			/*transpose_rhs=*/b.getBoolAttr(false))
					   .getOutput();
		// Expand back.
		SmallVector<ReassociationIndices> outReassoc;
		ReassociationIndices mOutIdx;
		for (int64_t i = 0; i < outTy.getRank() - 1; ++i)
			mOutIdx.push_back(i);
		outReassoc.push_back(mOutIdx);
		outReassoc.push_back(ReassociationIndices{outTy.getRank() - 1});
		Value expanded =
			tensor::ExpandShapeOp::create(rewriter, loc, outTy, mm, outReassoc);
		rewriter.replaceOp(op, expanded);
		return success();
	}
};

// tensor.pad → qnn.pad. Supports CONSTANT padding only; mirror/edge are
// follow-ups. Padding values must be static.
struct LowerTensorPad : OpRewritePattern<tensor::PadOp> {
	using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(
		tensor::PadOp op, PatternRewriter &rewriter) const override {
		if (!op.getStaticLow().size() || !op.getStaticHigh().size())
			return failure();
		auto outTy = cast<RankedTensorType>(op.getResult().getType());
		int64_t rank = outTy.getRank();
		SmallVector<int32_t> padArr;
		padArr.reserve(2 * rank);
		for (int64_t i = 0; i < rank; ++i) {
			int64_t lo = op.getStaticLow()[i];
			int64_t hi = op.getStaticHigh()[i];
			if (lo == ShapedType::kDynamic || hi == ShapedType::kDynamic)
				return failure();
			padArr.push_back(static_cast<int32_t>(lo));
			padArr.push_back(static_cast<int32_t>(hi));
		}
		// Read constant padding value from the body. For tensor.pad, the body
		// has a single yield of the constant.
		Block &body = op.getRegion().front();
		auto *yield = body.getTerminator();
		if (!yield || yield->getNumOperands() != 1)
			return failure();
		float padConst = 0.0f;
		if (auto cstOp =
				yield->getOperand(0).getDefiningOp<arith::ConstantOp>()) {
			if (auto fAttr = dyn_cast<FloatAttr>(cstOp.getValue())) {
				padConst = fAttr.getValue().convertToFloat();
			} else if (auto iAttr = dyn_cast<IntegerAttr>(cstOp.getValue())) {
				padConst = static_cast<float>(iAttr.getInt());
			}
		}
		Builder b(rewriter.getContext());
		auto qnnPad = PadOp::create(rewriter, op.getLoc(), outTy,
			op.getSource(), b.getI32ArrayAttr(padArr),
			b.getI32IntegerAttr(/*scheme=CONSTANT*/ 0),
			b.getF32FloatAttr(padConst));
		rewriter.replaceOp(op, qnnPad.getOutput());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertLinalgToQNNPass
	: public PassWrapper<ConvertLinalgToQNNPass, OperationPass<>> {
	StringRef getArgument() const final {
		return "merlin-convert-linalg-to-qnn";
	}
	StringRef getDescription() const final {
		return "Pattern-match recognized linalg ops and rewrite to qnn.* ops "
			   "(post-global-optimization). Replaces the Python recognizers in "
			   "kernels/qnn/recognizers/.";
	}
	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<QNNDialect, linalg::LinalgDialect,
			tensor::TensorDialect, arith::ArithDialect, quant::QuantDialect>();
	}
	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		// Conv/Pool/MatMul patterns: structured ops first. Conv must beat
		// ElementWiseNeuron (benefit=2 below) so the conv-rescale fusion path
		// sees the rescale generic before it gets standalone-converted.
		patterns.add<LowerConv2dQGeneric>(&getContext(), /*benefit=*/3);
		patterns.add<LowerConv2dFp, LowerDepthwiseConv2dQGeneric,
			LowerNhwcMaxPool, LowerNhwcSumPool, LowerMatMul,
			LowerMatMulTransposeB, LowerContractionGeneric>(&getContext());
		// Dequantize re-enabled 2026-05-09 for boundary-op coverage on
		// QNN_GPU: standalone qnn.dequantize verified on board (mean 2.9 ms
		// for 1x3x320x320xi8→f32). Quantize remains disabled — QNN_GPU's
		// graphFinalize still rejects it (rc=6022), QNN_HTA cannot accept
		// it by spec (all-int8-graph constraint).
		patterns.add<LowerBinaryQuantize>(&getContext(), /*benefit=*/2);
		patterns.add<LowerDequantize, LowerQuantize>(&getContext());
		// 2-input dequant-with-bias (yolov8 stem f32-out tail) — runs at higher
		// benefit than the 1-input LowerDequantize so it wins when both shapes
		// are present.
		patterns.add<LowerDequantizeWithBias>(&getContext(), /*benefit=*/2);
		// 2-input rescale-and-requantize-to-i8 with bias (yolov8 int8 conv-tail
		// dispatches). Highest benefit so it wins over LowerDequantizeWithBias
		// when the output type is i8.
		// benefit=1 so LowerConv2dQGeneric (benefit=3) fires FIRST. Then on
		// the next iteration this pattern sees its main input as a qnn.conv2d
		// (or a collapse_shape of one) and can absorb the bias into the
		// conv2d's bias operand instead of emitting a separate i32
		// ElementWiseAdd that HTA refuses.
		patterns.add<LowerRescaleQuantizeWithBias>(
			&getContext(), /*benefit=*/1);
		// 3-input residual+SiLU+quantize body (yolov8n int8 backbone residual
		// blocks). Highest benefit overall.
		patterns.add<LowerConvBiasSiluResidualQuantize>(&getContext(),
			/*benefit=*/4);
		patterns.add<LowerFp32BiasSiLU>(&getContext(), /*benefit=*/3);
		patterns.add<LowerReduceSumGeneric>(&getContext(), /*benefit=*/2);
		patterns.add<LowerFp32ResidualBiasSiLU>(&getContext(), /*benefit=*/4);
		patterns.add<EraseDeadLinalgFill>(&getContext());
		patterns.add<LowerFp32BiasAdd>(&getContext(), /*benefit=*/2);
		// fp32 multi-N matmul-like (yolov8 1×1 conv on CHW activations).
		patterns.add<LowerSpatialMatmulFp>(&getContext(), /*benefit=*/3);
		// fp32 NCHW max-pool from generalized 5-iter linalg.generic (yolov8
		// SPPF).
		patterns.add<LowerMaxPoolGeneric>(&getContext(), /*benefit=*/3);
		// Activation before binary so a single-input neuron beats a binary
		// misclassification.
		patterns.add<LowerElementWiseNeuron>(&getContext(), /*benefit=*/2);
		// Standalone sigmoid body (`1/(1+exp(-x))`) over fp32 — yolov8
		// detect-head form that ElementWiseNeuron's named-op matcher doesn't
		// pick up.
		patterns.add<LowerStandaloneSigmoid>(&getContext(), /*benefit=*/3);
		patterns.add<LowerElementWiseBinary>(&getContext());
		// Tensor-shape lowerings — only the explicit model-level ones.
		// tensor.collapse_shape / tensor.expand_shape are NOT auto-converted:
		// IREE generates them internally for layout transforms and turning
		// them into qnn.reshape leaks QNN ops into IR that downstream
		// dispatch creation expects to be plain tensor ops.
		patterns.add<LowerTensorConcat, LowerLinalgTranspose>(&getContext());
		// Phase 4b: yolov8 coverage. Generic-transpose at lower benefit than
		// ElementWiseNeuron (so single-input pointwise activations beat
		// transpose-misclassification) but higher than ElementWiseBinary.
		patterns.add<LowerGenericTranspose>(&getContext(), /*benefit=*/1);
		patterns.add<LowerTensorPad>(&getContext());
		// im2col-matmul takes priority over the conv-q pattern (which won't
		// match anyway since im2col already happened) and over the standalone
		// contraction pattern (which requires 2D-only).
		patterns.add<LowerIm2ColMatmul>(&getContext(), /*benefit=*/4);
		// LowerDequantize is now registered above with LowerQuantize.

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createConvertLinalgToQNNPass() {
	return std::make_unique<ConvertLinalgToQNNPass>();
}

} // namespace mlir::iree_compiler::QNN
