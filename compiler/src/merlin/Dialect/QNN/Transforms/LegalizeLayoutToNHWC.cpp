// NHWC layout legalization for the QNN dialect.
//
// Inserts `linalg.transpose` ops on the inputs/outputs of any
// `linalg.conv_2d_nchw_fchw{,_q}` and `linalg.pooling_nchw_max` so subsequent
// linalg→qnn conversion sees the NHWC form QNN's verifier requires.
// Canonicalization folds cancelling transposes through tensor.pad /
// linalg.broadcast / linalg.generic boundaries when shapes line up.
//
// Replaces the upstream IREE `iree-preprocessing-convert-conv-to-channels-last`
// pass which produces invalid IR for `linalg.conv_2d_nchw_fchw_q` (i32
// zero-points end up in shape-typed slots; verifier rejects).

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::QNN {
namespace {

// Insert a `linalg.transpose` from NCHW (N=0, C=1, H=2, W=3) to NHWC
// (N=0, H=1, W=2, C=3). Returns the transposed value.
static Value insertNCHWtoNHWC(Location loc, Value v, OpBuilder &b) {
	auto srcType = cast<RankedTensorType>(v.getType());
	assert(srcType.getRank() == 4 && "expected rank-4 NCHW tensor");
	ArrayRef<int64_t> srcShape = srcType.getShape();
	SmallVector<int64_t, 4> dstShape{
		srcShape[0], srcShape[2], srcShape[3], srcShape[1]};
	Value empty =
		tensor::EmptyOp::create(b, loc, dstShape, srcType.getElementType());
	SmallVector<int64_t, 4> perm{0, 2, 3, 1};
	auto t = linalg::TransposeOp::create(b, loc, v, empty, perm);
	return t.getResult()[0];
}

// Reverse: NHWC -> NCHW.
static Value insertNHWCtoNCHW(Location loc, Value v, OpBuilder &b) {
	auto srcType = cast<RankedTensorType>(v.getType());
	assert(srcType.getRank() == 4 && "expected rank-4 NHWC tensor");
	ArrayRef<int64_t> srcShape = srcType.getShape();
	SmallVector<int64_t, 4> dstShape{
		srcShape[0], srcShape[3], srcShape[1], srcShape[2]};
	Value empty =
		tensor::EmptyOp::create(b, loc, dstShape, srcType.getElementType());
	SmallVector<int64_t, 4> perm{0, 3, 1, 2};
	auto t = linalg::TransposeOp::create(b, loc, v, empty, perm);
	return t.getResult()[0];
}

// Convert FCHW weight (oc, ic, kh, kw) to HWCF (kh, kw, ic, oc) via
// linalg.transpose with perm=[2,3,1,0].
static Value insertFCHWtoHWCF(Location loc, Value v, OpBuilder &b) {
	auto srcType = cast<RankedTensorType>(v.getType());
	assert(srcType.getRank() == 4 && "expected rank-4 FCHW weight");
	ArrayRef<int64_t> srcShape = srcType.getShape();
	SmallVector<int64_t, 4> dstShape{
		srcShape[2], srcShape[3], srcShape[1], srcShape[0]};
	Value empty =
		tensor::EmptyOp::create(b, loc, dstShape, srcType.getElementType());
	SmallVector<int64_t, 4> perm{2, 3, 1, 0};
	auto t = linalg::TransposeOp::create(b, loc, v, empty, perm);
	return t.getResult()[0];
}

// Pattern: rewrite linalg.conv_2d_nchw_fchw_q to a linalg.conv_2d_nhwc_hwcf_q
// surrounded by transposes. The tensor.pad (low/high) is rewritten to use
// NHWC dim indices.
struct RewriteNCHWConvQ : OpRewritePattern<linalg::Conv2DNchwFchwQOp> {
	using OpRewritePattern<linalg::Conv2DNchwFchwQOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwQOp op,
		PatternRewriter &rewriter) const override {
		Location loc = op.getLoc();
		// ins(input, weight, in_zp, w_zp); outs(init).
		Value input = op.getInputs()[0];
		Value weight = op.getInputs()[1];
		Value inZp = op.getInputs()[2];
		Value wZp = op.getInputs()[3];
		Value init = op.getOutputs()[0];

		// Insert NCHW→NHWC on input + init (init is rank-4 NCHW too).
		Value inputNHWC = insertNCHWtoNHWC(loc, input, rewriter);
		Value weightHWCF = insertFCHWtoHWCF(loc, weight, rewriter);
		Value initNHWC = insertNCHWtoNHWC(loc, init, rewriter);

		// Build the NHWC quantized conv.
		auto strides = op.getStrides();
		auto dilations = op.getDilations();
		Value convNHWC = linalg::Conv2DNhwcHwcfQOp::create(rewriter, loc,
			initNHWC.getType(), ValueRange{inputNHWC, weightHWCF, inZp, wZp},
			ValueRange{initNHWC}, strides, dilations)
							 .getResult(0);

		// Transpose result back to NCHW for downstream consumers.
		Value convNCHW = insertNHWCtoNCHW(loc, convNHWC, rewriter);
		rewriter.replaceOp(op, convNCHW);
		return success();
	}
};

// Generic-form NCHW conv: matches `linalg.generic` ops whose convolution
// shape (per linalg::isaConvolutionOpInterface) places the output channel
// dim BEFORE the output spatial dims. Emits transposes around the op and
// rewrites its indexing maps to the NHWC form. Handles both rank-4
// ([N,C,H,W]) and rank-3 ([C,H,W] — yolov8's batch-stripped form) shapes.
//
// Strategy: do NOT rebuild the linalg.generic from scratch. Instead, swap
// the indexing maps in-place so the same body produces the same result
// once the inputs/outputs are in NHWC form. The IREE rewriter handles the
// SSA wiring after we replace the op's result with a NHWC->NCHW transpose
// of the rewritten generic's result.
struct RewriteNCHWGenericConv : OpRewritePattern<linalg::GenericOp> {
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
		if (dims.outputImage.size() != 2 || dims.filterLoop.size() != 2)
			return failure();
		if (dims.outputChannel.size() != 1)
			return failure();

		// Detect NCHW: outputChannel iter dim sits BEFORE the outputImage dims.
		// (NHWC has it after.) This must be true for both the conv body and
		// the output indexing map to be consistent.
		int64_t lastImage =
			*std::max_element(dims.outputImage.begin(), dims.outputImage.end());
		int64_t cPos = dims.outputChannel[0];
		if (cPos > lastImage)
			return failure(); // already NHWC

		// Bail on cases the simple rank-3/rank-4 transpose can't handle.
		if (op.getNumDpsInputs() < 1 || op.getNumDpsInits() != 1)
			return failure();

		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		int64_t outRank = outTy.getRank();
		// Output is either [OC, oH, oW] (rank-3) or [N, OC, oH, oW] (rank-4).
		if (outRank != 3 && outRank != 4)
			return failure();

		Location loc = op.getLoc();
		Builder b(rewriter.getContext());

		// Build the perm vector that takes NCHW -> NHWC for the OUTPUT.
		// rank-3: [OC, oH, oW] -> [oH, oW, OC]   perm = [1, 2, 0]
		// rank-4: [N, OC, oH, oW] -> [N, oH, oW, OC]  perm = [0, 2, 3, 1]
		SmallVector<int64_t> outPerm;
		if (outRank == 3)
			outPerm = {1, 2, 0};
		else
			outPerm = {0, 2, 3, 1};

		// Permute the output type's shape.
		SmallVector<int64_t> outNHWCShape;
		for (int64_t p : outPerm)
			outNHWCShape.push_back(outTy.getShape()[p]);
		auto outNHWCTy =
			RankedTensorType::get(outNHWCShape, outTy.getElementType());

		// Build explicit NHWC/HWCF convolution maps. A simple channel-dim
		// permutation is insufficient for generic-form conv because transposing
		// weights from FCHW to HWCF also moves the input-channel reduction loop
		// behind the filter loops. The desired loop order is:
		//   rank-3: H, W, OC, KH, KW, IC
		//   rank-4: N, H, W, OC, KH, KW, IC
		int64_t numLoops = outRank + 3;
		if (op.getNumLoops() != numLoops || dims.strides.size() != 2 ||
			dims.dilations.size() != 2)
			return failure();

		auto d = [&](unsigned pos) { return rewriter.getAffineDimExpr(pos); };
		int64_t strideH = dims.strides[0];
		int64_t strideW = dims.strides[1];
		int64_t dilationH = dims.dilations[0];
		int64_t dilationW = dims.dilations[1];

		SmallVector<AffineMap> newMaps;
		newMaps.reserve(op.getNumDpsInputs() + op.getNumDpsInits());
		if (outRank == 3) {
			AffineExpr oh = d(0), ow = d(1), oc = d(2);
			AffineExpr kh = d(3), kw = d(4), ic = d(5);
			newMaps.push_back(AffineMap::get(numLoops, 0,
				{oh * strideH + kh * dilationH, ow * strideW + kw * dilationW,
					ic},
				rewriter.getContext()));
			newMaps.push_back(AffineMap::get(
				numLoops, 0, {kh, kw, ic, oc}, rewriter.getContext()));
			for (int64_t i = 2; i < op.getNumDpsInputs(); ++i)
				newMaps.push_back(
					AffineMap::get(numLoops, 0, {}, rewriter.getContext()));
			newMaps.push_back(AffineMap::get(
				numLoops, 0, {oh, ow, oc}, rewriter.getContext()));
		} else {
			AffineExpr n = d(0), oh = d(1), ow = d(2), oc = d(3);
			AffineExpr kh = d(4), kw = d(5), ic = d(6);
			newMaps.push_back(AffineMap::get(numLoops, 0,
				{n, oh * strideH + kh * dilationH,
					ow * strideW + kw * dilationW, ic},
				rewriter.getContext()));
			newMaps.push_back(AffineMap::get(
				numLoops, 0, {kh, kw, ic, oc}, rewriter.getContext()));
			for (int64_t i = 2; i < op.getNumDpsInputs(); ++i)
				newMaps.push_back(
					AffineMap::get(numLoops, 0, {}, rewriter.getContext()));
			newMaps.push_back(AffineMap::get(
				numLoops, 0, {n, oh, ow, oc}, rewriter.getContext()));
		}
		SmallVector<utils::IteratorType> newIters(
			outRank, utils::IteratorType::parallel);
		newIters.append(3, utils::IteratorType::reduction);

		// Transpose each TENSOR-typed input that has a non-singleton (non-
		// affine_map<()->()>) indexing map and whose shape needs the same
		// OC/spatial swap. Heuristic: if the input's tensor rank == output's
		// tensor rank AND both are >= 3, apply outPerm-style transpose. For
		// weight ops which are typically [OC, IC, KH, KW], we apply a
		// separate "fchw -> hwcf" if it's rank-4 and matches the output
		// channel layout.
		SmallVector<Value> newOperands;
		for (Value inV : op.getDpsInputs()) {
			auto rt = dyn_cast<RankedTensorType>(inV.getType());
			if (!rt) {
				// Scalar (e.g. zero-point i32 const) — pass through.
				newOperands.push_back(inV);
				continue;
			}
			if (rt.getRank() == outRank) {
				// Same rank as output: same OC/spatial swap.
				SmallVector<int64_t> permutedShape;
				for (int64_t p : outPerm)
					permutedShape.push_back(rt.getShape()[p]);
				auto permTy =
					RankedTensorType::get(permutedShape, rt.getElementType());
				Value empty = tensor::EmptyOp::create(
					rewriter, loc, permutedShape, rt.getElementType());
				Value tr = linalg::TransposeOp::create(
					rewriter, loc, inV, empty, outPerm)
							   .getResult()[0];
				newOperands.push_back(tr);
				(void)permTy;
			} else if (rt.getRank() == 4 && outRank == 3) {
				// Common case: weight is rank-4 [OC, IC, KH, KW] for a rank-3
				// [C, H, W] activation. Convert weight FCHW -> HWCF (perm
				// [2,3,1,0]) so it lines up with the NHWC body indexing.
				Value tr = insertFCHWtoHWCF(loc, inV, rewriter);
				newOperands.push_back(tr);
			} else {
				// Other shapes (rare). Leave alone; if they were broken before
				// the rewrite, they still are.
				newOperands.push_back(inV);
			}
		}

		// Transpose the init/output operand to NHWC space.
		Value initV = op.getDpsInits()[0];
		auto initTy = cast<RankedTensorType>(initV.getType());
		SmallVector<int64_t> initPermShape;
		for (int64_t p : outPerm)
			initPermShape.push_back(initTy.getShape()[p]);
		Value initEmpty = tensor::EmptyOp::create(
			rewriter, loc, initPermShape, initTy.getElementType());
		Value initNHWC = linalg::TransposeOp::create(
			rewriter, loc, initV, initEmpty, outPerm)
							 .getResult()[0];

		// Build the new linalg.generic with rewritten indexing maps + iters.
		SmallVector<Type> newResultTypes{outNHWCTy};
		SmallVector<Value> allOperands;
		allOperands.append(newOperands.begin(), newOperands.end());
		allOperands.push_back(initNHWC);

		// Split the operands list into ins / outs for the GenericOp::create
		// overload that takes them separately.
		auto newGeneric = linalg::GenericOp::create(rewriter, loc,
			newResultTypes, /*inputs=*/newOperands,
			/*outputs=*/ValueRange{initNHWC}, newMaps, newIters);
		// Move the body region from the old op to the new one (body is
		// identical: same yield computation on the same block-arg layout).
		rewriter.inlineRegionBefore(op.getRegion(), newGeneric.getRegion(),
			newGeneric.getRegion().begin());

		// Transpose result NHWC -> back to original NCHW.
		SmallVector<int64_t> invOutPerm(outPerm.size());
		for (size_t i = 0; i < outPerm.size(); ++i)
			invOutPerm[outPerm[i]] = (int64_t)i;
		SmallVector<int64_t> origShape(
			outTy.getShape().begin(), outTy.getShape().end());
		Value backEmpty = tensor::EmptyOp::create(
			rewriter, loc, origShape, outTy.getElementType());
		Value resNCHW = linalg::TransposeOp::create(
			rewriter, loc, newGeneric.getResult(0), backEmpty, invOutPerm)
							.getResult()[0];
		rewriter.replaceOp(op, resNCHW);
		return success();
	}
};

// Fallback for generalized NCHW quantized conv forms that
// `isaConvolutionOpInterface` rejects (post-dispatch-creation 3-input form
// with a scalar zero-point input has empty indexing map for the zp, which
// upstream conv detection refuses). Hand-roll the rewrite by inspecting
// indexing maps directly.
//
// Recognized shape (rank-3, no batch — yolov8 dispatch-collapsed form):
//   indexing_maps = [
//     (d0..d5) -> (d3, d1*sH + d4*dH, d2*sW + d5*dW),   // input [IC, H, W]
//     (d0..d5) -> (d0, d3, d4, d5),                      // weight
//     [OC,IC,KH,KW] (d0..d5) -> (),                                    // zp
//     scalar (opt'l) (d0..d5) -> (d0, d1, d2)]                          //
//     output [OC, H, W]
//   iters = [par, par, par, red, red, red]
// where d0=OC, d1=oH, d2=oW, d3=IC, d4=KH, d5=KW.
struct RewriteNCHWGenericConvExplicit : OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (op.getNumLoops() != 6 || op.getNumDpsInits() != 1)
			return failure();
		auto iters = op.getIteratorTypesArray();
		SmallVector<unsigned> parIters, redIters;
		for (size_t i = 0; i < iters.size(); ++i) {
			if (iters[i] == utils::IteratorType::parallel)
				parIters.push_back(i);
			else if (iters[i] == utils::IteratorType::reduction)
				redIters.push_back(i);
			else
				return failure();
		}
		if (parIters.size() != 3 || redIters.size() != 3)
			return failure();

		auto maps = op.getIndexingMapsArray();
		if (maps.size() < 3)
			return failure();
		auto outMap = maps.back();
		if (outMap.getNumResults() != 3)
			return failure();
		// Output map must be (d0, d1, d2) in that exact order (NCHW: OC first).
		auto outD0 = dyn_cast<AffineDimExpr>(outMap.getResult(0));
		auto outD1 = dyn_cast<AffineDimExpr>(outMap.getResult(1));
		auto outD2 = dyn_cast<AffineDimExpr>(outMap.getResult(2));
		if (!outD0 || !outD1 || !outD2)
			return failure();
		// OC iter must be parallel, first among the parallel iters, and come
		// BEFORE the spatial iters numerically.
		unsigned cIter = outD0.getPosition();
		unsigned hIter = outD1.getPosition();
		unsigned wIter = outD2.getPosition();
		if (iters[cIter] != utils::IteratorType::parallel ||
			iters[hIter] != utils::IteratorType::parallel ||
			iters[wIter] != utils::IteratorType::parallel)
			return failure();
		if (cIter > hIter)
			return failure(); // already NHWC ordering

		auto outTy = cast<RankedTensorType>(op.getResult(0).getType());
		if (outTy.getRank() != 3)
			return failure();

		// Find weight map: rank-4 with results (d0, d3, d4, d5) — d0=cIter,
		// and the 3 remaining dims are the reduction iters in some order.
		auto inputMap = maps[0];
		auto weightMap = maps[1];
		if (inputMap.getNumResults() != 3 || weightMap.getNumResults() != 4)
			return failure();
		auto wOcD = dyn_cast<AffineDimExpr>(weightMap.getResult(0));
		auto wIcD = dyn_cast<AffineDimExpr>(weightMap.getResult(1));
		auto wKhD = dyn_cast<AffineDimExpr>(weightMap.getResult(2));
		auto wKwD = dyn_cast<AffineDimExpr>(weightMap.getResult(3));
		if (!wOcD || !wIcD || !wKhD || !wKwD)
			return failure();
		if (wOcD.getPosition() != cIter)
			return failure();
		unsigned icIter = wIcD.getPosition();
		unsigned khIter = wKhD.getPosition();
		unsigned kwIter = wKwD.getPosition();
		auto isRed = [&](unsigned i) {
			return iters[i] == utils::IteratorType::reduction;
		};
		if (!isRed(icIter) || !isRed(khIter) || !isRed(kwIter))
			return failure();

		// Input map: (icIter, hIter * sH + khIter * dH, wIter * sW + kwIter *
		// dW).
		auto parseStrideDilation = [&](AffineExpr e, unsigned outerIter,
									   unsigned kernelIter, int64_t *strideOut,
									   int64_t *dilOut) -> bool {
			// Expression form: outerIter * stride + kernelIter * dilation.
			// Either side may have implicit '* 1'. Also accept the kernel-only
			// case where the spatial dim is just kernelIter (stride=1,
			// dilation=1 and the outer iter is 0-valued — but that would be a
			// degenerate shape).
			auto bin = dyn_cast<AffineBinaryOpExpr>(e);
			if (!bin || bin.getKind() != AffineExprKind::Add)
				return false;
			auto extract = [&](AffineExpr side, unsigned iter,
							   int64_t *coef) -> bool {
				if (auto d = dyn_cast<AffineDimExpr>(side);
					d && d.getPosition() == iter) {
					*coef = 1;
					return true;
				}
				auto mul = dyn_cast<AffineBinaryOpExpr>(side);
				if (!mul || mul.getKind() != AffineExprKind::Mul)
					return false;
				AffineExpr dimSide, constSide;
				if (isa<AffineDimExpr>(mul.getLHS()) &&
					isa<AffineConstantExpr>(mul.getRHS())) {
					dimSide = mul.getLHS();
					constSide = mul.getRHS();
				} else if (isa<AffineDimExpr>(mul.getRHS()) &&
					isa<AffineConstantExpr>(mul.getLHS())) {
					dimSide = mul.getRHS();
					constSide = mul.getLHS();
				} else {
					return false;
				}
				auto d = cast<AffineDimExpr>(dimSide);
				if (d.getPosition() != iter)
					return false;
				*coef = cast<AffineConstantExpr>(constSide).getValue();
				return true;
			};
			if (extract(bin.getLHS(), outerIter, strideOut) &&
				extract(bin.getRHS(), kernelIter, dilOut))
				return true;
			if (extract(bin.getRHS(), outerIter, strideOut) &&
				extract(bin.getLHS(), kernelIter, dilOut))
				return true;
			return false;
		};
		auto inICD = dyn_cast<AffineDimExpr>(inputMap.getResult(0));
		if (!inICD || inICD.getPosition() != icIter)
			return failure();
		int64_t strideH = 0, dilH = 0, strideW = 0, dilW = 0;
		if (!parseStrideDilation(
				inputMap.getResult(1), hIter, khIter, &strideH, &dilH))
			return failure();
		if (!parseStrideDilation(
				inputMap.getResult(2), wIter, kwIter, &strideW, &dilW))
			return failure();

		Location loc = op.getLoc();
		Builder b(rewriter.getContext());

		// Build new indexing maps using a normalized iter order:
		// (oh, ow, oc, kh, kw, ic).
		int64_t numLoops = 6;
		auto dim = [&](unsigned pos) { return rewriter.getAffineDimExpr(pos); };
		AffineExpr oh = dim(0), ow = dim(1), oc = dim(2);
		AffineExpr kh = dim(3), kw = dim(4), ic = dim(5);
		SmallVector<AffineMap> newMaps;
		newMaps.push_back(AffineMap::get(numLoops, 0,
			{oh * strideH + kh * dilH, ow * strideW + kw * dilW, ic},
			rewriter.getContext()));
		newMaps.push_back(AffineMap::get(
			numLoops, 0, {kh, kw, ic, oc}, rewriter.getContext()));
		// Pass through any extra scalar inputs (e.g. zp).
		for (int64_t i = 2; i < op.getNumDpsInputs(); ++i) {
			// Original map may be empty (scalar) or rank-3+. For scalars keep
			// empty; for shaped inputs of matching output rank apply outPerm
			// [1,2,0].
			auto m = maps[i];
			if (m.getNumResults() == 0)
				newMaps.push_back(
					AffineMap::get(numLoops, 0, {}, rewriter.getContext()));
			else
				return failure(); // not yet handled
		}
		newMaps.push_back(
			AffineMap::get(numLoops, 0, {oh, ow, oc}, rewriter.getContext()));
		SmallVector<utils::IteratorType> newIters(
			3, utils::IteratorType::parallel);
		newIters.append(3, utils::IteratorType::reduction);

		// Transpose operands: input [IC, H, W] -> [H, W, IC] (perm [1,2,0]);
		// weight [OC, IC, KH, KW] -> [KH, KW, IC, OC] (FCHW -> HWCF, perm
		// [2,3,1,0]); output empty [OC, H, W] -> [H, W, OC] (perm [1,2,0]).
		SmallVector<int64_t> outPerm{1, 2, 0};
		Value inputV = op.getDpsInputs()[0];
		auto inputTy = cast<RankedTensorType>(inputV.getType());
		SmallVector<int64_t> inputPermShape;
		for (int64_t p : outPerm)
			inputPermShape.push_back(inputTy.getShape()[p]);
		Value inputEmpty = tensor::EmptyOp::create(
			rewriter, loc, inputPermShape, inputTy.getElementType());
		Value inputNHWC = linalg::TransposeOp::create(
			rewriter, loc, inputV, inputEmpty, outPerm)
							  .getResult()[0];

		Value weightV = op.getDpsInputs()[1];
		Value weightHWCF = insertFCHWtoHWCF(loc, weightV, rewriter);

		SmallVector<Value> newInputs{inputNHWC, weightHWCF};
		for (int64_t i = 2; i < op.getNumDpsInputs(); ++i)
			newInputs.push_back(op.getDpsInputs()[i]);

		Value initV = op.getDpsInits()[0];
		auto initTy = cast<RankedTensorType>(initV.getType());
		SmallVector<int64_t> initPermShape;
		for (int64_t p : outPerm)
			initPermShape.push_back(initTy.getShape()[p]);
		Value initEmpty = tensor::EmptyOp::create(
			rewriter, loc, initPermShape, initTy.getElementType());
		Value initNHWC = linalg::TransposeOp::create(
			rewriter, loc, initV, initEmpty, outPerm)
							 .getResult()[0];

		SmallVector<int64_t> outNHWCShape;
		for (int64_t p : outPerm)
			outNHWCShape.push_back(outTy.getShape()[p]);
		auto outNHWCTy =
			RankedTensorType::get(outNHWCShape, outTy.getElementType());

		auto newGeneric = linalg::GenericOp::create(rewriter, loc,
			TypeRange{outNHWCTy}, /*inputs=*/newInputs,
			/*outputs=*/ValueRange{initNHWC}, newMaps, newIters);
		rewriter.inlineRegionBefore(op.getRegion(), newGeneric.getRegion(),
			newGeneric.getRegion().begin());

		// Transpose result NHWC [H, W, OC] -> back to NCHW [OC, H, W]
		// (perm [2, 0, 1]).
		SmallVector<int64_t> invPerm{2, 0, 1};
		Value backEmpty = tensor::EmptyOp::create(
			rewriter, loc, outTy.getShape(), outTy.getElementType());
		Value resNCHW = linalg::TransposeOp::create(
			rewriter, loc, newGeneric.getResult(0), backEmpty, invPerm)
							.getResult()[0];
		rewriter.replaceOp(op, resNCHW);
		return success();
	}
};

// Strip the `_q` (quantized) wrapping from a `linalg.conv_2d_nhwc_hwcf_q`
// when both input and weight zero-points are 0. The resulting plain
// `linalg.conv_2d_nhwc_hwcf` has the same numerics (no zp adjustment
// needed) but is recognized by upstream passes like
// `iree-linalg-ext-convert-conv-to-im2col-op` that treat the q variant as
// unsupported.
struct StripZeroZpQConv : OpRewritePattern<linalg::Conv2DNhwcHwcfQOp> {
	using OpRewritePattern<linalg::Conv2DNhwcHwcfQOp>::OpRewritePattern;
	LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfQOp op,
		PatternRewriter &rewriter) const override {
		Value inZp = op.getInputs()[2];
		Value wZp = op.getInputs()[3];
		auto isZeroI32 = [&](Value v) -> bool {
			auto cst = v.getDefiningOp<arith::ConstantOp>();
			if (!cst)
				return false;
			auto a = dyn_cast<IntegerAttr>(cst.getValue());
			return a && a.getInt() == 0;
		};
		if (!isZeroI32(inZp) || !isZeroI32(wZp))
			return failure();
		rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(op,
			op.getResult(0).getType(),
			ValueRange{op.getInputs()[0], op.getInputs()[1]},
			ValueRange{op.getOutputs()[0]}, op.getStridesAttr(),
			op.getDilationsAttr());
		return success();
	}
};

struct LegalizeLayoutToNHWCPass
	: public PassWrapper<LegalizeLayoutToNHWCPass, OperationPass<>> {
	StringRef getArgument() const final {
		return "merlin-qnn-legalize-layout-to-nhwc";
	}
	StringRef getDescription() const final {
		return "Rewrite linalg NCHW convs to NHWC by surrounding them with "
			   "linalg.transpose ops; required before "
			   "merlin-convert-linalg-to-qnn "
			   "since the QNN dialect's verifier rejects NCHW spatial ops.";
	}
	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
			arith::ArithDialect>();
	}
	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<RewriteNCHWConvQ, RewriteNCHWGenericConv,
			RewriteNCHWGenericConvExplicit, StripZeroZpQConv>(&getContext());
		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createLegalizeLayoutToNHWCPass() {
	return std::make_unique<LegalizeLayoutToNHWCPass>();
}

} // namespace mlir::iree_compiler::QNN
