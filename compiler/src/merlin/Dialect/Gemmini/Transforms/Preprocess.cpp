//===- Preprocess.cpp - Gemmini-friendly linalg-level rewrites -----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//===---------------------------------------------------------------------===//
//
// Linalg-level rewrites that benefit Gemmini regardless of whether we use the
// tensor-domain Gemmini ops (lowerBackToIREE=true) or stay on linalg.matmul
// for inside-dispatch RoCC codegen (lowerBackToIREE=false).
//
// Today this contains one pattern: ConvertDequantMaxPoolQuantPattern. The
// canonical QDQ-quantized max-pool dispatch in dronet/yolov8n wraps an f32
// max-pool with a `sitofp + mulf scale` dequant and a `divf scale + roundeven
// + clip + fptosi` requant. When the dequant and quant scales bit-match (the
// usual case for symmetric per-tensor quant), max() is positive-scalar-
// equivariant, so `divf(max(c*x_i), c) = max(x_i)` and the saturate+roundeven
// of an exact i8 is a no-op. Folding this round-trip away gives IREE's
// standard linalg.generic codegen an i8 max-pool to vectorize, instead of
// 11 MB of scalar f32 elementwise ops for dronet's dispatch_3.
//
// See the 2026-05-26 dev log / plan smooth-conjuring-lemur.md (Opt #1).
//
//===---------------------------------------------------------------------===//

#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINIPREPROCESSPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

static std::optional<double> getScalarFloatConstant(Value value) {
	Attribute attr;
	if (!matchPattern(value, m_Constant(&attr)))
		return std::nullopt;
	if (auto fa = dyn_cast<FloatAttr>(attr))
		return fa.getValueAsDouble();
	if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
		if (!dense.isSplat())
			return std::nullopt;
		return dense.getSplatValue<APFloat>().convertToDouble();
	}
	return std::nullopt;
}

static std::optional<double> resolveFloatValue(
	Value value, linalg::GenericOp genericOp) {
	if (auto blockArg = dyn_cast<BlockArgument>(value)) {
		if (blockArg.getOwner() == &genericOp.getRegion().front() &&
			blockArg.getArgNumber() < genericOp.getNumDpsInputs()) {
			return getScalarFloatConstant(
				genericOp.getDpsInputs()[blockArg.getArgNumber()]);
		}
	}
	return getScalarFloatConstant(value);
}

// Recognize the f32→i8 requantize body:
//   divf scale; roundeven; addf 0; max -128; min 127; fptosi
// Returns the divf scale constant on success.
static std::optional<double> matchQuantBody(linalg::GenericOp op) {
	Block &body = op.getRegion().front();
	SmallVector<Operation *> ops;
	for (Operation &nested : body.without_terminator())
		ops.push_back(&nested);
	auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
	if (!yield || ops.size() != 6)
		return std::nullopt;
	auto divf = dyn_cast<arith::DivFOp>(ops[0]);
	auto rnd = dyn_cast<math::RoundEvenOp>(ops[1]);
	auto add = dyn_cast<arith::AddFOp>(ops[2]);
	auto maxF = dyn_cast<arith::MaximumFOp>(ops[3]);
	auto minF = dyn_cast<arith::MinimumFOp>(ops[4]);
	auto cast = dyn_cast<arith::FPToSIOp>(ops[5]);
	if (!divf || !rnd || !add || !maxF || !minF || !cast)
		return std::nullopt;
	Value bbIn = body.getArgument(0);
	if (divf.getLhs() != bbIn)
		return std::nullopt;
	if (rnd.getOperand() != divf.getResult())
		return std::nullopt;
	if (add.getLhs() != rnd.getResult())
		return std::nullopt;
	if (maxF.getLhs() != add.getResult())
		return std::nullopt;
	if (minF.getLhs() != maxF.getResult())
		return std::nullopt;
	if (cast.getIn() != minF.getResult())
		return std::nullopt;
	if (yield.getOperand(0) != cast.getResult())
		return std::nullopt;
	auto scale = resolveFloatValue(divf.getRhs(), op);
	auto zp = resolveFloatValue(add.getRhs(), op);
	auto qmin = resolveFloatValue(maxF.getRhs(), op);
	auto qmax = resolveFloatValue(minF.getRhs(), op);
	if (!scale || !zp || !qmin || !qmax)
		return std::nullopt;
	if (*zp != 0.0 || *qmin != -128.0 || *qmax != 127.0)
		return std::nullopt;
	return scale;
}

// Recognize a permutation-only linalg.generic — single input, identity-yield
// body (`linalg.yield %in`), non-identity input indexing map, identity output
// indexing map.  Returns the input map (which represents the permutation as
// applied to the iteration space of the transpose op).
static std::optional<AffineMap> matchPermutationOnly(linalg::GenericOp op) {
	if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
		return std::nullopt;
	Block &body = op.getRegion().front();
	if (body.getOperations().size() != 1)
		return std::nullopt;
	auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
	if (!yield || yield.getNumOperands() != 1)
		return std::nullopt;
	if (yield->getOperand(0) != body.getArgument(0))
		return std::nullopt;
	auto maps = op.getIndexingMapsArray();
	if (maps.size() != 2)
		return std::nullopt;
	if (!maps[1].isIdentity())
		return std::nullopt;
	return maps[0];
}

// Recognize the i8→f32 dequant body: sitofp; mulf scale; yield.
static std::optional<double> matchDequantBody(linalg::GenericOp op) {
	if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
		return std::nullopt;
	Block &body = op.getRegion().front();
	SmallVector<Operation *> ops;
	for (Operation &nested : body.without_terminator())
		ops.push_back(&nested);
	auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
	if (!yield || ops.size() != 2)
		return std::nullopt;
	auto sitofp = dyn_cast<arith::SIToFPOp>(ops[0]);
	auto mulf = dyn_cast<arith::MulFOp>(ops[1]);
	if (!sitofp || !mulf)
		return std::nullopt;
	Value bbIn = body.getArgument(0);
	if (sitofp.getIn() != bbIn)
		return std::nullopt;
	if (mulf.getLhs() != sitofp.getResult())
		return std::nullopt;
	if (yield.getOperand(0) != mulf.getResult())
		return std::nullopt;
	auto inTy = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
	if (!inTy || !inTy.getElementType().isSignlessInteger(8))
		return std::nullopt;
	return resolveFloatValue(mulf.getRhs(), op);
}

// 5-loop sliding-window max-pool linalg.generic with body
// `arith.maximumf(in, out)` (commutative; either operand order).
static bool isMaximumFPoolBody(linalg::GenericOp op) {
	Block &body = op.getRegion().front();
	auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
	if (!yield || yield.getNumOperands() != 1)
		return false;
	auto maxF = yield->getOperand(0).getDefiningOp<arith::MaximumFOp>();
	if (!maxF)
		return false;
	if (body.getNumArguments() != 3)
		return false;
	Value bbInput = body.getArgument(0);
	Value bbOut = body.getArgument(2);
	bool ok = (maxF.getLhs() == bbInput && maxF.getRhs() == bbOut) ||
		(maxF.getLhs() == bbOut && maxF.getRhs() == bbInput);
	if (!ok)
		return false;
	// Body must be exactly maximumf + yield — nothing else.
	int count = 0;
	for (Operation &nested : body.without_terminator()) {
		(void)nested;
		++count;
	}
	return count == 1;
}

struct ConvertDequantMaxPoolQuantPattern final
	: OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp quantOp, PatternRewriter &rewriter) const override {
		if (!quantOp.hasPureTensorSemantics())
			return failure();
		if (quantOp.getNumDpsInputs() != 1 || quantOp.getNumDpsInits() != 1)
			return failure();
		auto outTy = dyn_cast<RankedTensorType>(quantOp.getResult(0).getType());
		if (!outTy || !outTy.getElementType().isSignlessInteger(8))
			return failure();
		auto maps = quantOp.getIndexingMapsArray();
		if (maps.size() != 2)
			return failure();
		if (!maps[0].isIdentity() || !maps[1].isIdentity())
			return failure();
		auto quantScale = matchQuantBody(quantOp);
		if (!quantScale)
			return failure();

		Value quantIn = quantOp.getDpsInputs()[0];
		auto poolOp = quantIn.getDefiningOp<linalg::GenericOp>();
		if (!poolOp || !poolOp.hasPureTensorSemantics())
			return failure();
		if (poolOp.getNumLoops() != 5)
			return failure();
		if (poolOp.getNumDpsInputs() != 2 || poolOp.getNumDpsInits() != 1)
			return failure();
		if (!quantIn.hasOneUse())
			return failure();
		if (!isMaximumFPoolBody(poolOp))
			return failure();
		auto iterTypes = poolOp.getIteratorTypesArray();
		if (iterTypes.size() != 5)
			return failure();
		if (iterTypes[0] != utils::IteratorType::parallel ||
			iterTypes[1] != utils::IteratorType::parallel ||
			iterTypes[2] != utils::IteratorType::parallel ||
			iterTypes[3] != utils::IteratorType::reduction ||
			iterTypes[4] != utils::IteratorType::reduction)
			return failure();

		auto poolMaps = poolOp.getIndexingMapsArray();
		if (poolMaps.size() != 3)
			return failure();
		linalg::GenericOp dequantOp = nullptr;
		linalg::GenericOp transposeOp = nullptr;
		Value shapeCarrier = nullptr;
		AffineMap windowMap;
		AffineMap kernelMap;
		AffineMap outputMap = poolMaps[2];
		// Try each pool input: it's either the dequant directly, or a
		// permutation-only generic that wraps the dequant (IREE's pre-
		// bufferization form keeps the input-layout transpose as a separate
		// op, which gets fused into the pool's indexing map only after
		// dispatch creation). When we walk through a transpose, the final
		// pool window map is `transpose.inputMap.compose(pool.windowMap)`
		// so the new i8 pool reads directly from the original i8 tensor in
		// its native layout.
		for (int idx = 0; idx < 2; ++idx) {
			Value v = poolOp.getDpsInputs()[idx];
			auto producer = v.getDefiningOp<linalg::GenericOp>();
			if (!producer)
				continue;
			AffineMap thisWindow = poolMaps[idx];
			Value source = v;
			linalg::GenericOp candidateTranspose = nullptr;
			if (!matchDequantBody(producer)) {
				if (auto permMap = matchPermutationOnly(producer)) {
					if (!source.hasOneUse())
						continue;
					thisWindow = permMap->compose(thisWindow);
					candidateTranspose = producer;
					source = producer.getDpsInputs()[0];
					producer = source.getDefiningOp<linalg::GenericOp>();
					if (!producer)
						continue;
				}
			}
			if (matchDequantBody(producer)) {
				dequantOp = producer;
				transposeOp = candidateTranspose;
				windowMap = thisWindow;
				shapeCarrier = poolOp.getDpsInputs()[1 - idx];
				kernelMap = poolMaps[1 - idx];
				break;
			}
		}
		if (!dequantOp || !shapeCarrier)
			return failure();
		if (!dequantOp.getResult(0).hasOneUse())
			return failure();
		auto dequantScale = matchDequantBody(dequantOp);
		if (!dequantScale)
			return failure();

		auto asF32 = [](double d) {
			return llvm::APFloat((float)d).convertToFloat();
		};
		if (asF32(*dequantScale) != asF32(*quantScale))
			return failure();

		Value poolInit = poolOp.getDpsInits()[0];
		auto fillOp = poolInit.getDefiningOp<linalg::FillOp>();
		if (!fillOp || !poolInit.hasOneUse())
			return failure();
		auto fillValue = getScalarFloatConstant(fillOp.getDpsInputs()[0]);
		if (!fillValue || !std::isinf(*fillValue) || *fillValue >= 0.0)
			return failure();

		Location loc = quantOp.getLoc();
		Value origI8 = dequantOp.getDpsInputs()[0];
		auto shapeCarrierTy = cast<RankedTensorType>(shapeCarrier.getType());
		Value i8Kernel = rewriter.create<tensor::EmptyOp>(
			loc, shapeCarrierTy.getShape(), rewriter.getIntegerType(8));
		Value emptyOut = rewriter.create<tensor::EmptyOp>(
			loc, outTy.getShape(), outTy.getElementType());
		Value initVal = rewriter.create<arith::ConstantOp>(
			loc, rewriter.getI8IntegerAttr(-128));
		Value initI8 = rewriter
						   .create<linalg::FillOp>(
							   loc, ValueRange{initVal}, ValueRange{emptyOut})
						   .getResult(0);

		SmallVector<AffineMap> newMaps = {windowMap, kernelMap, outputMap};
		SmallVector<utils::IteratorType> newIterTypes = iterTypes;
		Value i8Pool =
			rewriter
				.create<linalg::GenericOp>(loc,
					/*resultTensorTypes=*/TypeRange{outTy},
					/*inputs=*/ValueRange{origI8, i8Kernel},
					/*outputs=*/ValueRange{initI8},
					/*indexingMaps=*/newMaps,
					/*iteratorTypes=*/newIterTypes,
					[&](OpBuilder &b, Location nestedLoc, ValueRange args) {
						Value maxv = b.create<arith::MaxSIOp>(
							nestedLoc, args[2], args[0]);
						b.create<linalg::YieldOp>(nestedLoc, maxv);
					})
				.getResult(0);

		rewriter.replaceOp(quantOp, i8Pool);
		rewriter.eraseOp(poolOp);
		rewriter.eraseOp(fillOp);
		if (transposeOp)
			rewriter.eraseOp(transposeOp);
		rewriter.eraseOp(dequantOp);
		return success();
	}
};

// Materialize a `linalg.transpose` for every `linalg.matmul` whose RHS
// indexing_maps encode an N×K (b-transposed) layout. Goal: convert the
// "b-only-transposed" matmul that `LowerBufferizedLinalgMatmulToTileMatmul`
// currently declines to lower (FireSim Shuttle Gemmini hangs on the
// b-transpose path) into a regular K×N matmul preceded by an explicit
// transpose. When the RHS is a constant (dronet's dispatch_11/20/29 all
// use `__constant_tensor_NxKxi8`), IREE folds the transpose at compile
// time so the runtime cost is zero. Once this rewrite fires, those three
// dispatches reach `tryMatchChainedResidualRescale` and the partial-fold
// (Opt #3) lights up.
//
// Match conditions:
//  - tensor-domain `linalg.matmul` with explicit `indexing_maps`
//  - LHS map is identity (d0,d2), output map (d0,d1) — only RHS encodes
//    the transpose (d1,d2)
//  - All operand types ranked 2D
struct MaterializeBTransposeAsLinalgTransposePattern final
	: OpRewritePattern<linalg::MatmulOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::MatmulOp op, PatternRewriter &rewriter) const override {
		if (!op.hasPureTensorSemantics())
			return failure();
		if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
			return failure();

		auto maps = op.getIndexingMapsArray();
		if (maps.size() != 3)
			return failure();
		MLIRContext *ctx = op.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineExpr d2 = getAffineDimExpr(2, ctx);
		AffineMap lhsKxNMap = AffineMap::get(3, 0, {d0, d2}, ctx);
		AffineMap rhsKxNMap = AffineMap::get(3, 0, {d2, d1}, ctx);
		AffineMap rhsNxKMap = AffineMap::get(3, 0, {d1, d2}, ctx);
		AffineMap outMap = AffineMap::get(3, 0, {d0, d1}, ctx);
		// Only fire on b-only-transpose: LHS identity, RHS N×K, output
		// identity.
		if (maps[0] != lhsKxNMap)
			return failure();
		if (maps[1] != rhsNxKMap)
			return failure();
		if (maps[2] != outMap)
			return failure();

		Value rhs = op.getDpsInputs()[1];
		auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
		if (!rhsTy || rhsTy.getRank() != 2)
			return failure();
		// Build a transpose to convert N×K → K×N.
		Location loc = op.getLoc();
		int64_t N = rhsTy.getDimSize(0);
		int64_t K = rhsTy.getDimSize(1);
		auto transposedTy =
			RankedTensorType::get({K, N}, rhsTy.getElementType());
		Value init = rewriter.create<tensor::EmptyOp>(
			loc, transposedTy.getShape(), transposedTy.getElementType());
		Value transposed = rewriter
							   .create<linalg::TransposeOp>(
								   loc, rhs, init, ArrayRef<int64_t>{1, 0})
							   ->getResult(0);

		// Re-emit the matmul with default (K×N) indexing.
		SmallVector<AffineMap> newMaps = {lhsKxNMap, rhsKxNMap, outMap};
		auto newMatmul = rewriter.create<linalg::MatmulOp>(loc,
			op.getResultTypes(), ValueRange{op.getDpsInputs()[0], transposed},
			ValueRange{op.getDpsInits()[0]});
		newMatmul.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
		// Preserve compilation_info / attrs so the IREE dispatch pipeline
		// hooks still apply.
		for (auto &attr : op->getAttrs()) {
			if (attr.getName() == "indexing_maps")
				continue;
			if (attr.getName() == "operandSegmentSizes")
				continue;
			newMatmul->setAttr(attr.getName(), attr.getValue());
		}
		rewriter.replaceOp(op, newMatmul.getResults());
		return success();
	}
};

// Split a 1-input linalg.generic whose body BEGINS with the canonical 8-op
// i32→i8 QDQ rescale chain (sitofp; mulf scale_in; divf scale_int;
// roundeven; addf 0; max -128; min 127; fptosi) and then CONTINUES with
// non-rescale logic (SiLU, layernorm tail, custom activations, etc.). This
// lights up yolov8n's matmul+SiLU+rescale dispatches: after the split,
// the first half is a clean QDQ rescale that the matmul-rescale fusion
// (Opt #0 in LowerTileToISA) picks up; the second half stays as a smaller
// elementwise dispatch that IREE codegens normally.
//
// Bit-exact because we re-emit the same instruction sequence verbatim —
// just sliced at the i8 boundary. The intermediate `fptosi → i8` is what
// IREE's downstream codegen already materializes anyway.
struct SplitRescaleHeadFromActivationTailPattern final
	: OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp gen, PatternRewriter &rewriter) const override {
		if (!gen.hasPureTensorSemantics())
			return failure();
		if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1)
			return failure();
		auto inTy = dyn_cast<RankedTensorType>(gen.getDpsInputs()[0].getType());
		auto outTy = dyn_cast<RankedTensorType>(gen.getResult(0).getType());
		if (!inTy || !inTy.getElementType().isSignlessInteger(32))
			return failure();
		if (!outTy || !outTy.getElementType().isSignlessInteger(8))
			return failure();
		// All identity-mapped — the "head" rescale must be element-wise so
		// it can be split off cleanly.
		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 2)
			return failure();
		if (!maps[0].isIdentity() || !maps[1].isIdentity())
			return failure();

		Block &body = gen.getRegion().front();
		SmallVector<Operation *> ops;
		for (Operation &nested : body.without_terminator())
			ops.push_back(&nested);
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!yield || yield.getNumOperands() != 1)
			return failure();
		// Need MORE than the trivial 8-op rescale; otherwise this is the
		// shape `ConvertRequantizePattern` / `tryMatchI32ToI8Rescale`
		// already handle on their own.
		if (ops.size() <= 8)
			return failure();

		// The first 8 ops MUST be the canonical QDQ rescale.
		auto sf = dyn_cast<arith::SIToFPOp>(ops[0]);
		auto mf = dyn_cast<arith::MulFOp>(ops[1]);
		auto df = dyn_cast<arith::DivFOp>(ops[2]);
		auto rd = dyn_cast<math::RoundEvenOp>(ops[3]);
		auto af = dyn_cast<arith::AddFOp>(ops[4]);
		auto mx = dyn_cast<arith::MaximumFOp>(ops[5]);
		auto mn = dyn_cast<arith::MinimumFOp>(ops[6]);
		auto fp = dyn_cast<arith::FPToSIOp>(ops[7]);
		if (!sf || !mf || !df || !rd || !af || !mx || !mn || !fp)
			return failure();
		Value mmIn = body.getArgument(0);
		if (sf.getIn() != mmIn)
			return failure();
		if (mf.getLhs() != sf.getResult())
			return failure();
		if (df.getLhs() != mf.getResult())
			return failure();
		if (rd.getOperand() != df.getResult())
			return failure();
		if (af.getLhs() != rd.getResult())
			return failure();
		if (mx.getLhs() != af.getResult())
			return failure();
		if (mn.getLhs() != mx.getResult())
			return failure();
		if (fp.getIn() != mn.getResult())
			return failure();
		// Validate the constants: zp=0, qmin=-128, qmax=127.
		auto check = [&](Value v, double exp) {
			auto cst = v.getDefiningOp<arith::ConstantOp>();
			if (!cst)
				return false;
			auto attr = dyn_cast<FloatAttr>(cst.getValue());
			return attr && attr.getValueAsDouble() == exp;
		};
		if (!check(af.getRhs(), 0.0))
			return failure();
		if (!check(mx.getRhs(), -128.0))
			return failure();
		if (!check(mn.getRhs(), 127.0))
			return failure();
		auto scaleInCst = mf.getRhs().getDefiningOp<arith::ConstantOp>();
		auto scaleIntCst = df.getRhs().getDefiningOp<arith::ConstantOp>();
		if (!scaleInCst || !scaleIntCst)
			return failure();

		// Build generic 1: same shape, only the first 8 ops, yield the
		// fptosi result.
		Location loc = gen.getLoc();
		Value mmI8Empty = rewriter.create<tensor::EmptyOp>(
			loc, outTy.getShape(), rewriter.getIntegerType(8));
		Value scaleInVal = mf.getRhs();
		Value scaleIntVal = df.getRhs();
		Value zeroVal = af.getRhs();
		Value cMinVal = mx.getRhs();
		Value cMaxVal = mn.getRhs();
		Value firstRescale =
			rewriter
				.create<linalg::GenericOp>(loc,
					/*resultTensorTypes=*/
					TypeRange{RankedTensorType::get(
						outTy.getShape(), rewriter.getIntegerType(8))},
					/*inputs=*/ValueRange{gen.getDpsInputs()[0]},
					/*outputs=*/ValueRange{mmI8Empty},
					/*indexingMaps=*/SmallVector<AffineMap>{maps[0], maps[1]},
					/*iteratorTypes=*/gen.getIteratorTypesArray(),
					[&](OpBuilder &b, Location nl, ValueRange args) {
						Value a = b.create<arith::SIToFPOp>(
							nl, b.getF32Type(), args[0]);
						Value bv = b.create<arith::MulFOp>(nl, a, scaleInVal);
						Value c = b.create<arith::DivFOp>(nl, bv, scaleIntVal);
						Value d = b.create<math::RoundEvenOp>(nl, c);
						Value e = b.create<arith::AddFOp>(nl, d, zeroVal);
						Value f = b.create<arith::MaximumFOp>(nl, e, cMinVal);
						Value g = b.create<arith::MinimumFOp>(nl, f, cMaxVal);
						Value h = b.create<arith::FPToSIOp>(
							nl, b.getIntegerType(8), g);
						b.create<linalg::YieldOp>(nl, h);
					})
				.getResult(0);

		// Build generic 2: same shape, ins = i8 intermediate, outs = the
		// original outs. Clone the body's remaining ops, remapping the
		// fptosi result (`fp.getResult()`) to the new bb-arg.
		Value newOutsEmpty = rewriter.create<tensor::EmptyOp>(
			loc, outTy.getShape(), outTy.getElementType());
		linalg::GenericOp tail = rewriter.create<linalg::GenericOp>(loc,
			/*resultTensorTypes=*/TypeRange{outTy},
			/*inputs=*/ValueRange{firstRescale},
			/*outputs=*/ValueRange{newOutsEmpty},
			/*indexingMaps=*/SmallVector<AffineMap>{maps[0], maps[1]},
			/*iteratorTypes=*/gen.getIteratorTypesArray(),
			/*bodyBuild=*/nullptr);
		{
			OpBuilder::InsertionGuard g(rewriter);
			Block *newBlock = rewriter.createBlock(&tail.getRegion(),
				tail.getRegion().end(),
				TypeRange{rewriter.getIntegerType(8), outTy.getElementType()},
				SmallVector<Location>{loc, loc});
			rewriter.setInsertionPointToStart(newBlock);
			IRMapping mapping;
			// Replace uses of the original fptosi result with the new bb-arg.
			mapping.map(fp.getResult(), newBlock->getArgument(0));
			for (size_t i = 8; i < ops.size(); ++i) {
				rewriter.clone(*ops[i], mapping);
			}
			// Re-emit the yield, mapping through.
			Value yieldedVal = yield.getOperand(0);
			Value newYielded = mapping.lookupOrDefault(yieldedVal);
			rewriter.create<linalg::YieldOp>(loc, newYielded);
		}
		rewriter.replaceOp(gen, tail.getResult(0));
		return success();
	}
};

// Opt #B (2026-05-26): fuse a `linalg.generic` broadcast (small N-element
// tensor → MxN with indexing map (d0,d1)->(d1) on input, identity on
// output, body `linalg.yield %in`) into the downstream `linalg.generic
// addi` (bias-add to matmul output). After this rewrite, the bias-add
// reads the SMALL source directly with broadcast indexing, the broadcast
// generic becomes dead, and `LowerTileToISA::findBiasAddConsumer` sees a
// rank-1 bias input — triggering `repeatingBias=true` on the emitted
// tile_matmul, which saves the per-dispatch ~25K MVIN-D cycles of
// loading a fully-materialized MxN i32 bias buffer.
//
// Why post-global-opt is the right stage: HoistIntoGlobalsPass runs LATER
// and would otherwise materialize the broadcast as a separate initializer
// dispatch (the 401KB `__hoisted_tensor_3136x32xi32` for dronet's
// dispatch_2). Rewriting here keeps the small tensor as the bias-add's
// only bias input — IREE's hoist then only promotes the 32-element
// constant, not the full 3136×32.
struct FuseBroadcastIntoBiasAddPattern final
	: OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp addOp, PatternRewriter &rewriter) const override {
		if (!addOp.hasPureTensorSemantics())
			return failure();
		if (addOp.getNumDpsInputs() != 2 || addOp.getNumDpsInits() != 1)
			return failure();
		auto outTy = dyn_cast<RankedTensorType>(addOp.getResult(0).getType());
		if (!outTy || !outTy.getElementType().isSignlessInteger(32))
			return failure();
		// Body must be exactly `arith.addi(in0, in1)`.
		Block &body = addOp.getRegion().front();
		if (body.getNumArguments() != 3)
			return failure();
		auto term = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!term || term.getNumOperands() != 1)
			return failure();
		auto addi = term->getOperand(0).getDefiningOp<arith::AddIOp>();
		if (!addi)
			return failure();
		{
			Value a = body.getArgument(0), b = body.getArgument(1);
			Value lhs = addi.getLhs(), rhs = addi.getRhs();
			if (!((lhs == a && rhs == b) || (lhs == b && rhs == a)))
				return failure();
		}
		auto maps = addOp.getIndexingMapsArray();
		if (maps.size() != 3)
			return failure();
		// We require the OUTPUT and at least one INPUT to be identity (the
		// matmul-output side); the OTHER input is the bias candidate.
		if (!maps[2].isIdentity())
			return failure();

		MLIRContext *ctx = addOp.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineMap broadcastMap = AffineMap::get(2, 0, {d1}, ctx);

		// Try each input as the bias candidate.
		for (int biasIdx = 0; biasIdx < 2; ++biasIdx) {
			int otherIdx = 1 - biasIdx;
			if (!maps[otherIdx].isIdentity())
				continue;
			// Only proceed if the current bias input has identity indexing
			// (otherwise the fusion has already happened or the pattern is
			// different).
			if (!maps[biasIdx].isIdentity())
				continue;
			Value biasInput = addOp.getDpsInputs()[biasIdx];
			auto biasProducer = biasInput.getDefiningOp<linalg::GenericOp>();
			if (!biasProducer)
				continue;
			if (biasProducer.getNumDpsInputs() != 1 ||
				biasProducer.getNumDpsInits() != 1)
				continue;
			auto producerMaps = biasProducer.getIndexingMapsArray();
			if (producerMaps.size() != 2)
				continue;
			if (producerMaps[0] != broadcastMap)
				continue;
			if (!producerMaps[1].isIdentity())
				continue;
			// Body must be `linalg.yield %in` (identity broadcast).
			Block &pbody = biasProducer.getRegion().front();
			if (pbody.getOperations().size() != 1)
				continue;
			auto pyield = dyn_cast<linalg::YieldOp>(pbody.getTerminator());
			if (!pyield || pyield.getNumOperands() != 1)
				continue;
			if (pyield->getOperand(0) != pbody.getArgument(0))
				continue;
			// Source tensor must be 1D (Nxi32). We rewrite the addi to read
			// it directly with broadcast indexing.
			Value broadcastSrc = biasProducer.getDpsInputs()[0];
			auto srcTy = dyn_cast<RankedTensorType>(broadcastSrc.getType());
			if (!srcTy || srcTy.getRank() != 1)
				continue;
			// Only one user of the broadcast result — if anything else
			// also reads it, leave it alone (would require keeping the
			// broadcast op).
			if (!biasInput.hasOneUse())
				continue;

			SmallVector<AffineMap> newMaps;
			for (int i = 0; i < 3; ++i)
				newMaps.push_back(i == biasIdx ? broadcastMap : maps[i]);
			SmallVector<Value> newInputs = addOp.getDpsInputs();
			newInputs[biasIdx] = broadcastSrc;

			rewriter.modifyOpInPlace(addOp, [&]() {
				addOp.setIndexingMapsAttr(
					rewriter.getAffineMapArrayAttr(newMaps));
				addOp->setOperand(biasIdx, broadcastSrc);
			});
			rewriter.eraseOp(biasProducer);
			return success();
		}
		return failure();
	}
};

// Opt#G — flatten yolov8n's 4D `matmul_like_*` QDQ generic into a 2D
// `linalg.matmul`. Form to match (post-global-opt, before dispatch creation):
//
//   linalg.generic {indexing_maps = [
//       (d0,d1,d2,d3) -> (d3, d1, d2),  // op0 = feat[c, h, w]
//       (d0,d1,d2,d3) -> (d0, d3),      // op1 = weight[f, c]
//       (d0,d1,d2,d3) -> (),            // zp0 (scalar)
//       (d0,d1,d2,d3) -> (),            // zp1 (scalar)
//       (d0,d1,d2,d3) -> (d0, d1, d2)], // out[f, h, w]
//     iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
//     ins(feat, weight, zp0, zp1 : tensor<CxHxWxi8>, tensor<FxCxi8>,
//                                   i32, i32)
//     outs(out_init : tensor<FxHxWxi32>) {
//   ^bb0(%in: i8, %in_w: i8, %in_zp0: i32, %in_zp1: i32, %out: i32):
//     %a = arith.extsi %in : i8 to i32
//     %a_z = arith.subi %a, %in_zp0
//     %b = arith.extsi %in_w : i8 to i32
//     %b_z = arith.subi %b, %in_zp1
//     %m = arith.muli %a_z, %b_z
//     %r = arith.addi %out, %m
//     linalg.yield %r
//   }
//
// When both zero-points are 0 (the common case for pt2e-quantized models),
// the subi ops are no-ops and the body collapses to canonical
// extsi/extsi/muli/addi. We rewrite to:
//   1. tensor.collapse_shape feat[C,H,W]    → [C, H*W]
//   2. tensor.collapse_shape out_init[F,H,W]→ [F, H*W]
//   3. linalg.matmul (weight as A=[F,C], feat_flat as B=[C,H*W], output map
//      [(d0,d2), (d2,d1), (d0,d1)]) — same b-only-transposed form
//      MaterializeBTransposeAsLinalgTransposePattern produces, so all
//      downstream patterns (ConvertNamedMatmulPattern, Opt#0/#3 folds)
//      pick it up naturally.
//   4. tensor.expand_shape result[F, H*W]   → [F, H, W]
//
// IREE's standard fold-reshape passes collapse the (collapse → matmul →
// expand) sandwich so no extra dispatches are generated; the dispatch sees
// only the matmul.
//
// Without this rewrite, every yolov8n matmul_like_* dispatch (matmul_like_
// 32x6400x32 at 167M cycles, matmul_like_64x1600x64 at 59M, etc., ~20
// total) falls through to scalar CPU codegen.
struct FlattenMatmulLike4DToMatmulPattern final
	: OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp gen, PatternRewriter &rewriter) const override {
		if (!gen.hasPureTensorSemantics())
			return failure();
		if (gen.getNumLoops() != 4)
			return failure();
		auto iters = gen.getIteratorTypesArray();
		if (iters[0] != utils::IteratorType::parallel ||
			iters[1] != utils::IteratorType::parallel ||
			iters[2] != utils::IteratorType::parallel ||
			iters[3] != utils::IteratorType::reduction)
			return failure();
		if (gen.getNumDpsInputs() != 4 || gen.getNumDpsInits() != 1)
			return failure();

		auto maps = gen.getIndexingMapsArray();
		if (maps.size() != 5)
			return failure();
		MLIRContext *ctx = gen.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineExpr d2 = getAffineDimExpr(2, ctx);
		AffineExpr d3 = getAffineDimExpr(3, ctx);
		AffineMap featMap = AffineMap::get(4, 0, {d3, d1, d2}, ctx);
		AffineMap weightMap = AffineMap::get(4, 0, {d0, d3}, ctx);
		AffineMap scalarMap = AffineMap::get(4, 0, {}, ctx);
		AffineMap outMap = AffineMap::get(4, 0, {d0, d1, d2}, ctx);
		if (maps[0] != featMap || maps[1] != weightMap ||
			maps[2] != scalarMap || maps[3] != scalarMap || maps[4] != outMap)
			return failure();

		// Body: extsi, subi, extsi, subi, muli, addi, yield.
		Block &body = gen.getRegion().front();
		if (body.getNumArguments() != 5)
			return failure();
		SmallVector<Operation *> ops;
		for (Operation &nested : body.without_terminator())
			ops.push_back(&nested);
		if (ops.size() != 6)
			return failure();
		auto sf0 = dyn_cast<arith::ExtSIOp>(ops[0]);
		auto sb0 = dyn_cast<arith::SubIOp>(ops[1]);
		auto sf1 = dyn_cast<arith::ExtSIOp>(ops[2]);
		auto sb1 = dyn_cast<arith::SubIOp>(ops[3]);
		auto mul = dyn_cast<arith::MulIOp>(ops[4]);
		auto add = dyn_cast<arith::AddIOp>(ops[5]);
		auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
		if (!sf0 || !sb0 || !sf1 || !sb1 || !mul || !add || !yield)
			return failure();
		Value bbFeat = body.getArgument(0);
		Value bbW = body.getArgument(1);
		Value bbZp0 = body.getArgument(2);
		Value bbZp1 = body.getArgument(3);
		Value bbOut = body.getArgument(4);
		if (sf0.getIn() != bbFeat || sf1.getIn() != bbW)
			return failure();
		if (sb0.getLhs() != sf0.getResult() || sb0.getRhs() != bbZp0)
			return failure();
		if (sb1.getLhs() != sf1.getResult() || sb1.getRhs() != bbZp1)
			return failure();
		if (mul.getLhs() != sb0.getResult() || mul.getRhs() != sb1.getResult())
			return failure();
		if (!((add.getLhs() == bbOut && add.getRhs() == mul.getResult()) ||
				(add.getRhs() == bbOut && add.getLhs() == mul.getResult())))
			return failure();
		if (yield.getOperand(0) != add.getResult())
			return failure();

		// Require zero-points to be statically 0 — otherwise we'd need to
		// fold the zp arithmetic into a bias adjustment, which is doable
		// but out of scope for this rewrite.
		auto isZeroI32 = [](Value v) {
			auto cst = v.getDefiningOp<arith::ConstantOp>();
			if (!cst)
				return false;
			auto attr = dyn_cast<IntegerAttr>(cst.getValue());
			return attr && attr.getInt() == 0;
		};
		if (!isZeroI32(gen.getDpsInputs()[2]) ||
			!isZeroI32(gen.getDpsInputs()[3]))
			return failure();

		// Type checks.
		Value feat = gen.getDpsInputs()[0];
		Value weight = gen.getDpsInputs()[1];
		Value outInit = gen.getDpsInits()[0];
		auto featTy = dyn_cast<RankedTensorType>(feat.getType());
		auto wTy = dyn_cast<RankedTensorType>(weight.getType());
		auto outTy = dyn_cast<RankedTensorType>(outInit.getType());
		if (!featTy || !wTy || !outTy)
			return failure();
		if (featTy.getRank() != 3 || wTy.getRank() != 2 || outTy.getRank() != 3)
			return failure();
		if (!featTy.getElementType().isSignlessInteger(8))
			return failure();
		if (!wTy.getElementType().isSignlessInteger(8))
			return failure();
		if (!outTy.getElementType().isSignlessInteger(32))
			return failure();
		if (!featTy.hasStaticShape() || !outTy.hasStaticShape())
			return failure();
		// feat[C,H,W], weight[F,C], out[F,H,W].
		int64_t C = featTy.getDimSize(0);
		int64_t H = featTy.getDimSize(1);
		int64_t W = featTy.getDimSize(2);
		int64_t F = wTy.getDimSize(0);
		if (wTy.getDimSize(1) != C)
			return failure();
		if (outTy.getDimSize(0) != F || outTy.getDimSize(1) != H ||
			outTy.getDimSize(2) != W)
			return failure();

		Location loc = gen.getLoc();
		// Collapse feat[C,H,W] → [C, H*W]; collapse out_init[F,H,W] →
		// [F, H*W].
		SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};
		auto featFlatTy =
			RankedTensorType::get({C, H * W}, featTy.getElementType());
		auto outFlatTy =
			RankedTensorType::get({F, H * W}, outTy.getElementType());
		Value featFlat = rewriter.create<tensor::CollapseShapeOp>(
			loc, featFlatTy, feat, reassoc);
		Value outFlat = rewriter.create<tensor::CollapseShapeOp>(
			loc, outFlatTy, outInit, reassoc);

		// linalg.matmul with weight as A (M×K=F×C) and feat_flat as B
		// (K×N=C×HW). Default linalg.matmul indexing maps are
		// [(d0,d2), (d2,d1), (d0,d1)] which match feat_flat's K×N
		// layout naturally.
		//
		// Note: tried alternative orientations (operand swap + double
		// transpose to get the "big M" geometry that dronet's fast
		// matmuls use) — produced bit-identical vmfb md5sum, indicating
		// IREE normalizes operand orientation downstream. The choice of
		// orientation at THIS level doesn't affect generated machine
		// code, so we use the simplest form here.
		AffineExpr e0 = getAffineDimExpr(0, ctx);
		AffineExpr e1 = getAffineDimExpr(1, ctx);
		AffineExpr e2 = getAffineDimExpr(2, ctx);
		SmallVector<AffineMap> matmulMaps = {
			AffineMap::get(3, 0, {e0, e2}, ctx), // lhs M×K
			AffineMap::get(3, 0, {e2, e1}, ctx), // rhs K×N
			AffineMap::get(3, 0, {e0, e1}, ctx), // out M×N
		};
		auto matmul =
			rewriter.create<linalg::MatmulOp>(loc, TypeRange{outFlatTy},
				ValueRange{weight, featFlat}, ValueRange{outFlat});
		matmul.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(matmulMaps));

		// Expand back to [F, H, W].
		Value expanded = rewriter.create<tensor::ExpandShapeOp>(
			loc, outTy, matmul.getResult(0), reassoc);

		rewriter.replaceOp(gen, expanded);
		return success();
	}
};

struct GemminiPreprocessPass final
	: public impl::GemminiPreprocessPassBase<GemminiPreprocessPass> {
	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		// SplitRescaleHeadFromActivationTailPattern is defined but not
		// registered: at post-global-opt, IREE has already broken the
		// 28-op chained body into ~8 small generics. The chain only
		// re-unifies as a single linalg.generic AFTER bufferization. The
		// head-split fold lives in LowerTileToISA.cpp instead, as a fall-
		// back when the simple matmul+rescale match fails.
		patterns.add<ConvertDequantMaxPoolQuantPattern,
			MaterializeBTransposeAsLinalgTransposePattern,
			FuseBroadcastIntoBiasAddPattern,
			FlattenMatmulLike4DToMatmulPattern>(&getContext());
		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

// createGemminiPreprocessPass() is auto-generated as a friend of
// GemminiPreprocessPassBase by GEN_PASS_DEF_*; do not redeclare here.

} // namespace mlir::iree_compiler::Gemmini
