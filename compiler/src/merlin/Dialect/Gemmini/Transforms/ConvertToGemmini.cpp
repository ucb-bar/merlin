#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include <optional>
#include <utility>

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINICONVERTTOGEMMINIPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

static std::optional<int64_t> getIntegerFromAttribute(Attribute attr) {
	if (!attr)
		return std::nullopt;
	if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
		return intAttr.getInt();
	}
	if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
		if (!dense.isSplat())
			return std::nullopt;
		return dense.getSplatValue<APInt>().getSExtValue();
	}
	return std::nullopt;
}

static std::optional<double> getFloatFromAttribute(Attribute attr) {
	if (!attr)
		return std::nullopt;
	if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
		return floatAttr.getValueAsDouble();
	}
	if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
		if (!dense.isSplat())
			return std::nullopt;
		return dense.getSplatValue<APFloat>().convertToDouble();
	}
	return std::nullopt;
}

static std::optional<int64_t> getScalarIntConstant(Value value) {
	Attribute attr;
	if (matchPattern(value, m_Constant(&attr))) {
		return getIntegerFromAttribute(attr);
	}
	return std::nullopt;
}

static std::optional<double> getScalarFloatConstant(Value value) {
	Attribute attr;
	if (matchPattern(value, m_Constant(&attr))) {
		return getFloatFromAttribute(attr);
	}
	return std::nullopt;
}

static std::optional<int64_t> resolveIntValue(
	Value value, linalg::GenericOp genericOp) {
	if (auto blockArg = dyn_cast<BlockArgument>(value)) {
		if (blockArg.getOwner() == &genericOp.getRegion().front() &&
			blockArg.getArgNumber() < genericOp.getNumDpsInputs()) {
			return getScalarIntConstant(
				genericOp.getDpsInputs()[blockArg.getArgNumber()]);
		}
	}
	return getScalarIntConstant(value);
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

static bool isZeroResultMap(AffineMap map) {
	return map && map.getNumResults() == 0;
}

static bool isRankedI8Tensor(Value value, int rank) {
	auto type = dyn_cast<RankedTensorType>(value.getType());
	return type && type.getRank() == rank &&
		type.getElementType().isSignlessInteger(8);
}

static bool isRankedI32Tensor(Value value, int rank) {
	auto type = dyn_cast<RankedTensorType>(value.getType());
	return type && type.getRank() == rank &&
		type.getElementType().isSignlessInteger(32);
}

static bool isRankedF32Tensor(Value value, int rank) {
	auto type = dyn_cast<RankedTensorType>(value.getType());
	return type && type.getRank() == rank && type.getElementType().isF32();
}

static bool extractDimTimesConstant(
	AffineExpr expr, unsigned &dim, int64_t &coeff) {
	if (auto d = dyn_cast<AffineDimExpr>(expr)) {
		dim = d.getPosition();
		coeff = 1;
		return true;
	}

	auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
	if (!bin || bin.getKind() != AffineExprKind::Mul)
		return false;

	if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
		if (auto d = dyn_cast<AffineDimExpr>(bin.getRHS())) {
			dim = d.getPosition();
			coeff = c.getValue();
			return true;
		}
	}
	if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
		if (auto d = dyn_cast<AffineDimExpr>(bin.getLHS())) {
			dim = d.getPosition();
			coeff = c.getValue();
			return true;
		}
	}
	return false;
}

static bool extractAffineAddOfDimTerms(AffineExpr expr, unsigned &dimA,
	int64_t &coeffA, unsigned &dimB, int64_t &coeffB) {
	auto add = dyn_cast<AffineBinaryOpExpr>(expr);
	if (!add || add.getKind() != AffineExprKind::Add)
		return false;
	return extractDimTimesConstant(add.getLHS(), dimA, coeffA) &&
		extractDimTimesConstant(add.getRHS(), dimB, coeffB);
}

static bool matchInt8I8I32AccumulatorBody(
	linalg::GenericOp op, int64_t &lhsZp, int64_t &rhsZp) {
	Block &block = op.getRegion().front();
	if (block.empty())
		return false;

	SmallVector<Operation *> ops;
	for (Operation &nested : block.without_terminator())
		ops.push_back(&nested);
	auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
	if (!yield || yield.getNumOperands() != 1)
		return false;

	lhsZp = 0;
	rhsZp = 0;

	if (block.getNumArguments() == 3) {
		// lhs, rhs, out
		if (ops.size() != 4)
			return false;

		auto extL = dyn_cast<arith::ExtSIOp>(ops[0]);
		auto extR = dyn_cast<arith::ExtSIOp>(ops[1]);
		auto mul = dyn_cast<arith::MulIOp>(ops[2]);
		auto add = dyn_cast<arith::AddIOp>(ops[3]);
		if (!extL || !extR || !mul || !add)
			return false;

		Value lhsArg = block.getArgument(0);
		Value rhsArg = block.getArgument(1);
		Value outArg = block.getArgument(2);

		if (extL.getIn() != lhsArg || extR.getIn() != rhsArg)
			return false;

		auto addLhs = add.getLhs();
		auto addRhs = add.getRhs();
		bool addMatches = ((addLhs == outArg && addRhs == mul.getResult()) ||
			(addRhs == outArg && addLhs == mul.getResult()));
		if (!addMatches)
			return false;

		bool mulMatches = ((mul.getLhs() == extL.getResult() &&
							   mul.getRhs() == extR.getResult()) ||
			(mul.getRhs() == extL.getResult() &&
				mul.getLhs() == extR.getResult()));
		if (!mulMatches)
			return false;

		return yield.getOperand(0) == add.getResult();
	}

	if (block.getNumArguments() == 5) {
		// lhs, rhs, lhs_zp, rhs_zp, out
		if (ops.size() != 6)
			return false;

		auto extL = dyn_cast<arith::ExtSIOp>(ops[0]);
		auto subL = dyn_cast<arith::SubIOp>(ops[1]);
		auto extR = dyn_cast<arith::ExtSIOp>(ops[2]);
		auto subR = dyn_cast<arith::SubIOp>(ops[3]);
		auto mul = dyn_cast<arith::MulIOp>(ops[4]);
		auto add = dyn_cast<arith::AddIOp>(ops[5]);
		if (!extL || !subL || !extR || !subR || !mul || !add)
			return false;

		Value lhsArg = block.getArgument(0);
		Value rhsArg = block.getArgument(1);
		Value lhsZpArg = block.getArgument(2);
		Value rhsZpArg = block.getArgument(3);
		Value outArg = block.getArgument(4);

		if (extL.getIn() != lhsArg || extR.getIn() != rhsArg)
			return false;
		if (subL.getLhs() != extL.getResult() || subL.getRhs() != lhsZpArg) {
			return false;
		}
		if (subR.getLhs() != extR.getResult() || subR.getRhs() != rhsZpArg) {
			return false;
		}

		bool mulMatches = ((mul.getLhs() == subL.getResult() &&
							   mul.getRhs() == subR.getResult()) ||
			(mul.getRhs() == subL.getResult() &&
				mul.getLhs() == subR.getResult()));
		if (!mulMatches)
			return false;

		bool addMatches =
			((add.getLhs() == outArg && add.getRhs() == mul.getResult()) ||
				(add.getRhs() == outArg && add.getLhs() == mul.getResult()));
		if (!addMatches)
			return false;

		auto maybeLhsZp = resolveIntValue(lhsZpArg, op);
		auto maybeRhsZp = resolveIntValue(rhsZpArg, op);
		if (!maybeLhsZp || !maybeRhsZp)
			return false;

		lhsZp = *maybeLhsZp;
		rhsZp = *maybeRhsZp;
		return yield.getOperand(0) == add.getResult();
	}

	return false;
}

static bool matchMatmulMaps(linalg::GenericOp op) {
	auto maps = op.getIndexingMapsArray();
	if (maps.size() != 3 && maps.size() != 5)
		return false;

	MLIRContext *ctx = op.getContext();
	AffineExpr d0 = getAffineDimExpr(0, ctx);
	AffineExpr d1 = getAffineDimExpr(1, ctx);
	AffineExpr d2 = getAffineDimExpr(2, ctx);

	AffineMap lhsMap = AffineMap::get(3, 0, {d0, d2}, ctx);
	AffineMap rhsMap = AffineMap::get(3, 0, {d1, d2}, ctx);
	AffineMap outMap = AffineMap::get(3, 0, {d0, d1}, ctx);

	if (maps[0] != lhsMap || maps[1] != rhsMap || maps.back() != outMap) {
		return false;
	}
	if (maps.size() == 5) {
		if (!isZeroResultMap(maps[2]) || !isZeroResultMap(maps[3])) {
			return false;
		}
	}
	return true;
}

static bool matchConv2DMaps(linalg::GenericOp op, int64_t &strideH,
	int64_t &strideW, int64_t &dilationH, int64_t &dilationW) {
	auto maps = op.getIndexingMapsArray();
	if (maps.size() != 3 && maps.size() != 5)
		return false;

	AffineMap inMap = maps[0];
	AffineMap filMap = maps[1];
	AffineMap outMap = maps.back();

	if (inMap.getNumDims() != 6 || filMap.getNumDims() != 6 ||
		outMap.getNumDims() != 6) {
		return false;
	}
	if (inMap.getNumResults() != 3 || filMap.getNumResults() != 4 ||
		outMap.getNumResults() != 3) {
		return false;
	}

	auto outR0 = dyn_cast<AffineDimExpr>(outMap.getResult(0));
	auto outR1 = dyn_cast<AffineDimExpr>(outMap.getResult(1));
	auto outR2 = dyn_cast<AffineDimExpr>(outMap.getResult(2));
	if (!outR0 || !outR1 || !outR2)
		return false;
	if (outR0.getPosition() != 0 || outR1.getPosition() != 1 ||
		outR2.getPosition() != 2) {
		return false;
	}

	auto fil0 = dyn_cast<AffineDimExpr>(filMap.getResult(0));
	auto fil1 = dyn_cast<AffineDimExpr>(filMap.getResult(1));
	auto fil2 = dyn_cast<AffineDimExpr>(filMap.getResult(2));
	auto fil3 = dyn_cast<AffineDimExpr>(filMap.getResult(3));
	if (!fil0 || !fil1 || !fil2 || !fil3)
		return false;
	if (fil0.getPosition() != 0 || fil1.getPosition() != 3 ||
		fil2.getPosition() != 4 || fil3.getPosition() != 5) {
		return false;
	}

	auto in0 = dyn_cast<AffineDimExpr>(inMap.getResult(0));
	if (!in0 || in0.getPosition() != 3)
		return false;

	unsigned dimA = 0, dimB = 0;
	int64_t coeffA = 0, coeffB = 0;
	if (!extractAffineAddOfDimTerms(
			inMap.getResult(1), dimA, coeffA, dimB, coeffB)) {
		return false;
	}
	if (!((dimA == 1 && dimB == 4) || (dimA == 4 && dimB == 1)))
		return false;
	strideH = (dimA == 1) ? coeffA : coeffB;
	dilationH = (dimA == 4) ? coeffA : coeffB;
	if (dimA == 1 && dimB == 4) {
		// already set
	} else {
		std::swap(strideH, dilationH);
	}

	if (!extractAffineAddOfDimTerms(
			inMap.getResult(2), dimA, coeffA, dimB, coeffB)) {
		return false;
	}
	if (!((dimA == 2 && dimB == 5) || (dimA == 5 && dimB == 2)))
		return false;
	strideW = (dimA == 2) ? coeffA : coeffB;
	dilationW = (dimA == 5) ? coeffA : coeffB;
	if (!(dimA == 2 && dimB == 5)) {
		std::swap(strideW, dilationW);
	}

	if (maps.size() == 5) {
		if (!isZeroResultMap(maps[2]) || !isZeroResultMap(maps[3])) {
			return false;
		}
	}
	return strideH > 0 && strideW > 0 && dilationH > 0 && dilationW > 0;
}

struct ConvertMatmulPattern final : OpRewritePattern<linalg::GenericOp> {
	ConvertMatmulPattern(
		MLIRContext *ctx, const GemminiTransformOptions &options)
		: OpRewritePattern(ctx), options(options) {}

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!options.enableMatmul)
			return failure();
		if (!op.hasPureTensorSemantics())
			return failure();
		if (op.getNumLoops() != 3)
			return failure();
		if (op.getNumDpsInputs() != 2 && op.getNumDpsInputs() != 4)
			return failure();
		if (op.getNumDpsInits() != 1)
			return failure();

		if (!isRankedI8Tensor(op.getDpsInputs()[0], 2) ||
			!isRankedI8Tensor(op.getDpsInputs()[1], 2) ||
			!isRankedI32Tensor(op.getResult(0), 2)) {
			return failure();
		}

		if (!matchMatmulMaps(op))
			return failure();

		int64_t lhsZp = 0;
		int64_t rhsZp = 0;
		if (!matchInt8I8I32AccumulatorBody(op, lhsZp, rhsZp))
			return failure();

		rewriter.replaceOpWithNewOp<Gemmini::MatmulOp>(op, op.getResultTypes(),
			op.getDpsInputs()[0], op.getDpsInputs()[1],
			rewriter.getI64IntegerAttr(lhsZp),
			rewriter.getI64IntegerAttr(rhsZp),
			Gemmini::DataflowAttr::get(
				op.getContext(), options.defaultDataflow));
		return success();
	}

	GemminiTransformOptions options;
};

struct ConvertConv2DPattern final : OpRewritePattern<linalg::GenericOp> {
	ConvertConv2DPattern(
		MLIRContext *ctx, const GemminiTransformOptions &options)
		: OpRewritePattern(ctx), options(options) {}

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!options.enableConv2D)
			return failure();
		if (!op.hasPureTensorSemantics())
			return failure();
		if (op.getNumLoops() != 6)
			return failure();
		if (op.getNumDpsInputs() != 2 && op.getNumDpsInputs() != 4)
			return failure();
		if (op.getNumDpsInits() != 1)
			return failure();

		if (!isRankedI8Tensor(op.getDpsInputs()[0], 3) ||
			!isRankedI8Tensor(op.getDpsInputs()[1], 4) ||
			!isRankedI32Tensor(op.getResult(0), 3)) {
			return failure();
		}

		int64_t strideH = 0, strideW = 0, dilationH = 0, dilationW = 0;
		if (!matchConv2DMaps(op, strideH, strideW, dilationH, dilationW)) {
			return failure();
		}

		int64_t inputZp = 0;
		int64_t filterZp = 0;
		if (!matchInt8I8I32AccumulatorBody(op, inputZp, filterZp))
			return failure();

		rewriter.replaceOpWithNewOp<Gemmini::Conv2DOp>(op, op.getResultTypes(),
			op.getDpsInputs()[0], op.getDpsInputs()[1],
			rewriter.getI64IntegerAttr(strideH),
			rewriter.getI64IntegerAttr(strideW),
			rewriter.getI64IntegerAttr(dilationH),
			rewriter.getI64IntegerAttr(dilationW),
			rewriter.getI64IntegerAttr(inputZp),
			rewriter.getI64IntegerAttr(filterZp),
			Gemmini::DataflowAttr::get(
				op.getContext(), options.defaultDataflow));
		return success();
	}

	GemminiTransformOptions options;
};

struct ConvertRequantizePattern final : OpRewritePattern<linalg::GenericOp> {
	ConvertRequantizePattern(
		MLIRContext *ctx, const GemminiTransformOptions &options)
		: OpRewritePattern(ctx), options(options) {}

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!options.enableRequantize)
			return failure();
		if (!op.hasPureTensorSemantics())
			return failure();
		if (op.getNumLoops() < 1)
			return failure();
		if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
			return failure();

		if (!isRankedF32Tensor(op.getDpsInputs()[0],
				cast<RankedTensorType>(op.getDpsInputs()[0].getType())
					.getRank())) {
			return failure();
		}

		auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!outType || !outType.getElementType().isSignlessInteger(8)) {
			return failure();
		}

		Block &block = op.getRegion().front();
		SmallVector<Operation *> ops;
		for (Operation &nested : block.without_terminator())
			ops.push_back(&nested);
		auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
		if (!yield || ops.size() != 6)
			return failure();

		auto div = dyn_cast<arith::DivFOp>(ops[0]);
		auto rnd = dyn_cast<math::RoundEvenOp>(ops[1]);
		auto add = dyn_cast<arith::AddFOp>(ops[2]);
		auto maxf = dyn_cast<arith::MaximumFOp>(ops[3]);
		auto minf = dyn_cast<arith::MinimumFOp>(ops[4]);
		auto cast = dyn_cast<arith::FPToSIOp>(ops[5]);
		if (!div || !rnd || !add || !maxf || !minf || !cast)
			return failure();

		Value inputArg = block.getArgument(0);
		if (div.getLhs() != inputArg)
			return failure();
		if (rnd.getOperand() != div.getResult())
			return failure();
		if (add.getLhs() != rnd.getResult())
			return failure();
		if (maxf.getLhs() != add.getResult())
			return failure();
		if (minf.getLhs() != maxf.getResult())
			return failure();
		if (cast.getIn() != minf.getResult())
			return failure();
		if (yield.getOperand(0) != cast.getResult())
			return failure();

		auto scale = resolveFloatValue(div.getRhs(), op);
		auto zp = resolveFloatValue(add.getRhs(), op);
		auto qmin = resolveFloatValue(maxf.getRhs(), op);
		auto qmax = resolveFloatValue(minf.getRhs(), op);
		if (!scale || !zp || !qmin || !qmax)
			return failure();

		rewriter.replaceOpWithNewOp<Gemmini::RequantizeOp>(op,
			op.getResultTypes(), op.getDpsInputs()[0],
			rewriter.getF32FloatAttr(static_cast<float>(*scale)),
			rewriter.getI64IntegerAttr(static_cast<int64_t>(*zp)),
			rewriter.getI64IntegerAttr(static_cast<int64_t>(*qmin)),
			rewriter.getI64IntegerAttr(static_cast<int64_t>(*qmax)));
		return success();
	}

	GemminiTransformOptions options;
};

struct ConvertClampPattern final : OpRewritePattern<linalg::GenericOp> {
	ConvertClampPattern(
		MLIRContext *ctx, const GemminiTransformOptions &options)
		: OpRewritePattern(ctx), options(options) {}

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!options.enableClamp)
			return failure();
		if (!op.hasPureTensorSemantics())
			return failure();
		if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1)
			return failure();

		auto inType =
			dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
		auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!inType || !outType || !inType.getElementType().isF32() ||
			!outType.getElementType().isF32() || inType != outType) {
			return failure();
		}

		Block &block = op.getRegion().front();
		SmallVector<Operation *> ops;
		for (Operation &nested : block.without_terminator())
			ops.push_back(&nested);
		auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
		if (!yield || ops.size() != 4)
			return failure();

		auto cmpMin = dyn_cast<arith::CmpFOp>(ops[0]);
		auto selMin = dyn_cast<arith::SelectOp>(ops[1]);
		auto cmpMax = dyn_cast<arith::CmpFOp>(ops[2]);
		auto selMax = dyn_cast<arith::SelectOp>(ops[3]);
		if (!cmpMin || !selMin || !cmpMax || !selMax)
			return failure();

		Value inArg = block.getArgument(0);
		Value minArg = block.getArgument(1);
		Value maxArg = block.getArgument(2);

		if (cmpMin.getPredicate() != arith::CmpFPredicate::ULT)
			return failure();
		if (cmpMin.getLhs() != inArg || cmpMin.getRhs() != minArg)
			return failure();

		if (selMin.getCondition() != cmpMin.getResult() ||
			selMin.getTrueValue() != minArg ||
			selMin.getFalseValue() != inArg) {
			return failure();
		}

		if (cmpMax.getPredicate() != arith::CmpFPredicate::UGT)
			return failure();
		if (cmpMax.getLhs() != selMin.getResult() ||
			cmpMax.getRhs() != maxArg) {
			return failure();
		}

		if (selMax.getCondition() != cmpMax.getResult() ||
			selMax.getTrueValue() != maxArg ||
			selMax.getFalseValue() != selMin.getResult()) {
			return failure();
		}

		if (yield.getOperand(0) != selMax.getResult())
			return failure();

		auto minValue = resolveFloatValue(minArg, op);
		auto maxValue = resolveFloatValue(maxArg, op);
		if (!minValue || !maxValue)
			return failure();

		rewriter.replaceOpWithNewOp<Gemmini::ClampOp>(op, op.getResultTypes(),
			op.getDpsInputs()[0],
			rewriter.getF32FloatAttr(static_cast<float>(*minValue)),
			rewriter.getF32FloatAttr(static_cast<float>(*maxValue)));
		return success();
	}

	GemminiTransformOptions options;
};

struct ConvertToGemminiPass final
	: public impl::GemminiConvertToGemminiPassBase<ConvertToGemminiPass> {
	ConvertToGemminiPass() = default;
	explicit ConvertToGemminiPass(const GemminiTransformOptions &options)
		: options(options) {}

	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<ConvertMatmulPattern, ConvertConv2DPattern,
			ConvertRequantizePattern, ConvertClampPattern>(
			&getContext(), options);

		if (failed(applyPatternsAndFoldGreedily(
				getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}

	GemminiTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createConvertToGemminiPass(
	const GemminiTransformOptions &options) {
	return std::make_unique<ConvertToGemminiPass>(options);
}

} // namespace mlir::iree_compiler::Gemmini
