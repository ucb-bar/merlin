#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINILOWERTOIREEPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

static SmallVector<Value> getIdentityDynamicDims(
	OpBuilder &builder, Location loc, Value source) {
	SmallVector<Value> dims;
	auto type = cast<RankedTensorType>(source.getType());
	for (int64_t i = 0, e = type.getRank(); i < e; ++i) {
		if (type.isDynamicDim(i)) {
			dims.push_back(builder.create<tensor::DimOp>(loc, source, i));
		}
	}
	return dims;
}

static Value createEmptyForType(OpBuilder &builder, Location loc,
	RankedTensorType type, ValueRange dynamicDims) {
	return builder.create<tensor::EmptyOp>(
		loc, type.getShape(), type.getElementType(), dynamicDims);
}

static Value createZeroFilledTensor(OpBuilder &builder, Location loc,
	RankedTensorType type, ValueRange dynamicDims) {
	Value empty = createEmptyForType(builder, loc, type, dynamicDims);
	Type elemType = type.getElementType();
	Value zero;
	if (elemType.isSignlessInteger()) {
		zero = builder.create<arith::ConstantIntOp>(
			loc, cast<IntegerType>(elemType), 0);
	} else if (elemType.isF32()) {
		zero = builder.create<arith::ConstantFloatOp>(
			loc, cast<FloatType>(elemType), APFloat(0.0f));
	} else {
		llvm_unreachable("unsupported element type for zero fill");
	}
	return builder.create<linalg::FillOp>(loc, zero, empty).getResult(0);
}

static SmallVector<Value> getMatmulResultDynamicDims(OpBuilder &builder,
	Location loc, Value lhs, Value rhs, RankedTensorType resultType) {
	SmallVector<Value> dims;
	if (resultType.isDynamicDim(0)) {
		dims.push_back(builder.create<tensor::DimOp>(loc, lhs, 0));
	}
	if (resultType.isDynamicDim(1)) {
		dims.push_back(builder.create<tensor::DimOp>(loc, rhs, 0));
	}
	return dims;
}

static Value computeConvOutDim(OpBuilder &builder, Location loc, Value inSize,
	Value kernelSize, int64_t stride, int64_t dilation) {
	Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
	Value cStride = builder.create<arith::ConstantIndexOp>(loc, stride);
	Value cDilation = builder.create<arith::ConstantIndexOp>(loc, dilation);

	Value km1 = builder.create<arith::SubIOp>(loc, kernelSize, c1);
	Value effKernelMinusOne =
		builder.create<arith::MulIOp>(loc, km1, cDilation);
	Value effKernel = builder.create<arith::AddIOp>(loc, effKernelMinusOne, c1);
	Value numer = builder.create<arith::SubIOp>(loc, inSize, effKernel);
	Value div = builder.create<arith::DivUIOp>(loc, numer, cStride);
	return builder.create<arith::AddIOp>(loc, div, c1);
}

static SmallVector<Value> getConvResultDynamicDims(
	OpBuilder &builder, Location loc, Gemmini::Conv2DOp op) {
	SmallVector<Value> dims;
	auto resultType = cast<RankedTensorType>(op.getResult().getType());

	Value inputH = builder.create<tensor::DimOp>(loc, op.getInput(), 1);
	Value inputW = builder.create<tensor::DimOp>(loc, op.getInput(), 2);
	Value filterF = builder.create<tensor::DimOp>(loc, op.getFilter(), 0);
	Value filterKH = builder.create<tensor::DimOp>(loc, op.getFilter(), 2);
	Value filterKW = builder.create<tensor::DimOp>(loc, op.getFilter(), 3);

	Value outH = computeConvOutDim(builder, loc, inputH, filterKH,
		static_cast<int64_t>(op.getStrideH()),
		static_cast<int64_t>(op.getDilationH()));
	Value outW = computeConvOutDim(builder, loc, inputW, filterKW,
		static_cast<int64_t>(op.getStrideW()),
		static_cast<int64_t>(op.getDilationW()));

	if (resultType.isDynamicDim(0))
		dims.push_back(filterF);
	if (resultType.isDynamicDim(1))
		dims.push_back(outH);
	if (resultType.isDynamicDim(2))
		dims.push_back(outW);

	return dims;
}

struct LowerMatmulPattern final : OpRewritePattern<Gemmini::MatmulOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::MatmulOp op, PatternRewriter &rewriter) const override {
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		Location loc = op.getLoc();

		SmallVector<Value> dynamicDims = getMatmulResultDynamicDims(
			rewriter, loc, op.getLhs(), op.getRhs(), resultType);
		Value init =
			createZeroFilledTensor(rewriter, loc, resultType, dynamicDims);

		MLIRContext *ctx = op.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineExpr d2 = getAffineDimExpr(2, ctx);

		SmallVector<AffineMap> maps = {
			AffineMap::get(3, 0, {d0, d2}, ctx),
			AffineMap::get(3, 0, {d1, d2}, ctx),
			AffineMap::get(3, 0, {d0, d1}, ctx),
		};
		SmallVector<utils::IteratorType> iterators = {
			utils::IteratorType::parallel,
			utils::IteratorType::parallel,
			utils::IteratorType::reduction,
		};

		int64_t lhsZp = static_cast<int64_t>(op.getLhsZeroPoint());
		int64_t rhsZp = static_cast<int64_t>(op.getRhsZeroPoint());

		auto generic = rewriter.create<linalg::GenericOp>(loc, resultType,
			ValueRange{op.getLhs(), op.getRhs()}, ValueRange{init}, maps,
			iterators, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
				Value lhs = b.create<arith::ExtSIOp>(
					nestedLoc, b.getI32Type(), args[0]);
				Value rhs = b.create<arith::ExtSIOp>(
					nestedLoc, b.getI32Type(), args[1]);
				if (lhsZp != 0) {
					Value zp =
						b.create<arith::ConstantIntOp>(nestedLoc, lhsZp, 32);
					lhs = b.create<arith::SubIOp>(nestedLoc, lhs, zp);
				}
				if (rhsZp != 0) {
					Value zp =
						b.create<arith::ConstantIntOp>(nestedLoc, rhsZp, 32);
					rhs = b.create<arith::SubIOp>(nestedLoc, rhs, zp);
				}
				Value prod = b.create<arith::MulIOp>(nestedLoc, lhs, rhs);
				Value acc = b.create<arith::AddIOp>(nestedLoc, args[2], prod);
				b.create<linalg::YieldOp>(nestedLoc, acc);
			});

		rewriter.replaceOp(op, generic.getResults());
		return success();
	}
};

struct LowerMatmulTilePattern final : OpRewritePattern<Gemmini::MatmulTileOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::MatmulTileOp op, PatternRewriter &rewriter) const override {
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		Location loc = op.getLoc();

		SmallVector<Value> dynamicDims = getMatmulResultDynamicDims(
			rewriter, loc, op.getLhs(), op.getRhs(), resultType);
		Value init =
			createZeroFilledTensor(rewriter, loc, resultType, dynamicDims);

		MLIRContext *ctx = op.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx);
		AffineExpr d1 = getAffineDimExpr(1, ctx);
		AffineExpr d2 = getAffineDimExpr(2, ctx);

		SmallVector<AffineMap> maps = {
			AffineMap::get(3, 0, {d0, d2}, ctx),
			AffineMap::get(3, 0, {d1, d2}, ctx),
			AffineMap::get(3, 0, {d0, d1}, ctx),
		};
		SmallVector<utils::IteratorType> iterators = {
			utils::IteratorType::parallel,
			utils::IteratorType::parallel,
			utils::IteratorType::reduction,
		};

		int64_t lhsZp = static_cast<int64_t>(op.getLhsZeroPoint());
		int64_t rhsZp = static_cast<int64_t>(op.getRhsZeroPoint());

		auto generic = rewriter.create<linalg::GenericOp>(loc, resultType,
			ValueRange{op.getLhs(), op.getRhs()}, ValueRange{init}, maps,
			iterators, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
				Value lhs = b.create<arith::ExtSIOp>(
					nestedLoc, b.getI32Type(), args[0]);
				Value rhs = b.create<arith::ExtSIOp>(
					nestedLoc, b.getI32Type(), args[1]);
				if (lhsZp != 0) {
					Value zp =
						b.create<arith::ConstantIntOp>(nestedLoc, lhsZp, 32);
					lhs = b.create<arith::SubIOp>(nestedLoc, lhs, zp);
				}
				if (rhsZp != 0) {
					Value zp =
						b.create<arith::ConstantIntOp>(nestedLoc, rhsZp, 32);
					rhs = b.create<arith::SubIOp>(nestedLoc, rhs, zp);
				}
				Value prod = b.create<arith::MulIOp>(nestedLoc, lhs, rhs);
				Value acc = b.create<arith::AddIOp>(nestedLoc, args[2], prod);
				b.create<linalg::YieldOp>(nestedLoc, acc);
			});

		rewriter.replaceOp(op, generic.getResults());
		return success();
	}
};

struct LowerConv2DPattern final : OpRewritePattern<Gemmini::Conv2DOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::Conv2DOp op, PatternRewriter &rewriter) const override {
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		Location loc = op.getLoc();

		SmallVector<Value> dynamicDims =
			getConvResultDynamicDims(rewriter, loc, op);
		Value init =
			createZeroFilledTensor(rewriter, loc, resultType, dynamicDims);

		MLIRContext *ctx = op.getContext();
		AffineExpr d0 = getAffineDimExpr(0, ctx); // f
		AffineExpr d1 = getAffineDimExpr(1, ctx); // oh
		AffineExpr d2 = getAffineDimExpr(2, ctx); // ow
		AffineExpr d3 = getAffineDimExpr(3, ctx); // c
		AffineExpr d4 = getAffineDimExpr(4, ctx); // kh
		AffineExpr d5 = getAffineDimExpr(5, ctx); // kw

		int64_t sh = static_cast<int64_t>(op.getStrideH());
		int64_t sw = static_cast<int64_t>(op.getStrideW());
		int64_t dh = static_cast<int64_t>(op.getDilationH());
		int64_t dw = static_cast<int64_t>(op.getDilationW());

		SmallVector<AffineMap> maps = {
			AffineMap::get(
				6, 0, {d3, d1 * sh + d4 * dh, d2 * sw + d5 * dw}, ctx),
			AffineMap::get(6, 0, {d0, d3, d4, d5}, ctx),
			AffineMap::get(6, 0, {d0, d1, d2}, ctx),
		};
		SmallVector<utils::IteratorType> iterators = {
			utils::IteratorType::parallel,
			utils::IteratorType::parallel,
			utils::IteratorType::parallel,
			utils::IteratorType::reduction,
			utils::IteratorType::reduction,
			utils::IteratorType::reduction,
		};

		int64_t inputZp = static_cast<int64_t>(op.getInputZeroPoint());
		int64_t filterZp = static_cast<int64_t>(op.getFilterZeroPoint());

		auto generic = rewriter.create<linalg::GenericOp>(loc, resultType,
			ValueRange{op.getInput(), op.getFilter()}, ValueRange{init}, maps,
			iterators, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
				Value in = b.create<arith::ExtSIOp>(
					nestedLoc, b.getI32Type(), args[0]);
				Value fil = b.create<arith::ExtSIOp>(
					nestedLoc, b.getI32Type(), args[1]);
				if (inputZp != 0) {
					Value zp =
						b.create<arith::ConstantIntOp>(nestedLoc, inputZp, 32);
					in = b.create<arith::SubIOp>(nestedLoc, in, zp);
				}
				if (filterZp != 0) {
					Value zp =
						b.create<arith::ConstantIntOp>(nestedLoc, filterZp, 32);
					fil = b.create<arith::SubIOp>(nestedLoc, fil, zp);
				}
				Value prod = b.create<arith::MulIOp>(nestedLoc, in, fil);
				Value acc = b.create<arith::AddIOp>(nestedLoc, args[2], prod);
				b.create<linalg::YieldOp>(nestedLoc, acc);
			});

		rewriter.replaceOp(op, generic.getResults());
		return success();
	}
};

struct LowerRequantizePattern final : OpRewritePattern<Gemmini::RequantizeOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::RequantizeOp op, PatternRewriter &rewriter) const override {
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		Location loc = op.getLoc();

		SmallVector<Value> dynamicDims =
			getIdentityDynamicDims(rewriter, loc, op.getInput());
		Value empty =
			createEmptyForType(rewriter, loc, resultType, dynamicDims);

		float scale = op.getScale().convertToDouble();
		float zp = static_cast<float>(op.getZeroPoint());
		float qmin = static_cast<float>(op.getQmin());
		float qmax = static_cast<float>(op.getQmax());

		auto map = AffineMap::getMultiDimIdentityMap(
			resultType.getRank(), op.getContext());
		SmallVector<AffineMap> maps = {map, map};
		SmallVector<utils::IteratorType> iterators(
			resultType.getRank(), utils::IteratorType::parallel);

		auto generic = rewriter.create<linalg::GenericOp>(loc, resultType,
			ValueRange{op.getInput()}, ValueRange{empty}, maps, iterators,
			[&](OpBuilder &b, Location nestedLoc, ValueRange args) {
				Value cScale = b.create<arith::ConstantFloatOp>(
					nestedLoc, b.getF32Type(), APFloat(scale));
				Value cZp = b.create<arith::ConstantFloatOp>(
					nestedLoc, b.getF32Type(), APFloat(zp));
				Value cQmin = b.create<arith::ConstantFloatOp>(
					nestedLoc, b.getF32Type(), APFloat(qmin));
				Value cQmax = b.create<arith::ConstantFloatOp>(
					nestedLoc, b.getF32Type(), APFloat(qmax));

				Value div = b.create<arith::DivFOp>(nestedLoc, args[0], cScale);
				Value round = b.create<math::RoundEvenOp>(nestedLoc, div);
				Value add = b.create<arith::AddFOp>(nestedLoc, round, cZp);
				Value maxf = b.create<arith::MaximumFOp>(nestedLoc, add, cQmin);
				Value minf =
					b.create<arith::MinimumFOp>(nestedLoc, maxf, cQmax);
				Value cast =
					b.create<arith::FPToSIOp>(nestedLoc, b.getI8Type(), minf);
				b.create<linalg::YieldOp>(nestedLoc, cast);
			});

		rewriter.replaceOp(op, generic.getResults());
		return success();
	}
};

struct LowerClampPattern final : OpRewritePattern<Gemmini::ClampOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::ClampOp op, PatternRewriter &rewriter) const override {
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		Location loc = op.getLoc();

		SmallVector<Value> dynamicDims =
			getIdentityDynamicDims(rewriter, loc, op.getInput());
		Value empty =
			createEmptyForType(rewriter, loc, resultType, dynamicDims);

		float minValue = op.getMinValue().convertToDouble();
		float maxValue = op.getMaxValue().convertToDouble();

		auto map = AffineMap::getMultiDimIdentityMap(
			resultType.getRank(), op.getContext());
		SmallVector<AffineMap> maps = {map, map};
		SmallVector<utils::IteratorType> iterators(
			resultType.getRank(), utils::IteratorType::parallel);

		auto generic = rewriter.create<linalg::GenericOp>(loc, resultType,
			ValueRange{op.getInput()}, ValueRange{empty}, maps, iterators,
			[&](OpBuilder &b, Location nestedLoc, ValueRange args) {
				Value cMin = b.create<arith::ConstantFloatOp>(
					nestedLoc, b.getF32Type(), APFloat(minValue));
				Value cMax = b.create<arith::ConstantFloatOp>(
					nestedLoc, b.getF32Type(), APFloat(maxValue));
				Value cmpMin = b.create<arith::CmpFOp>(
					nestedLoc, arith::CmpFPredicate::ULT, args[0], cMin);
				Value selMin =
					b.create<arith::SelectOp>(nestedLoc, cmpMin, cMin, args[0]);
				Value cmpMax = b.create<arith::CmpFOp>(
					nestedLoc, arith::CmpFPredicate::UGT, selMin, cMax);
				Value selMax =
					b.create<arith::SelectOp>(nestedLoc, cmpMax, cMax, selMin);
				b.create<linalg::YieldOp>(nestedLoc, selMax);
			});

		rewriter.replaceOp(op, generic.getResults());
		return success();
	}
};

struct LowerGemminiToIREEPass final
	: public impl::GemminiLowerToIREEPassBase<LowerGemminiToIREEPass> {
	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<LowerMatmulPattern, LowerMatmulTilePattern,
			LowerConv2DPattern, LowerRequantizePattern, LowerClampPattern>(
			&getContext());

		if (failed(applyPatternsAndFoldGreedily(
				getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLowerGemminiToIREEPass() {
	return std::make_unique<LowerGemminiToIREEPass>();
}

} // namespace mlir::iree_compiler::Gemmini
