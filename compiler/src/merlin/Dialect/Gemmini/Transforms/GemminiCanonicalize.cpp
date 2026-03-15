#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINICANONICALIZEPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

struct FoldClampOfClampPattern final : OpRewritePattern<Gemmini::ClampOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::ClampOp op, PatternRewriter &rewriter) const override {
		auto prev = op.getInput().getDefiningOp<Gemmini::ClampOp>();
		if (!prev)
			return failure();

		float minValue = std::max(prev.getMinValue().convertToDouble(),
			op.getMinValue().convertToDouble());
		float maxValue = std::min(prev.getMaxValue().convertToDouble(),
			op.getMaxValue().convertToDouble());

		rewriter.replaceOpWithNewOp<Gemmini::ClampOp>(op,
			op.getResult().getType(), prev.getInput(),
			rewriter.getF32FloatAttr(minValue),
			rewriter.getF32FloatAttr(maxValue));
		return success();
	}
};

struct ElideIdentityClampPattern final : OpRewritePattern<Gemmini::ClampOp> {
	using OpRewritePattern::OpRewritePattern;

	LogicalResult matchAndRewrite(
		Gemmini::ClampOp op, PatternRewriter &rewriter) const override {
		float minValue = op.getMinValue().convertToDouble();
		float maxValue = op.getMaxValue().convertToDouble();
		if (minValue <= -std::numeric_limits<float>::max() &&
			maxValue >= std::numeric_limits<float>::max()) {
			rewriter.replaceOp(op, op.getInput());
			return success();
		}
		return failure();
	}
};

struct GemminiCanonicalizePass final
	: public impl::GemminiCanonicalizePassBase<GemminiCanonicalizePass> {
	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<FoldClampOfClampPattern, ElideIdentityClampPattern>(
			&getContext());

		if (failed(applyPatternsAndFoldGreedily(
				getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGemminiCanonicalizeFuncPass() {
	return std::make_unique<GemminiCanonicalizePass>();
}

} // namespace mlir::iree_compiler::Gemmini
