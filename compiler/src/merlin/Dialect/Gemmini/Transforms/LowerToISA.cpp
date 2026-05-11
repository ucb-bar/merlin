#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::Gemmini {

#define GEN_PASS_DEF_GEMMINILOWERTOISAPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

namespace {

struct LowerMatmulToTilePattern final : OpRewritePattern<Gemmini::MatmulOp> {
	LowerMatmulToTilePattern(
		MLIRContext *ctx, int64_t tileM, int64_t tileN, int64_t tileK)
		: OpRewritePattern(ctx), tileM(tileM), tileN(tileN), tileK(tileK) {}

	LogicalResult matchAndRewrite(
		Gemmini::MatmulOp op, PatternRewriter &rewriter) const override {
		if (tileM <= 0 || tileN <= 0 || tileK <= 0)
			return failure();

		auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
		auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
		auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
		if (!lhsType || !rhsType || !resultType)
			return failure();
		if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
			resultType.hasStaticShape()) {
			rewriter.replaceOpWithNewOp<Gemmini::MatmulTileOp>(op,
				op.getResult().getType(), op.getLhs(), op.getRhs(),
				op.getLhsZeroPoint(), op.getRhsZeroPoint(), op.getDataflow(),
				static_cast<uint64_t>(tileM), static_cast<uint64_t>(tileN),
				static_cast<uint64_t>(tileK));
			return success();
		}

		return failure();
	}

	int64_t tileM;
	int64_t tileN;
	int64_t tileK;
};

struct LowerToISAPass final
	: public impl::GemminiLowerToISAPassBase<LowerToISAPass> {
	LowerToISAPass() = default;
	explicit LowerToISAPass(const GemminiTransformOptions &options)
		: options(options) {}

	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<LowerMatmulToTilePattern>(
			&getContext(), options.tileM, options.tileN, options.tileK);

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}

	GemminiTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerToISAPass(
	const GemminiTransformOptions &options) {
	return std::make_unique<LowerToISAPass>(options);
}

} // namespace mlir::iree_compiler::Gemmini
