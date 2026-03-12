#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::NPU {
namespace {

struct LowerKernelMatmulToSchedulePattern
	: public OpRewritePattern<NPUKernel::MatmulOp> {
	using OpRewritePattern<NPUKernel::MatmulOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		NPUKernel::MatmulOp op, PatternRewriter &rewriter) const override {
		rewriter.replaceOpWithNewOp<NPUSchedule::MatmulTileOp>(
			op, op->getResultTypes(), op.getLhs(), op.getRhs());
		return success();
	}
};

struct LowerKernelUKernelToSchedulePattern
	: public OpRewritePattern<NPUKernel::UKernelGenericOp> {
	using OpRewritePattern<NPUKernel::UKernelGenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(NPUKernel::UKernelGenericOp op,
		PatternRewriter &rewriter) const override {
		rewriter.replaceOpWithNewOp<NPUSchedule::UKernelLaunchOp>(
			op, op.getResult().getType(), op.getSymbol(), op.getInputs());
		return success();
	}
};

struct ConvertNPUKernelToSchedulePass
	: public PassWrapper<ConvertNPUKernelToSchedulePass,
		  OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNPUKernelToSchedulePass)

	StringRef getArgument() const final {
		return "convert-npu-kernel-to-schedule";
	}
	StringRef getDescription() const final {
		return "Lower npu_kernel.matmul/ukernel_generic to npu_schedule ops";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUKernel::NPUKernelDialect>();
		registry.insert<NPUSchedule::NPUScheduleDialect>();
	}

	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<LowerKernelMatmulToSchedulePattern>(&getContext());
		patterns.add<LowerKernelUKernelToSchedulePattern>(&getContext());

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createConvertNPUKernelToSchedulePass() {
	return std::make_unique<ConvertNPUKernelToSchedulePass>();
}

void registerConvertNPUKernelToSchedulePass() {
	PassRegistration<ConvertNPUKernelToSchedulePass>();
}

} // namespace mlir::iree_compiler::NPU
