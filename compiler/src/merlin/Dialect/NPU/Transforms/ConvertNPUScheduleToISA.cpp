#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::NPU {
namespace {

static IntegerAttr i64(PatternRewriter &rewriter, int64_t v) {
	return rewriter.getI64IntegerAttr(v);
}

static void emitMatmulWeightLoad(
	Location loc, PatternRewriter &rewriter, bool matmulUseMxu1Weights) {
	if (matmulUseMxu1Weights) {
		rewriter.create<NPUISA::DmaLoadMxu1Op>(loc, i64(rewriter, 1),
			i64(rewriter, 2048), i64(rewriter, 512), i64(rewriter, 0));
	} else {
		rewriter.create<NPUISA::DmaLoadMxu0Op>(loc, i64(rewriter, 1),
			i64(rewriter, 2048), i64(rewriter, 512), i64(rewriter, 0));
	}
}

static Value emitMatmulSkeleton(Location loc, Value lhs, Value rhs,
	Type resultType, PatternRewriter &rewriter, bool matmulUseMxu1Weights) {
	// Deterministic ISA skeleton for a single matmul tile / ukernel launch.
	rewriter.create<NPUISA::DmaLoadOp>(loc, i64(rewriter, 2), i64(rewriter, 0),
		i64(rewriter, 2048), i64(rewriter, 0));
	rewriter.create<NPUISA::DmaWaitOp>(loc, i64(rewriter, 0));
	emitMatmulWeightLoad(loc, rewriter, matmulUseMxu1Weights);
	rewriter.create<NPUISA::DmaWaitOp>(loc, i64(rewriter, 0));
	auto matmul = rewriter.create<NPUISA::MatmulMxu0Op>(loc, resultType, lhs,
		rhs, i64(rewriter, 0), i64(rewriter, 2), i64(rewriter, 1));
	return matmul.getResult();
}

static Value emitSoftmaxVectorChain(
	Location loc, Value input, PatternRewriter &rewriter) {
	auto scaled = rewriter.create<NPUISA::VMulOp>(loc, input.getType(), input,
		input, i64(rewriter, 4), i64(rewriter, 3), i64(rewriter, 2));
	auto exp = rewriter.create<NPUISA::VExpOp>(loc, input.getType(),
		scaled.getResult(), i64(rewriter, 5), i64(rewriter, 4));
	auto sum = rewriter.create<NPUISA::VReduceSumOp>(loc, input.getType(),
		exp.getResult(), i64(rewriter, 6), i64(rewriter, 5));
	auto inv = rewriter.create<NPUISA::VRcpOp>(loc, input.getType(),
		sum.getResult(), i64(rewriter, 7), i64(rewriter, 6));
	auto soft = rewriter.create<NPUISA::VMulOp>(loc, input.getType(),
		exp.getResult(), inv.getResult(), i64(rewriter, 8), i64(rewriter, 5),
		i64(rewriter, 7));
	return soft.getResult();
}

struct LowerScheduleMatmulToISAPattern
	: public OpRewritePattern<NPUSchedule::MatmulTileOp> {
	LowerScheduleMatmulToISAPattern(
		MLIRContext *context, bool matmulUseMxu1Weights)
		: OpRewritePattern<NPUSchedule::MatmulTileOp>(context),
		  matmulUseMxu1Weights(matmulUseMxu1Weights) {}

	LogicalResult matchAndRewrite(NPUSchedule::MatmulTileOp op,
		PatternRewriter &rewriter) const override {
		Value result = emitMatmulSkeleton(op.getLoc(), op.getLhs(), op.getRhs(),
			op.getResult().getType(), rewriter, matmulUseMxu1Weights);
		rewriter.replaceOp(op, result);
		return success();
	}

	bool matmulUseMxu1Weights = false;
};

struct LowerScheduleSoftmaxToISAPattern
	: public OpRewritePattern<NPUSchedule::SoftmaxFragmentOp> {
	using OpRewritePattern<NPUSchedule::SoftmaxFragmentOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(NPUSchedule::SoftmaxFragmentOp op,
		PatternRewriter &rewriter) const override {
		Value result =
			emitSoftmaxVectorChain(op.getLoc(), op.getInput(), rewriter);
		rewriter.replaceOp(op, result);
		return success();
	}
};

struct LowerScheduleUKernelToISAPattern
	: public OpRewritePattern<NPUSchedule::UKernelLaunchOp> {
	LowerScheduleUKernelToISAPattern(
		MLIRContext *context, const NPULoweringOptions &options)
		: OpRewritePattern<NPUSchedule::UKernelLaunchOp>(context),
		  options(options) {}

	LogicalResult matchAndRewrite(NPUSchedule::UKernelLaunchOp op,
		PatternRewriter &rewriter) const override {
		llvm::StringRef symbol = op.getSymbol();
		auto inputs = op.getInputs();

		if (symbol.starts_with("npu_uk_matmul_")) {
			if (inputs.size() < 2) {
				return rewriter.notifyMatchFailure(
					op, "matmul ukernel expects >=2 tensor inputs");
			}
			Value result = emitMatmulSkeleton(op.getLoc(), inputs[0], inputs[1],
				op.getResult().getType(), rewriter,
				options.matmulUseMxu1Weights);
			rewriter.replaceOp(op, result);
			return success();
		}

		if (symbol.starts_with("npu_uk_gemma_mlp_")) {
			if (inputs.size() < 2) {
				return rewriter.notifyMatchFailure(
					op, "gemma_mlp ukernel expects >=2 tensor inputs");
			}
			// Mirrors model_npu/configs/programs/gemma_mlp.py shape.
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 0), i64(rewriter, 0x0000), i64(rewriter, 512),
				i64(rewriter, 0));
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 1), i64(rewriter, 0x0200), i64(rewriter, 512),
				i64(rewriter, 1));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));
			rewriter.create<NPUISA::DmaLoadOp>(op.getLoc(), i64(rewriter, 0),
				i64(rewriter, 0x2000), i64(rewriter, 2048), i64(rewriter, 2));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 2));

			auto gate = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), inputs[0], inputs[1],
				i64(rewriter, 1), i64(rewriter, 0), i64(rewriter, 0));
			auto up = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), inputs[0], inputs[1],
				i64(rewriter, 2), i64(rewriter, 0), i64(rewriter, 1));
			auto fused = rewriter.create<NPUISA::VMulOp>(op.getLoc(),
				op.getResult().getType(), gate.getResult(), up.getResult(),
				i64(rewriter, 6), i64(rewriter, 1), i64(rewriter, 2));

			rewriter.create<NPUISA::DmaStoreOp>(op.getLoc(), fused.getResult(),
				i64(rewriter, 6), i64(rewriter, 0x3000), i64(rewriter, 2048),
				i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));

			rewriter.replaceOp(op, fused.getResult());
			return success();
		}

		if (symbol.starts_with("npu_uk_gemma_attention_")) {
			if (inputs.size() < 2) {
				return rewriter.notifyMatchFailure(
					op, "gemma_attention ukernel expects >=2 tensor inputs");
			}
			// Mirrors model_npu/configs/programs/gemma_attention.py shape.
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 0), i64(rewriter, 0x2000), i64(rewriter, 256),
				i64(rewriter, 0));
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 1), i64(rewriter, 0x3000), i64(rewriter, 256),
				i64(rewriter, 1));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));
			rewriter.create<NPUISA::DmaLoadOp>(op.getLoc(), i64(rewriter, 0),
				i64(rewriter, 0x0000), i64(rewriter, 1024), i64(rewriter, 2));
			rewriter.create<NPUISA::DmaLoadOp>(op.getLoc(), i64(rewriter, 2),
				i64(rewriter, 0x4000), i64(rewriter, 2048), i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 2));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));

			auto scores = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), inputs[0], inputs[1],
				i64(rewriter, 3), i64(rewriter, 0), i64(rewriter, 0));
			auto softmax = emitSoftmaxVectorChain(
				op.getLoc(), scores.getResult(), rewriter);
			auto output = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), softmax, inputs[1], i64(rewriter, 9),
				i64(rewriter, 8), i64(rewriter, 1));

			rewriter.create<NPUISA::DmaStoreOp>(op.getLoc(), output.getResult(),
				i64(rewriter, 9), i64(rewriter, 0x5000), i64(rewriter, 2048),
				i64(rewriter, 1));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));

			rewriter.replaceOp(op, output.getResult());
			return success();
		}

		if (!options.allowUnknownUkernelFallback) {
			return rewriter.notifyMatchFailure(op,
				"unknown ukernel symbol family with "
				"fallback disabled");
		}

		if (inputs.size() < 2) {
			return rewriter.notifyMatchFailure(
				op, "unknown ukernel symbol requires at least 2 tensor inputs");
		}
		Value result = emitMatmulSkeleton(op.getLoc(), inputs[0], inputs[1],
			op.getResult().getType(), rewriter, options.matmulUseMxu1Weights);
		rewriter.replaceOp(op, result);
		return success();
	}

	NPULoweringOptions options;
};

struct ConvertNPUScheduleToISAPass
	: public PassWrapper<ConvertNPUScheduleToISAPass, OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNPUScheduleToISAPass)

	explicit ConvertNPUScheduleToISAPass(const NPULoweringOptions &options)
		: options(options) {}
	ConvertNPUScheduleToISAPass() = default;

	StringRef getArgument() const final {
		return "convert-npu-schedule-to-isa";
	}
	StringRef getDescription() const final {
		return "Lower npu_schedule matmul/ukernel ops to npu_isa skeleton ops";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUSchedule::NPUScheduleDialect>();
		registry.insert<NPUISA::NPUISADialect>();
	}

	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<LowerScheduleMatmulToISAPattern>(
			&getContext(), options.matmulUseMxu1Weights);
		patterns.add<LowerScheduleSoftmaxToISAPattern>(&getContext());
		patterns.add<LowerScheduleUKernelToISAPattern>(&getContext(), options);

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}

	NPULoweringOptions options;
};

} // namespace

std::unique_ptr<Pass> createConvertNPUScheduleToISAPass(
	const NPULoweringOptions &options) {
	return std::make_unique<ConvertNPUScheduleToISAPass>(options);
}

std::unique_ptr<Pass> createConvertNPUScheduleToISAPass() {
	return std::make_unique<ConvertNPUScheduleToISAPass>();
}

void registerConvertNPUScheduleToISAPass() {
	PassRegistration<ConvertNPUScheduleToISAPass>();
}

} // namespace mlir::iree_compiler::NPU
