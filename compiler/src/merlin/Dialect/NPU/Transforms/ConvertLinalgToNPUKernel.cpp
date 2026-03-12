#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::NPU {
namespace {

static bool isDimExpr(AffineExpr expr, unsigned position) {
	auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr);
	return dimExpr && dimExpr.getPosition() == position;
}

static bool isMatmulLikeGeneric(linalg::GenericOp op) {
	if (!op.hasPureTensorSemantics()) {
		return false;
	}
	if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
		op->getNumResults() != 1) {
		return false;
	}

	auto maps = op.getIndexingMapsArray();
	if (maps.size() != 3) {
		return false;
	}
	AffineMap lhsMap = maps[0];
	AffineMap rhsMap = maps[1];
	AffineMap outMap = maps[2];
	if (lhsMap.getNumDims() != 3 || rhsMap.getNumDims() != 3 ||
		outMap.getNumDims() != 3) {
		return false;
	}
	if (lhsMap.getNumResults() != 2 || rhsMap.getNumResults() != 2 ||
		outMap.getNumResults() != 2) {
		return false;
	}

	bool lhsMatches =
		isDimExpr(lhsMap.getResult(0), 0) && isDimExpr(lhsMap.getResult(1), 2);
	bool rhsMatches = (isDimExpr(rhsMap.getResult(0), 2) &&
						  isDimExpr(rhsMap.getResult(1), 1)) ||
		(isDimExpr(rhsMap.getResult(0), 1) &&
			isDimExpr(rhsMap.getResult(1), 2));
	bool outMatches =
		isDimExpr(outMap.getResult(0), 0) && isDimExpr(outMap.getResult(1), 1);
	return lhsMatches && rhsMatches && outMatches;
}

static std::string getTypeMnemonic(Type type) {
	std::string text;
	llvm::raw_string_ostream os(text);
	type.print(os);
	os.flush();

	std::string out;
	out.reserve(text.size());
	for (char c : text) {
		if (llvm::isAlnum(static_cast<unsigned char>(c)) || c == '_') {
			out.push_back(c);
		}
	}
	return out.empty() ? "unknown" : out;
}

static std::string inferMatmulUkernelSymbol(
	Value lhs, Value rhs, Type resultType) {
	auto lhsShaped = llvm::dyn_cast<ShapedType>(lhs.getType());
	auto rhsShaped = llvm::dyn_cast<ShapedType>(rhs.getType());
	auto outShaped = llvm::dyn_cast<ShapedType>(resultType);
	if (!lhsShaped || !rhsShaped || !outShaped) {
		return "npu_uk_matmul_generic";
	}

	std::string lhsElem = getTypeMnemonic(lhsShaped.getElementType());
	std::string rhsElem = getTypeMnemonic(rhsShaped.getElementType());
	std::string outElem = getTypeMnemonic(outShaped.getElementType());
	return "npu_uk_matmul_" + lhsElem + "_" + rhsElem + "_" + outElem;
}

struct LowerMatmulToNPUKernelPattern
	: public OpRewritePattern<linalg::MatmulOp> {
	using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::MatmulOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2 || op->getNumResults() != 1) {
			return rewriter.notifyMatchFailure(
				op, "expected tensor matmul with 2 inputs and 1 result");
		}

		Value lhs = op.getDpsInputOperand(0)->get();
		Value rhs = op.getDpsInputOperand(1)->get();

		rewriter.replaceOpWithNewOp<NPUKernel::MatmulOp>(
			op, op->getResultTypes(), lhs, rhs);
		return success();
	}
};

struct LowerMatmulGenericToNPUKernelUKernelPattern
	: public OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!isMatmulLikeGeneric(op)) {
			return rewriter.notifyMatchFailure(
				op, "not a matmul-like linalg.generic");
		}

		Value lhs = op.getDpsInputOperand(0)->get();
		Value rhs = op.getDpsInputOperand(1)->get();
		auto symbol = rewriter.getStringAttr(
			inferMatmulUkernelSymbol(lhs, rhs, op.getResult(0).getType()));

		rewriter.replaceOpWithNewOp<NPUKernel::UKernelGenericOp>(
			op, op.getResult(0).getType(), symbol, ValueRange{lhs, rhs});
		return success();
	}
};

struct ConvertLinalgToNPUKernelPass
	: public PassWrapper<ConvertLinalgToNPUKernelPass,
		  OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLinalgToNPUKernelPass)

	StringRef getArgument() const final {
		return "convert-linalg-to-npu-kernel";
	}
	StringRef getDescription() const final {
		return "Lower linalg.matmul and matmul-like linalg.generic to "
			   "npu_kernel ops";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUKernel::NPUKernelDialect>();
	}

	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<LowerMatmulToNPUKernelPattern>(&getContext());
		patterns.add<LowerMatmulGenericToNPUKernelUKernelPattern>(
			&getContext());

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createConvertLinalgToNPUKernelPass() {
	return std::make_unique<ConvertLinalgToNPUKernelPass>();
}

void registerConvertLinalgToNPUKernelPass() {
	PassRegistration<ConvertLinalgToNPUKernelPass>();
}

} // namespace mlir::iree_compiler::NPU
