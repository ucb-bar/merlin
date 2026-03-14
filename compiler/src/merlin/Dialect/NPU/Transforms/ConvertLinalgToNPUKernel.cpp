#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
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

static std::string inferAttentionUkernelSymbol(Value query, Type resultType) {
	auto queryShaped = llvm::dyn_cast<ShapedType>(query.getType());
	auto outShaped = llvm::dyn_cast<ShapedType>(resultType);
	if (!queryShaped || !outShaped) {
		return "npu_uk_gemma_attention_generic";
	}

	std::string queryElem = getTypeMnemonic(queryShaped.getElementType());
	std::string outElem = getTypeMnemonic(outShaped.getElementType());
	return "npu_uk_gemma_attention_" + queryElem + "_" + outElem;
}

static bool isBF16OrF32Type(Type type) {
	auto floatType = llvm::dyn_cast<FloatType>(type);
	return floatType && (floatType.isBF16() || floatType.isF32());
}

static bool shouldDemoteContractionToBF16(
	Value lhs, Value rhs, Type resultType) {
	auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
	auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
	auto outTy = llvm::dyn_cast<RankedTensorType>(resultType);
	return lhsTy && rhsTy && outTy && isBF16OrF32Type(lhsTy.getElementType()) &&
		isBF16OrF32Type(rhsTy.getElementType()) &&
		outTy.getElementType().isF32();
}

static Value createElementwiseFloatTensorCast(
	PatternRewriter &rewriter, Location loc, Value input, Value outputInit) {
	auto inTy = llvm::dyn_cast<RankedTensorType>(input.getType());
	auto outTy = llvm::dyn_cast<RankedTensorType>(outputInit.getType());
	if (!inTy || !outTy || inTy.getRank() != outTy.getRank()) {
		return {};
	}
	if (!llvm::isa<FloatType>(inTy.getElementType()) ||
		!llvm::isa<FloatType>(outTy.getElementType())) {
		return {};
	}
	if (inTy.getElementType() == outTy.getElementType()) {
		return input;
	}

	MLIRContext *ctx = rewriter.getContext();
	int64_t rank = outTy.getRank();
	AffineMap identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
	SmallVector<AffineMap> maps = {identity, identity};
	SmallVector<utils::IteratorType> iterators(
		rank, utils::IteratorType::parallel);

	auto castOp = rewriter.create<linalg::GenericOp>(loc, outTy,
		ValueRange{input}, ValueRange{outputInit}, maps, iterators,
		[&](OpBuilder &b, Location nestedLoc, ValueRange args) {
			auto inElem = llvm::cast<FloatType>(inTy.getElementType());
			auto outElem = llvm::cast<FloatType>(outTy.getElementType());
			Value casted;
			if (inElem.getWidth() > outElem.getWidth()) {
				casted = b.create<arith::TruncFOp>(
					nestedLoc, outTy.getElementType(), args[0]);
			} else {
				casted = b.create<arith::ExtFOp>(
					nestedLoc, outTy.getElementType(), args[0]);
			}
			b.create<linalg::YieldOp>(nestedLoc, casted);
		});
	return castOp.getResult(0);
}

static Value createEmptyLikeWithElementType(PatternRewriter &rewriter,
	Location loc, RankedTensorType sourceType, Value source,
	Type targetElemType) {
	auto targetType = RankedTensorType::get(
		sourceType.getShape(), targetElemType, sourceType.getEncoding());
	SmallVector<Value> dynamicDims;
	for (int64_t i = 0; i < sourceType.getRank(); ++i) {
		if (sourceType.isDynamicDim(i)) {
			dynamicDims.push_back(
				rewriter.create<tensor::DimOp>(loc, source, i));
		}
	}
	return rewriter.create<tensor::EmptyOp>(
		loc, targetType.getShape(), targetElemType, dynamicDims);
}

static LogicalResult replaceWithDemotedMatmulAndExtf(PatternRewriter &rewriter,
	Operation *op, Value lhs, Value rhs, Value outputInit,
	Type originalResultType) {
	auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
	auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
	auto outTy = llvm::dyn_cast<RankedTensorType>(originalResultType);
	if (!lhsTy || !rhsTy || !outTy || !outTy.getElementType().isF32()) {
		return failure();
	}

	// Keep shape/encoding while demoting contraction accumulation/output to
	// bf16.
	Value lhsBF16 = lhs;
	if (!lhsTy.getElementType().isBF16()) {
		Value lhsEmpty = createEmptyLikeWithElementType(
			rewriter, op->getLoc(), lhsTy, lhs, rewriter.getBF16Type());
		lhsBF16 = createElementwiseFloatTensorCast(
			rewriter, op->getLoc(), lhs, lhsEmpty);
		if (!lhsBF16) {
			return failure();
		}
	}

	Value rhsBF16 = rhs;
	if (!rhsTy.getElementType().isBF16()) {
		Value rhsEmpty = createEmptyLikeWithElementType(
			rewriter, op->getLoc(), rhsTy, rhs, rewriter.getBF16Type());
		rhsBF16 = createElementwiseFloatTensorCast(
			rewriter, op->getLoc(), rhs, rhsEmpty);
		if (!rhsBF16) {
			return failure();
		}
	}

	auto demotedOutTy = RankedTensorType::get(
		outTy.getShape(), rewriter.getBF16Type(), outTy.getEncoding());
	auto demotedMatmul = rewriter.create<NPUKernel::MatmulOp>(
		op->getLoc(), demotedOutTy, lhsBF16, rhsBF16);

	Value widened = createElementwiseFloatTensorCast(
		rewriter, op->getLoc(), demotedMatmul.getResult(), outputInit);
	if (!widened) {
		return failure();
	}

	rewriter.replaceOp(op, widened);
	return success();
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
		Value outputInit = op.getDpsInitOperand(0)->get();

		if (shouldDemoteContractionToBF16(
				lhs, rhs, op.getResult(0).getType())) {
			if (succeeded(
					replaceWithDemotedMatmulAndExtf(rewriter, op.getOperation(),
						lhs, rhs, outputInit, op.getResult(0).getType()))) {
				return success();
			}
		}

		rewriter.replaceOpWithNewOp<NPUKernel::MatmulOp>(
			op, op->getResultTypes(), lhs, rhs);
		return success();
	}
};

struct LowerBatchMatmulToNPUKernelPattern
	: public OpRewritePattern<linalg::BatchMatmulOp> {
	using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::BatchMatmulOp op, PatternRewriter &rewriter) const override {
		if (op.getNumDpsInputs() != 2 || op->getNumResults() != 1) {
			return rewriter.notifyMatchFailure(
				op, "expected tensor batch_matmul with 2 inputs and 1 result");
		}

		Value lhs = op.getDpsInputOperand(0)->get();
		Value rhs = op.getDpsInputOperand(1)->get();
		Value outputInit = op.getDpsInitOperand(0)->get();

		if (shouldDemoteContractionToBF16(
				lhs, rhs, op.getResult(0).getType())) {
			if (succeeded(
					replaceWithDemotedMatmulAndExtf(rewriter, op.getOperation(),
						lhs, rhs, outputInit, op.getResult(0).getType()))) {
				return success();
			}
		}

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

struct LowerSoftmaxToNPUSchedulePattern
	: public OpRewritePattern<linalg::SoftmaxOp> {
	using OpRewritePattern<linalg::SoftmaxOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::SoftmaxOp op, PatternRewriter &rewriter) const override {
		if (!op.hasPureTensorSemantics()) {
			return rewriter.notifyMatchFailure(
				op, "expected pure tensor linalg.softmax");
		}
		if (op.getNumDpsInputs() != 1 || op->getNumResults() != 1) {
			return rewriter.notifyMatchFailure(
				op, "expected linalg.softmax with 1 input and 1 result");
		}

		Value input = op.getDpsInputOperand(0)->get();
		rewriter.replaceOpWithNewOp<NPUSchedule::SoftmaxFragmentOp>(
			op, op->getResult(0).getType(), input);
		return success();
	}
};

struct LowerIreeAttentionToNPUKernelUKernelPattern : public RewritePattern {
	explicit LowerIreeAttentionToNPUKernelUKernelPattern(MLIRContext *context)
		: RewritePattern("iree_linalg_ext.attention", 1, context) {}

	LogicalResult matchAndRewrite(
		Operation *op, PatternRewriter &rewriter) const override {
		if (op->getNumResults() != 1) {
			return rewriter.notifyMatchFailure(
				op, "expected single-result iree_linalg_ext.attention");
		}

		auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(op);
		if (!dpsOp) {
			return rewriter.notifyMatchFailure(
				op, "expected destination-style attention op");
		}

		SmallVector<Value> tensorInputs;
		for (Value input : dpsOp.getDpsInputs()) {
			if (isa<TensorType>(input.getType())) {
				tensorInputs.push_back(input);
			}
		}
		if (tensorInputs.size() < 2) {
			return rewriter.notifyMatchFailure(
				op, "expected at least two tensor inputs");
		}

		auto symbol = rewriter.getStringAttr(inferAttentionUkernelSymbol(
			tensorInputs.front(), op->getResult(0).getType()));
		rewriter.replaceOpWithNewOp<NPUKernel::UKernelGenericOp>(
			op, op->getResult(0).getType(), symbol, tensorInputs);
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
		return "Lower global-opt linalg/attention forms to NPU kernel/schedule "
			   "ops";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUKernel::NPUKernelDialect>();
		registry.insert<NPUSchedule::NPUScheduleDialect>();
	}

	void runOnOperation() override {
		RewritePatternSet patterns(&getContext());
		patterns.add<LowerMatmulToNPUKernelPattern>(&getContext());
		patterns.add<LowerBatchMatmulToNPUKernelPattern>(&getContext());
		patterns.add<LowerMatmulGenericToNPUKernelUKernelPattern>(
			&getContext());
		patterns.add<LowerSoftmaxToNPUSchedulePattern>(&getContext());
		patterns.add<LowerIreeAttentionToNPUKernelUKernelPattern>(
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
