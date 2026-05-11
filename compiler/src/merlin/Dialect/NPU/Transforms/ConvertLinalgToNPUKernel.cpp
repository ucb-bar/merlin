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

// Returns true iff `op` has all-parallel iterators and every indexing map is an
// identity-equivalent map (a dim permutation with no reduction, no broadcast).
// This is what makes an op a candidate for binary/unary elementwise binding.
static bool isPurelyParallelElementwise(linalg::GenericOp op) {
	if (!op.hasPureTensorSemantics() || op->getNumResults() != 1) {
		return false;
	}
	for (utils::IteratorType it : op.getIteratorTypesArray()) {
		if (it != utils::IteratorType::parallel) {
			return false;
		}
	}
	// Output map must be an identity (no transpose).
	AffineMap outMap = op.getIndexingMapsArray().back();
	if (!outMap.isIdentity()) {
		return false;
	}
	// Input maps must each be identity too — i.e. no broadcast/permutation.
	// This is strict on purpose; broadcasts and transposes need separate
	// handling because their DRAM layout differs from the kernel ABI.
	for (AffineMap m : llvm::ArrayRef(op.getIndexingMapsArray()).drop_back(1)) {
		if (!m.isIdentity()) {
			return false;
		}
	}
	return true;
}

// Inspects the linalg body for a single binary arith op matching `opName`
// (e.g. "arith.addf") whose operands are the two block arguments and whose
// result is yielded. Returns true on exact match. Rejects bodies with any
// other compute op to keep the pattern unambiguous.
static bool isBinaryElementwiseBody(Block &body, StringRef opName) {
	if (body.getNumArguments() < 2) {
		return false;
	}
	BlockArgument a = body.getArgument(0);
	BlockArgument b = body.getArgument(1);
	Operation *computeOp = nullptr;
	Operation *yieldOp = nullptr;
	for (Operation &op : body) {
		if (isa<linalg::YieldOp>(op)) {
			yieldOp = &op;
			continue;
		}
		if (computeOp) {
			return false; // Only one compute op allowed.
		}
		computeOp = &op;
	}
	if (!computeOp || !yieldOp ||
		computeOp->getName().getStringRef() != opName) {
		return false;
	}
	if (computeOp->getNumOperands() != 2 || computeOp->getNumResults() != 1) {
		return false;
	}
	// Operands must be the two block args (in either order for commutative
	// ops, but for subf/divf we additionally require canonical order later).
	Value x = computeOp->getOperand(0);
	Value y = computeOp->getOperand(1);
	if (!((x == a && y == b) || (x == b && y == a))) {
		return false;
	}
	if (yieldOp->getNumOperands() != 1 ||
		yieldOp->getOperand(0) != computeOp->getResult(0)) {
		return false;
	}
	return true;
}

// For non-commutative binary ops (subf, divf), we must preserve operand order.
// Returns the (lhs, rhs) pair from the linalg inputs in the order the body
// consumes them, or std::nullopt if the body reverses them (which would flip
// the op's semantics).
static std::optional<std::pair<Value, Value>> orderedBinaryInputs(
	linalg::GenericOp op, StringRef opName) {
	Block &body = op.getBlock()->getParent()->front();
	BlockArgument a = body.getArgument(0);
	Operation *computeOp = nullptr;
	for (Operation &o : body) {
		if (isa<linalg::YieldOp>(o))
			continue;
		computeOp = &o;
		break;
	}
	if (!computeOp || computeOp->getName().getStringRef() != opName) {
		return std::nullopt;
	}
	Value lhs = op.getDpsInputOperand(0)->get();
	Value rhs = op.getDpsInputOperand(1)->get();
	// If the body reads arg0 as lhs of the op, we keep input order.
	// If it reads arg1 as lhs, we need to swap inputs to match the op order.
	if (computeOp->getOperand(0) == a) {
		return std::make_pair(lhs, rhs);
	}
	return std::make_pair(rhs, lhs);
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

// Binds a linalg.generic with one "reduction" iterator and a single
// `arith.addf` body to `npu_uk_reduction_sum`. This is the row-sum shape:
//   ins(%x : tensor<MxKxbf16>) outs(%init : tensor<Mxbf16>) { addf %in, %out }
// with iterators [parallel, reduction].
struct LowerRowReductionSumToUKernelPattern
	: public OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!op.hasPureTensorSemantics() || op->getNumResults() != 1 ||
			op.getNumDpsInputs() != 1) {
			return rewriter.notifyMatchFailure(
				op, "not single-input reduction");
		}
		// Exactly one reduction iterator.
		int reductionCount = 0;
		for (utils::IteratorType it : op.getIteratorTypesArray()) {
			if (it == utils::IteratorType::reduction) {
				++reductionCount;
			}
		}
		if (reductionCount != 1) {
			return rewriter.notifyMatchFailure(
				op, "need exactly one reduction dim");
		}
		// Body: single `arith.addf` combining %in with %out, yielded.
		Block &body = op->getRegion(0).front();
		if (body.getNumArguments() != 2) {
			return rewriter.notifyMatchFailure(op, "body args != 2");
		}
		Operation *computeOp = nullptr;
		Operation *yieldOp = nullptr;
		for (Operation &o : body) {
			if (isa<linalg::YieldOp>(o)) {
				yieldOp = &o;
				continue;
			}
			if (computeOp) {
				return rewriter.notifyMatchFailure(op, "multi-op body");
			}
			computeOp = &o;
		}
		if (!computeOp || !yieldOp ||
			computeOp->getName().getStringRef() != "arith.addf") {
			return rewriter.notifyMatchFailure(op, "body is not single addf");
		}
		// Output must be bf16 to match the kernel ABI.
		auto outTy =
			llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
		if (!outTy || !llvm::cast<FloatType>(outTy.getElementType()).isBF16()) {
			return rewriter.notifyMatchFailure(op, "output not bf16");
		}

		Value input = op.getDpsInputOperand(0)->get();
		auto symbol = rewriter.getStringAttr("npu_uk_reduction_sum");
		rewriter.replaceOpWithNewOp<NPUKernel::UKernelGenericOp>(
			op, op.getResult(0).getType(), symbol, ValueRange{input});
		return success();
	}
};

// Binds a linalg.generic body with a single `arith.<op>` to the corresponding
// `npu_uk_elementwise_<suffix>` manifest kernel. `commutative=true` means
// operand order doesn't matter; for subf/divf we preserve it.
struct LowerElementwiseGenericToUKernelPattern
	: public OpRewritePattern<linalg::GenericOp> {
	using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

	LowerElementwiseGenericToUKernelPattern(MLIRContext *ctx, StringRef arithOp,
		StringRef kernelSuffix, bool commutative)
		: OpRewritePattern(ctx), arithOp(arithOp.str()),
		  kernelSuffix(kernelSuffix.str()), commutative(commutative) {}

	LogicalResult matchAndRewrite(
		linalg::GenericOp op, PatternRewriter &rewriter) const override {
		if (!isPurelyParallelElementwise(op)) {
			return rewriter.notifyMatchFailure(op, "not parallel elementwise");
		}
		if (op.getNumDpsInputs() != 2) {
			return rewriter.notifyMatchFailure(op, "not a binary op");
		}
		if (!isBinaryElementwiseBody(op->getRegion(0).front(), arithOp)) {
			return rewriter.notifyMatchFailure(op, "body mismatch");
		}

		Value lhs;
		Value rhs;
		if (commutative) {
			lhs = op.getDpsInputOperand(0)->get();
			rhs = op.getDpsInputOperand(1)->get();
		} else {
			auto ordered = orderedBinaryInputs(op, arithOp);
			if (!ordered) {
				return rewriter.notifyMatchFailure(op, "cannot order operands");
			}
			lhs = ordered->first;
			rhs = ordered->second;
		}

		auto symbol =
			rewriter.getStringAttr("npu_uk_elementwise_" + kernelSuffix);
		rewriter.replaceOpWithNewOp<NPUKernel::UKernelGenericOp>(
			op, op.getResult(0).getType(), symbol, ValueRange{lhs, rhs});
		return success();
	}

	std::string arithOp;
	std::string kernelSuffix;
	bool commutative;
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
		// Row-reduction: must register BEFORE elementwise because a
		// reduction-iterator addf body would otherwise match the strict
		// elementwise matcher's isPurelyParallelElementwise check... wait,
		// it doesn't, because that matcher rejects non-parallel iterators.
		// But registering reduction first is still clearer intent.
		patterns.add<LowerRowReductionSumToUKernelPattern>(&getContext());
		// Elementwise: register last so matmul-like patterns win when a body
		// contains addf+mulf (that's a contraction, not an elementwise add).
		patterns.add<LowerElementwiseGenericToUKernelPattern>(
			&getContext(), "arith.addf", "add", /*commutative=*/true);
		patterns.add<LowerElementwiseGenericToUKernelPattern>(
			&getContext(), "arith.mulf", "mul", /*commutative=*/true);
		patterns.add<LowerElementwiseGenericToUKernelPattern>(
			&getContext(), "arith.subf", "sub", /*commutative=*/false);
		patterns.add<LowerElementwiseGenericToUKernelPattern>(
			&getContext(), "arith.divf", "div", /*commutative=*/false);
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
