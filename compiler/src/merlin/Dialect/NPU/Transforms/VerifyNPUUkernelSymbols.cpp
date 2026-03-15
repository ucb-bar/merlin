#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::iree_compiler::NPU {
namespace {

enum class UkernelFamily {
	Matmul,
	GemmaMLP,
	GemmaAttention,
	Unknown,
};

static UkernelFamily classifySymbol(StringRef symbol) {
	if (symbol.starts_with("npu_uk_matmul_")) {
		return UkernelFamily::Matmul;
	}
	if (symbol.starts_with("npu_uk_gemma_mlp_")) {
		return UkernelFamily::GemmaMLP;
	}
	if (symbol.starts_with("npu_uk_gemma_attention_")) {
		return UkernelFamily::GemmaAttention;
	}
	return UkernelFamily::Unknown;
}

static bool isF8E4M3FN(Type type) {
	if (!isa<FloatType>(type)) {
		return false;
	}
	std::string typeString;
	llvm::raw_string_ostream os(typeString);
	os << type;
	os.flush();
	return typeString == "f8E4M3FN";
}

static bool isF32(Type type) {
	return isa<Float32Type>(type);
}

static bool dimsCompatible(int64_t lhs, int64_t rhs) {
	return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) ||
		lhs == rhs;
}

static bool dimsCompatible3(int64_t a, int64_t b, int64_t c) {
	return dimsCompatible(a, b) && dimsCompatible(a, c) && dimsCompatible(b, c);
}

static LogicalResult verifyMatmulLikeShapes(Operation *op, StringRef symbol,
	Value lhs, Value rhs, Type resultType, bool enforceFp8f8f32) {
	auto lhsTy = dyn_cast<ShapedType>(lhs.getType());
	auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
	auto outTy = dyn_cast<ShapedType>(resultType);
	if (!lhsTy || !rhsTy || !outTy || !lhsTy.hasRank() || !rhsTy.hasRank() ||
		!outTy.hasRank() || lhsTy.getRank() < 2 || rhsTy.getRank() < 2 ||
		outTy.getRank() < 2) {
		op->emitOpError() << "symbol '" << symbol
						  << "' expects ranked tensors with rank >= 2";
		return failure();
	}

	int64_t lhsRank = lhsTy.getRank();
	int64_t rhsRank = rhsTy.getRank();
	int64_t outRank = outTy.getRank();
	if (lhsRank - 2 != rhsRank - 2 || lhsRank - 2 != outRank - 2) {
		op->emitOpError() << "symbol '" << symbol
						  << "' expects equal batch rank across lhs/rhs/result";
		return failure();
	}

	for (int64_t i = 0, e = lhsRank - 2; i < e; ++i) {
		if (!dimsCompatible3(lhsTy.getDimSize(i), rhsTy.getDimSize(i),
				outTy.getDimSize(i))) {
			op->emitOpError()
				<< "symbol '" << symbol
				<< "' has incompatible batch dimensions at index " << i;
			return failure();
		}
	}

	// Accept both rhs layouts (..xKxN) and (..xNxK).
	auto lhsM = lhsTy.getDimSize(lhsRank - 2);
	auto lhsK = lhsTy.getDimSize(lhsRank - 1);
	auto rhs0 = rhsTy.getDimSize(rhsRank - 2);
	auto rhs1 = rhsTy.getDimSize(rhsRank - 1);
	auto outM = outTy.getDimSize(outRank - 2);
	auto outN = outTy.getDimSize(outRank - 1);

	bool mMatches = dimsCompatible(lhsM, outM);
	bool standardCompatible =
		dimsCompatible(lhsK, rhs0) && dimsCompatible(rhs1, outN);
	bool transposedCompatible =
		dimsCompatible(lhsK, rhs1) && dimsCompatible(rhs0, outN);

	if (!mMatches || (!standardCompatible && !transposedCompatible)) {
		op->emitOpError()
			<< "symbol '" << symbol
			<< "' has incompatible matmul dimensions for lhs/rhs/result";
		return failure();
	}

	if (!enforceFp8f8f32) {
		return success();
	}

	if (!isF8E4M3FN(lhsTy.getElementType()) ||
		!isF8E4M3FN(rhsTy.getElementType()) || !isF32(outTy.getElementType())) {
		op->emitOpError()
			<< "strict verification requires f8E4M3FN,f8E4M3FN->f32 for '"
			<< symbol << "'";
		return failure();
	}
	return success();
}

static LogicalResult verifyAttentionShapes(
	Operation *op, StringRef symbol, ValueRange inputs, Type resultType) {
	if (inputs.size() < 3) {
		op->emitOpError() << "symbol '" << symbol
						  << "' expects at least 3 tensor inputs";
		return failure();
	}

	auto qTy = dyn_cast<ShapedType>(inputs[0].getType());
	auto kTy = dyn_cast<ShapedType>(inputs[1].getType());
	auto vTy = dyn_cast<ShapedType>(inputs[2].getType());
	auto outTy = dyn_cast<ShapedType>(resultType);
	if (!qTy || !kTy || !vTy || !outTy || !qTy.hasRank() || !kTy.hasRank() ||
		!vTy.hasRank() || !outTy.hasRank() || qTy.getRank() < 3 ||
		kTy.getRank() < 3 || vTy.getRank() < 3 || outTy.getRank() < 3) {
		op->emitOpError()
			<< "symbol '" << symbol
			<< "' expects ranked attention tensors with rank >= 3";
		return failure();
	}

	int64_t qRank = qTy.getRank();
	if (kTy.getRank() != qRank || vTy.getRank() != qRank ||
		outTy.getRank() != qRank) {
		op->emitOpError() << "symbol '" << symbol
						  << "' expects equal tensor rank across q/k/v/result";
		return failure();
	}

	for (int64_t i = 0, e = qRank - 2; i < e; ++i) {
		if (!dimsCompatible3(
				qTy.getDimSize(i), kTy.getDimSize(i), vTy.getDimSize(i)) ||
			!dimsCompatible(qTy.getDimSize(i), outTy.getDimSize(i))) {
			op->emitOpError()
				<< "symbol '" << symbol
				<< "' has incompatible batch dimensions at index " << i;
			return failure();
		}
	}

	if (!dimsCompatible(
			qTy.getDimSize(qRank - 2), outTy.getDimSize(qRank - 2))) {
		op->emitOpError() << "symbol '" << symbol
						  << "' expects query M to match output M";
		return failure();
	}

	return success();
}

template <typename OpTy>
static LogicalResult verifyUkernelOp(OpTy op, bool strict) {
	StringRef symbol = op.getSymbol();
	UkernelFamily family = classifySymbol(symbol);
	if (family == UkernelFamily::Unknown) {
		if (strict) {
			op.emitOpError()
				<< "unknown ukernel symbol family: '" << symbol << "'";
			return failure();
		}
		return success();
	}

	if (op.getInputs().size() < 2) {
		op.emitOpError() << "ukernel symbol '" << symbol
						 << "' expects at least 2 tensor inputs";
		return failure();
	}

	switch (family) {
		case UkernelFamily::Matmul:
			return verifyMatmulLikeShapes(op, symbol, op.getInputs()[0],
				op.getInputs()[1], op.getResult().getType(),
				/*enforceFp8f8f32=*/false);
		case UkernelFamily::GemmaMLP:
			return verifyMatmulLikeShapes(op, symbol, op.getInputs()[0],
				op.getInputs()[1], op.getResult().getType(),
				/*enforceFp8f8f32=*/strict);
		case UkernelFamily::GemmaAttention:
			return verifyAttentionShapes(
				op, symbol, op.getInputs(), op.getResult().getType());
		case UkernelFamily::Unknown:
			break;
	}
	return success();
}

struct VerifyNPUUkernelSymbolsPass
	: public PassWrapper<VerifyNPUUkernelSymbolsPass, OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyNPUUkernelSymbolsPass)

	explicit VerifyNPUUkernelSymbolsPass(const NPUUkernelVerifyOptions &options)
		: strict(options.strict) {}
	VerifyNPUUkernelSymbolsPass() = default;

	StringRef getArgument() const final {
		return "npu-verify-ukernel-symbols";
	}
	StringRef getDescription() const final {
		return "Verify NPU ukernel symbol families, shapes, and element types";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUKernel::NPUKernelDialect>();
		registry.insert<NPUSchedule::NPUScheduleDialect>();
	}

	void runOnOperation() override {
		bool anyFailure = false;

		getOperation()->walk([&](NPUKernel::UKernelGenericOp op) {
			if (failed(verifyUkernelOp(op, strict))) {
				anyFailure = true;
			}
		});

		getOperation()->walk([&](NPUSchedule::UKernelLaunchOp op) {
			if (failed(verifyUkernelOp(op, strict))) {
				anyFailure = true;
			}
		});

		if (anyFailure) {
			signalPassFailure();
		}
	}

	bool strict = true;
};

} // namespace

std::unique_ptr<Pass> createVerifyNPUUkernelSymbolsPass(
	const NPUUkernelVerifyOptions &options) {
	return std::make_unique<VerifyNPUUkernelSymbolsPass>(options);
}

void registerVerifyNPUUkernelSymbolsPass() {
	PassRegistration<VerifyNPUUkernelSymbolsPass>();
}

} // namespace mlir::iree_compiler::NPU
