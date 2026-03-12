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

static LogicalResult verifyMatmulShapes(Operation *op, StringRef symbol,
	Value lhs, Value rhs, Type resultType, bool enforceFp8f8f32) {
	auto lhsTy = dyn_cast<ShapedType>(lhs.getType());
	auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
	auto outTy = dyn_cast<ShapedType>(resultType);
	if (!lhsTy || !rhsTy || !outTy || lhsTy.getRank() != 2 ||
		rhsTy.getRank() != 2 || outTy.getRank() != 2) {
		op->emitOpError() << "symbol '" << symbol
						  << "' expects rank-2 shaped tensors";
		return failure();
	}

	// Accept both rhs layouts (KxN) and (NxK) for current frontend flexibility.
	auto lhsK = lhsTy.getDimSize(1);
	auto rhs0 = rhsTy.getDimSize(0);
	auto rhs1 = rhsTy.getDimSize(1);
	auto outM = outTy.getDimSize(0);
	auto outN = outTy.getDimSize(1);

	auto lhsM = lhsTy.getDimSize(0);
	bool mMatches = ShapedType::isDynamic(lhsM) ||
		ShapedType::isDynamic(outM) || lhsM == outM;
	bool standardCompatible =
		(ShapedType::isDynamic(lhsK) || ShapedType::isDynamic(rhs0) ||
			lhsK == rhs0) &&
		(ShapedType::isDynamic(rhs1) || ShapedType::isDynamic(outN) ||
			rhs1 == outN);
	bool transposedCompatible =
		(ShapedType::isDynamic(lhsK) || ShapedType::isDynamic(rhs1) ||
			lhsK == rhs1) &&
		(ShapedType::isDynamic(rhs0) || ShapedType::isDynamic(outN) ||
			rhs0 == outN);

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

	if (failed(verifyMatmulShapes(op, symbol, op.getInputs()[0],
			op.getInputs()[1], op.getResult().getType(),
			strict && family != UkernelFamily::Matmul))) {
		return failure();
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
