#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;
using namespace mlir::iree_compiler::Gemmini;

namespace {

static bool isFP8(Type type) {
	return isa<Float8E4M3FNType>(type);
}

static bool isFP8MatmulLike(Type lhs, Type rhs, Type out) {
	return isFP8(lhs) && isFP8(rhs) &&
		(cast<FloatType>(out).isF32() || cast<FloatType>(out).isBF16());
}

static LogicalResult verifyMatmulTypes(
	Operation *op, Value lhs, Value rhs, Value result) {
	auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
	auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
	auto resultType = dyn_cast<RankedTensorType>(result.getType());
	if (!lhsType || !rhsType || !resultType) {
		return op->emitOpError() << "expects ranked tensor operands and result";
	}
	Type lhsElem = lhsType.getElementType();
	Type rhsElem = rhsType.getElementType();
	Type outElem = resultType.getElementType();
	bool isInt8Path = lhsElem.isSignlessInteger(8) &&
		rhsElem.isSignlessInteger(8) && outElem.isSignlessInteger(32);
	bool isFP8Path =
		isa<FloatType>(outElem) && isFP8MatmulLike(lhsElem, rhsElem, outElem);
	if (!isInt8Path && !isFP8Path) {
		return op->emitOpError()
			<< "expects either i8/i8->i32 or f8E4M3FN/f8E4M3FN->{bf16|f32}";
	}
	if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
		resultType.getRank() != 2) {
		return op->emitOpError() << "expects rank-2 lhs/rhs/result tensors";
	}

	int64_t lhsM = lhsType.getDimSize(0);
	int64_t lhsK = lhsType.getDimSize(1);
	int64_t rhsN = rhsType.getDimSize(0);
	int64_t rhsK = rhsType.getDimSize(1);
	int64_t outM = resultType.getDimSize(0);
	int64_t outN = resultType.getDimSize(1);

	if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
		lhsK != rhsK) {
		return op->emitOpError() << "lhs K and rhs K mismatch";
	}
	if (!ShapedType::isDynamic(lhsM) && !ShapedType::isDynamic(outM) &&
		lhsM != outM) {
		return op->emitOpError() << "lhs M and result M mismatch";
	}
	if (!ShapedType::isDynamic(rhsN) && !ShapedType::isDynamic(outN) &&
		rhsN != outN) {
		return op->emitOpError() << "rhs N and result N mismatch";
	}

	return success();
}

} // namespace

LogicalResult MatmulOp::verify() {
	return verifyMatmulTypes(getOperation(), getLhs(), getRhs(), getResult());
}

LogicalResult MatmulTileOp::verify() {
	if (failed(verifyMatmulTypes(
			getOperation(), getLhs(), getRhs(), getResult()))) {
		return failure();
	}
	if (getTileM() == 0 || getTileN() == 0 || getTileK() == 0) {
		return emitOpError() << "tile sizes must be positive";
	}
	return success();
}

LogicalResult Conv2DOp::verify() {
	auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
	auto filterType = dyn_cast<RankedTensorType>(getFilter().getType());
	auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
	if (!inputType || !filterType || !resultType) {
		return emitOpError() << "expects ranked tensor operands and result";
	}

	if (!inputType.getElementType().isSignlessInteger(8) ||
		!filterType.getElementType().isSignlessInteger(8) ||
		!resultType.getElementType().isSignlessInteger(32)) {
		return emitOpError() << "expects i8 input/filter and i32 result";
	}

	if (inputType.getRank() != 3 || filterType.getRank() != 4 ||
		resultType.getRank() != 3) {
		return emitOpError() << "expects input CHW rank-3, filter FCHW rank-4, "
								"result FHW rank-3";
	}

	if (getStrideH() == 0 || getStrideW() == 0 || getDilationH() == 0 ||
		getDilationW() == 0) {
		return emitOpError() << "stride and dilation must be positive";
	}

	int64_t inC = inputType.getDimSize(0);
	int64_t filF = filterType.getDimSize(0);
	int64_t filC = filterType.getDimSize(1);
	int64_t outF = resultType.getDimSize(0);

	if (!ShapedType::isDynamic(inC) && !ShapedType::isDynamic(filC) &&
		inC != filC) {
		return emitOpError() << "input/filter channel mismatch";
	}
	if (!ShapedType::isDynamic(filF) && !ShapedType::isDynamic(outF) &&
		filF != outF) {
		return emitOpError() << "filter/output channel mismatch";
	}

	return success();
}

LogicalResult RequantizeOp::verify() {
	auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
	auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
	if (!inputType || !resultType) {
		return emitOpError() << "expects ranked tensor input and result";
	}
	if (inputType.getRank() != resultType.getRank()) {
		return emitOpError() << "input/result rank mismatch";
	}
	if (!inputType.getElementType().isF32() ||
		!resultType.getElementType().isSignlessInteger(8)) {
		return emitOpError() << "expects f32 input and i8 result";
	}
	return success();
}

LogicalResult ClampOp::verify() {
	auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
	auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
	if (!inputType || !resultType) {
		return emitOpError() << "expects ranked tensor input and result";
	}
	if (inputType != resultType) {
		return emitOpError() << "expects identical input/result tensor types";
	}
	if (!inputType.getElementType().isF32()) {
		return emitOpError() << "expects f32 tensor";
	}
	if (getMinValue().convertToDouble() > getMaxValue().convertToDouble()) {
		return emitOpError() << "minValue must be <= maxValue";
	}
	return success();
}

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.cpp.inc"
