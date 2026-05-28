#include "compiler/src/merlin/Dialect/QNN/IR/QNNDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::QNN;

#include "compiler/src/merlin/Dialect/QNN/IR/QNNDialect.cpp.inc"

#define GET_OP_CLASSES
#include "compiler/src/merlin/Dialect/QNN/IR/QNNOps.cpp.inc"

void QNNDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "compiler/src/merlin/Dialect/QNN/IR/QNNOps.cpp.inc"
		>();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

namespace {

// Common helper: ensure rank-4 NHWC input with the given dimension
// invariants. Returns LogicalResult; on failure emits a diagnostic on `op`.
template <typename Op>
LogicalResult verifyRank4NHWC(
	Op op, RankedTensorType ty, llvm::StringRef name) {
	if (ty.getRank() != 4) {
		return op.emitOpError()
			<< name << " must be rank 4 (NHWC); got rank " << ty.getRank();
	}
	return success();
}

} // namespace

LogicalResult Conv2dOp::verify() {
	auto inputType = cast<RankedTensorType>(getInput().getType());
	if (failed(verifyRank4NHWC(*this, inputType, "input")))
		return failure();
	auto weightType = cast<RankedTensorType>(getWeight().getType());
	if (weightType.getRank() != 4) {
		return emitOpError() << "weight must be rank 4 (HWIO); got rank "
							 << weightType.getRank();
	}
	// stride / pad_amount / dilation length checks.
	if (getStride().size() != 2)
		return emitOpError() << "stride must have exactly 2 entries [sh, sw]";
	if (getPadAmount().size() != 4)
		return emitOpError() << "pad_amount must have exactly 4 entries "
							 << "[pad_top, pad_bottom, pad_left, pad_right]";
	if (getDilation().size() != 2)
		return emitOpError() << "dilation must have exactly 2 entries [dh, dw]";
	if (getGroup() < 1)
		return emitOpError() << "group must be >= 1";
	return success();
}

LogicalResult DepthwiseConv2dOp::verify() {
	auto inputType = cast<RankedTensorType>(getInput().getType());
	if (failed(verifyRank4NHWC(*this, inputType, "input")))
		return failure();
	if (getStride().size() != 2)
		return emitOpError() << "stride must have exactly 2 entries [sh, sw]";
	if (getPadAmount().size() != 4)
		return emitOpError() << "pad_amount must have exactly 4 entries";
	if (getDilation().size() != 2)
		return emitOpError() << "dilation must have exactly 2 entries [dh, dw]";
	return success();
}

LogicalResult PoolMax2dOp::verify() {
	auto inputType = cast<RankedTensorType>(getInput().getType());
	if (failed(verifyRank4NHWC(*this, inputType, "input")))
		return failure();
	if (getFilterSize().size() != 2)
		return emitOpError()
			<< "filter_size must have exactly 2 entries [kh, kw]";
	if (getStride().size() != 2)
		return emitOpError() << "stride must have exactly 2 entries";
	if (getPadAmount().size() != 4)
		return emitOpError() << "pad_amount must have exactly 4 entries";
	return success();
}

LogicalResult PoolAvg2dOp::verify() {
	auto inputType = cast<RankedTensorType>(getInput().getType());
	if (failed(verifyRank4NHWC(*this, inputType, "input")))
		return failure();
	if (getFilterSize().size() != 2)
		return emitOpError()
			<< "filter_size must have exactly 2 entries [kh, kw]";
	if (getStride().size() != 2)
		return emitOpError() << "stride must have exactly 2 entries";
	if (getPadAmount().size() != 4)
		return emitOpError() << "pad_amount must have exactly 4 entries";
	return success();
}
