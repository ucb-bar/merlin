#include "compiler/src/merlin/Dialect/QNN/Transforms/GenericMatchUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::QNN {

namespace {

// Strips through any number of side-effect-free unary cast-ish ops and
// returns the producer of the chain (or `op` itself if no cast).
Operation *peelCasts(Operation *op) {
	while (op && op->getNumOperands() == 1 &&
		(isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, arith::TruncFOp,
			arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
			arith::FPToUIOp>(op))) {
		Operation *prev = op->getOperand(0).getDefiningOp();
		if (!prev)
			break;
		op = prev;
	}
	return op;
}

// Collect non-yield ops of `block` in program order.
SmallVector<Operation *> collectBodyOps(Block &block) {
	SmallVector<Operation *> ops;
	for (Operation &o : block.getOperations()) {
		if (!isa<linalg::YieldOp>(o))
			ops.push_back(&o);
	}
	return ops;
}

bool isI32IntegerConst(Value v, int64_t *out) {
	auto cst = v.getDefiningOp<arith::ConstantOp>();
	if (!cst)
		return false;
	auto iAttr = dyn_cast<IntegerAttr>(cst.getValue());
	if (!iAttr)
		return false;
	*out = iAttr.getInt();
	return true;
}

} // namespace

bool isF32Const(Value v, double *out) {
	auto cst = v.getDefiningOp<arith::ConstantOp>();
	if (!cst)
		return false;
	auto fAttr = dyn_cast<FloatAttr>(cst.getValue());
	if (!fAttr)
		return false;
	*out = fAttr.getValueAsDouble();
	return true;
}

// Extract the i32 zp value when `v` is either an inline integer constant
// OR a linalg.generic block argument that maps to an `ins` operand which
// is itself a scalar constant (yolov8's quant-conv shape: zp arrives via
// a 3rd `i32` operand on the generic, not as an inline constant).
static bool isI32IntegerConstOrBlockArgConst(Value v, int64_t *out) {
	if (isI32IntegerConst(v, out))
		return true;
	auto blockArg = dyn_cast<BlockArgument>(v);
	if (!blockArg)
		return false;
	auto *parentOp = blockArg.getOwner()->getParentOp();
	auto generic = dyn_cast_or_null<linalg::GenericOp>(parentOp);
	if (!generic)
		return false;
	unsigned argIdx = blockArg.getArgNumber();
	// GenericOp body args are: ins[0], ins[1], ..., outs[0], ...
	unsigned numIns = (unsigned)generic.getNumDpsInputs();
	if (argIdx >= numIns)
		return false; // out arg, not a zp
	Value insOperand = generic.getDpsInputs()[argIdx];
	return isI32IntegerConst(insOperand, out);
}

bool matchQuantConvBody(Block &block, int64_t *inZpOut, int64_t *wZpOut) {
	*inZpOut = 0;
	*wZpOut = 0;
	auto ops = collectBodyOps(block);

	// Folded shape (zero zp): extsi, extsi, muli, addi.
	if (ops.size() == 4 && isa<arith::ExtSIOp>(ops[0]) &&
		isa<arith::ExtSIOp>(ops[1]) && isa<arith::MulIOp>(ops[2]) &&
		isa<arith::AddIOp>(ops[3])) {
		return true;
	}

	// Non-folded shape: extsi(in), subi(zp_in), extsi(w), subi(zp_w),
	// muli, addi (or a permutation thereof). Walk back from the addi
	// accumulate to find the muli, then walk muli's operands to extract
	// optional subi-with-const-zp.
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	auto *acc = yield->getOperand(0).getDefiningOp();
	if (!isa_and_nonnull<arith::AddIOp>(acc))
		return false;
	Value mulVal;
	for (Value v : acc->getOperands()) {
		if (auto *p = v.getDefiningOp(); p && isa<arith::MulIOp>(p)) {
			mulVal = v;
			break;
		}
	}
	if (!mulVal)
		return false;
	auto *mul = mulVal.getDefiningOp();

	auto extractZp = [&](Value side, int64_t *zp) -> bool {
		auto *p = side.getDefiningOp();
		if (!p)
			return false;
		if (isa<arith::ExtSIOp>(p)) {
			*zp = 0;
			return true;
		}
		if (auto sub = dyn_cast<arith::SubIOp>(p)) {
			// zp can be either an inline constant OR a block arg whose ins-side
			// operand is a constant (yolov8 quant-conv form).
			if (isI32IntegerConstOrBlockArgConst(sub.getRhs(), zp)) {
				return sub.getLhs().getDefiningOp<arith::ExtSIOp>() != nullptr;
			}
			if (isI32IntegerConstOrBlockArgConst(sub.getLhs(), zp)) {
				*zp = -*zp;
				return sub.getRhs().getDefiningOp<arith::ExtSIOp>() != nullptr;
			}
		}
		return false;
	};
	if (!extractZp(mul->getOperand(0), inZpOut))
		return false;
	if (!extractZp(mul->getOperand(1), wZpOut))
		return false;
	return true;
}

bool matchDequantBody(Block &block, double *scaleOut, int64_t *zpOut) {
	*scaleOut = 1.0;
	*zpOut = 0;
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	// Walk back: yield <- mulf(_, scale) <- sitofp(_) <- subi(_, zp) <-
	// extsi(blockarg).
	auto *mulf = yield->getOperand(0).getDefiningOp();
	if (!isa_and_nonnull<arith::MulFOp>(mulf))
		return false;
	// Scale: one of the operands is a const float.
	bool gotScale = false;
	Value cvtChain;
	for (Value v : mulf->getOperands()) {
		if (isF32Const(v, scaleOut)) {
			gotScale = true;
		} else {
			cvtChain = v;
		}
	}
	if (!gotScale || !cvtChain)
		return false;
	auto *cvt = cvtChain.getDefiningOp();
	if (!isa_and_nonnull<arith::SIToFPOp>(cvt))
		return false;
	Value cvtIn = cvt->getOperand(0);
	// Direct sitofp from a block-arg (the fully-folded zp=0 i8→f32 chain
	// — yield <- mulf(scale, sitofp(blockarg))). This is the shape IREE's
	// global-opt produces for typical model dequant generics.
	if (isa<BlockArgument>(cvtIn)) {
		*zpOut = 0;
		return true;
	}
	auto *sub = cvtIn.getDefiningOp();
	if (auto subi = dyn_cast_or_null<arith::SubIOp>(sub)) {
		if (isI32IntegerConst(subi.getRhs(), zpOut)) {
			// ok
		} else if (isI32IntegerConst(subi.getLhs(), zpOut)) {
			*zpOut = -*zpOut;
		} else {
			return false;
		}
		return true;
	}
	// Folded zp=0 variant with explicit i8→i32 widening: sitofp(extsi(arg)).
	if (isa_and_nonnull<arith::ExtSIOp>(sub)) {
		*zpOut = 0;
		return true;
	}
	return false;
}

bool matchQuantizeBody(Block &block, double *scaleOut, int64_t *zpOut) {
	*scaleOut = 1.0;
	*zpOut = 0;
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	// yield <- fptosi <- [maximumf/minimumf clamp]* <- [roundeven] <-
	// addf(_, zp_f) <- divf(_, scale). The clamp + round are optional and
	// may be in either order; QNN doesn't need them since the runtime
	// emits its own clamp matching the output q-params.
	auto *fptosi = yield->getOperand(0).getDefiningOp();
	if (!isa_and_nonnull<arith::FPToSIOp>(fptosi))
		return false;
	Operation *cur = fptosi->getOperand(0).getDefiningOp();
	// Walk back through any single-operand pointwise tail ops we don't
	// care about: minimumf, maximumf (clamp), math.roundeven, math.round.
	while (cur &&
		(isa<arith::MinimumFOp, arith::MaximumFOp>(cur) ||
			(cur->getDialect() && cur->getDialect()->getNamespace() == "math" &&
				cur->getNumOperands() == 1))) {
		// For binary clamps (minimumf/maximumf) take the non-constant side.
		if (cur->getNumOperands() == 2) {
			Value next;
			double dummy = 0.0;
			if (isF32Const(cur->getOperand(0), &dummy)) {
				next = cur->getOperand(1);
			} else if (isF32Const(cur->getOperand(1), &dummy)) {
				next = cur->getOperand(0);
			} else {
				break;
			}
			cur = next.getDefiningOp();
		} else {
			cur = cur->getOperand(0).getDefiningOp();
		}
	}
	if (!cur)
		return false;
	auto addf = dyn_cast<arith::AddFOp>(cur);
	Value scaleSide;
	if (addf) {
		double zpF = 0.0;
		if (isF32Const(addf.getRhs(), &zpF)) {
			*zpOut = static_cast<int64_t>(zpF);
			scaleSide = addf.getLhs();
		} else if (isF32Const(addf.getLhs(), &zpF)) {
			*zpOut = static_cast<int64_t>(zpF);
			scaleSide = addf.getRhs();
		} else {
			return false;
		}
	} else {
		// No zp-add tail: scale-side is the divf/mulf directly.
		scaleSide = cur->getResult(0);
	}
	// Walk through any single-operand math op (round, etc.) between
	// addf and divf/mulf.
	auto *scaleOp = scaleSide.getDefiningOp();
	while (scaleOp && scaleOp->getNumOperands() == 1 && scaleOp->getDialect() &&
		scaleOp->getDialect()->getNamespace() == "math") {
		scaleOp = scaleOp->getOperand(0).getDefiningOp();
	}
	// divf(_, scale) OR mulf(_, 1/scale).
	if (auto divf = dyn_cast_or_null<arith::DivFOp>(scaleOp)) {
		if (!isF32Const(divf.getRhs(), scaleOut))
			return false;
		return true;
	}
	if (auto mulf = dyn_cast_or_null<arith::MulFOp>(scaleOp)) {
		double inv = 1.0;
		if (isF32Const(mulf.getRhs(), &inv) && inv != 0.0) {
			*scaleOut = 1.0 / inv;
			return true;
		}
		if (isF32Const(mulf.getLhs(), &inv) && inv != 0.0) {
			*scaleOut = 1.0 / inv;
			return true;
		}
	}
	return false;
}

bool matchConvRescaleBody(
	Block &block, double *scaleAccOut, double *scaleOutOut, int64_t *zpOut) {
	*scaleAccOut = 1.0;
	*scaleOutOut = 1.0;
	*zpOut = 0;
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	// yield <- fptosi <- [min/max clamp]* <- [round/cast] <- addf? <-
	// divf|mulf <- mulf|divf <- sitofp <- blockarg.
	auto *fptosi = yield->getOperand(0).getDefiningOp();
	if (!isa_and_nonnull<arith::FPToSIOp>(fptosi))
		return false;
	Operation *cur = fptosi->getOperand(0).getDefiningOp();
	// Walk back through clamps + round.
	while (cur &&
		(isa<arith::MinimumFOp, arith::MaximumFOp>(cur) ||
			(cur->getDialect() && cur->getDialect()->getNamespace() == "math" &&
				cur->getNumOperands() == 1))) {
		if (cur->getNumOperands() == 2) {
			Value next;
			double dummy = 0.0;
			if (isF32Const(cur->getOperand(0), &dummy)) {
				next = cur->getOperand(1);
			} else if (isF32Const(cur->getOperand(1), &dummy)) {
				next = cur->getOperand(0);
			} else {
				break;
			}
			cur = next.getDefiningOp();
		} else {
			cur = cur->getOperand(0).getDefiningOp();
		}
	}
	if (!cur)
		return false;
	// Optional zp tail: addf(_, zp_const).
	if (auto addf = dyn_cast<arith::AddFOp>(cur)) {
		double zpF = 0.0;
		if (isF32Const(addf.getRhs(), &zpF)) {
			*zpOut = static_cast<int64_t>(zpF);
			cur = addf.getLhs().getDefiningOp();
		} else if (isF32Const(addf.getLhs(), &zpF)) {
			*zpOut = static_cast<int64_t>(zpF);
			cur = addf.getRhs().getDefiningOp();
		}
	}
	// Walk through any single-operand math op (round) between addf and
	// divf/mulf.
	while (cur && cur->getNumOperands() == 1 && cur->getDialect() &&
		cur->getDialect()->getNamespace() == "math") {
		cur = cur->getOperand(0).getDefiningOp();
	}
	if (!cur)
		return false;
	// divf(_, scale_out)  OR  mulf(_, 1/scale_out): the "/scale_out" step.
	Operation *innerScaleOp = nullptr;
	if (auto divf = dyn_cast<arith::DivFOp>(cur)) {
		if (!isF32Const(divf.getRhs(), scaleOutOut))
			return false;
		innerScaleOp = divf.getLhs().getDefiningOp();
	} else if (auto mulf = dyn_cast<arith::MulFOp>(cur)) {
		double inv = 0.0;
		Value otherSide;
		if (isF32Const(mulf.getRhs(), &inv) && inv != 0.0) {
			*scaleOutOut = 1.0 / inv;
			otherSide = mulf.getLhs();
		} else if (isF32Const(mulf.getLhs(), &inv) && inv != 0.0) {
			*scaleOutOut = 1.0 / inv;
			otherSide = mulf.getRhs();
		} else {
			return false;
		}
		innerScaleOp = otherSide.getDefiningOp();
	} else {
		return false;
	}
	// Now expect mulf(_, scale_acc): the accumulator-side rescale.
	if (!innerScaleOp)
		return false;
	if (auto mulf = dyn_cast<arith::MulFOp>(innerScaleOp)) {
		Value cvtSide;
		if (isF32Const(mulf.getRhs(), scaleAccOut)) {
			cvtSide = mulf.getLhs();
		} else if (isF32Const(mulf.getLhs(), scaleAccOut)) {
			cvtSide = mulf.getRhs();
		} else {
			return false;
		}
		auto *cvt = cvtSide.getDefiningOp();
		if (!isa_and_nonnull<arith::SIToFPOp>(cvt))
			return false;
		if (!isa<BlockArgument>(cvt->getOperand(0)))
			return false;
		return true;
	}
	// Or no acc rescale: just sitofp.
	if (auto cvt = dyn_cast<arith::SIToFPOp>(innerScaleOp)) {
		*scaleAccOut = 1.0;
		if (!isa<BlockArgument>(cvt.getOperand()))
			return false;
		return true;
	}
	return false;
}

bool matchActivationBody(Block &block, ActivationKind *kindOut) {
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	auto *op = yield->getOperand(0).getDefiningOp();
	if (!op)
		return false;
	// Relu: maximumf(x, 0) or maxsi(x, 0).
	if (isa<arith::MaximumFOp>(op) || isa<arith::MaxSIOp>(op)) {
		*kindOut = ActivationKind::Relu;
		return true;
	}
	// Relu6: minimumf(maximumf(x, 0), 6).
	if (auto minOp = dyn_cast<arith::MinimumFOp>(op)) {
		auto *inner = minOp.getLhs().getDefiningOp();
		if (isa_and_nonnull<arith::MaximumFOp>(inner)) {
			*kindOut = ActivationKind::Relu6;
			return true;
		}
	}
	// Sigmoid / tanh have dedicated math ops.
	if (op->getDialect() && op->getDialect()->getNamespace() == "math") {
		StringRef name = op->getName().getStringRef();
		if (name.ends_with("tanh")) {
			*kindOut = ActivationKind::Tanh;
			return true;
		}
		if (name.ends_with("sigmoid")) {
			*kindOut = ActivationKind::Sigmoid;
			return true;
		}
	}
	return false;
}

namespace {

// Skip i8 min/max clamp pair (or any single clamp). Stops on the first op
// whose neither operand is an f32 constant.
Operation *skipClamps(Operation *op) {
	while (op && (isa<arith::MinimumFOp>(op) || isa<arith::MaximumFOp>(op))) {
		double dummy = 0.0;
		Value other;
		if (isF32Const(op->getOperand(0), &dummy))
			other = op->getOperand(1);
		else if (isF32Const(op->getOperand(1), &dummy))
			other = op->getOperand(0);
		else
			break;
		op = other.getDefiningOp();
	}
	return op;
}

// If op is addf(x, zp_const), strip it and return x; write zp into *zp.
Operation *skipZpAdd(Operation *op, int64_t *zp) {
	if (auto addf = dyn_cast_or_null<arith::AddFOp>(op)) {
		double zpF = 0.0;
		if (isF32Const(addf.getRhs(), &zpF)) {
			*zp = static_cast<int64_t>(zpF);
			return addf.getLhs().getDefiningOp();
		}
		if (isF32Const(addf.getLhs(), &zpF)) {
			*zp = static_cast<int64_t>(zpF);
			return addf.getRhs().getDefiningOp();
		}
	}
	return op;
}

// Skip single-operand math.* ops (round/floor/ceil).
Operation *skipMath(Operation *op) {
	while (op && op->getDialect() &&
		op->getDialect()->getNamespace() == "math" &&
		op->getNumOperands() == 1) {
		op = op->getOperand(0).getDefiningOp();
	}
	return op;
}

// Match `divf(_, scale)` or `mulf(_, 1/scale)` and return the inner value.
// On match writes scale to *scaleOut. Returns Value() on no-match.
Value matchScaleDownExpr(Operation *op, double *scaleOut) {
	if (auto divf = dyn_cast_or_null<arith::DivFOp>(op)) {
		if (isF32Const(divf.getRhs(), scaleOut))
			return divf.getLhs();
		return Value();
	}
	if (auto mulf = dyn_cast_or_null<arith::MulFOp>(op)) {
		double inv = 0.0;
		if (isF32Const(mulf.getRhs(), &inv) && inv != 0.0) {
			*scaleOut = 1.0 / inv;
			return mulf.getLhs();
		}
		if (isF32Const(mulf.getLhs(), &inv) && inv != 0.0) {
			*scaleOut = 1.0 / inv;
			return mulf.getRhs();
		}
	}
	return Value();
}

// Match `mulf(sitofp(?), scale)`. Returns the sitofp source operand.
Value matchDequantMul(Value v, double *scaleOut) {
	auto m = dyn_cast_or_null<arith::MulFOp>(v.getDefiningOp());
	if (!m)
		return Value();
	Value src;
	if (isF32Const(m.getRhs(), scaleOut))
		src = m.getLhs();
	else if (isF32Const(m.getLhs(), scaleOut))
		src = m.getRhs();
	else
		return Value();
	auto cvt = dyn_cast_or_null<arith::SIToFPOp>(src.getDefiningOp());
	if (!cvt)
		return Value();
	return cvt.getOperand();
}

} // namespace

// Body fragment shared by Fp32BiasSiLU and Fp32ResidualBiasSiLU: check
// that `siluRoot` computes `mulf(addf(a,b), sigmoid(addf(a,b)))` for two
// distinct BlockArgument operands a,b. Returns the addf op (= biased
// value) on match, or nullptr.
namespace {
arith::AddFOp matchBiasSiLUSubtree(Value siluRoot) {
	auto mul = dyn_cast_or_null<arith::MulFOp>(siluRoot.getDefiningOp());
	if (!mul)
		return nullptr;
	auto check = [&](Value addSide, Value sigSide) -> arith::AddFOp {
		auto addf = dyn_cast_or_null<arith::AddFOp>(addSide.getDefiningOp());
		if (!addf)
			return nullptr;
		if (!isa<BlockArgument>(addf.getLhs()) ||
			!isa<BlockArgument>(addf.getRhs()))
			return nullptr;
		auto sigDiv = dyn_cast_or_null<arith::DivFOp>(sigSide.getDefiningOp());
		if (!sigDiv)
			return nullptr;
		double one = 0.0;
		if (!isF32Const(sigDiv.getLhs(), &one) || std::abs(one - 1.0) > 1e-6)
			return nullptr;
		auto sigAdd =
			dyn_cast_or_null<arith::AddFOp>(sigDiv.getRhs().getDefiningOp());
		if (!sigAdd)
			return nullptr;
		Value expSide;
		double oneCk = 0.0;
		if (isF32Const(sigAdd.getRhs(), &oneCk) && std::abs(oneCk - 1.0) < 1e-6)
			expSide = sigAdd.getLhs();
		else if (isF32Const(sigAdd.getLhs(), &oneCk) &&
			std::abs(oneCk - 1.0) < 1e-6)
			expSide = sigAdd.getRhs();
		else
			return nullptr;
		auto *expOp = expSide.getDefiningOp();
		if (!expOp || !expOp->getDialect() ||
			expOp->getDialect()->getNamespace() != "math" ||
			!expOp->getName().getStringRef().ends_with("exp"))
			return nullptr;
		auto negf = dyn_cast_or_null<arith::NegFOp>(
			expOp->getOperand(0).getDefiningOp());
		if (!negf)
			return nullptr;
		if (negf.getOperand().getDefiningOp() != addf.getOperation())
			return nullptr;
		return addf;
	};
	if (auto r = check(mul->getOperand(0), mul->getOperand(1)))
		return r;
	return check(mul->getOperand(1), mul->getOperand(0));
}
} // namespace

bool matchFp32ResidualBiasSiLUBody(Block &block) {
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	// yield <- addf(residual_blockarg, silu_value)
	auto outerAdd = yield->getOperand(0).getDefiningOp<arith::AddFOp>();
	if (!outerAdd)
		return false;
	auto checkOrdering = [&](Value resSide, Value siluSide) -> bool {
		if (!isa<BlockArgument>(resSide))
			return false;
		return matchBiasSiLUSubtree(siluSide) != nullptr;
	};
	if (checkOrdering(outerAdd.getLhs(), outerAdd.getRhs()))
		return true;
	if (checkOrdering(outerAdd.getRhs(), outerAdd.getLhs()))
		return true;
	return false;
}

bool matchFp32BiasSiLUBody(Block &block) {
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;
	// yield <- mulf(addf(in,bias), sigmoid_path) — try both operand orderings.
	auto *mul = yield->getOperand(0).getDefiningOp();
	if (!isa_and_nonnull<arith::MulFOp>(mul))
		return false;
	auto check = [&](Value addSide, Value sigSide) -> bool {
		auto addf = dyn_cast_or_null<arith::AddFOp>(addSide.getDefiningOp());
		if (!addf)
			return false;
		if (!isa<BlockArgument>(addf.getLhs()) ||
			!isa<BlockArgument>(addf.getRhs()))
			return false;
		auto sigDiv = dyn_cast_or_null<arith::DivFOp>(sigSide.getDefiningOp());
		if (!sigDiv)
			return false;
		double one = 0.0;
		if (!isF32Const(sigDiv.getLhs(), &one) || std::abs(one - 1.0) > 1e-6)
			return false;
		auto sigAdd =
			dyn_cast_or_null<arith::AddFOp>(sigDiv.getRhs().getDefiningOp());
		if (!sigAdd)
			return false;
		Value expSide;
		double oneCk = 0.0;
		if (isF32Const(sigAdd.getRhs(), &oneCk) && std::abs(oneCk - 1.0) < 1e-6)
			expSide = sigAdd.getLhs();
		else if (isF32Const(sigAdd.getLhs(), &oneCk) &&
			std::abs(oneCk - 1.0) < 1e-6)
			expSide = sigAdd.getRhs();
		else
			return false;
		auto *expOp = expSide.getDefiningOp();
		if (!expOp || !expOp->getDialect() ||
			expOp->getDialect()->getNamespace() != "math" ||
			!expOp->getName().getStringRef().ends_with("exp"))
			return false;
		auto negf = dyn_cast_or_null<arith::NegFOp>(
			expOp->getOperand(0).getDefiningOp());
		if (!negf)
			return false;
		// negf operand should be the addf result (same instance).
		if (negf.getOperand().getDefiningOp() != addf.getOperation())
			return false;
		return true;
	};
	if (check(mul->getOperand(0), mul->getOperand(1)))
		return true;
	if (check(mul->getOperand(1), mul->getOperand(0)))
		return true;
	return false;
}

bool matchSiLURescaleBody(Block &block, double *sAccOut, double *sIntOut,
	double *sSigOut, double *sFinOut, int64_t *zpOut) {
	*sAccOut = 1.0;
	*sIntOut = 1.0;
	*sSigOut = 1.0;
	*sFinOut = 1.0;
	*zpOut = 0;
	auto *yield = block.getTerminator();
	if (!yield || yield->getNumOperands() != 1)
		return false;

	// FINAL: yield <- fptosi <- clamp <- addf(zp) <- round <- divf(sFin)
	//                              <- mulf(y_dq, sig_dq).
	auto *finalFptosi = yield->getOperand(0).getDefiningOp();
	if (!isa_and_nonnull<arith::FPToSIOp>(finalFptosi))
		return false;
	Operation *cur = finalFptosi->getOperand(0).getDefiningOp();
	cur = skipClamps(cur);
	cur = skipZpAdd(cur, zpOut);
	cur = skipMath(cur);
	Value innerProd = matchScaleDownExpr(cur, sFinOut);
	if (!innerProd)
		return false;
	auto outerMul = dyn_cast_or_null<arith::MulFOp>(innerProd.getDefiningOp());
	if (!outerMul)
		return false;

	// The outer mul is mulf(y_dq, sig_dq). Either operand could be y or sig;
	// try both orderings.
	auto matchPair = [&](Value yCand, Value sigCand) -> bool {
		double yDqScale = 0.0, sigDqScale = 0.0;
		Value yQI8 = matchDequantMul(yCand, &yDqScale);
		Value sigQI8 = matchDequantMul(sigCand, &sigDqScale);
		if (!yQI8 || !sigQI8)
			return false;
		*sIntOut = yDqScale;
		*sSigOut = sigDqScale;

		// y branch: yQI8 <- fptosi <- clamp <- addf(zp) <- round <- divf(sInt)
		//                <- mulf(sitofp(blockarg), sAcc).
		auto yFptosi = dyn_cast_or_null<arith::FPToSIOp>(yQI8.getDefiningOp());
		if (!yFptosi)
			return false;
		Operation *yc = yFptosi.getOperand().getDefiningOp();
		yc = skipClamps(yc);
		int64_t yZp = 0;
		yc = skipZpAdd(yc, &yZp);
		yc = skipMath(yc);
		double sIntCheck = 0.0;
		Value yInner = matchScaleDownExpr(yc, &sIntCheck);
		if (!yInner)
			return false;
		if (std::abs(sIntCheck - *sIntOut) > 1e-6)
			return false;
		auto yMulIn = dyn_cast_or_null<arith::MulFOp>(yInner.getDefiningOp());
		if (!yMulIn)
			return false;
		Value cvtSide;
		if (isF32Const(yMulIn.getRhs(), sAccOut))
			cvtSide = yMulIn.getLhs();
		else if (isF32Const(yMulIn.getLhs(), sAccOut))
			cvtSide = yMulIn.getRhs();
		else
			return false;
		auto cvt = dyn_cast_or_null<arith::SIToFPOp>(cvtSide.getDefiningOp());
		if (!cvt)
			return false;
		Value cvtOperand = cvt.getOperand();
		if (!isa<BlockArgument>(cvtOperand)) {
			auto addi =
				dyn_cast_or_null<arith::AddIOp>(cvtOperand.getDefiningOp());
			if (!addi || !isa<BlockArgument>(addi.getLhs()) ||
				!isa<BlockArgument>(addi.getRhs()))
				return false;
		}

		// sig branch: sigQI8 <- fptosi <- clamp <- addf(zp) <- round <-
		//   divf(sSig) <- divf(1.0, addf(exp(negf(y_dq)), 1.0)).
		auto sigFptosi =
			dyn_cast_or_null<arith::FPToSIOp>(sigQI8.getDefiningOp());
		if (!sigFptosi)
			return false;
		Operation *sc = sigFptosi.getOperand().getDefiningOp();
		sc = skipClamps(sc);
		int64_t sigZp = 0;
		sc = skipZpAdd(sc, &sigZp);
		sc = skipMath(sc);
		double sSigCheck = 0.0;
		Value sigInner = matchScaleDownExpr(sc, &sSigCheck);
		if (!sigInner)
			return false;
		if (std::abs(sSigCheck - *sSigOut) > 1e-6)
			return false;
		auto sigDiv = dyn_cast_or_null<arith::DivFOp>(sigInner.getDefiningOp());
		if (!sigDiv)
			return false;
		double oneCk = 0.0;
		if (!isF32Const(sigDiv.getLhs(), &oneCk) ||
			std::abs(oneCk - 1.0) > 1e-6)
			return false;
		auto sigAdd =
			dyn_cast_or_null<arith::AddFOp>(sigDiv.getRhs().getDefiningOp());
		if (!sigAdd)
			return false;
		Value expSide;
		double oneCk2 = 0.0;
		if (isF32Const(sigAdd.getRhs(), &oneCk2) &&
			std::abs(oneCk2 - 1.0) < 1e-6)
			expSide = sigAdd.getLhs();
		else if (isF32Const(sigAdd.getLhs(), &oneCk2) &&
			std::abs(oneCk2 - 1.0) < 1e-6)
			expSide = sigAdd.getRhs();
		else
			return false;
		auto *expOp = expSide.getDefiningOp();
		if (!expOp || !expOp->getDialect() ||
			expOp->getDialect()->getNamespace() != "math" ||
			!expOp->getName().getStringRef().ends_with("exp"))
			return false;
		auto negf = dyn_cast_or_null<arith::NegFOp>(
			expOp->getOperand(0).getDefiningOp());
		if (!negf)
			return false;
		// negf operand = y_dq (a fresh mulf(sitofp(q_y), sInt) with same
		// scale).
		double yDqAlt = 0.0;
		Value yQAlt = matchDequantMul(negf.getOperand(), &yDqAlt);
		if (!yQAlt || std::abs(yDqAlt - *sIntOut) > 1e-6)
			return false;
		*zpOut = yZp;
		return true;
	};

	if (matchPair(outerMul.getLhs(), outerMul.getRhs()))
		return true;
	if (matchPair(outerMul.getRhs(), outerMul.getLhs()))
		return true;
	return false;
}

Value recoverPadFromProducer(
	Value value, llvm::SmallVectorImpl<int32_t> &padAmountOut) {
	padAmountOut.assign(4, 0);
	auto pad = value.getDefiningOp<tensor::PadOp>();
	if (!pad)
		return value;
	auto low = pad.getStaticLow();
	auto high = pad.getStaticHigh();
	// Rank-4 NHWC: spatial dims at index 1, 2.
	if (low.size() == 4 && high.size() == 4) {
		padAmountOut[0] = static_cast<int32_t>(low[1]);
		padAmountOut[1] = static_cast<int32_t>(high[1]);
		padAmountOut[2] = static_cast<int32_t>(low[2]);
		padAmountOut[3] = static_cast<int32_t>(high[2]);
	} else if (low.size() == 3 && high.size() == 3) {
		// Rank-3 NHC (post-tile slice): spatial dims at index 0, 1.
		padAmountOut[0] = static_cast<int32_t>(low[0]);
		padAmountOut[1] = static_cast<int32_t>(high[0]);
		padAmountOut[2] = static_cast<int32_t>(low[1]);
		padAmountOut[3] = static_cast<int32_t>(high[1]);
	}
	return pad.getSource();
}

} // namespace mlir::iree_compiler::QNN
