// Within-body QDQ-roundtrip folder.
//
// The ONNX-QDQ → torch.export → IREE path that ultralytics uses for yolov8n
// int8 produces `linalg.generic` bodies with explicit dequantize-requantize
// cycles between every quant op. A typical conv-tail body for a SiLU node:
//
//   %x_f   = (acc_i32 + bias_i32) * s_acc                  // dequantize
//   %x_q   = quantize_to_i8(%x_f, s_x, zp_x)               // first quantize
//   %x_dq  = sitofp(%x_q) * s_x                            // dequant roundtrip
//   %sig_f = sigmoid(%x_dq)
//   %sig_q = quantize_to_i8(%sig_f, s_sig, zp_sig)
//   %sig_dq= sitofp(%sig_q) * s_sig                        // dequant roundtrip
//   %y_f   = %x_dq * %sig_dq                               // SiLU
//   yield  quantize_to_i8(%y_f, s_out, zp_out)
//
// The roundtrips (`sitofp(fptosi(...)) * s`) preserve the quantized value to
// within roundeven noise. For QNN's per-op tensor-quant semantics they're
// implicit boundaries — every QNN op already produces a quant.uniform-typed
// result. So we fold them out at the body level here, leaving:
//
//   %x_f   = (acc + bias) * s_acc
//   %sig_f = sigmoid(%x_f)
//   %y_f   = %x_f * %sig_f
//   yield  quantize_to_i8(%y_f, s_out, zp_out)
//
// which is the standard "conv-bias-SiLU-quantize" shape that downstream
// patterns can match.
//
// The transform is local — it only modifies SSA-defs within one linalg.generic
// region — so it does not perturb the dispatch/HAL structure.

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::QNN {
namespace {

// Returns the f32 constant value of v, or std::nullopt.
static std::optional<double> matchF32Const(Value v) {
	auto c = v.getDefiningOp<arith::ConstantOp>();
	if (!c)
		return std::nullopt;
	auto fa = dyn_cast<FloatAttr>(c.getValue());
	if (!fa)
		return std::nullopt;
	return fa.getValueAsDouble();
}
static bool isF32Eq(Value v, double target) {
	auto d = matchF32Const(v);
	return d && *d == target;
}

// Recognize `mulf(sitofp(fptosi(x_clamped_rescaled)), s)` and trace back to
// the value before the divf-by-s rescale. Returns Value() on no match.
//
// Expected chain ending at v:
//   mulf(_, s)               // post-quantize dequant
//   sitofp(_)
//   fptosi(_)
//   min(_, 127.0)
//   max(_, -128.0)
//   addf(_, zp_const)        // optional
//   math.roundeven(_)        // optional
//   divf(x, s)               // s matches the post-quantize mul factor
//   ----> return x
static Value tryPeelRoundtrip(Value v) {
	auto outerMulf = v.getDefiningOp<arith::MulFOp>();
	if (!outerMulf)
		return Value();
	Value cvtChain;
	std::optional<double> scaleOpt;
	if ((scaleOpt = matchF32Const(outerMulf.getRhs()))) {
		cvtChain = outerMulf.getLhs();
	} else if ((scaleOpt = matchF32Const(outerMulf.getLhs()))) {
		cvtChain = outerMulf.getRhs();
	} else {
		return Value();
	}
	double s = *scaleOpt;
	if (s == 0.0)
		return Value();
	auto sitofp = cvtChain.getDefiningOp<arith::SIToFPOp>();
	if (!sitofp)
		return Value();
	auto fptosi = sitofp.getOperand().getDefiningOp<arith::FPToSIOp>();
	if (!fptosi)
		return Value();
	Value cur = fptosi.getOperand();
	if (auto minf = cur.getDefiningOp<arith::MinimumFOp>()) {
		if (isF32Eq(minf.getRhs(), 127.0))
			cur = minf.getLhs();
		else if (isF32Eq(minf.getLhs(), 127.0))
			cur = minf.getRhs();
		else
			return Value();
	}
	if (auto maxf = cur.getDefiningOp<arith::MaximumFOp>()) {
		if (isF32Eq(maxf.getRhs(), -128.0))
			cur = maxf.getLhs();
		else if (isF32Eq(maxf.getLhs(), -128.0))
			cur = maxf.getRhs();
		else
			return Value();
	}
	if (auto addf = cur.getDefiningOp<arith::AddFOp>()) {
		// Zero-point add — allow but require it to be a constant (any value).
		if (matchF32Const(addf.getRhs()))
			cur = addf.getLhs();
		else if (matchF32Const(addf.getLhs()))
			cur = addf.getRhs();
		// If neither side is a constant the addf is something else (e.g. SiLU
		// sigmoid's `addf(exp, 1)`); don't accept the roundtrip.
		else
			return Value();
	}
	if (auto roundOp = cur.getDefiningOp<math::RoundEvenOp>()) {
		cur = roundOp.getOperand();
	}
	// Final step: divf(x, s_same) — the rescale matching the outer mul.
	auto divf = cur.getDefiningOp<arith::DivFOp>();
	if (!divf)
		return Value();
	auto sDivOpt = matchF32Const(divf.getRhs());
	if (!sDivOpt)
		return Value();
	// The two scales must match within tolerance — they represent the same
	// quantize-dequantize step.
	double sDiv = *sDivOpt;
	if (sDiv == 0.0)
		return Value();
	// Roundtrip's outer mul scale should be equal to the inner divf scale
	// (post-dequant = pre-quant value).
	if (std::abs(s - sDiv) > 1e-12 * std::max(std::abs(s), std::abs(sDiv)))
		return Value();
	return divf.getLhs();
}

struct FoldBodyQDQRoundtripPass
	: public PassWrapper<FoldBodyQDQRoundtripPass, OperationPass<>> {
	StringRef getArgument() const final {
		return "merlin-qnn-fold-body-qdq-roundtrip";
	}
	StringRef getDescription() const final {
		return "Within `linalg.generic` bodies, replace "
			   "dequant-requant-roundtrip "
			   "subtrees `mulf(sitofp(fptosi(...divf(x, s)...)), s)` with `x`. "
			   "Required before merlin-convert-linalg-to-qnn to simplify "
			   "yolov8n "
			   "int8 fused-SiLU bodies whose ONNX-QDQ export inserts explicit "
			   "dequantize-requantize cycles between every quant op.";
	}
	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<arith::ArithDialect, linalg::LinalgDialect,
			math::MathDialect>();
	}
	void runOnOperation() override {
		bool dbg = ::getenv("MERLIN_FOLD_DBG") != nullptr;
		SmallVector<Operation *> work;
		getOperation()->walk([&](linalg::GenericOp op) { work.push_back(op); });
		if (dbg)
			llvm::errs() << "[fold-dbg] visiting " << work.size()
						 << " generics\n";
		bool changed = false;
		for (Operation *op : work) {
			auto generic = cast<linalg::GenericOp>(op);
			if (generic.getRegion().empty())
				continue;
			Block &body = generic.getRegion().front();
			// Iterate to fixed point — folding one roundtrip may expose
			// another.
			const int kMaxIter = 16;
			for (int it = 0; it < kMaxIter; ++it) {
				bool localChange = false;
				// Walk a copy of the op list to allow erasure mid-iteration.
				SmallVector<Operation *> ops;
				for (Operation &o : body.without_terminator())
					ops.push_back(&o);
				for (Operation *bodyOp : ops) {
					auto mulf = dyn_cast<arith::MulFOp>(bodyOp);
					if (!mulf)
						continue;
					Value folded = tryPeelRoundtrip(mulf.getResult());
					if (!folded)
						continue;
					if (dbg)
						llvm::errs() << "[fold-dbg] folded a roundtrip\n";
					// Replace all uses of mulf.getResult() with `folded`.
					mulf.getResult().replaceAllUsesWith(folded);
					localChange = true;
					changed = true;
				}
				if (!localChange)
					break;
				// Clean up any now-dead arith/math ops left behind.
				SmallVector<Operation *> deadCandidates;
				for (Operation &o : body.without_terminator())
					deadCandidates.push_back(&o);
				for (Operation *o : llvm::reverse(deadCandidates)) {
					if (o->use_empty() && isOpTriviallyDead(o))
						o->erase();
				}
			}
		}
		(void)changed;
	}
};

} // namespace

std::unique_ptr<Pass> createFoldBodyQDQRoundtripPass() {
	return std::make_unique<FoldBodyQDQRoundtripPass>();
}

} // namespace mlir::iree_compiler::QNN
