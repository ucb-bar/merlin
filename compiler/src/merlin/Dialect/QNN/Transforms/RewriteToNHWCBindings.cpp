// Network-wide activation tensor layout rewrite: NCHW (collapsed) → NHWC.
//
// **Purpose.** Yolov8n int8 dispatches carry rank-3 activation tensors with
// shape `<C, H, W>` (the N=1 batch is elided by IREE's global opts). HTA's op
// set has no `Transpose` op (only `TransposeConv2d`); our `qnn.conv2d`
// emitters internally wrap the conv with transposes when the source is CHW,
// producing `qnn.transpose` ops at codegen that HTA refuses. By rewriting
// every cross-dispatch activation tensor from `<C, H, W>` to `<H, W, C>`
// BEFORE dispatch creation, we eliminate the need for in-conv transposes
// and unlock ~40-60 yolov8 dispatches for HTA.
//
// **Phase.** Runs at preprocessing — same scope as `LegalizeLayoutToNHWC`
// (module level). Operates on `linalg.generic` + `tensor.empty` +
// `linalg.broadcast` + named-op convs before they're encapsulated in
// `flow.dispatch.workgroups` by dispatch creation.
//
// **Approach.** Three-walk pass:
//   1. Survey: identify rank-3 tensor values that flow through activation
//      paths (i.e., are produced by a conv or rescale-quantize generic and
//      consumed by the next conv's input). Build a set of SSA values to
//      rewrite.
//   2. Rewrite: for each tensor value, allocate a permuted-shape replacement
//      and update its producer to emit the new shape + all consumers to
//      read the new shape. Indexing maps on `linalg.generic` consumers get
//      their channel-axis dimension moved from position 0 → position 2.
//   3. Boundary bridge: insert two `linalg.transpose` ops at the module
//      function entry (NCHW→NHWC) and exit (NHWC→NCHW) so the user-visible
//      function signature is unchanged.
//
// **Status (as of 2026-05-12).** Skeleton landed; full implementation
// pending. See `docs/dev_blog/2026-05-11-nhwc-binding-rewrite.md` for the
// design + acceptance criteria.

#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::QNN {
namespace {

// A rank-3 tensor `<C, H, W>` is HTA-rewriteable if:
//   - rank == 3
//   - element type is integer (i8/i32) — fp32 boundaries are unfixable
//   - the tensor is produced by a `linalg.generic` whose body is conv-like
//     (matchQuantConvBody) OR is read into a conv-like generic
// If any of these don't hold, leave the tensor alone (likely a constant,
// reshape intermediate, or scalar feed).
// Same disambiguation for rank-4 NCHW `<N, C, H, W>` with batch N=1.
// yolov8 has N=1 throughout, with square spatial. The signature:
//   shape = [a, b, c, d] with a == 1:
//     NCHW iff c == d and b != c   (channels at axis 1, square spatial)
//     NHWC iff b == c and c != d   (square spatial, channels at axis 3)
// Strict rank-4 NCHW disambiguation: only accept shapes that unambiguously
// match `<1, C, H, W>` with square spatial. yolov8 uses this form for the
// model input boundary. Reject all other rank-4 shapes (including
// canonicalization artifacts like `<H, W, 1, C>` permutations) to avoid
// false-positive layout rewrites that produce malformed IR.
static bool isLikelyNCHW4(RankedTensorType t) {
	if (t.getRank() != 4)
		return false;
	auto s = t.getShape();
	if (s[0] != 1)
		return false; // require N=1 explicitly
	int64_t b = s[1], c = s[2], d = s[3];
	if (b <= 0 || c <= 0 || d <= 0)
		return false;
	// Signature: c == d (square spatial) AND b != c.
	// Accepts: <1, 3, 320, 320>, <1, 64, 40, 40>, <1, 384, 10, 10>, etc.
	// Rejects: <1, 80, 80, 48> (NHWC), <80, 80, 1, 48> (unusual perm),
	//          <1, 4, 4, 4> (all-square ambiguous).
	if (c != d)
		return false;
	if (b == c)
		return false; // all-square ambiguous — reject conservatively
	if (b > 4096)
		return false;
	if (c < 2)
		return false;
	return true;
}

// Disambiguate rank-3 tensor layouts using the square-spatial signature.
//
// yolov8 (and most CNNs) have square H==W spatial maps throughout. After our
// preprocessing-phase legalize-to-NHWC inserts transposes around convs, the
// IR contains BOTH layouts. The disambiguation rule:
//
//   shape = [a, b, c]:
//     if b == c and a != b  →  NCHW  (channels=a, spatial=b×c square)
//     if a == b and b != c  →  NHWC  (spatial=a×b square, channels=c)
//     if a == b == c        →  ambiguous; default NCHW (treat as channel-first)
//
// Non-square spatial cases fall back to a shape-magnitude heuristic.
static bool isLikelyNCHW(RankedTensorType t) {
	if (t.getRank() != 3)
		return false;
	auto s = t.getShape();
	int64_t a = s[0], b = s[1], c = s[2];
	if (a <= 0 || b <= 0 || c <= 0)
		return false;
	// Square-spatial signature (the reliable case for yolov8).
	if (b == c && a != b)
		return true;
	if (a == b && b != c)
		return false;
	if (a == b && b == c)
		return true; // all-equal: default NCHW
	// Non-square spatial fallback: channel typically smaller than spatial in
	// early layers, larger in deep layers. Use the "spatial > channel by
	// 4x" rule as a conservative NCHW marker; otherwise assume NHWC.
	if (a > 1024)
		return false;
	// Last two dims must be plausibly spatial (within 2x of each other).
	if (b > 2 * c || c > 2 * b)
		return false;
	return a * 4 <= std::max(b, c);
}

static bool isActivationCandidate(Value v) {
	auto t = dyn_cast<RankedTensorType>(v.getType());
	if (!t || t.getRank() != 3)
		return false;
	Type elt = t.getElementType();
	if (!elt.isInteger(8) && !elt.isInteger(32))
		return false;
	// Only rewrite NCHW-form tensors. NHWC-form ones already match our target.
	if (!isLikelyNCHW(t))
		return false;
	// Heuristic: the producer is either a tensor.empty, a linalg.generic with
	// ≥1 reduction iter (conv/matmul-like), or a `linalg.fill` whose user is
	// a conv-like generic.
	Operation *defOp = v.getDefiningOp();
	if (!defOp) {
		// Block argument (function input) — accept; the boundary-transpose
		// step will handle it.
		return true;
	}
	if (isa<tensor::EmptyOp>(defOp))
		return true;
	if (auto g = dyn_cast<linalg::GenericOp>(defOp)) {
		for (auto it : g.getIteratorTypesArray()) {
			if (it == utils::IteratorType::reduction)
				return true;
		}
		// Pure elementwise — only rewrite if any of its consumers is itself
		// conv-like (i.e., this is the rescale-tail feeding the next conv).
		for (Operation *user : v.getUsers()) {
			if (auto cg = dyn_cast<linalg::GenericOp>(user)) {
				for (auto it : cg.getIteratorTypesArray()) {
					if (it == utils::IteratorType::reduction)
						return true;
				}
			}
		}
		return false;
	}
	return false;
}

// NCHW→NHWC permutation for rank-3 collapsed form.
// (rank-3, batch elided: C=axis 0, H=axis 1, W=axis 2 → H, W, C)
static constexpr int64_t kNchwToNhwcPerm[] = {1, 2, 0};

struct RewriteToNHWCBindingsPass
	: public PassWrapper<RewriteToNHWCBindingsPass, OperationPass<>> {
	StringRef getArgument() const final {
		return "merlin-qnn-rewrite-to-nhwc-bindings";
	}
	StringRef getDescription() const final {
		return "Rewrite cross-dispatch activation tensors from <C, H, W> "
			   "(NCHW-collapsed) to <H, W, C> (NHWC) so QNN_HTA-targeted convs "
			   "no longer need internal qnn.transpose bridges. Inserts "
			   "boundary "
			   "NCHW↔NHWC transposes at the module function entry/exit so the "
			   "user-visible signature is unchanged.";
	}
	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
			arith::ArithDialect, func::FuncDialect>();
	}

	// ---- Walk 1: Survey ----
	llvm::SetVector<Value> surveyActivationValues(Operation *root) {
		llvm::SetVector<Value> result;
		root->walk([&](Operation *op) {
			for (Value v : op->getResults()) {
				if (isActivationCandidate(v))
					result.insert(v);
			}
		});
		return result;
	}

	// Permute an AffineMap's result list using `perm`. Indices: new result[i]
	// is the old result at position perm[i].
	static AffineMap permuteAffineMapResults(
		AffineMap m, ArrayRef<int64_t> perm) {
		assert((int64_t)m.getNumResults() == (int64_t)perm.size());
		SmallVector<AffineExpr> newResults;
		newResults.reserve(perm.size());
		for (int64_t p : perm)
			newResults.push_back(m.getResult(p));
		return AffineMap::get(
			m.getNumDims(), m.getNumSymbols(), newResults, m.getContext());
	}

	// Permute a tensor's shape: result[i] = in[perm[i]].
	static SmallVector<int64_t> permuteShape(
		ArrayRef<int64_t> in, ArrayRef<int64_t> perm) {
		SmallVector<int64_t> out;
		out.reserve(perm.size());
		for (int64_t p : perm)
			out.push_back(in[p]);
		return out;
	}

	// Rewrite a single candidate value's type from <C, H, W> to <H, W, C>.
	// Updates the producer op AND every consumer's operand-indexing map.
	// Returns true if successful, false if the rewrite couldn't be applied
	// (in which case the candidate is skipped — IR untouched).
	bool rewriteOneCandidate(Value v) {
		auto oldTy = cast<RankedTensorType>(v.getType());
		auto perm = ArrayRef<int64_t>(kNchwToNhwcPerm, 3);
		auto newShape = permuteShape(oldTy.getShape(), perm);
		auto newTy = RankedTensorType::get(newShape, oldTy.getElementType());

		Operation *defOp = v.getDefiningOp();
		if (!defOp)
			return false; // block args handled by boundary pass

		// ---- Update producer ----
		OpBuilder b(defOp);
		if (auto empty = dyn_cast<tensor::EmptyOp>(defOp)) {
			Value newEmpty = tensor::EmptyOp::create(
				b, empty.getLoc(), newShape, oldTy.getElementType());
			v.replaceAllUsesWith(newEmpty);
			empty.erase();
			// Recursively rewrite consumers of the newEmpty (already triggered
			// by the replaceAllUsesWith).
			// Note: this also covers operand-type changes since the SSA value
			// type changed.
			return true;
		}

		if (auto fill = dyn_cast<linalg::FillOp>(defOp)) {
			// linalg.fill outs needs new type. Since fill is purely a write,
			// its outs init is a tensor.empty (or similar) — that will be
			// updated separately when its candidate is processed. Just change
			// the result type.
			fill.getResult(0).setType(newTy);
			return true;
		}

		if (auto generic = dyn_cast<linalg::GenericOp>(defOp)) {
			// Find which result slot v occupies.
			unsigned resultIdx = 0;
			bool found = false;
			for (unsigned i = 0; i < generic.getNumResults(); ++i) {
				if (generic.getResult(i) == v) {
					resultIdx = i;
					found = true;
					break;
				}
			}
			if (!found)
				return false;
			// Permute the corresponding output indexing map.
			unsigned outsMapIdx = generic.getNumDpsInputs() + resultIdx;
			auto maps = generic.getIndexingMapsArray();
			if (outsMapIdx >= maps.size() ||
				maps[outsMapIdx].getNumResults() != 3)
				return false;
			maps[outsMapIdx] = permuteAffineMapResults(maps[outsMapIdx], perm);
			generic.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
			// Update result type.
			v.setType(newTy);
			// Update outs init operand: replace with a permuted-shape empty.
			Value outsInit = generic.getDpsInits()[resultIdx];
			auto outsTy = dyn_cast<RankedTensorType>(outsInit.getType());
			if (outsTy && outsTy.getRank() == 3) {
				OpBuilder ib(generic);
				Value newOutsInit = tensor::EmptyOp::create(
					ib, outsInit.getLoc(), newShape, outsTy.getElementType());
				generic.getDpsInitsMutable()[resultIdx].assign(newOutsInit);
			}
			return true;
		}

		if (auto bcast = dyn_cast<linalg::BroadcastOp>(defOp)) {
			// Bias broadcast: source is rank-1 (channel), output is rank-3
			// <C, H, W>. After rewrite: output <H, W, C>; the broadcast
			// dimensions must move from [1, 2] (broadcasting along H, W) to
			// [0, 1] (broadcasting along H, W, since C is now at position 2).
			auto dims = bcast.getDimensions();
			SmallVector<int64_t> newDims;
			for (int64_t d : dims) {
				// Each old broadcast dim d (in NCHW <C,H,W>) maps to the
				// position in NHWC <H,W,C>: old dim 1 (H) → new 0; old dim 2
				// (W) → new 1; old dim 0 (C) → new 2.
				int64_t newD = -1;
				for (int64_t i = 0; i < 3; ++i) {
					if (perm[i] == d) {
						newD = i;
						break;
					}
				}
				if (newD < 0)
					return false;
				newDims.push_back(newD);
			}
			bcast.setDimensions(newDims);
			bcast.getResult()[0].setType(newTy);
			// Update outs init.
			Value outsInit = bcast.getDpsInits()[0];
			auto outsTy = dyn_cast<RankedTensorType>(outsInit.getType());
			if (outsTy && outsTy.getRank() == 3) {
				OpBuilder ib(bcast);
				Value newOutsInit = tensor::EmptyOp::create(
					ib, outsInit.getLoc(), newShape, outsTy.getElementType());
				bcast.getDpsInitsMutable()[0].assign(newOutsInit);
			}
			return true;
		}

		// Producers we don't (yet) handle — skip cleanly.
		return false;
	}

	// After producer rewrite, the SSA value type has changed (via setType
	// or replaceAllUsesWith). Walk consumer ops and rewrite their per-
	// operand indexing maps (linalg.generic) or attribute lists (broadcast,
	// etc.) to match the new physical layout.
	bool rewriteConsumersOfValue(Value v) {
		auto perm = ArrayRef<int64_t>(kNchwToNhwcPerm, 3);
		for (OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
			Operation *user = use.getOwner();
			if (auto gen = dyn_cast<linalg::GenericOp>(user)) {
				unsigned operandIdx = use.getOperandNumber();
				auto maps = gen.getIndexingMapsArray();
				if (operandIdx >= maps.size())
					continue;
				auto &m = maps[operandIdx];
				if (m.getNumResults() != 3)
					continue;
				maps[operandIdx] = permuteAffineMapResults(m, perm);
				OpBuilder b(gen);
				gen.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
			}
			// Other consumer ops (tensor.collapse_shape, tensor.expand_shape,
			// linalg.fill outs operand): use of v has the new type via SSA
			// propagation — no per-op attribute changes needed for plain
			// identity consumers.
		}
		return true;
	}

	// ---- Walk 3: Boundary transposes ----
	// TODO(Y-Ph7d): insert single NCHW→NHWC linalg.transpose at function
	// entry for each rank-3 i8/i32 argument, NHWC→NCHW on each return.

	void runOnOperation() override;
};

// ---------------------------------------------------------------------------
// Y-Ph7c-2: DialectConversion-based atomic rewrite
// ---------------------------------------------------------------------------

class NCHWToNHWCTypeConverter : public TypeConverter {
  public:
	NCHWToNHWCTypeConverter() {
		addConversion([](Type t) { return t; });
		addConversion([](RankedTensorType t) -> Type {
			if (isLikelyNCHW(t)) {
				static constexpr int64_t perm[] = {1, 2, 0};
				SmallVector<int64_t> newShape;
				newShape.reserve(3);
				for (int64_t p : perm)
					newShape.push_back(t.getShape()[p]);
				return RankedTensorType::get(newShape, t.getElementType());
			}
			if (isLikelyNCHW4(t)) {
				// rank-4 NCHW <N, C, H, W> → NHWC <N, H, W, C>, perm [0, 2, 3,
				// 1]
				static constexpr int64_t perm[] = {0, 2, 3, 1};
				SmallVector<int64_t> newShape;
				newShape.reserve(4);
				for (int64_t p : perm)
					newShape.push_back(t.getShape()[p]);
				return RankedTensorType::get(newShape, t.getElementType());
			}
			return t;
		});
		// Source materialization: if we ever need to bridge a converted type
		// back to its source (e.g., at a partial-conversion boundary), emit a
		// linalg.transpose. For now we use unrealized_conversion_cast as a
		// placeholder — it'll get DCE'd if both endpoints get converted.
		addSourceMaterialization([](OpBuilder &b, Type t, ValueRange v,
									 Location loc) -> Value {
			if (v.size() != 1)
				return Value();
			return UnrealizedConversionCastOp::create(b, loc, TypeRange{t}, v)
				.getResult(0);
		});
		addTargetMaterialization([](OpBuilder &b, Type t, ValueRange v,
									 Location loc) -> Value {
			if (v.size() != 1)
				return Value();
			return UnrealizedConversionCastOp::create(b, loc, TypeRange{t}, v)
				.getResult(0);
		});
	}
};

static SmallVector<int64_t> permuteShapeFn(
	ArrayRef<int64_t> in, ArrayRef<int64_t> perm) {
	SmallVector<int64_t> out;
	out.reserve(perm.size());
	for (int64_t p : perm)
		out.push_back(in[p]);
	return out;
}

static AffineMap permuteAffineMapResultsFn(
	AffineMap m, ArrayRef<int64_t> perm) {
	SmallVector<AffineExpr> newResults;
	newResults.reserve(perm.size());
	for (int64_t p : perm)
		newResults.push_back(m.getResult(p));
	return AffineMap::get(
		m.getNumDims(), m.getNumSymbols(), newResults, m.getContext());
}

// tensor.empty: trivial — just create a new empty with the permuted shape.
struct ConvertEmpty : OpConversionPattern<tensor::EmptyOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Type oldTy = op.getResult().getType();
		Type newTy = getTypeConverter()->convertType(oldTy);
		if (newTy == oldTy)
			return failure();
		auto newRTy = cast<RankedTensorType>(newTy);
		rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
			op, newRTy.getShape(), newRTy.getElementType());
		return success();
	}
};

// linalg.fill: bypass outs cascade ordering by creating a fresh
// tensor.empty with the converted shape. The original outs operand
// becomes orphaned and DCE'd.
struct ConvertFill : OpConversionPattern<linalg::FillOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Type oldOutTy = op.getResult(0).getType();
		auto newOutRTy = dyn_cast<RankedTensorType>(
			getTypeConverter()->convertType(oldOutTy));
		if (!newOutRTy || newOutRTy == oldOutTy)
			return failure();
		Value newOutsInit = tensor::EmptyOp::create(rewriter, op.getLoc(),
			newOutRTy.getShape(), newOutRTy.getElementType());
		auto newOp = linalg::FillOp::create(rewriter, op.getLoc(),
			adaptor.getInputs(), ValueRange{newOutsInit});
		rewriter.replaceOp(op, newOp.getResults());
		return success();
	}
};

// linalg.broadcast: rewrite the `dimensions` attribute when the result
// layout changes. NCHW <C,H,W> broadcasts a rank-1 <C> with dimensions=[1,2]
// (broadcast across H, W). NHWC <H,W,C> broadcasts the same <C> with
// dimensions=[0,1] (still broadcasting across H, W, but they're now at
// positions 0, 1).
struct ConvertBroadcast : OpConversionPattern<linalg::BroadcastOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(linalg::BroadcastOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		// Only handle rank-3 NCHW output → NHWC.
		Value outsInit = adaptor.getInit();
		auto outsTy = dyn_cast<RankedTensorType>(outsInit.getType());
		if (!outsTy || outsTy.getRank() != 3)
			return failure();
		Value oldInit = op.getInit();
		auto oldOutsTy = dyn_cast<RankedTensorType>(oldInit.getType());
		if (!oldOutsTy || oldOutsTy == outsTy)
			return failure();
		// The original dimensions list contains indices INTO the original
		// (NCHW) output shape. Remap each via the perm [1, 2, 0]:
		// old NCHW dim 0 (C) → new NHWC dim 2
		// old NCHW dim 1 (H) → new NHWC dim 0
		// old NCHW dim 2 (W) → new NHWC dim 1
		static constexpr int64_t oldToNew[3] = {2, 0, 1};
		SmallVector<int64_t> newDims;
		newDims.reserve(op.getDimensions().size());
		for (int64_t d : op.getDimensions()) {
			if (d < 0 || d > 2)
				return failure();
			newDims.push_back(oldToNew[d]);
		}
		// Make sure outsInit has the converted type.
		Type expectedOutsTy =
			getTypeConverter()->convertType(op.getInit().getType());
		if (outsInit.getType() != expectedOutsTy) {
			outsInit = UnrealizedConversionCastOp::create(rewriter, op.getLoc(),
				TypeRange{expectedOutsTy}, ValueRange{outsInit})
						   .getResult(0);
		}
		auto newOp = linalg::BroadcastOp::create(rewriter, op.getLoc(),
			adaptor.getInput(), outsInit,
			rewriter.getDenseI64ArrayAttr(newDims));
		rewriter.replaceOp(op, newOp.getResults());
		return success();
	}
};

// linalg.generic: the heart of the rewrite. Operand types come from the
// OpAdaptor (post-conversion). For each operand whose type was rewritten,
// we permute its indexing map's results in lockstep. Result types are
// permuted too if their original was NCHW.
struct ConvertGeneric : OpConversionPattern<linalg::GenericOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		static constexpr int64_t perm[] = {1, 2, 0};
		auto permArr = ArrayRef<int64_t>(perm, 3);
		// Compute new result types.
		SmallVector<Type> newResultTypes;
		newResultTypes.reserve(op.getNumResults());
		bool anyResultChange = false;
		for (Type t : op.getResultTypes()) {
			Type nt = getTypeConverter()->convertType(t);
			newResultTypes.push_back(nt);
			if (nt != t)
				anyResultChange = true;
		}
		// Check whether any operand type changed.
		bool anyOperandChange = false;
		auto oldOperands = op.getOperands();
		auto newOperands = adaptor.getOperands();
		for (auto [oldV, newV] : llvm::zip(oldOperands, newOperands)) {
			if (oldV.getType() != newV.getType()) {
				anyOperandChange = true;
				break;
			}
		}
		if (!anyOperandChange && !anyResultChange)
			return failure();
		// Permute indexing maps for each operand whose type changed.
		static constexpr int64_t perm4[] = {0, 2, 3, 1};
		auto perm4Arr = ArrayRef<int64_t>(perm4, 4);
		SmallVector<AffineMap> newMaps;
		auto oldMaps = op.getIndexingMapsArray();
		newMaps.reserve(oldMaps.size());
		for (size_t i = 0; i < oldMaps.size(); ++i) {
			bool changed = false;
			if (i < newOperands.size()) {
				if (newOperands[i].getType() != oldOperands[i].getType())
					changed = true;
			} else {
				size_t resultIdx = i - newOperands.size();
				if (resultIdx < newResultTypes.size() &&
					newResultTypes[resultIdx] != op.getResultTypes()[resultIdx])
					changed = true;
			}
			AffineMap m = oldMaps[i];
			if (changed && m.getNumResults() == 3) {
				newMaps.push_back(permuteAffineMapResultsFn(m, permArr));
			} else if (changed && m.getNumResults() == 4) {
				newMaps.push_back(permuteAffineMapResultsFn(m, perm4Arr));
			} else {
				newMaps.push_back(m);
			}
		}
		// Force operand types to match what the type converter expects
		// (handles cascade-ordering when adaptor returns pre-conversion
		// values).
		SmallVector<Value> newIns;
		newIns.reserve(adaptor.getInputs().size());
		for (auto [i, v] : llvm::enumerate(adaptor.getInputs())) {
			Type expected =
				getTypeConverter()->convertType(op.getInputs()[i].getType());
			if (v.getType() != expected) {
				v = UnrealizedConversionCastOp::create(
					rewriter, op.getLoc(), TypeRange{expected}, ValueRange{v})
						.getResult(0);
			}
			newIns.push_back(v);
		}
		SmallVector<Value> newOuts;
		newOuts.reserve(adaptor.getOutputs().size());
		for (auto [i, v] : llvm::enumerate(adaptor.getOutputs())) {
			Type expected =
				getTypeConverter()->convertType(op.getOutputs()[i].getType());
			if (v.getType() != expected) {
				v = UnrealizedConversionCastOp::create(
					rewriter, op.getLoc(), TypeRange{expected}, ValueRange{v})
						.getResult(0);
			}
			newOuts.push_back(v);
		}
		auto newOp = linalg::GenericOp::create(rewriter, op.getLoc(),
			newResultTypes, ValueRange(newIns), ValueRange(newOuts), newMaps,
			op.getIteratorTypesArray());
		// Move the body region from the old op.
		rewriter.inlineRegionBefore(
			op.getRegion(), newOp.getRegion(), newOp.getRegion().begin());
		rewriter.replaceOp(op, newOp.getResults());
		return success();
	}
};

// tensor.collapse_shape: passes through cleanly when both input and output
// are converted to NHWC consistently. The reassociation indices stay valid
// because the conversion permutes BOTH endpoints by the same scheme:
//   <1, C, H, W> reassoc [[0,1],[2],[3]] → <C, H, W>
// after conversion becomes:
//   <1, H, W, C> reassoc [[0,1],[2],[3]] → <H, W, C>
// — same reassoc still produces the matching rank-3 result.
struct ConvertCollapseShape : OpConversionPattern<tensor::CollapseShapeOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(tensor::CollapseShapeOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Value newSrc = adaptor.getSrc();
		Type srcTyChanged = (newSrc.getType() != op.getSrc().getType())
			? newSrc.getType()
			: Type();
		auto newSrcRTy = dyn_cast<RankedTensorType>(newSrc.getType());
		if (!newSrcRTy)
			return failure();

		// Case 1: Source unchanged → simple result-type-only conversion (if
		// any).
		if (!srcTyChanged) {
			Type newOutTy =
				getTypeConverter()->convertType(op.getResult().getType());
			if (newOutTy == op.getResult().getType())
				return failure();
			auto newOutRTy = dyn_cast<RankedTensorType>(newOutTy);
			if (!newOutRTy)
				return failure();
			rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
				op, newOutRTy, newSrc, op.getReassociationIndices());
			return success();
		}

		// Case 2: Source got permuted NCHW→NHWC. The reassoc must rotate so it
		// still collapses meaningful groups. Yolov8 cases:
		//   a) rank-3 <C,H,W> reassoc [[0],[1,2]] → <C, H*W>
		//      becomes rank-3 <H,W,C> reassoc [[0,1],[2]] → <H*W, C>
		//   b) rank-3 <C,H,W> reassoc [[0,1],[2]] → <C*H, W>   (rare)
		//      becomes rank-3 <H,W,C> reassoc [[0],[1,2]] → <H, W*C>
		//   c) rank-4 <N,C,H,W> reassoc [[0,1],[2],[3]] → <N*C, H, W>
		//      becomes rank-4 <N,H,W,C> reassoc [[0],[1],[2,3]] → <N, H, W*C>
		//   d) rank-4 → rank-3 collapse of batch: <1,C,H,W> reassoc
		//      [[0,1],[2],[3]] → <C,H,W>. After conv: <1,H,W,C> reassoc
		//      [[0,1],[2],[3]] still works → <H,W,C>. SAME reassoc OK.
		auto reassoc = op.getReassociationIndices();
		int srcRank = newSrcRTy.getRank();
		// Determine if SAME reassoc is still valid (case d): result rank
		// matches src rank minus the difference, and groups don't span the
		// permuted-channel boundary.
		auto isCase_d = [&]() {
			// Case d: rank-4 → rank-3, source N moved (perm [0,2,3,1] keeps N
			// at position 0). The reassoc [[0,1],[2],[3]] still merges {0,1}
			// which is now {N, H}. We want {N, H}? No, we want {N, ...} merged
			// with C (was {N, C} at NCHW positions [0,1]). After perm, C is at
			// 3, not adjacent to 0. So same reassoc doesn't preserve semantic.
			// ACTUALLY for case d (collapse rank-4 to rank-3), the goal is to
			// drop N=1, producing rank-3. In NHWC, N is still at position 0 so
			// reassoc [[0],[1],[2],[3]]→[[0,1],...] would still merge N+H not
			// N+C. So case d is NOT generally safe; skip.
			return false;
		};

		if (srcRank == 3) {
			// Result was rank-2. Old reassoc collapses 3→2; rotate the reassoc
			// group containing channel-axis-0 to be at the end of the new
			// reassoc list, since after perm [1,2,0] channel is at position 2.
			// The "group containing 0" in old maps to "group containing 2" in
			// new. We rotate the group list to swap positions.
			if (reassoc.size() != 2)
				return failure();
			auto &g0 = reassoc[0], &g1 = reassoc[1];
			// Find which group contains 0 (the old channel).
			bool g0HasC = std::find(g0.begin(), g0.end(), 0) != g0.end();
			bool g1HasC = std::find(g1.begin(), g1.end(), 0) != g1.end();
			if (g0HasC == g1HasC)
				return failure();
			// Build new reassoc: swap and remap each old index by perm.
			// Old indices [0,1,2] map to new positions where perm[i]=old:
			//   old 0 → new 2; old 1 → new 0; old 2 → new 1
			static constexpr int64_t oldToNew[3] = {2, 0, 1};
			SmallVector<SmallVector<int64_t, 2>, 2> newGroups(2);
			auto remap = [&](ReassociationIndices grp,
							 SmallVector<int64_t, 2> &out) {
				for (int64_t i : grp)
					out.push_back(oldToNew[i]);
				std::sort(out.begin(), out.end());
			};
			// Group containing C (old idx 0) goes to position 1 (last).
			// Group containing spatial dims goes to position 0.
			remap(g0HasC ? g1 : g0, newGroups[0]);
			remap(g0HasC ? g0 : g1, newGroups[1]);
			// Build new output shape: collapse groups via product.
			SmallVector<int64_t> newOutShape;
			for (auto &grp : newGroups) {
				int64_t prod = 1;
				for (int64_t i : grp)
					prod *= newSrcRTy.getShape()[i];
				newOutShape.push_back(prod);
			}
			auto newOutRTy =
				RankedTensorType::get(newOutShape, newSrcRTy.getElementType());
			SmallVector<ReassociationIndices, 2> newReassoc;
			for (auto &g : newGroups)
				newReassoc.push_back(ReassociationIndices(g.begin(), g.end()));
			rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
				op, newOutRTy, newSrc, newReassoc);
			return success();
		}

		// Rank-4 source: keep the simple pass-through (case d-like) for N=1
		// collapse. Other rank-4 cases are not handled here.
		Type newOutTy =
			getTypeConverter()->convertType(op.getResult().getType());
		auto newOutRTy = dyn_cast<RankedTensorType>(newOutTy);
		if (!newOutRTy)
			return failure();
		rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
			op, newOutRTy, newSrc, op.getReassociationIndices());
		return success();
	}
};

// tensor.expand_shape: dual of collapse_shape. Same logic.
struct ConvertExpandShape : OpConversionPattern<tensor::ExpandShapeOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(tensor::ExpandShapeOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Type oldOutTy = op.getResult().getType();
		Type newOutTy = getTypeConverter()->convertType(oldOutTy);
		Value newSrc = adaptor.getSrc();
		if (newOutTy == oldOutTy && newSrc.getType() == op.getSrc().getType())
			return failure();
		auto newOutRTy = dyn_cast<RankedTensorType>(newOutTy);
		if (!newOutRTy)
			return failure();
		rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
			op, newOutRTy, newSrc, op.getReassociationIndices());
		return success();
	}
};

// tensor.concat: the `dim` attribute indexes into the result tensor's
// shape. NCHW concat axis 0 (rank-3 channel-first) maps to NHWC axis 2.
// Other axes (spatial) shift positions correspondingly.
struct ConvertConcat : OpConversionPattern<tensor::ConcatOp> {
	using OpConversionPattern::OpConversionPattern;
	LogicalResult matchAndRewrite(tensor::ConcatOp op, OpAdaptor adaptor,
		ConversionPatternRewriter &rewriter) const override {
		Type oldOutTy = op.getResult().getType();
		Type newOutTy = getTypeConverter()->convertType(oldOutTy);
		if (newOutTy == oldOutTy)
			return failure();
		auto newOutRTy = dyn_cast<RankedTensorType>(newOutTy);
		if (!newOutRTy)
			return failure();
		int64_t oldDim = op.getDim();
		int64_t newDim = oldDim;
		if (newOutRTy.getRank() == 3) {
			// perm [1, 2, 0] inverse: new dim = position of old dim in perm.
			// old 0 (C) → new 2; old 1 (H) → new 0; old 2 (W) → new 1.
			static constexpr int64_t map3[3] = {2, 0, 1};
			if (oldDim < 0 || oldDim >= 3)
				return failure();
			newDim = map3[oldDim];
		} else if (newOutRTy.getRank() == 4) {
			// perm [0, 2, 3, 1] inverse: old N=0 → new 0; old C=1 → new 3;
			// old H=2 → new 1; old W=3 → new 2.
			static constexpr int64_t map4[4] = {0, 3, 1, 2};
			if (oldDim < 0 || oldDim >= 4)
				return failure();
			newDim = map4[oldDim];
		}
		rewriter.replaceOpWithNewOp<tensor::ConcatOp>(
			op, newOutRTy, newDim, adaptor.getInputs());
		return success();
	}
};

void RewriteToNHWCBindingsPass::runOnOperation() {
	Operation *root = getOperation();
	bool dbg = ::getenv("MERLIN_NHWC_DBG") != nullptr;
	if (dbg) {
		llvm::SetVector<Value> activations = surveyActivationValues(root);
		llvm::errs() << "[nhwc-rewrite] candidates: " << activations.size()
					 << " rank-3 activation values (post-disambiguation)\n";
	}

	NCHWToNHWCTypeConverter converter;
	ConversionTarget target(getContext());
	// All ops default LEGAL; only specific op kinds become illegal when
	// their result/operand types are converter-changeable. The materializer
	// (unrealized_conversion_cast) bridges between converted and unconverted
	// edges; cleanup happens in Y-Ph7d.
	target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
	target.addDynamicallyLegalOp<tensor::EmptyOp, linalg::FillOp,
		linalg::BroadcastOp, linalg::GenericOp, tensor::CollapseShapeOp,
		tensor::ExpandShapeOp, tensor::ConcatOp>([&](Operation *op) {
		for (Type t : op->getResultTypes()) {
			if (converter.convertType(t) != t)
				return false;
		}
		for (Value v : op->getOperands()) {
			if (converter.convertType(v.getType()) != v.getType())
				return false;
		}
		return true;
	});
	target.addLegalOp<UnrealizedConversionCastOp>();

	RewritePatternSet patterns(&getContext());
	patterns.add<ConvertEmpty, ConvertFill, ConvertBroadcast, ConvertGeneric,
		ConvertCollapseShape, ConvertExpandShape, ConvertConcat>(
		converter, &getContext());

	if (failed(applyPartialConversion(root, target, std::move(patterns)))) {
		if (dbg)
			llvm::errs()
				<< "[nhwc-rewrite] partial conversion failed; IR may be "
				   "partially rewritten\n";
		signalPassFailure();
	}
}

} // namespace

std::unique_ptr<Pass> createRewriteToNHWCBindingsPass() {
	return std::make_unique<RewriteToNHWCBindingsPass>();
}

// ---------------------------------------------------------------------------
// Y-Ph7d: Boundary transpose materialization
// ---------------------------------------------------------------------------
// RewriteToNHWCBindings leaves `unrealized_conversion_cast` ops at the
// boundary between converted (NHWC) and unconverted (NCHW) tensors. This
// pass replaces each cast with a real `linalg.transpose` op that performs
// the layout permutation. Adjacent transposes (created when a cast's
// output is consumed by another cast going back) get folded by
// downstream canonicalization.

namespace {

// Compute the permutation that takes `fromShape` to `toShape` assuming
// they are layout permutations of each other. Returns empty on no match.
static SmallVector<int64_t> inferPerm(
	ArrayRef<int64_t> fromShape, ArrayRef<int64_t> toShape) {
	if (fromShape.size() != toShape.size())
		return {};
	SmallVector<int64_t> perm;
	perm.reserve(fromShape.size());
	SmallVector<bool> used(fromShape.size(), false);
	for (int64_t dim : toShape) {
		bool found = false;
		for (size_t i = 0; i < fromShape.size(); ++i) {
			if (!used[i] && fromShape[i] == dim) {
				perm.push_back(static_cast<int64_t>(i));
				used[i] = true;
				found = true;
				break;
			}
		}
		if (!found)
			return {};
	}
	return perm;
}

struct LowerNHWCCastsToTransposesPass
	: public PassWrapper<LowerNHWCCastsToTransposesPass, OperationPass<>> {
	StringRef getArgument() const final {
		return "merlin-qnn-lower-nhwc-casts-to-transposes";
	}
	StringRef getDescription() const final {
		return "Replace `unrealized_conversion_cast` between rank-3/rank-4 "
			   "tensor types of permuted shape (NCHW↔NHWC) with explicit "
			   "`linalg.transpose` ops. Required after RewriteToNHWCBindings "
			   "to materialize the layout-change boundaries.";
	}
	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
	}
	void runOnOperation() override {
		SmallVector<UnrealizedConversionCastOp> work;
		getOperation()->walk([&](UnrealizedConversionCastOp op) {
			if (op.getNumOperands() != 1 || op.getNumResults() != 1)
				return;
			auto inTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
			auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
			if (!inTy || !outTy)
				return;
			if (inTy == outTy) {
				// Identity cast — DCE.
				work.push_back(op);
				return;
			}
			if (inTy.getRank() != outTy.getRank())
				return;
			if (inTy.getElementType() != outTy.getElementType())
				return;
			work.push_back(op);
		});
		for (auto castOp : work) {
			OpBuilder b(castOp);
			auto inV = castOp.getOperand(0);
			auto inTy = mlir::cast<RankedTensorType>(inV.getType());
			auto outTy =
				mlir::cast<RankedTensorType>(castOp.getResult(0).getType());
			if (inTy == outTy) {
				castOp.getResult(0).replaceAllUsesWith(inV);
				castOp.erase();
				continue;
			}
			SmallVector<int64_t> perm =
				inferPerm(inTy.getShape(), outTy.getShape());
			if (perm.empty())
				continue;
			Value emptyDest = tensor::EmptyOp::create(
				b, castOp.getLoc(), outTy.getShape(), outTy.getElementType());
			Value transposed = linalg::TransposeOp::create(
				b, castOp.getLoc(), inV, emptyDest, perm)
								   .getResult()[0];
			castOp.getResult(0).replaceAllUsesWith(transposed);
			castOp.erase();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createLowerNHWCCastsToTransposesPass() {
	return std::make_unique<LowerNHWCCastsToTransposesPass>();
}

} // namespace mlir::iree_compiler::QNN
