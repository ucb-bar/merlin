// TileNPUKernelToSchedule.cpp
//
// Generates scf.for tile loops around NPUKernel::UKernelGenericOp and
// NPUKernel::MatmulOp at manifest tile-shape granularity, emitting
// NPUSchedule::UKernelLaunchOp / MatmulTileOp inside the innermost body.
//
// Strategy:
//   1. Read tile_shape from the manifest (loaded once at pass entry, keyed
//      by kernel symbol prefix). Default to {32, 32} when no manifest is
//      provided or the kernel's `tile_shape` is absent.
//   2. For each NPUKernel op at full-tensor granularity with statically-shaped
//      ranked-tensor operands, compute the per-dim tile count from the result
//      shape rounded up to `tile_shape`.
//   3. Build nested scf.for loops over the tile grid. Inside the innermost
//      body, extract tile slices from each input via tensor.extract_slice,
//      call the scheduled op, and insert the result back into the iteration
//      argument via tensor.insert_slice.
//   4. Replace the original op with the outermost loop result.
//
// This first cut handles 2D cases (elementwise binary ops, row-reduction).
// Matmul is deferred to a separate path because it needs a K-reduction dim
// handled inside the kernel (no outer K-loop).

#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUKernelDialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::iree_compiler::NPU {
namespace {

// Per-kernel tile shape loaded from the manifest.
struct KernelTileSpec {
	// Public tile dims the kernel accepts per invocation. Defaults to {32, 32}.
	// For multi-tile kernels (e.g. tiled_matmul_2x1) this may be larger.
	SmallVector<int64_t, 4> tileShape;
};

using TileSpecMap = llvm::StringMap<KernelTileSpec>;

// Parse the manifest JSON and populate `out` with (symbol_prefix -> tile_shape)
// entries. If manifestPath is empty or unreadable, `out` is left empty and the
// pass falls back to the default {32, 32}.
static void loadManifestTileShapes(StringRef manifestPath, TileSpecMap &out) {
	if (manifestPath.empty()) {
		return;
	}
	auto buf = llvm::MemoryBuffer::getFile(manifestPath);
	if (!buf) {
		return;
	}
	auto parsed = llvm::json::parse((*buf)->getBuffer());
	if (!parsed) {
		llvm::consumeError(parsed.takeError());
		return;
	}
	const llvm::json::Object *root = parsed->getAsObject();
	if (!root)
		return;
	const llvm::json::Object *kernels = root->getObject("kernels");
	if (!kernels)
		return;
	for (const auto &entry : *kernels) {
		const llvm::json::Object *kernelObj = entry.second.getAsObject();
		if (!kernelObj)
			continue;
		llvm::StringRef symbolPrefix;
		if (auto s = kernelObj->getString("symbol_prefix")) {
			symbolPrefix = *s;
		} else {
			continue;
		}
		KernelTileSpec spec;
		const llvm::json::Array *tileShape = kernelObj->getArray("tile_shape");
		if (tileShape) {
			for (const auto &v : *tileShape) {
				if (auto i = v.getAsInteger()) {
					spec.tileShape.push_back(*i);
				}
			}
		}
		if (spec.tileShape.empty()) {
			spec.tileShape = {32, 32};
		}
		out.insert({symbolPrefix, std::move(spec)});
	}
}

// Resolve the tile shape for a given kernel symbol. Falls back to {32, 32}
// when the manifest doesn't describe the symbol. Matmul ops (which have no
// symbol attribute) always use {32, 32} until the manifest gains multi-tile
// matmul variants.
static SmallVector<int64_t, 4> resolveTileShape(
	const TileSpecMap &specs, StringRef symbol) {
	auto it = specs.find(symbol);
	if (it != specs.end()) {
		return it->second.tileShape;
	}
	// Prefix-match for symbols like "npu_uk_elementwise_add_<suffix>".
	for (const auto &entry : specs) {
		if (symbol.starts_with(entry.getKey())) {
			return entry.getValue().tileShape;
		}
	}
	return {32, 32};
}

// Validates a tensor type is a rank-2 statically-shaped tensor we know how
// to tile today. Returns its shape on success, nullopt otherwise.
static std::optional<std::array<int64_t, 2>> staticRank2Shape(Type t) {
	auto rt = llvm::dyn_cast<RankedTensorType>(t);
	if (!rt || rt.getRank() != 2 || !rt.hasStaticShape()) {
		return std::nullopt;
	}
	return std::array<int64_t, 2>{rt.getShape()[0], rt.getShape()[1]};
}

// Emits: for (i = 0; i < outerBound; i += tileOuter) {
//          for (j = 0; j < innerBound; j += tileInner) {
//            <body builder>
//          }
//        }
// and returns the outermost scf.for's single result (the final output tensor).
//
// The body builder receives:
//   - `b`: the inner-loop OpBuilder
//   - `loc`: the original op's location
//   - `ivOuter`, `ivInner`: the induction variables (tile offsets)
//   - `outputIter`: the current iteration-arg tensor (the partial output)
// and must yield a Value that is the updated output tensor for this tile.
template <typename BodyFn>
static Value emitTileLoopNest(OpBuilder &rewriter, Location loc,
	int64_t outerBound, int64_t innerBound, int64_t tileOuter,
	int64_t tileInner, Value initOutput, BodyFn &&bodyFn) {
	auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
	auto cTileOuter = rewriter.create<arith::ConstantIndexOp>(loc, tileOuter);
	auto cTileInner = rewriter.create<arith::ConstantIndexOp>(loc, tileInner);
	auto cOuterBound = rewriter.create<arith::ConstantIndexOp>(loc, outerBound);
	auto cInnerBound = rewriter.create<arith::ConstantIndexOp>(loc, innerBound);

	auto outer = rewriter.create<scf::ForOp>(loc, c0, cOuterBound, cTileOuter,
		ValueRange{initOutput},
		[&](OpBuilder &ob, Location oloc, Value ivOuter,
			ValueRange outerIterArgs) {
			auto inner = ob.create<scf::ForOp>(oloc, c0, cInnerBound,
				cTileInner, ValueRange{outerIterArgs.front()},
				[&](OpBuilder &ib, Location iloc, Value ivInner,
					ValueRange innerIterArgs) {
					Value updated = bodyFn(
						ib, iloc, ivOuter, ivInner, innerIterArgs.front());
					ib.create<scf::YieldOp>(iloc, ValueRange{updated});
				});
			ob.create<scf::YieldOp>(oloc, inner.getResults());
		});
	return outer.getResult(0);
}

// Extract a (tileOuter x tileInner) slice of `source` starting at
// (ivOuter, ivInner). Returns a rank-2 tensor with the tile shape.
static Value extractTile2D(OpBuilder &b, Location loc, Value source,
	Value ivOuter, Value ivInner, int64_t tileOuter, int64_t tileInner) {
	auto sourceTy = llvm::cast<RankedTensorType>(source.getType());
	SmallVector<OpFoldResult, 2> offsets = {ivOuter, ivInner};
	SmallVector<OpFoldResult, 2> sizes = {
		b.getIndexAttr(tileOuter), b.getIndexAttr(tileInner)};
	SmallVector<OpFoldResult, 2> strides = {
		b.getIndexAttr(1), b.getIndexAttr(1)};
	auto tileTy = RankedTensorType::get(
		{tileOuter, tileInner}, sourceTy.getElementType());
	return b.create<tensor::ExtractSliceOp>(
		loc, tileTy, source, offsets, sizes, strides);
}

// Insert `tile` into `dest` at (ivOuter, ivInner). Returns the updated tensor.
static Value insertTile2D(OpBuilder &b, Location loc, Value tile, Value dest,
	Value ivOuter, Value ivInner, int64_t tileOuter, int64_t tileInner) {
	SmallVector<OpFoldResult, 2> offsets = {ivOuter, ivInner};
	SmallVector<OpFoldResult, 2> sizes = {
		b.getIndexAttr(tileOuter), b.getIndexAttr(tileInner)};
	SmallVector<OpFoldResult, 2> strides = {
		b.getIndexAttr(1), b.getIndexAttr(1)};
	return b.create<tensor::InsertSliceOp>(
		loc, tile, dest, offsets, sizes, strides);
}

// Emits an empty rank-2 tensor of the given shape and element type, to be
// used as the initial value of the outermost scf.for's iter-arg.
static Value emitEmptyOutput(
	OpBuilder &b, Location loc, ArrayRef<int64_t> shape, Type elementType) {
	return b.create<tensor::EmptyOp>(loc, shape, elementType);
}

// Rewrites a binary elementwise UKernelGenericOp on (M, N) tensors into a
// tile-loop nest. Each tile iteration emits a UKernelLaunchOp with the tile
// slices as inputs.
static LogicalResult tileBinaryElementwise(NPUKernel::UKernelGenericOp op,
	PatternRewriter &rewriter, ArrayRef<int64_t> tileShape) {
	if (op.getInputs().size() != 2)
		return failure();
	auto lhs = op.getInputs()[0];
	auto rhs = op.getInputs()[1];
	auto resultTy = op.getResult().getType();
	auto lhsShape = staticRank2Shape(lhs.getType());
	auto rhsShape = staticRank2Shape(rhs.getType());
	auto resShape = staticRank2Shape(resultTy);
	if (!lhsShape || !rhsShape || !resShape)
		return failure();
	if ((*lhsShape)[0] != (*resShape)[0] || (*lhsShape)[1] != (*resShape)[1]) {
		return failure();
	}
	if ((*rhsShape)[0] != (*resShape)[0] || (*rhsShape)[1] != (*resShape)[1]) {
		return failure();
	}
	int64_t outerBound = (*resShape)[0];
	int64_t innerBound = (*resShape)[1];
	int64_t tileOuter = tileShape.size() > 0 ? tileShape[0] : 32;
	int64_t tileInner = tileShape.size() > 1 ? tileShape[1] : 32;
	// Only rewrite if the tensor is strictly larger than one tile — otherwise
	// the 1:1 conversion path (ConvertNPUKernelToSchedule) already handles it.
	if (outerBound <= tileOuter && innerBound <= tileInner) {
		return failure();
	}
	// Tile bounds must divide the tensor; we don't handle padding in this
	// first cut. Pad the IR upstream or bump tile_shape when that's needed.
	if (outerBound % tileOuter != 0 || innerBound % tileInner != 0) {
		return failure();
	}

	Location loc = op.getLoc();
	auto resTy = llvm::cast<RankedTensorType>(resultTy);
	Value init = emitEmptyOutput(
		rewriter, loc, resTy.getShape(), resTy.getElementType());

	StringRef symbol = op.getSymbol();
	auto tileTy =
		RankedTensorType::get({tileOuter, tileInner}, resTy.getElementType());
	Value result = emitTileLoopNest(rewriter, loc, outerBound, innerBound,
		tileOuter, tileInner, init,
		[&](OpBuilder &b, Location iloc, Value ivOuter, Value ivInner,
			Value acc) -> Value {
			Value lhsTile = extractTile2D(
				b, iloc, lhs, ivOuter, ivInner, tileOuter, tileInner);
			Value rhsTile = extractTile2D(
				b, iloc, rhs, ivOuter, ivInner, tileOuter, tileInner);
			auto launch = b.create<NPUSchedule::UKernelLaunchOp>(iloc, tileTy,
				b.getStringAttr(symbol), ValueRange{lhsTile, rhsTile});
			return insertTile2D(b, iloc, launch.getResult(), acc, ivOuter,
				ivInner, tileOuter, tileInner);
		});

	rewriter.replaceOp(op, result);
	return success();
}

// Pattern: tile binary-elementwise NPUKernel::UKernelGenericOp.
struct TileBinaryElementwisePattern
	: public OpRewritePattern<NPUKernel::UKernelGenericOp> {
	TileBinaryElementwisePattern(MLIRContext *ctx, const TileSpecMap *specs)
		: OpRewritePattern(ctx), specs(specs) {}

	LogicalResult matchAndRewrite(NPUKernel::UKernelGenericOp op,
		PatternRewriter &rewriter) const override {
		if (op.getInputs().size() != 2) {
			return rewriter.notifyMatchFailure(op, "not binary");
		}
		// Skip matmul ukernels — they are tiled by TileMatmulPattern, which
		// understands the K-reduction dimension.
		if (op.getSymbol().starts_with("npu_uk_matmul")) {
			return rewriter.notifyMatchFailure(op, "matmul handled elsewhere");
		}
		auto tileShape = resolveTileShape(*specs, op.getSymbol());
		return tileBinaryElementwise(op, rewriter, tileShape);
	}

	const TileSpecMap *specs;
};

// Symbol names of the K-accumulator matmul variants. The compiler emits one
// of these per K-tile invocation; the order is fixed by manifest convention.
constexpr StringLiteral kMatmulAccFirst = "npu_uk_matmul_acc_first";
constexpr StringLiteral kMatmulAccMid = "npu_uk_matmul_acc_mid";
constexpr StringLiteral kMatmulAccLast = "npu_uk_matmul_acc_last";

// Emit one ukernel_launch per K-tile, choosing the variant by position in the
// K-accumulator chain. The launches share an implicit MXU accumulator across
// invocations: `_first` overwrites it, `_mid` adds into it, `_last` adds and
// then drains the result tile back to DRAM.
//
// Returns the SSA value produced by the `_last` launch — the tile that should
// be insert_slice'd into the (M, N) iter-arg.
static Value emitKAccumulatorChain(OpBuilder &b, Location loc, int64_t kTiles,
	int64_t mTile, int64_t nTile, int64_t kTile, Value lhs, Value rhs,
	Value ivM, Value ivN, Type tileElementType) {
	auto tileType = RankedTensorType::get({mTile, nTile}, tileElementType);
	Value lastResult;
	for (int64_t k = 0; k < kTiles; ++k) {
		auto kOffset = b.create<arith::ConstantIndexOp>(loc, k * kTile);
		// Slice A[m..m+mTile, k..k+kTile] and B[k..k+kTile, n..n+nTile].
		SmallVector<OpFoldResult, 2> aOffsets = {ivM, kOffset.getResult()};
		SmallVector<OpFoldResult, 2> aSizes = {
			b.getIndexAttr(mTile), b.getIndexAttr(kTile)};
		SmallVector<OpFoldResult, 2> aStrides = {
			b.getIndexAttr(1), b.getIndexAttr(1)};
		auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
		auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
		auto aTileType =
			RankedTensorType::get({mTile, kTile}, lhsType.getElementType());
		auto bTileType =
			RankedTensorType::get({kTile, nTile}, rhsType.getElementType());
		Value aTile = b.create<tensor::ExtractSliceOp>(
			loc, aTileType, lhs, aOffsets, aSizes, aStrides);
		SmallVector<OpFoldResult, 2> bOffsets = {kOffset.getResult(), ivN};
		SmallVector<OpFoldResult, 2> bSizes = {
			b.getIndexAttr(kTile), b.getIndexAttr(nTile)};
		Value bTile = b.create<tensor::ExtractSliceOp>(
			loc, bTileType, rhs, bOffsets, bSizes, aStrides);

		StringRef symbol;
		if (k == 0) {
			symbol = kMatmulAccFirst;
		} else if (k == kTiles - 1) {
			symbol = kMatmulAccLast;
		} else {
			symbol = kMatmulAccMid;
		}
		auto launch = b.create<NPUSchedule::UKernelLaunchOp>(
			loc, tileType, b.getStringAttr(symbol), ValueRange{aTile, bTile});
		// Only the LAST variant materializes a tensor we read; the first/mid
		// results are unused. They still have to be valid SSA per the op
		// definition (results = (outs AnyTensor)); leave them as-is and rely on
		// downstream lowering / DCE to drop them.
		if (k == kTiles - 1) {
			lastResult = launch.getResult();
		}
	}
	return lastResult;
}

// Tile a matmul op (NPUKernel::MatmulOp or matmul-symboled UKernelGenericOp)
// into an scf.for nest over (M, N) tiles, with the K dimension unrolled inside
// each (m, n) body via emitKAccumulatorChain.
//
// `lhs` is M×K, `rhs` is K×N, result is M×N. All shapes must be statically
// known and divide the chosen tile shape evenly.
static LogicalResult tileMatmulShared(Operation *op, Value lhs, Value rhs,
	Type resultType, PatternRewriter &rewriter, ArrayRef<int64_t> tileShape) {
	auto lhsShape = staticRank2Shape(lhs.getType());
	auto rhsShape = staticRank2Shape(rhs.getType());
	auto resTy = llvm::dyn_cast<RankedTensorType>(resultType);
	if (!lhsShape || !rhsShape || !resTy || resTy.getRank() != 2 ||
		!resTy.hasStaticShape()) {
		return failure();
	}
	int64_t M = (*lhsShape)[0];
	int64_t K = (*lhsShape)[1];
	int64_t Kr = (*rhsShape)[0];
	int64_t N = (*rhsShape)[1];
	if (K != Kr || resTy.getShape()[0] != M || resTy.getShape()[1] != N) {
		return failure();
	}
	int64_t mTile = tileShape.size() > 0 ? tileShape[0] : 32;
	int64_t nTile = tileShape.size() > 1 ? tileShape[1] : 32;
	// Manifest doesn't carry a separate K-tile dimension today; use mTile
	// since that's the kernel's accumulator stripe.
	int64_t kTile = mTile;
	if (M % mTile || N % nTile || K % kTile) {
		return failure();
	}
	// If K fits in a single tile, the 1:1 ConvertNPUKernelToSchedule path
	// handles the matmul. Tile only when there is a real K-loop to emit.
	int64_t kTiles = K / kTile;
	if (kTiles <= 1) {
		return failure();
	}

	Location loc = op->getLoc();
	Value init = emitEmptyOutput(
		rewriter, loc, resTy.getShape(), resTy.getElementType());
	Value result = emitTileLoopNest(rewriter, loc, M, N, mTile, nTile, init,
		[&](OpBuilder &b, Location iloc, Value ivM, Value ivN,
			Value acc) -> Value {
			Value lastTile = emitKAccumulatorChain(b, iloc, kTiles, mTile,
				nTile, kTile, lhs, rhs, ivM, ivN, resTy.getElementType());
			return insertTile2D(b, iloc, lastTile, acc, ivM, ivN, mTile, nTile);
		});
	rewriter.replaceOp(op, result);
	return success();
}

// Pattern: tile a 2D NPUKernel::MatmulOp with K-accumulator variants.
struct TileMatmulOpPattern : public OpRewritePattern<NPUKernel::MatmulOp> {
	TileMatmulOpPattern(MLIRContext *ctx, const TileSpecMap *specs)
		: OpRewritePattern(ctx), specs(specs) {}

	LogicalResult matchAndRewrite(
		NPUKernel::MatmulOp op, PatternRewriter &rewriter) const override {
		// MatmulOp has no symbol; resolve a default tile shape via the
		// "npu_uk_matmul" prefix lookup (or fall back to {32, 32}).
		auto tileShape = resolveTileShape(*specs, "npu_uk_matmul");
		return tileMatmulShared(op.getOperation(), op.getLhs(), op.getRhs(),
			op.getResult().getType(), rewriter, tileShape);
	}

	const TileSpecMap *specs;
};

// Pattern: tile a matmul-symboled NPUKernel::UKernelGenericOp.
struct TileMatmulUKernelPattern
	: public OpRewritePattern<NPUKernel::UKernelGenericOp> {
	TileMatmulUKernelPattern(MLIRContext *ctx, const TileSpecMap *specs)
		: OpRewritePattern(ctx), specs(specs) {}

	LogicalResult matchAndRewrite(NPUKernel::UKernelGenericOp op,
		PatternRewriter &rewriter) const override {
		if (!op.getSymbol().starts_with("npu_uk_matmul")) {
			return rewriter.notifyMatchFailure(op, "not a matmul ukernel");
		}
		// The matmul-as-ukernel form takes (lhs, rhs) inputs in that order.
		if (op.getInputs().size() != 2) {
			return rewriter.notifyMatchFailure(
				op, "matmul ukernel needs 2 inputs");
		}
		auto tileShape = resolveTileShape(*specs, op.getSymbol());
		return tileMatmulShared(op.getOperation(), op.getInputs()[0],
			op.getInputs()[1], op.getResult().getType(), rewriter, tileShape);
	}

	const TileSpecMap *specs;
};

// Symbol names for the flash-attention K/V-tile chain.
constexpr StringLiteral kAttentionAccFirst = "npu_uk_attention_acc_first";
constexpr StringLiteral kAttentionAccMid = "npu_uk_attention_acc_mid";
constexpr StringLiteral kAttentionAccLast = "npu_uk_attention_acc_last";

// Validate a tensor type is a static rank-2 or rank-3 shape and return its
// trailing two dims (seq, head). For 3D inputs the leading dim is treated as a
// batch dim. Returns std::nullopt for unsupported shapes.
struct AttentionTensorShape {
	int64_t batch; // 1 if input was rank-2
	int64_t seq;
	int64_t head;
};

static std::optional<AttentionTensorShape> attentionShape(Type t) {
	auto rt = llvm::dyn_cast<RankedTensorType>(t);
	if (!rt || !rt.hasStaticShape())
		return std::nullopt;
	if (rt.getRank() == 2) {
		return AttentionTensorShape{1, rt.getShape()[0], rt.getShape()[1]};
	}
	if (rt.getRank() == 3) {
		return AttentionTensorShape{
			rt.getShape()[0], rt.getShape()[1], rt.getShape()[2]};
	}
	return std::nullopt;
}

// Slice an attention operand along its sequence dim. Handles both rank-2
// (seq, head) and rank-3 (batch, seq, head) inputs. ivBatch is unused for
// rank-2; pass any valid Value.
static Value sliceSeqTile(OpBuilder &b, Location loc, Value source,
	Value ivBatch, Value ivSeq, int64_t seqTile) {
	auto sourceTy = llvm::cast<RankedTensorType>(source.getType());
	if (sourceTy.getRank() == 2) {
		SmallVector<OpFoldResult, 2> offsets = {ivSeq, b.getIndexAttr(0)};
		SmallVector<OpFoldResult, 2> sizes = {
			b.getIndexAttr(seqTile), b.getIndexAttr(sourceTy.getShape()[1])};
		SmallVector<OpFoldResult, 2> strides = {
			b.getIndexAttr(1), b.getIndexAttr(1)};
		auto tileTy = RankedTensorType::get(
			{seqTile, sourceTy.getShape()[1]}, sourceTy.getElementType());
		return b.create<tensor::ExtractSliceOp>(
			loc, tileTy, source, offsets, sizes, strides);
	}
	SmallVector<OpFoldResult, 3> offsets = {ivBatch, ivSeq, b.getIndexAttr(0)};
	SmallVector<OpFoldResult, 3> sizes = {b.getIndexAttr(1),
		b.getIndexAttr(seqTile), b.getIndexAttr(sourceTy.getShape()[2])};
	SmallVector<OpFoldResult, 3> strides = {
		b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1)};
	auto tileTy = RankedTensorType::get(
		{1, seqTile, sourceTy.getShape()[2]}, sourceTy.getElementType());
	return b.create<tensor::ExtractSliceOp>(
		loc, tileTy, source, offsets, sizes, strides);
}

// Insert an attention output tile back into the running result tensor.
static Value insertSeqTile(OpBuilder &b, Location loc, Value tile, Value dest,
	Value ivBatch, Value ivQ, int64_t qTile) {
	auto destTy = llvm::cast<RankedTensorType>(dest.getType());
	if (destTy.getRank() == 2) {
		SmallVector<OpFoldResult, 2> offsets = {ivQ, b.getIndexAttr(0)};
		SmallVector<OpFoldResult, 2> sizes = {
			b.getIndexAttr(qTile), b.getIndexAttr(destTy.getShape()[1])};
		SmallVector<OpFoldResult, 2> strides = {
			b.getIndexAttr(1), b.getIndexAttr(1)};
		return b.create<tensor::InsertSliceOp>(
			loc, tile, dest, offsets, sizes, strides);
	}
	SmallVector<OpFoldResult, 3> offsets = {ivBatch, ivQ, b.getIndexAttr(0)};
	SmallVector<OpFoldResult, 3> sizes = {b.getIndexAttr(1),
		b.getIndexAttr(qTile), b.getIndexAttr(destTy.getShape()[2])};
	SmallVector<OpFoldResult, 3> strides = {
		b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(1)};
	return b.create<tensor::InsertSliceOp>(
		loc, tile, dest, offsets, sizes, strides);
}

// Emit one attention_acc_* launch per K/V tile, sharing implicit running
// state (max, denom, partial output) across invocations. Returns the result
// of the `_last` launch, which carries the final attention output for the
// (batch, q_block) pair.
static Value emitAttentionKVChain(OpBuilder &b, Location loc, int64_t kvTiles,
	int64_t qTile, int64_t kvTile, Value q, Value k, Value v, Value mask,
	Value ivBatch, Value ivQ, Type tileElementType,
	ArrayRef<int64_t> qTileShape) {
	auto qTileTy = sliceSeqTile(b, loc, q, ivBatch, ivQ, qTile);
	(void)qTileTy; // sliceSeqTile is invoked below per-iteration to keep IR
				   // local; this no-op silences an unused-variable warning.
	auto outTileType = RankedTensorType::get(qTileShape, tileElementType);
	Value lastResult;
	for (int64_t kv = 0; kv < kvTiles; ++kv) {
		auto kvOffset = b.create<arith::ConstantIndexOp>(loc, kv * kvTile);
		Value qTileVal = sliceSeqTile(b, loc, q, ivBatch, ivQ, qTile);
		Value kTile =
			sliceSeqTile(b, loc, k, ivBatch, kvOffset.getResult(), kvTile);
		Value vTile =
			sliceSeqTile(b, loc, v, ivBatch, kvOffset.getResult(), kvTile);
		SmallVector<Value, 4> launchInputs = {qTileVal, kTile, vTile};
		if (mask) {
			Value maskTile = sliceSeqTile(
				b, loc, mask, ivBatch, kvOffset.getResult(), kvTile);
			launchInputs.push_back(maskTile);
		}
		StringRef symbol;
		if (kv == 0) {
			symbol = kAttentionAccFirst;
		} else if (kv == kvTiles - 1) {
			symbol = kAttentionAccLast;
		} else {
			symbol = kAttentionAccMid;
		}
		auto launch = b.create<NPUSchedule::UKernelLaunchOp>(
			loc, outTileType, b.getStringAttr(symbol), launchInputs);
		if (kv == kvTiles - 1) {
			lastResult = launch.getResult();
		}
	}
	return lastResult;
}

// Pattern: tile a single-tile attention ukernel (any "npu_uk_attention" or
// "npu_uk_gemma_attention" symbol) into an scf.for nest over (batch, q-block)
// with the K/V dimension unrolled inside each body via emitAttentionKVChain.
struct TileAttentionUKernelPattern
	: public OpRewritePattern<NPUKernel::UKernelGenericOp> {
	TileAttentionUKernelPattern(MLIRContext *ctx, const TileSpecMap *specs)
		: OpRewritePattern(ctx), specs(specs) {}

	LogicalResult matchAndRewrite(NPUKernel::UKernelGenericOp op,
		PatternRewriter &rewriter) const override {
		StringRef sym = op.getSymbol();
		bool isAttention = sym.starts_with("npu_uk_attention") ||
			sym.starts_with("npu_uk_gemma_attention");
		// Don't recurse on already-tiled attention variants.
		if (!isAttention || sym.starts_with("npu_uk_attention_acc")) {
			return rewriter.notifyMatchFailure(op, "not an untiled attention");
		}
		auto inputs = op.getInputs();
		if (inputs.size() < 3) {
			return rewriter.notifyMatchFailure(op, "need at least Q, K, V");
		}
		Value q = inputs[0];
		Value k = inputs[1];
		Value v = inputs[2];
		Value mask = inputs.size() > 3 ? inputs[3] : Value{};

		auto qShape = attentionShape(q.getType());
		auto kShape = attentionShape(k.getType());
		auto vShape = attentionShape(v.getType());
		auto resultTy =
			llvm::dyn_cast<RankedTensorType>(op.getResult().getType());
		if (!qShape || !kShape || !vShape || !resultTy) {
			return rewriter.notifyMatchFailure(
				op, "non-static attention shape");
		}
		if (kShape->seq != vShape->seq) {
			return rewriter.notifyMatchFailure(op, "K and V seq mismatch");
		}
		auto tileShape = resolveTileShape(*specs, sym);
		int64_t qTile = tileShape.size() > 0 ? tileShape[0] : 32;
		int64_t kvTile = tileShape.size() > 1 ? tileShape[1] : 32;
		if (qShape->seq % qTile || kShape->seq % kvTile) {
			return rewriter.notifyMatchFailure(
				op, "shape not divisible by tile");
		}
		int64_t kvTiles = kShape->seq / kvTile;
		// If the K/V dimension fits in a single tile, fall back to the
		// existing 1:1 conversion of the original attention symbol — no need
		// to introduce running-state plumbing for a single launch.
		if (kvTiles <= 1) {
			return rewriter.notifyMatchFailure(
				op, "single-tile, no chain needed");
		}

		Location loc = op.getLoc();
		Value initOut = emitEmptyOutput(
			rewriter, loc, resultTy.getShape(), resultTy.getElementType());

		auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
		auto cQTile = rewriter.create<arith::ConstantIndexOp>(loc, qTile);
		auto cQBound =
			rewriter.create<arith::ConstantIndexOp>(loc, qShape->seq);

		auto runBatch = [&](OpBuilder &bb, Location bloc, Value ivBatch,
							Value batchAcc) -> Value {
			auto qLoop = bb.create<scf::ForOp>(bloc, c0, cQBound, cQTile,
				ValueRange{batchAcc},
				[&](OpBuilder &qb, Location qloc, Value ivQ,
					ValueRange qIterArgs) {
					SmallVector<int64_t, 3> qTileResultShape;
					if (resultTy.getRank() == 2) {
						qTileResultShape = {qTile, resultTy.getShape()[1]};
					} else {
						qTileResultShape = {1, qTile, resultTy.getShape()[2]};
					}
					Value lastTile = emitAttentionKVChain(qb, qloc, kvTiles,
						qTile, kvTile, q, k, v, mask, ivBatch, ivQ,
						resultTy.getElementType(), qTileResultShape);
					Value updated = insertSeqTile(qb, qloc, lastTile,
						qIterArgs.front(), ivBatch, ivQ, qTile);
					qb.create<scf::YieldOp>(qloc, ValueRange{updated});
				});
			return qLoop.getResult(0);
		};

		Value finalResult;
		if (resultTy.getRank() == 2) {
			finalResult = runBatch(rewriter, loc, c0.getResult(), initOut);
		} else {
			auto cBatchBound =
				rewriter.create<arith::ConstantIndexOp>(loc, qShape->batch);
			auto cBatchStep = rewriter.create<arith::ConstantIndexOp>(loc, 1);
			auto batchLoop = rewriter.create<scf::ForOp>(loc, c0, cBatchBound,
				cBatchStep, ValueRange{initOut},
				[&](OpBuilder &bb, Location bloc, Value ivBatch,
					ValueRange batchIterArgs) {
					Value updated =
						runBatch(bb, bloc, ivBatch, batchIterArgs.front());
					bb.create<scf::YieldOp>(bloc, ValueRange{updated});
				});
			finalResult = batchLoop.getResult(0);
		}
		rewriter.replaceOp(op, finalResult);
		return success();
	}

	const TileSpecMap *specs;
};

struct TileNPUKernelToSchedulePass
	: public PassWrapper<TileNPUKernelToSchedulePass, OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileNPUKernelToSchedulePass)

	TileNPUKernelToSchedulePass() = default;
	TileNPUKernelToSchedulePass(const TileNPUKernelToSchedulePass &other)
		: PassWrapper(other) {}

	StringRef getArgument() const final {
		return "tile-npu-kernel-to-schedule";
	}
	StringRef getDescription() const final {
		return "Wrap NPUKernel::UKernelGenericOp/MatmulOp in scf.for tile "
			   "loops "
			   "at manifest tile_shape granularity and emit npu_schedule ops";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUKernel::NPUKernelDialect>();
		registry.insert<NPUSchedule::NPUScheduleDialect>();
		registry.insert<scf::SCFDialect>();
		registry.insert<tensor::TensorDialect>();
		registry.insert<arith::ArithDialect>();
	}

	Option<std::string> manifestPath{*this, "kernel-manifest",
		llvm::cl::desc("Path to SaturnNPU kernel manifest.json"),
		llvm::cl::init("")};

	void runOnOperation() override {
		TileSpecMap specs;
		loadManifestTileShapes(manifestPath, specs);

		RewritePatternSet patterns(&getContext());
		patterns.add<TileBinaryElementwisePattern>(&getContext(), &specs);
		patterns.add<TileMatmulOpPattern>(&getContext(), &specs);
		patterns.add<TileMatmulUKernelPattern>(&getContext(), &specs);
		patterns.add<TileAttentionUKernelPattern>(&getContext(), &specs);

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
		}
	}
};

} // namespace

std::unique_ptr<Pass> createTileNPUKernelToSchedulePass() {
	return std::make_unique<TileNPUKernelToSchedulePass>();
}

void registerTileNPUKernelToSchedulePass() {
	PassRegistration<TileNPUKernelToSchedulePass>();
}

} // namespace mlir::iree_compiler::NPU
