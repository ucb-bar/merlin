/// FoldFP8ScalesAroundContractions — eliminate QDQ chains around matmuls.
///
/// This standalone pass operates on standard linalg/arith/tensor ops and does
/// NOT produce any NPU dialect ops.  It is designed for preparing the IR for
/// kernel extraction by human kernel writers.
///
/// The pass recognises two patterns:
///
///   1. **Weight dequant chain feeding matmul:**
///      `extf(f8E4M3FN → f32) → truncf(f32 → bf16) → matmul input`
///      Rewires the matmul input to use the original f8E4M3FN tensor directly.
///
///   2. **Activation quantize-dequant (QDQ) chain feeding matmul:**
///      `truncf(f32 → f8E4M3FN) → extf(f8E4M3FN → f32) → truncf(f32 → bf16) → matmul`
///      Removes the dequant (extf + truncf) and feeds the f8E4M3FN tensor
///      from the quantize step directly to the matmul.
///
/// After rewiring inputs to fp8, the pass also changes the matmul accumulator
/// from f32 to bf16, since the NPU MXU accumulates in bf16.
///
/// Mathematical justification (zero_point = 0, per-tensor scale):
///   matmul(dequant(w), dequant(x)) = matmul(w_fp8 * s_w, x_fp8 * s_x)
///                                  = matmul(w_fp8, x_fp8) * (s_w * s_x)
/// The combined scale is a compile-time constant that can be applied after the
/// matmul as a single broadcast multiply.

#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::NPU {
namespace {

/// Check if a Type is f8E4M3FN.
static bool isF8E4M3FN(Type t) {
  return t.isF8E4M3FN();
}

/// Check if a Type is bf16.
static bool isBF16(Type t) {
  return t.isBF16();
}

/// Check if a Type is f32.
static bool isF32(Type t) {
  return t.isF32();
}

/// Get the element type of a tensor value.
static Type getElementType(Value v) {
  if (auto shaped = dyn_cast<ShapedType>(v.getType()))
    return shaped.getElementType();
  return {};
}

/// Trace a value backward through type conversions, reshapes, transposes,
/// and scale multiplies to find the deepest fp8 source.
///
/// Traces through:
///   - linalg.generic with extf/truncf body (type conversion chains)
///   - linalg.generic with mulf body + constant input (scale multiply)
///   - linalg.generic with clamp body (clamp ops in quantize path)
///   - linalg.generic that is a simple copy/broadcast (identity yield)
///   - tensor.expand_shape / tensor.collapse_shape (reshapes)
///   - linalg.transpose
///
/// Returns the deepest fp8 source found, or the original value if no fp8
/// source exists.
static Value traceToFP8Source(Value v) {
  Value current = v;
  for (int depth = 0; depth < 24; ++depth) {
    auto defOp = current.getDefiningOp();
    if (!defOp)
      break;

    // If current value is already fp8, return it
    if (isF8E4M3FN(getElementType(current)))
      return current;

    // Trace through tensor.expand_shape
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
      current = expandOp.getSrc();
      continue;
    }

    // Trace through tensor.collapse_shape
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(defOp)) {
      current = collapseOp.getSrc();
      continue;
    }

    // Trace through linalg.transpose
    if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp)) {
      current = transposeOp.getInput();
      continue;
    }

    // Check for linalg.generic
    auto generic = dyn_cast<linalg::GenericOp>(defOp);
    if (!generic)
      break;

    // Count body ops (excluding yield)
    Block &body = generic.getRegion().front();
    SmallVector<Operation *> bodyOps;
    for (Operation &op : body) {
      if (!isa<linalg::YieldOp>(op))
        bodyOps.push_back(&op);
    }

    // --- Single-input generics (type conversion, clamp) ---
    if (generic.getNumDpsInputs() == 1 && generic.getNumDpsInits() == 1 &&
        bodyOps.size() == 1) {
      Operation *bodyOp = bodyOps[0];

      // Type conversion: extf, truncf
      if (isa<arith::ExtFOp, arith::TruncFOp>(bodyOp)) {
        Value input = generic.getDpsInputOperand(0)->get();
        if (isF8E4M3FN(getElementType(input)))
          return input;
        current = input;
        continue;
      }

      // Clamp (part of quantize path): maximumf, minimumf
      if (isa<arith::MaximumFOp, arith::MinimumFOp>(bodyOp)) {
        current = generic.getDpsInputOperand(0)->get();
        continue;
      }

      // Identity copy / broadcast (body just yields the input arg)
      if (auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator())) {
        // The single body op might be absent if body is just yield
      }
    }

    // --- Single-input generic with identity body (broadcast/copy) ---
    if (generic.getNumDpsInputs() == 1 && generic.getNumDpsInits() == 1 &&
        bodyOps.empty()) {
      // Body is just `linalg.yield %arg0` — identity/broadcast
      current = generic.getDpsInputOperand(0)->get();
      continue;
    }

    // --- Two-input generic with mulf body (scale multiply) ---
    // Pattern: linalg.generic {mulf} ins(%data, %scale) or ins(%scale, %data)
    // Skip through the scale multiply to trace the data operand.
    if (generic.getNumDpsInputs() == 2 && generic.getNumDpsInits() == 1 &&
        bodyOps.size() == 1 && isa<arith::MulFOp>(bodyOps[0])) {
      Value in0 = generic.getDpsInputOperand(0)->get();
      Value in1 = generic.getDpsInputOperand(1)->get();
      auto in0Type = cast<RankedTensorType>(in0.getType());
      auto in1Type = cast<RankedTensorType>(in1.getType());

      // One of the inputs should be a "smaller" tensor (scale) or a constant.
      // Heuristic: trace through the input with MORE elements (the data tensor).
      int64_t in0Size = in0Type.getNumElements();
      int64_t in1Size = in1Type.getNumElements();

      // If one input is a scalar or much smaller, it's the scale → trace the other
      if (in0Size > in1Size) {
        current = in0;
      } else if (in1Size > in0Size) {
        current = in1;
      } else {
        // Same size — check if one is defined by a constant
        bool in0Const = in0.getDefiningOp() &&
                        isa<arith::ConstantOp>(in0.getDefiningOp());
        bool in1Const = in1.getDefiningOp() &&
                        isa<arith::ConstantOp>(in1.getDefiningOp());
        if (in1Const && !in0Const)
          current = in0;
        else if (in0Const && !in1Const)
          current = in1;
        else
          break; // Can't determine which is the scale
      }
      continue;
    }

    // --- Clamp pattern with 2 ops (cmpf + select pairs) ---
    if (generic.getNumDpsInputs() == 1 && generic.getNumDpsInits() == 1 &&
        bodyOps.size() == 2) {
      bool hasCompare =
          llvm::any_of(bodyOps, [](Operation *op) {
            return isa<arith::CmpFOp>(op);
          });
      bool hasSelect =
          llvm::any_of(bodyOps, [](Operation *op) {
            return isa<arith::SelectOp>(op);
          });
      if (hasCompare && hasSelect) {
        current = generic.getDpsInputOperand(0)->get();
        continue;
      }
    }

    // Unrecognized pattern — stop tracing
    break;
  }
  return current;
}

/// Stored shape-op descriptor (extracted before any rewriting).
struct ShapeOpInfo {
  enum Kind { Transpose, Expand, Collapse, Broadcast };
  Kind kind;
  SmallVector<int64_t> resultShape;
  // For transpose:
  SmallVector<int64_t> permutation;
  // For expand/collapse:
  SmallVector<SmallVector<int64_t>> reassociation;
};

/// Walk backward from matmulInput toward fp8Source, recording shape-changing
/// ops.  Skip type-conversion generics (extf/truncf) and scale multiplies.
/// Returns true if we successfully traced back to fp8Source's shape.
static bool collectShapeOps(Value fp8Source, Value matmulInput,
                            SmallVectorImpl<ShapeOpInfo> &ops) {
  auto fp8Shape = cast<RankedTensorType>(fp8Source.getType()).getShape();
  Value cursor = matmulInput;

  for (int depth = 0; depth < 24; ++depth) {
    auto curShape = cast<RankedTensorType>(cursor.getType()).getShape();
    if (curShape == fp8Shape)
      return true;

    auto *def = cursor.getDefiningOp();
    if (!def)
      return false;

    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(def)) {
      ShapeOpInfo info;
      info.kind = ShapeOpInfo::Expand;
      info.resultShape.assign(curShape.begin(), curShape.end());
      for (auto &indices : expand.getReassociationIndices()) {
        SmallVector<int64_t> idx(indices.begin(), indices.end());
        info.reassociation.push_back(std::move(idx));
      }
      ops.push_back(std::move(info));
      cursor = expand.getSrc();
      continue;
    }

    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(def)) {
      ShapeOpInfo info;
      info.kind = ShapeOpInfo::Collapse;
      info.resultShape.assign(curShape.begin(), curShape.end());
      for (auto &indices : collapse.getReassociationIndices()) {
        SmallVector<int64_t> idx(indices.begin(), indices.end());
        info.reassociation.push_back(std::move(idx));
      }
      ops.push_back(std::move(info));
      cursor = collapse.getSrc();
      continue;
    }

    if (auto transpose = dyn_cast<linalg::TransposeOp>(def)) {
      ShapeOpInfo info;
      info.kind = ShapeOpInfo::Transpose;
      info.resultShape.assign(curShape.begin(), curShape.end());
      auto perm = transpose.getPermutation();
      info.permutation.assign(perm.begin(), perm.end());
      ops.push_back(std::move(info));
      cursor = transpose.getInput();
      continue;
    }

    // Skip through linalg.generic type conversions
    if (auto generic = dyn_cast<linalg::GenericOp>(def)) {
      if (generic.getNumDpsInputs() == 1) {
        // Check for identity/broadcast (different indexing maps = broadcast)
        auto maps = generic.getIndexingMapsArray();
        if (maps[0] != maps[1]) {
          // Broadcast — record it as a shape op
          ShapeOpInfo info;
          info.kind = ShapeOpInfo::Broadcast;
          info.resultShape.assign(curShape.begin(), curShape.end());
          ops.push_back(std::move(info));
        }
        cursor = generic.getDpsInputOperand(0)->get();
        continue;
      }
      if (generic.getNumDpsInputs() == 2) {
        // Scale multiply — trace through larger input
        auto in0 = cast<RankedTensorType>(
            generic.getDpsInputOperand(0)->get().getType());
        auto in1 = cast<RankedTensorType>(
            generic.getDpsInputOperand(1)->get().getType());
        cursor = (in0.getNumElements() >= in1.getNumElements())
                     ? generic.getDpsInputOperand(0)->get()
                     : generic.getDpsInputOperand(1)->get();
        continue;
      }
    }

    return false;
  }
  return false;
}

/// Apply the collected shape ops to fp8Source, creating new ops with fp8 types.
static Value replayShapeOps(Value fp8Source,
                            ArrayRef<ShapeOpInfo> shapeOps,
                            PatternRewriter &rewriter, Location loc) {
  Value current = fp8Source;
  Type fp8Elem = cast<RankedTensorType>(fp8Source.getType()).getElementType();

  // Replay in reverse (we collected backward, replay forward)
  for (auto it = shapeOps.rbegin(); it != shapeOps.rend(); ++it) {
    const auto &info = *it;
    auto newResultType = RankedTensorType::get(info.resultShape, fp8Elem);

    switch (info.kind) {
    case ShapeOpInfo::Transpose: {
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, info.resultShape, fp8Elem);
      auto transposeOp = rewriter.create<linalg::TransposeOp>(
          loc, current, init, info.permutation);
      current = transposeOp.getResult()[0];
      break;
    }
    case ShapeOpInfo::Expand: {
      SmallVector<ReassociationIndices> reassoc;
      for (auto &r : info.reassociation)
        reassoc.push_back(ReassociationIndices(r.begin(), r.end()));
      current = rewriter.create<tensor::ExpandShapeOp>(
          loc, newResultType, current, reassoc);
      break;
    }
    case ShapeOpInfo::Collapse: {
      SmallVector<ReassociationIndices> reassoc;
      for (auto &r : info.reassociation)
        reassoc.push_back(ReassociationIndices(r.begin(), r.end()));
      current = rewriter.create<tensor::CollapseShapeOp>(
          loc, newResultType, current, reassoc);
      break;
    }
    case ShapeOpInfo::Broadcast: {
      // Create a linalg.generic identity broadcast with fp8 types
      int64_t srcRank = cast<RankedTensorType>(current.getType()).getRank();
      int64_t dstRank = newResultType.getRank();
      // Build identity maps for broadcast
      SmallVector<AffineExpr> srcExprs, dstExprs;
      MLIRContext *ctx = rewriter.getContext();
      for (int64_t i = 0; i < dstRank; ++i)
        dstExprs.push_back(getAffineDimExpr(i, ctx));
      // For broadcast, src map drops the broadcast dims
      // Simple heuristic: if src has fewer dims, use trailing dims
      int64_t offset = dstRank - srcRank;
      for (int64_t i = 0; i < srcRank; ++i)
        srcExprs.push_back(getAffineDimExpr(i + offset, ctx));

      auto srcMap = AffineMap::get(dstRank, 0, srcExprs, ctx);
      auto dstMap = AffineMap::get(dstRank, 0, dstExprs, ctx);
      SmallVector<utils::IteratorType> iterTypes(
          dstRank, utils::IteratorType::parallel);

      Value init = rewriter.create<tensor::EmptyOp>(
          loc, info.resultShape, fp8Elem);
      auto genericOp = rewriter.create<linalg::GenericOp>(
          loc, newResultType, ValueRange{current}, ValueRange{init},
          ArrayRef<AffineMap>{srcMap, dstMap}, iterTypes,
          [](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
          });
      current = genericOp.getResult(0);
      break;
    }
    }
  }
  return current;
}

/// Adapt fp8Source to match the matmul input's expected shape.
/// If shapes already match, returns fp8Source directly.
/// Otherwise, collects shape-changing ops from the chain and replays them.
static Value adaptFP8ToShape(Value fp8Source, Value matmulInput,
                             PatternRewriter &rewriter, Location loc) {
  auto targetShape = cast<RankedTensorType>(matmulInput.getType()).getShape();
  auto fp8Shape = cast<RankedTensorType>(fp8Source.getType()).getShape();
  if (fp8Shape == targetShape)
    return fp8Source;

  SmallVector<ShapeOpInfo> shapeOps;
  if (!collectShapeOps(fp8Source, matmulInput, shapeOps))
    return nullptr;

  return replayShapeOps(fp8Source, shapeOps, rewriter, loc);
}

/// Pattern: Rewire batch_matmul inputs from bf16 to fp8 sources.
///
/// Matches:  linalg.batch_matmul ins(%lhs_bf16, %rhs_bf16) outs(%acc_f32)
/// Where lhs_bf16 and/or rhs_bf16 trace back to f8E4M3FN tensors through
/// extf/truncf chains.
///
/// Produces a linalg.generic with the SAME indexing maps as the original
/// batch_matmul, but with fp8 input types and bf16 accumulator.
/// Body: extf(fp8→bf16), mulf, addf.
struct FoldFP8IntoBatchMatmul
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();

    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value init = op.getDpsInitOperand(0)->get();

    // Trace both inputs to fp8 sources
    Value lhsFP8 = traceToFP8Source(lhs);
    Value rhsFP8 = traceToFP8Source(rhs);

    bool lhsIsFP8 = isF8E4M3FN(getElementType(lhsFP8));
    bool rhsIsFP8 = isF8E4M3FN(getElementType(rhsFP8));

    // At least one input must trace to fp8
    if (!lhsIsFP8 && !rhsIsFP8)
      return failure();

    auto loc = op.getLoc();

    // Adapt fp8 sources to match matmul input shapes
    Value newLhs = lhsIsFP8 ? adaptFP8ToShape(lhsFP8, lhs, rewriter, loc)
                             : lhs;
    Value newRhs = rhsIsFP8 ? adaptFP8ToShape(rhsFP8, rhs, rewriter, loc)
                             : rhs;

    // Fall back to original if adaptation failed
    if (!newLhs) newLhs = lhs;
    if (!newRhs) newRhs = rhs;

    // At least one input must actually be fp8
    if (!isF8E4M3FN(getElementType(newLhs)) &&
        !isF8E4M3FN(getElementType(newRhs)))
      return failure();

    // Change accumulator from f32 to bf16
    auto initType = cast<RankedTensorType>(init.getType());
    Value newInit;
    if (isF32(initType.getElementType())) {
      auto bf16InitType = RankedTensorType::get(
          initType.getShape(), rewriter.getBF16Type());
      auto zeroAttr = rewriter.getFloatAttr(rewriter.getBF16Type(), 0.0);
      Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, bf16InitType.getShape(), rewriter.getBF16Type());
      newInit = rewriter.create<linalg::FillOp>(loc, zero, emptyTensor)
                    .getResult(0);
    } else {
      newInit = init;
    }

    // Use the ORIGINAL batch_matmul's indexing maps (handles any rank correctly)
    auto indexingMaps = op.getIndexingMapsArray();
    auto iteratorTypes = op.getIteratorTypesArray();
    auto newInitType = cast<RankedTensorType>(newInit.getType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, newInitType, ValueRange{newLhs, newRhs}, ValueRange{newInit},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value lhsElem = args[0];
          Value rhsElem = args[1];
          Value accElem = args[2];
          Type accType = accElem.getType();
          Value lhsExt = b.create<arith::ExtFOp>(loc, accType, lhsElem);
          Value rhsExt = b.create<arith::ExtFOp>(loc, accType, rhsElem);
          Value prod = b.create<arith::MulFOp>(loc, lhsExt, rhsExt);
          Value sum = b.create<arith::AddFOp>(loc, accElem, prod);
          b.create<linalg::YieldOp>(loc, sum);
        });

    Value result = genericOp.getResult(0);

    // If the original result was f32, cast bf16→f32 to maintain downstream types
    if (isF32(initType.getElementType())) {
      MLIRContext *ctx = rewriter.getContext();
      auto f32ResultType = RankedTensorType::get(
          initType.getShape(), rewriter.getF32Type());
      SmallVector<utils::IteratorType> castIterTypes(
          initType.getRank(), utils::IteratorType::parallel);
      SmallVector<AffineExpr> dimExprs;
      for (int64_t i = 0; i < initType.getRank(); ++i)
        dimExprs.push_back(getAffineDimExpr(i, ctx));
      auto identityMap = AffineMap::get(
          initType.getRank(), 0, dimExprs, ctx);

      Value emptyF32 = rewriter.create<tensor::EmptyOp>(
          loc, initType.getShape(), rewriter.getF32Type());
      auto castOp = rewriter.create<linalg::GenericOp>(
          loc, f32ResultType, ValueRange{result}, ValueRange{emptyF32},
          ArrayRef<AffineMap>{identityMap, identityMap}, castIterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value ext = b.create<arith::ExtFOp>(
                loc, rewriter.getF32Type(), args[0]);
            b.create<linalg::YieldOp>(loc, ext);
          });
      result = castOp.getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Same pattern for linalg.matmul (non-batched).
struct FoldFP8IntoMatmul
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();

    Value lhs = op.getDpsInputOperand(0)->get();
    Value rhs = op.getDpsInputOperand(1)->get();
    Value init = op.getDpsInitOperand(0)->get();

    Value lhsFP8 = traceToFP8Source(lhs);
    Value rhsFP8 = traceToFP8Source(rhs);

    bool lhsIsFP8 = isF8E4M3FN(getElementType(lhsFP8));
    bool rhsIsFP8 = isF8E4M3FN(getElementType(rhsFP8));

    if (!lhsIsFP8 && !rhsIsFP8)
      return failure();

    auto loc = op.getLoc();

    Value newLhs = lhsIsFP8 ? adaptFP8ToShape(lhsFP8, lhs, rewriter, loc)
                             : lhs;
    Value newRhs = rhsIsFP8 ? adaptFP8ToShape(rhsFP8, rhs, rewriter, loc)
                             : rhs;

    if (!newLhs) newLhs = lhs;
    if (!newRhs) newRhs = rhs;

    if (!isF8E4M3FN(getElementType(newLhs)) &&
        !isF8E4M3FN(getElementType(newRhs)))
      return failure();

    auto initType = cast<RankedTensorType>(init.getType());
    Value newInit;
    if (isF32(initType.getElementType())) {
      auto zeroAttr = rewriter.getFloatAttr(rewriter.getBF16Type(), 0.0);
      Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, initType.getShape(), rewriter.getBF16Type());
      newInit = rewriter.create<linalg::FillOp>(loc, zero, emptyTensor)
                    .getResult(0);
    } else {
      newInit = init;
    }

    auto indexingMaps = op.getIndexingMapsArray();
    auto iteratorTypes = op.getIteratorTypesArray();
    auto newInitType = cast<RankedTensorType>(newInit.getType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, newInitType, ValueRange{newLhs, newRhs}, ValueRange{newInit},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Type accType = args[2].getType();
          Value lhsExt = b.create<arith::ExtFOp>(loc, accType, args[0]);
          Value rhsExt = b.create<arith::ExtFOp>(loc, accType, args[1]);
          Value prod = b.create<arith::MulFOp>(loc, lhsExt, rhsExt);
          Value sum = b.create<arith::AddFOp>(loc, args[2], prod);
          b.create<linalg::YieldOp>(loc, sum);
        });

    Value result = genericOp.getResult(0);
    if (isF32(initType.getElementType())) {
      MLIRContext *ctx = rewriter.getContext();
      auto f32ResultType = RankedTensorType::get(
          initType.getShape(), rewriter.getF32Type());
      SmallVector<AffineExpr> dimExprs;
      for (int64_t i = 0; i < initType.getRank(); ++i)
        dimExprs.push_back(getAffineDimExpr(i, ctx));
      auto identityMap = AffineMap::get(
          initType.getRank(), 0, dimExprs, ctx);
      SmallVector<utils::IteratorType> castIterTypes(
          initType.getRank(), utils::IteratorType::parallel);
      Value emptyF32 = rewriter.create<tensor::EmptyOp>(
          loc, initType.getShape(), rewriter.getF32Type());
      auto castOp = rewriter.create<linalg::GenericOp>(
          loc, f32ResultType, ValueRange{result}, ValueRange{emptyF32},
          ArrayRef<AffineMap>{identityMap, identityMap}, castIterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value ext = b.create<arith::ExtFOp>(
                loc, rewriter.getF32Type(), args[0]);
            b.create<linalg::YieldOp>(loc, ext);
          });
      result = castOp.getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Pass definition.
struct FoldFP8ScalesAroundContractionsPass
    : public PassWrapper<FoldFP8ScalesAroundContractionsPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FoldFP8ScalesAroundContractionsPass)

  StringRef getArgument() const override {
    return "fold-fp8-scales-around-contractions";
  }

  StringRef getDescription() const override {
    return "Fold FP8 QDQ chains into contraction ops, rewiring matmul inputs "
           "from bf16 to f8E4M3FN and changing accumulators from f32 to bf16. "
           "Produces standard linalg ops (no NPU dialect).";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FoldFP8IntoBatchMatmul>(ctx);
    patterns.add<FoldFP8IntoMatmul>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFoldFP8ScalesAroundContractionsPass() {
  return std::make_unique<FoldFP8ScalesAroundContractionsPass>();
}

void registerFoldFP8ScalesAroundContractionsPass() {
  PassRegistration<FoldFP8ScalesAroundContractionsPass>();
}

} // namespace mlir::iree_compiler::NPU
