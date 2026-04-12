/// FuseF32IntermediateConversions — collapse f32 intermediates in fp8↔bf16
/// type conversion chains.
///
/// The PyTorch export produces chains like:
///   bf16 → extf → f32 → [clamp] → truncf → f8E4M3FN   (requantization)
///   f8E4M3FN → extf → f32 → truncf → bf16              (dequantization)
///
/// MLIR supports direct truncf/extf between bf16 and f8E4M3FN, so the f32
/// intermediates are unnecessary. This pass fuses them:
///
///   Pattern A: generic{extf fp8→f32} → generic{truncf f32→bf16}
///              ⟹ generic{extf fp8→bf16}
///
///   Pattern B: generic{extf bf16→f32} → generic{truncf f32→fp8}
///              ⟹ generic{truncf bf16→fp8}
///
///   Pattern C: generic{extf bf16→f32} → generic{clamp f32} → generic{truncf f32→fp8}
///              ⟹ generic{truncf bf16→fp8}  (clamp is safe: bf16 max < fp8 max)
///
/// This pass operates on standard linalg/arith ops only.

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

/// Check if a linalg.generic is a single-op type conversion (extf or truncf).
/// Returns the body op if so, nullptr otherwise.
static Operation *getSingleConversionOp(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return nullptr;

  Block &body = generic.getRegion().front();
  Operation *singleOp = nullptr;
  for (Operation &op : body) {
    if (isa<linalg::YieldOp>(op))
      continue;
    if (singleOp)
      return nullptr; // more than one non-yield op
    singleOp = &op;
  }

  if (!singleOp || !isa<arith::ExtFOp, arith::TruncFOp>(singleOp))
    return nullptr;

  return singleOp;
}

/// Check if a linalg.generic is a clamp operation (cmpf+select pairs or
/// maximumf/minimumf).
static bool isClampGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;

  Block &body = generic.getRegion().front();
  bool hasCmp = false, hasSelect = false;
  bool hasMax = false, hasMin = false;
  for (Operation &op : body) {
    if (isa<linalg::YieldOp>(op))
      continue;
    if (isa<arith::CmpFOp>(op))
      hasCmp = true;
    if (isa<arith::SelectOp>(op))
      hasSelect = true;
    if (isa<arith::MaximumFOp, arith::MaxNumFOp>(op))
      hasMax = true;
    if (isa<arith::MinimumFOp, arith::MinNumFOp>(op))
      hasMin = true;
  }
  return (hasCmp && hasSelect) || hasMax || hasMin;
}

/// Get element type of a tensor value.
static Type getElemType(Value v) {
  if (auto shaped = dyn_cast<ShapedType>(v.getType()))
    return shaped.getElementType();
  return {};
}

/// Fuse extf(A→f32) followed by truncf(f32→B) into a single conversion.
/// Handles optional clamp ops between them.
struct FuseExtfTruncfChain : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp truncfGeneric,
                                PatternRewriter &rewriter) const override {
    // This must be a truncf generic with f32 input
    Operation *truncfOp = getSingleConversionOp(truncfGeneric);
    if (!truncfOp || !isa<arith::TruncFOp>(truncfOp))
      return failure();

    Value truncfInput = truncfGeneric.getDpsInputOperand(0)->get();
    Type truncfInputElem = getElemType(truncfInput);
    if (!truncfInputElem || !truncfInputElem.isF32())
      return failure();

    // Walk backward through optional clamp ops to find the extf
    Value cursor = truncfInput;
    for (int depth = 0; depth < 4; ++depth) {
      auto defOp = cursor.getDefiningOp();
      if (!defOp)
        return failure();

      auto prevGeneric = dyn_cast<linalg::GenericOp>(defOp);
      if (!prevGeneric)
        return failure();

      // Is this an extf generic?
      Operation *prevBodyOp = getSingleConversionOp(prevGeneric);
      if (prevBodyOp && isa<arith::ExtFOp>(prevBodyOp)) {
        // Found the extf. Check that input is fp8 or bf16 (not f32).
        Value extfInput = prevGeneric.getDpsInputOperand(0)->get();
        Type srcElem = getElemType(extfInput);
        if (!srcElem || srcElem.isF32())
          return failure();

        Type dstElem = getElemType(truncfGeneric.getResult(0));
        if (!dstElem)
          return failure();

        // Don't fuse if src and dst are the same type
        if (srcElem == dstElem)
          return failure();

        // Create a direct conversion: src → dst (no f32 intermediate)
        auto resultType =
            cast<RankedTensorType>(truncfGeneric.getResult(0).getType());
        auto newResultType =
            RankedTensorType::get(resultType.getShape(), dstElem);

        Value init = rewriter.create<tensor::EmptyOp>(
            truncfGeneric.getLoc(), newResultType.getShape(), dstElem);

        auto indexingMaps = truncfGeneric.getIndexingMapsArray();
        auto iteratorTypes = truncfGeneric.getIteratorTypesArray();

        bool isNarrowing =
            srcElem.getIntOrFloatBitWidth() > dstElem.getIntOrFloatBitWidth();

        auto fused = rewriter.create<linalg::GenericOp>(
            truncfGeneric.getLoc(), newResultType, ValueRange{extfInput},
            ValueRange{init}, indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value converted;
              if (isNarrowing)
                converted =
                    b.create<arith::TruncFOp>(loc, dstElem, args[0]);
              else
                converted =
                    b.create<arith::ExtFOp>(loc, dstElem, args[0]);
              b.create<linalg::YieldOp>(loc, converted);
            });

        rewriter.replaceOp(truncfGeneric, fused.getResult(0));
        return success();
      }

      // Is this a clamp? Skip through it.
      if (isClampGeneric(prevGeneric)) {
        if (prevGeneric.getNumDpsInputs() != 1)
          return failure();
        cursor = prevGeneric.getDpsInputOperand(0)->get();
        continue;
      }

      return failure();
    }
    return failure();
  }
};

/// Pass definition.
struct FuseF32IntermediateConversionsPass
    : public PassWrapper<FuseF32IntermediateConversionsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FuseF32IntermediateConversionsPass)

  StringRef getArgument() const override {
    return "fuse-f32-intermediate-conversions";
  }

  StringRef getDescription() const override {
    return "Fuse f32 intermediate type conversions in fp8/bf16 chains. "
           "Collapses extf(X→f32)→truncf(f32→Y) into direct extf/truncf(X→Y).";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseExtfTruncfChain>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseF32IntermediateConversionsPass() {
  return std::make_unique<FuseF32IntermediateConversionsPass>();
}

void registerFuseF32IntermediateConversionsPass() {
  PassRegistration<FuseF32IntermediateConversionsPass>();
}

} // namespace mlir::iree_compiler::NPU
