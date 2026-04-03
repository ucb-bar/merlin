// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 1: Convert data movement ops to cuda_tile IR.
//
// Handles: tensor.extract_slice, tensor.insert_slice, linalg.copy,
//          linalg.transpose, tensor.collapse_shape, tensor.expand_shape,
//          linalg.broadcast
//
// The pass analyzes ops and annotates them with cuda_tile metadata attributes.

#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::iree_compiler::CudaTile {

#define GEN_PASS_DEF_CUDATILECONVERTDATAMOVEMENTPASS
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"

namespace {

/// Returns the cuda_tile element type string (e.g. "f32", "f16", "i32").
static std::string getCudaTileElementTypeStr(Type type) {
  if (type.isF32()) return "f32";
  if (type.isF16()) return "f16";
  if (type.isBF16()) return "bf16";
  if (type.isF64()) return "f64";
  if (type.isInteger(32)) return "i32";
  if (type.isInteger(16)) return "i16";
  if (type.isInteger(8)) return "i8";
  if (type.isInteger(1)) return "i1";
  return "f32";
}

/// Compute row-major strides for a given shape.
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

/// Extract static shape from a RankedTensorType. Returns false if any dim is
/// dynamic.
static bool getStaticShape(Type type, SmallVectorImpl<int64_t> &shape) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    if (!tensorTy.hasStaticShape())
      return false;
    shape.assign(tensorTy.getShape().begin(), tensorTy.getShape().end());
    return true;
  }
  if (auto memrefTy = dyn_cast<MemRefType>(type)) {
    if (!memrefTy.hasStaticShape())
      return false;
    shape.assign(memrefTy.getShape().begin(), memrefTy.getShape().end());
    return true;
  }
  return false;
}

/// Get element type from a shaped type.
static Type getElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return {};
}

/// Attach common metadata attributes to an operation.
static void attachShapeAttrs(Operation *op, ArrayRef<int64_t> srcShape,
                             ArrayRef<int64_t> dstShape, Type elemType) {
  auto ctx = op->getContext();
  auto builder = Builder(ctx);

  op->setAttr("cuda_tile.src_shape", builder.getDenseI64ArrayAttr(srcShape));
  op->setAttr("cuda_tile.dst_shape", builder.getDenseI64ArrayAttr(dstShape));
  op->setAttr("cuda_tile.elem_type",
              builder.getStringAttr(getCudaTileElementTypeStr(elemType)));

  auto srcStrides = computeRowMajorStrides(srcShape);
  auto dstStrides = computeRowMajorStrides(dstShape);
  op->setAttr("cuda_tile.src_strides",
              builder.getDenseI64ArrayAttr(srcStrides));
  op->setAttr("cuda_tile.dst_strides",
              builder.getDenseI64ArrayAttr(dstStrides));
}

/// Analyze a linalg.copy op.
static void analyzeCopy(linalg::CopyOp op) {
  SmallVector<int64_t> srcShape, dstShape;
  if (!getStaticShape(op.getInputs()[0].getType(), srcShape))
    return;
  if (!getStaticShape(op.getOutputs()[0].getType(), dstShape))
    return;

  auto elemType = getElementType(op.getInputs()[0].getType());
  if (!elemType)
    return;

  // Check if strides differ (strided copy vs contiguous copy).
  auto srcStrides = computeRowMajorStrides(srcShape);
  auto dstStrides = computeRowMajorStrides(dstShape);

  op->setAttr("cuda_tile.kernel_class",
              StringAttr::get(op->getContext(), "copy"));
  attachShapeAttrs(op, srcShape, dstShape, elemType);
}

/// Analyze a linalg.transpose op.
static void analyzeTranspose(linalg::TransposeOp op) {
  SmallVector<int64_t> srcShape;
  if (!getStaticShape(op.getInput().getType(), srcShape))
    return;

  auto elemType = getElementType(op.getInput().getType());
  if (!elemType)
    return;

  auto perm = op.getPermutation();
  SmallVector<int64_t> permVec(perm.begin(), perm.end());

  // Compute destination shape from permutation.
  SmallVector<int64_t> dstShape(srcShape.size());
  for (size_t i = 0; i < srcShape.size(); ++i) {
    dstShape[i] = srcShape[permVec[i]];
  }

  auto ctx = op->getContext();
  auto builder = Builder(ctx);
  op->setAttr("cuda_tile.kernel_class", StringAttr::get(ctx, "transpose"));
  attachShapeAttrs(op, srcShape, dstShape, elemType);
  op->setAttr("cuda_tile.permutation",
              builder.getDenseI64ArrayAttr(permVec));
}

/// Analyze a linalg.broadcast op.
static void analyzeBroadcast(linalg::BroadcastOp op) {
  SmallVector<int64_t> srcShape, dstShape;
  if (!getStaticShape(op.getInput().getType(), srcShape))
    return;
  if (!getStaticShape(op.getInit().getType(), dstShape))
    return;

  auto elemType = getElementType(op.getInput().getType());
  if (!elemType)
    return;

  auto dims = op.getDimensions();
  SmallVector<int64_t> dimsVec(dims.begin(), dims.end());

  auto ctx = op->getContext();
  auto builder = Builder(ctx);
  op->setAttr("cuda_tile.kernel_class", StringAttr::get(ctx, "broadcast"));
  attachShapeAttrs(op, srcShape, dstShape, elemType);
  op->setAttr("cuda_tile.broadcast_dims",
              builder.getDenseI64ArrayAttr(dimsVec));
}

/// Analyze tensor.extract_slice. We need static offsets, sizes, strides.
static void analyzeExtractSlice(tensor::ExtractSliceOp op) {
  // Only handle fully static cases.
  if (!op.getType().hasStaticShape())
    return;

  auto sourceType = op.getSourceType();
  if (!sourceType.hasStaticShape())
    return;

  auto offsets = op.getStaticOffsets();
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();

  // Check all are static.
  for (auto o : offsets)
    if (ShapedType::isDynamic(o))
      return;
  for (auto s : sizes)
    if (ShapedType::isDynamic(s))
      return;
  for (auto st : strides)
    if (ShapedType::isDynamic(st))
      return;

  SmallVector<int64_t> srcShape(sourceType.getShape().begin(),
                                sourceType.getShape().end());
  SmallVector<int64_t> dstShape(op.getType().getShape().begin(),
                                op.getType().getShape().end());

  auto elemType = sourceType.getElementType();
  auto ctx = op->getContext();
  auto builder = Builder(ctx);

  op->setAttr("cuda_tile.kernel_class",
              StringAttr::get(ctx, "extract_slice"));
  attachShapeAttrs(op, srcShape, dstShape, elemType);
  op->setAttr("cuda_tile.offsets",
              builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  offsets.begin(), offsets.end())));
  op->setAttr("cuda_tile.slice_sizes",
              builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  sizes.begin(), sizes.end())));
  op->setAttr("cuda_tile.slice_strides",
              builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  strides.begin(), strides.end())));
}

/// Analyze tensor.insert_slice.
static void analyzeInsertSlice(tensor::InsertSliceOp op) {
  auto sourceType = op.getSourceType();
  auto destType = op.getDestType();
  if (!sourceType.hasStaticShape() || !destType.hasStaticShape())
    return;

  auto offsets = op.getStaticOffsets();
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();

  for (auto o : offsets)
    if (ShapedType::isDynamic(o))
      return;
  for (auto s : sizes)
    if (ShapedType::isDynamic(s))
      return;
  for (auto st : strides)
    if (ShapedType::isDynamic(st))
      return;

  SmallVector<int64_t> srcShape(sourceType.getShape().begin(),
                                sourceType.getShape().end());
  SmallVector<int64_t> dstShape(destType.getShape().begin(),
                                destType.getShape().end());

  auto elemType = sourceType.getElementType();
  auto ctx = op->getContext();
  auto builder = Builder(ctx);

  op->setAttr("cuda_tile.kernel_class",
              StringAttr::get(ctx, "insert_slice"));
  attachShapeAttrs(op, srcShape, dstShape, elemType);
  op->setAttr("cuda_tile.offsets",
              builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  offsets.begin(), offsets.end())));
  op->setAttr("cuda_tile.slice_sizes",
              builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  sizes.begin(), sizes.end())));
  op->setAttr("cuda_tile.slice_strides",
              builder.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  strides.begin(), strides.end())));
}

/// Analyze tensor.collapse_shape / expand_shape.
static void analyzeReshape(Operation *op, StringRef kind) {
  SmallVector<int64_t> srcShape, dstShape;
  if (!getStaticShape(op->getOperand(0).getType(), srcShape))
    return;
  if (!getStaticShape(op->getResult(0).getType(), dstShape))
    return;

  auto elemType = getElementType(op->getOperand(0).getType());
  if (!elemType)
    return;

  op->setAttr("cuda_tile.kernel_class",
              StringAttr::get(op->getContext(), kind));
  attachShapeAttrs(op, srcShape, dstShape, elemType);
}

/// Detect transpose pattern in linalg.generic:
/// - All parallel iterators
/// - Pass-through body (yield %arg_in)
/// - Input and output maps differ by a permutation
/// Returns the permutation vector if it's a transpose, empty otherwise.
static SmallVector<int64_t>
detectTransposeGeneric(linalg::GenericOp genericOp) {
  // Must have exactly 1 input and 1 output.
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return {};

  // All iterators must be parallel.
  if (genericOp.getNumReductionLoops() > 0)
    return {};

  // Body must be a simple yield of the block argument (pass-through).
  Block &body = genericOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return {};
  // The yielded value must be the first block argument (the input element).
  if (yieldOp.getOperand(0) != body.getArgument(0))
    return {};

  // Check indexing maps: input map should be identity-like, output permuted
  // (or vice versa). Extract the permutation.
  auto maps = genericOp.getIndexingMapsArray();
  if (maps.size() != 2)
    return {};

  auto inputMap = maps[0];
  auto outputMap = maps[1];

  // Both maps must be permutations.
  if (!inputMap.isPermutation() || !outputMap.isPermutation())
    return {};

  // If both are identity, it's a copy not a transpose.
  if (inputMap.isIdentity() && outputMap.isIdentity())
    return {};

  // Compute the effective permutation: output_map^{-1} * input_map.
  // For typical IREE lowering: output is identity, input is permuted.
  // The transpose permutation P satisfies: dst[i] = src[P[i]]
  // output_map: (d0,d1) -> (d0,d1) [identity]
  // input_map:  (d0,d1) -> (d1,d0) [transposed]
  // So P = inverse of input_map composed with output_map.
  unsigned numDims = inputMap.getNumDims();
  SmallVector<int64_t> perm(numDims);

  // Get the output inverse permutation.
  SmallVector<unsigned> outputPerm;
  if (!outputMap.isIdentity()) {
    // General case: compose permutations.
    // For now, handle the common case where output is identity.
    for (unsigned i = 0; i < numDims; ++i) {
      auto expr = outputMap.getResult(i);
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        return {};
      outputPerm.push_back(dimExpr.getPosition());
    }
  }

  // Extract input permutation.
  SmallVector<unsigned> inputPerm;
  for (unsigned i = 0; i < numDims; ++i) {
    auto expr = inputMap.getResult(i);
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return {};
    inputPerm.push_back(dimExpr.getPosition());
  }

  // Compute effective permutation.
  // IREE lowering can produce either:
  //   input: (d0,d1)->(d1,d0), output: (d0,d1)->(d0,d1)  [input permuted]
  //   input: (d0,d1)->(d0,d1), output: (d0,d1)->(d1,d0)  [output permuted]
  // Both represent a transpose. We need the permutation P such that
  // dst[i] = src[P[i]].
  if (outputMap.isIdentity()) {
    // Input is permuted: input(d0,d1) -> (d_p0, d_p1).
    for (unsigned i = 0; i < numDims; ++i)
      perm[i] = inputPerm[i];
  } else if (inputMap.isIdentity()) {
    // Output is permuted: output(d0,d1) -> (d_p0, d_p1).
    // The outputPerm maps iteration dim → output dim.
    // The transpose is the inverse: for output dim i, which iteration dim?
    SmallVector<unsigned> inversePerm(numDims);
    for (unsigned i = 0; i < numDims; ++i)
      inversePerm[outputPerm[i]] = i;
    for (unsigned i = 0; i < numDims; ++i)
      perm[i] = inversePerm[i];
  } else {
    // Both permuted — general case.
    // Compose: perm = outputPerm^{-1} * inputPerm.
    SmallVector<unsigned> inverseOutput(numDims);
    for (unsigned i = 0; i < numDims; ++i)
      inverseOutput[outputPerm[i]] = i;
    for (unsigned i = 0; i < numDims; ++i)
      perm[i] = inputPerm[inverseOutput[i]];
  }

  // Verify it's not identity.
  bool isIdentity = true;
  for (unsigned i = 0; i < numDims; ++i) {
    if (perm[i] != (int64_t)i) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity)
    return {};

  return perm;
}

struct ConvertDataMovementPass
    : impl::CudaTileConvertDataMovementPassBase<ConvertDataMovementPass> {

  ConvertDataMovementPass() = default;
  explicit ConvertDataMovementPass(const CudaTileTransformOptions &options)
      : options(options) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    funcOp->walk([&](Operation *op) {
      if (auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
        analyzeCopy(copyOp);
      } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
        analyzeTranspose(transposeOp);
      } else if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(op)) {
        analyzeBroadcast(broadcastOp);
      } else if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
        analyzeExtractSlice(extractOp);
      } else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
        analyzeInsertSlice(insertOp);
      } else if (isa<tensor::CollapseShapeOp>(op)) {
        analyzeReshape(op, "collapse_shape");
      } else if (isa<tensor::ExpandShapeOp>(op)) {
        analyzeReshape(op, "expand_shape");
      } else if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        // Detect transpose lowered to linalg.generic by IREE.
        auto perm = detectTransposeGeneric(genericOp);
        if (!perm.empty()) {
          SmallVector<int64_t> srcShape, dstShape;
          if (!getStaticShape(genericOp.getDpsInputs()[0].getType(), srcShape))
            return;
          if (!getStaticShape(genericOp.getDpsInits()[0].getType(), dstShape))
            return;
          auto elemTy = getElementType(genericOp.getDpsInputs()[0].getType());
          if (!elemTy)
            return;

          auto ctx = op->getContext();
          auto builder = Builder(ctx);
          op->setAttr("cuda_tile.kernel_class",
                      StringAttr::get(ctx, "transpose"));
          attachShapeAttrs(op, srcShape, dstShape, elemTy);
          op->setAttr("cuda_tile.permutation",
                      builder.getDenseI64ArrayAttr(perm));
        }
      }
    });
  }

  CudaTileTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertDataMovementToCudaTilePass(
    const CudaTileTransformOptions &options) {
  return std::make_unique<ConvertDataMovementPass>(options);
}

} // namespace mlir::iree_compiler::CudaTile
