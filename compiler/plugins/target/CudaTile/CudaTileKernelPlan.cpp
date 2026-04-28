// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/CudaTile/CudaTileKernelPlan.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

static llvm::SmallVector<int64_t> getI64ArrayAttr(Operation *op,
                                                  StringRef name) {
  if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name)) {
    return llvm::SmallVector<int64_t>(attr.asArrayRef().begin(),
                                      attr.asArrayRef().end());
  }
  return {};
}

static llvm::SmallVector<int64_t> getStaticShapeFromType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (shaped.hasStaticShape()) {
      return llvm::SmallVector<int64_t>(shaped.getShape().begin(),
                                        shaped.getShape().end());
    }
    return {};
  }
  if (auto dispatchTensor =
          dyn_cast<IREE::TensorExt::DispatchTensorType>(type)) {
    if (auto shaped = dyn_cast<ShapedType>(dispatchTensor.getBoundType())) {
      if (shaped.hasStaticShape()) {
        return llvm::SmallVector<int64_t>(shaped.getShape().begin(),
                                          shaped.getShape().end());
      }
    }
  }
  return {};
}

static Type getElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  if (auto dispatchTensor =
          dyn_cast<IREE::TensorExt::DispatchTensorType>(type)) {
    if (auto shaped = dyn_cast<ShapedType>(dispatchTensor.getBoundType()))
      return shaped.getElementType();
  }
  return {};
}

static std::optional<int64_t> getStaticIndexValue(Value value) {
  if (!value)
    return int64_t{0};
  if (auto constantIndex = value.getDefiningOp<arith::ConstantIndexOp>())
    return constantIndex.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttr.getInt();
  }
  return std::nullopt;
}

static Type getElementTypeFromCudaTileString(MLIRContext *ctx,
                                             StringRef typeStr) {
  if (typeStr == "f32")
    return Float32Type::get(ctx);
  if (typeStr == "f16")
    return Float16Type::get(ctx);
  if (typeStr == "bf16")
    return BFloat16Type::get(ctx);
  if (typeStr == "f64")
    return Float64Type::get(ctx);
  if (typeStr == "i32")
    return IntegerType::get(ctx, 32);
  if (typeStr == "i16")
    return IntegerType::get(ctx, 16);
  if (typeStr == "i8")
    return IntegerType::get(ctx, 8);
  if (typeStr == "i1")
    return IntegerType::get(ctx, 1);
  return {};
}

static void extractPrimaryShapeAndType(CudaTileKernelPlan &plan) {
  Operation *op = plan.primaryOp;
  if (!op)
    return;

  plan.srcShape = getI64ArrayAttr(op, "cuda_tile.src_shape");
  plan.dstShape = getI64ArrayAttr(op, "cuda_tile.dst_shape");

  if (auto elemTypeAttr = op->getAttrOfType<TypeAttr>("cuda_tile.elem_type")) {
    plan.elementType = elemTypeAttr.getValue();
  } else if (auto elemTypeAttr =
                 op->getAttrOfType<StringAttr>("cuda_tile.elem_type")) {
    plan.elementType =
        getElementTypeFromCudaTileString(op->getContext(),
                                         elemTypeAttr.getValue());
  }

  if (plan.srcShape.empty() || !plan.elementType) {
    for (Value operand : op->getOperands()) {
      if (plan.srcShape.empty())
        plan.srcShape = getStaticShapeFromType(operand.getType());
      if (!plan.elementType)
        plan.elementType = getElementType(operand.getType());
      if (!plan.srcShape.empty() && plan.elementType)
        break;
    }
  }

  if (plan.dstShape.empty()) {
    for (Value result : op->getResults()) {
      plan.dstShape = getStaticShapeFromType(result.getType());
      if (!plan.dstShape.empty())
        break;
    }
  }

  if (plan.dstShape.empty())
    plan.dstShape = plan.srcShape;
}

static bool hasUnitSteps(llvm::ArrayRef<int64_t> values) {
  return llvm::all_of(values, [](int64_t v) { return v == 1; });
}

static bool isMultiplyAddGeneric(linalg::GenericOp genOp) {
  if (genOp.getNumDpsInputs() != 2 || genOp.getNumDpsInits() != 1)
    return false;

  Block &body = genOp.getRegion().front();
  auto ops = body.without_terminator();
  int opCount = 0;
  bool hasMul = false;
  bool hasAdd = false;
  for (auto &op : ops) {
    if (isa<arith::MulFOp, arith::MulIOp>(&op))
      hasMul = true;
    else if (isa<arith::AddFOp, arith::AddIOp>(&op))
      hasAdd = true;
    opCount++;
  }
  return opCount == 2 && hasMul && hasAdd;
}

static bool isGenericContraction(linalg::GenericOp genOp) {
  return genOp.getNumReductionLoops() > 0 && isMultiplyAddGeneric(genOp);
}

static bool collectAffineCoefficients(AffineExpr expr,
                                      llvm::SmallVectorImpl<int64_t> &coeffs,
                                      int64_t &constant,
                                      int64_t scale = 1) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    unsigned pos = dimExpr.getPosition();
    if (pos >= coeffs.size())
      return false;
    coeffs[pos] += scale;
    return true;
  }
  if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr)) {
    constant += scale * cstExpr.getValue();
    return true;
  }
  if (auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    switch (binaryExpr.getKind()) {
    case AffineExprKind::Add:
      return collectAffineCoefficients(binaryExpr.getLHS(), coeffs, constant,
                                       scale) &&
             collectAffineCoefficients(binaryExpr.getRHS(), coeffs, constant,
                                       scale);
    case AffineExprKind::Mul: {
      auto lhsCst = dyn_cast<AffineConstantExpr>(binaryExpr.getLHS());
      auto rhsCst = dyn_cast<AffineConstantExpr>(binaryExpr.getRHS());
      if (lhsCst)
        return collectAffineCoefficients(binaryExpr.getRHS(), coeffs, constant,
                                         scale * lhsCst.getValue());
      if (rhsCst)
        return collectAffineCoefficients(binaryExpr.getLHS(), coeffs, constant,
                                         scale * rhsCst.getValue());
      return false;
    }
    default:
      return false;
    }
  }
  return false;
}

static bool getLoopPositionOrUnitConstant(AffineExpr expr, int64_t extent,
                                          int64_t &loopPos) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    loopPos = dimExpr.getPosition();
    return true;
  }
  if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr);
      cstExpr && cstExpr.getValue() == 0 && extent == 1) {
    loopPos = -1;
    return true;
  }
  return false;
}

static llvm::SmallVector<int64_t> copyI64Array(llvm::ArrayRef<int64_t> values) {
  return llvm::SmallVector<int64_t>(values.begin(), values.end());
}

static void printI64Array(llvm::ArrayRef<int64_t> values,
                          llvm::raw_ostream &os) {
  os << "[";
  for (auto [idx, value] : llvm::enumerate(values))
    os << (idx ? ", " : "") << value;
  os << "]";
}

static void printBool(bool value, llvm::raw_ostream &os) {
  os << (value ? "true" : "false");
}

static void printOpName(Operation *op, llvm::raw_ostream &os) {
  if (!op) {
    os << "<none>";
    return;
  }
  os << "\"" << op->getName().getStringRef() << "\"";
}

static void printOptionalAttr(Attribute attr, llvm::raw_ostream &os) {
  if (!attr) {
    os << "<none>";
    return;
  }
  attr.print(os);
}

static llvm::SmallVector<int64_t>
extractTensorReductionDims(linalg::LinalgOp linalgOp) {
  llvm::SmallVector<int64_t> reductionDims;
  if (auto reduceOp = dyn_cast<linalg::ReduceOp>(linalgOp.getOperation())) {
    reductionDims.assign(reduceOp.getDimensions().begin(),
                         reduceOp.getDimensions().end());
    return reductionDims;
  }

  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  if (iteratorTypes.empty() || linalgOp.getDpsInputs().empty())
    return reductionDims;

  AffineMap inputMap;
  if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
    auto maps = genericOp.getIndexingMapsArray();
    if (!maps.empty())
      inputMap = maps.front();
  } else {
    inputMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  }

  for (auto [iterDim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType == utils::IteratorType::parallel)
      continue;
    if (!inputMap) {
      reductionDims.push_back(iterDim);
      continue;
    }
    for (unsigned tensorDim = 0; tensorDim < inputMap.getNumResults();
         ++tensorDim) {
      auto dimExpr = dyn_cast<AffineDimExpr>(inputMap.getResult(tensorDim));
      if (dimExpr && dimExpr.getPosition() == iterDim) {
        reductionDims.push_back(tensorDim);
        break;
      }
    }
  }
  return reductionDims;
}

static CudaTileSemanticKind classifyLinalgOp(linalg::LinalgOp linalgOp,
                                             const CudaTileConvPlan &convPlan) {
  if (isa<linalg::FillOp>(linalgOp.getOperation()))
    return CudaTileSemanticKind::Unknown;
  if (convPlan)
    return CudaTileSemanticKind::WindowedReduction;
  if (linalg::isaContractionOpInterface(linalgOp))
    return CudaTileSemanticKind::Contraction;
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp.getOperation()))
    return CudaTileSemanticKind::Contraction;
  if (isa<linalg::ReduceOp>(linalgOp.getOperation()))
    return CudaTileSemanticKind::Reduction;
  if (auto genOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
    if (isGenericContraction(genOp))
      return CudaTileSemanticKind::Contraction;
  }

  bool sawParallel = false;
  bool sawReduction = false;
  for (auto iteratorType : linalgOp.getIteratorTypesArray()) {
    if (iteratorType == utils::IteratorType::parallel)
      sawParallel = true;
    else if (iteratorType == utils::IteratorType::reduction)
      sawReduction = true;
  }
  if (sawReduction)
    return CudaTileSemanticKind::Reduction;
  if (sawParallel)
    return CudaTileSemanticKind::Map;
  return CudaTileSemanticKind::Unknown;
}

static CudaTileKernelKind
getKernelKindForSemantic(CudaTileSemanticKind semanticKind) {
  switch (semanticKind) {
  case CudaTileSemanticKind::DataMovement:
    return CudaTileKernelKind::Copy;
  case CudaTileSemanticKind::Map:
    return CudaTileKernelKind::Elementwise;
  case CudaTileSemanticKind::Reduction:
    return CudaTileKernelKind::Reduction;
  case CudaTileSemanticKind::Contraction:
    return CudaTileKernelKind::Matmul;
  case CudaTileSemanticKind::WindowedReduction:
    return CudaTileKernelKind::Conv;
  case CudaTileSemanticKind::FusedReductionElementwise:
    return CudaTileKernelKind::FusedReductionElementwise;
  case CudaTileSemanticKind::ControlFlow:
    return CudaTileKernelKind::SCF;
  case CudaTileSemanticKind::Unknown:
    return CudaTileKernelKind::Unsupported;
  }
  return CudaTileKernelKind::Unsupported;
}

static void recordLoopDims(linalg::LinalgOp linalgOp,
                           CudaTileKernelPlan &plan) {
  plan.parallelLoopDims.clear();
  plan.reductionLoopDims.clear();
  for (auto [idx, iteratorType] :
       llvm::enumerate(linalgOp.getIteratorTypesArray())) {
    if (iteratorType == utils::IteratorType::parallel)
      plan.parallelLoopDims.push_back(idx);
    else if (iteratorType == utils::IteratorType::reduction)
      plan.reductionLoopDims.push_back(idx);
  }
}

static int64_t findBindingForValue(const CudaTileKernelPlan &plan,
                                   Value value) {
  for (auto [idx, binding] : llvm::enumerate(plan.bindingShapes)) {
    if (binding.memref == value)
      return binding.binding >= 0 ? binding.binding : static_cast<int64_t>(idx);
  }
  if (auto loadOp =
          value.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>())
    return findBindingForValue(plan, loadOp.getSource());
  return -1;
}

static llvm::SmallVector<int64_t>
getBindingShape(const CudaTileKernelPlan &plan, int64_t bindingIndex) {
  for (const CudaTileBindingPlan &binding : plan.bindingShapes) {
    if (binding.binding == bindingIndex)
      return binding.shape;
  }
  if (bindingIndex >= 0 &&
      bindingIndex < static_cast<int64_t>(plan.bindingShapes.size()))
    return plan.bindingShapes[bindingIndex].shape;
  return {};
}

static CudaTileOperandPlan makeOperandPlan(CudaTileKernelPlan &plan,
                                           CudaTileOperandRole role,
                                           Operation *owner, Value value) {
  CudaTileOperandPlan operand;
  operand.role = role;
  operand.owner = owner;
  operand.value = value;
  operand.logicalShape = getStaticShapeFromType(value.getType());
  operand.binding = findBindingForValue(plan, value);
  operand.physicalShape = getBindingShape(plan, operand.binding);
  operand.isConstant = value.getDefiningOp<arith::ConstantOp>() != nullptr;

  if (auto loadOp =
          value.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>()) {
    operand.isDispatchLoad = true;
    operand.binding = findBindingForValue(plan, loadOp.getSource());
    operand.physicalShape = getStaticShapeFromType(loadOp.getSource().getType());
    operand.offsets = copyI64Array(loadOp.getStaticOffsets());
    operand.sizes = copyI64Array(loadOp.getStaticSizes());
    operand.strides = copyI64Array(loadOp.getStaticStrides());
  }

  if (operand.logicalShape.empty())
    operand.logicalShape = operand.physicalShape;
  if (operand.physicalShape.empty())
    operand.physicalShape = operand.logicalShape;
  return operand;
}

static int64_t getOnlyOutputBinding(const CudaTileKernelPlan &plan) {
  int64_t binding = -1;
  for (const CudaTileOperandPlan &operand : plan.operands) {
    if (!operand.isDispatchStore || operand.binding < 0)
      continue;
    if (binding >= 0 && binding != operand.binding)
      return -1;
    binding = operand.binding;
  }
  return binding;
}

static bool isDispatchTensorSlice(
    IREE::TensorExt::DispatchTensorLoadOp loadOp) {
  if (!loadOp)
    return false;
  return !loadOp.isLoadOfWholeSource() ||
         !hasUnitSteps(copyI64Array(loadOp.getStaticStrides()));
}

static bool detectRhsTransposed(linalg::LinalgOp linalgOp) {
  auto maps = linalgOp.getIndexingMapsArray();
  auto iterTypes = linalgOp.getIteratorTypesArray();
  if (maps.size() < 2)
    return false;

  AffineMap rhsMap = maps[1];
  for (unsigned dim = 0; dim < iterTypes.size(); ++dim) {
    if (iterTypes[dim] != utils::IteratorType::reduction)
      continue;
    if (rhsMap.getNumResults() < 2)
      return false;
    auto lastExpr = rhsMap.getResult(rhsMap.getNumResults() - 1);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(lastExpr))
      return dimExpr.getPosition() == dim;
    return false;
  }
  return false;
}

static bool isSupportedExplicitCudaTileSize(int64_t size) {
  return size > 0 && (size & (size - 1)) == 0;
}

static bool getDirectAffineDimPosition(AffineMap map, unsigned resultIndex,
                                       int64_t &position) {
  if (!map || resultIndex >= map.getNumResults())
    return false;
  auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(resultIndex));
  if (!dimExpr)
    return false;
  position = static_cast<int64_t>(dimExpr.getPosition());
  return true;
}

static void populateContractionScheduleTiles(CudaTileKernelPlan &plan,
                                             linalg::LinalgOp linalgOp,
                                             CudaTileContractionPlan &ctPlan) {
  ArrayRef<int64_t> workgroupTiles = plan.schedule.workgroupTileSizes;
  if (workgroupTiles.empty() || ctPlan.resultShape.size() != 2 ||
      plan.reductionLoopDims.size() != 1)
    return;

  auto maps = linalgOp.getIndexingMapsArray();
  unsigned outputMapIndex = linalgOp.getNumDpsInputs();
  if (outputMapIndex >= maps.size())
    return;

  int64_t mIter = -1, nIter = -1;
  if (!getDirectAffineDimPosition(maps[outputMapIndex], 0, mIter) ||
      !getDirectAffineDimPosition(maps[outputMapIndex], 1, nIter))
    return;
  int64_t kIter = plan.reductionLoopDims.front();
  if (mIter < 0 || nIter < 0 || kIter < 0 ||
      mIter >= static_cast<int64_t>(workgroupTiles.size()) ||
      nIter >= static_cast<int64_t>(workgroupTiles.size()) ||
      kIter >= static_cast<int64_t>(workgroupTiles.size()))
    return;

  int64_t tileM = workgroupTiles[mIter];
  int64_t tileN = workgroupTiles[nIter];
  int64_t tileK = workgroupTiles[kIter];
  if (!isSupportedExplicitCudaTileSize(tileM) ||
      !isSupportedExplicitCudaTileSize(tileN) ||
      !isSupportedExplicitCudaTileSize(tileK))
    return;

  ctPlan.tileM = tileM;
  ctPlan.tileN = tileN;
  ctPlan.tileK = tileK;
  ctPlan.hasScheduleTiles = true;
  ctPlan.scheduleSource = plan.schedule.source;
}

static void populateContractionPlan(CudaTileKernelPlan &plan) {
  bool isContraction = plan.semanticKind == CudaTileSemanticKind::Contraction;
  bool isPointwiseConvMatmul =
      plan.conv.mode == CudaTileConvLoweringMode::PointwiseMatmul;
  if (!plan.primaryOp || (!isContraction && !isPointwiseConvMatmul))
    return;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(plan.primaryOp);
  if (!linalgOp || linalgOp.getNumDpsInputs() < 2)
    return;

  CudaTileContractionPlan contraction;
  contraction.isValid = true;
  contraction.op = plan.primaryOp;

  auto dpsInputs = linalgOp.getDpsInputs();
  Value lhs = dpsInputs[0];
  Value rhs = dpsInputs[1];
  contraction.lhsShape = getStaticShapeFromType(lhs.getType());
  contraction.rhsShape = getStaticShapeFromType(rhs.getType());
  contraction.lhsBinding = findBindingForValue(plan, lhs);
  contraction.rhsBinding = findBindingForValue(plan, rhs);
  contraction.constantRhs = rhs.getDefiningOp<arith::ConstantOp>() ? rhs : Value();

  for (Value result : plan.primaryOp->getResults()) {
    contraction.resultShape = getStaticShapeFromType(result.getType());
    if (!contraction.resultShape.empty())
      break;
  }
  if (contraction.resultShape.empty())
    contraction.resultShape = plan.dstShape;

  if (!linalgOp.getDpsInits().empty())
    contraction.resultBinding = findBindingForValue(plan, linalgOp.getDpsInits()[0]);
  if (contraction.resultBinding < 0)
    contraction.resultBinding = getOnlyOutputBinding(plan);

  contraction.rhsTransposed = detectRhsTransposed(linalgOp);

  if (auto lhsLoadOp =
          lhs.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>()) {
    contraction.hasSlicedLhs = isDispatchTensorSlice(lhsLoadOp);
    contraction.slicedLhsSourceShape =
        getStaticShapeFromType(lhsLoadOp.getSource().getType());
    contraction.slicedLhsOffsets = copyI64Array(lhsLoadOp.getStaticOffsets());
    contraction.slicedLhsSizes = copyI64Array(lhsLoadOp.getStaticSizes());
    contraction.slicedLhsStrides = copyI64Array(lhsLoadOp.getStaticStrides());
  }

  if (!contraction.resultShape.empty()) {
    contraction.n = contraction.resultShape.back();
    contraction.m = 1;
    for (int64_t i = 0;
         i + 1 < static_cast<int64_t>(contraction.resultShape.size()); ++i)
      contraction.m *= contraction.resultShape[i];
    if (contraction.resultShape.size() <= 1)
      contraction.m = contraction.resultShape.front();
    contraction.hasRankFlattening = contraction.resultShape.size() > 2;
    if (contraction.resultShape.size() > 2) {
      contraction.batch = 1;
      for (int64_t i = 0;
           i + 2 < static_cast<int64_t>(contraction.resultShape.size()); ++i)
        contraction.batch *= contraction.resultShape[i];
    }
  }

  if (contraction.rhsTransposed) {
    if (!contraction.rhsShape.empty())
      contraction.k = contraction.rhsShape.back();
  } else if (!contraction.lhsShape.empty()) {
    contraction.k = contraction.lhsShape.back();
  }

  contraction.canUseMma =
      contraction.m > 0 && contraction.n > 0 && contraction.k > 0 &&
      !contraction.lhsShape.empty() && !contraction.rhsShape.empty() &&
      !contraction.resultShape.empty();
  populateContractionScheduleTiles(plan, linalgOp, contraction);
  plan.contraction = std::move(contraction);
}

static void populateSchedulePlan(Operation *innerModule,
                                 CudaTileKernelPlan &plan) {
  if (plan.primaryOp) {
    if (auto config = mlir::iree_compiler::getLoweringConfig(plan.primaryOp)) {
      plan.schedule.loweringConfig = config;
      plan.schedule.hasLoweringConfig = true;
      plan.schedule.source = CudaTileScheduleSource::IREECodegen;
      plan.schedule.workgroupTileSizes = config.getWorkgroupTileSizes();

      if (std::optional<unsigned> numLevels = config.getNumTilingLevels()) {
        for (unsigned level = 0; level < *numLevels; ++level) {
          llvm::SmallVector<int64_t> sizes =
              config.getStaticTilingLevelSizes(level, plan.primaryOp);
          if (!sizes.empty())
            plan.schedule.tilingLevelSizes.push_back(std::move(sizes));
        }
      } else {
        for (unsigned level = 0; level < 4; ++level) {
          if (!config.hasTilingLevel(level))
            continue;
          llvm::SmallVector<int64_t> sizes =
              config.getStaticTilingLevelSizes(level, plan.primaryOp);
          if (!sizes.empty())
            plan.schedule.tilingLevelSizes.push_back(std::move(sizes));
        }
      }

      if (auto gpuConfig = dyn_cast<IREE::GPU::LoweringConfigAttr>(config)) {
        plan.schedule.hasIREEGPULoweringConfig = true;
        plan.schedule.source = CudaTileScheduleSource::IREEGPU;
        plan.schedule.mmaKind = IREE::GPU::getMmaKind(gpuConfig);
      }
    }
  }

  innerModule->walk([&](FunctionOpInterface funcOp) {
    if (plan.schedule.hasTranslationInfo)
      return;
    auto translationInfo =
        funcOp->getAttrOfType<IREE::Codegen::TranslationInfoAttr>(
            "translation_info");
    if (!translationInfo)
      return;
    plan.schedule.translationInfo = translationInfo;
    plan.schedule.hasTranslationInfo = true;
    for (int64_t dim : translationInfo.getWorkgroupSize())
      plan.schedule.workgroupSize.push_back(dim);
    if (translationInfo.getSubgroupSize() != int64_t())
      plan.schedule.subgroupSize = translationInfo.getSubgroupSize();
  });
}

static void populatePrimaryOperandPlans(CudaTileKernelPlan &plan) {
  if (!plan.primaryOp)
    return;
  auto linalgOp = dyn_cast<linalg::LinalgOp>(plan.primaryOp);
  if (!linalgOp)
    return;

  for (Value input : linalgOp.getDpsInputs())
    plan.operands.push_back(
        makeOperandPlan(plan, CudaTileOperandRole::Input, plan.primaryOp,
                        input));
  for (Value init : linalgOp.getDpsInits())
    plan.operands.push_back(makeOperandPlan(plan, CudaTileOperandRole::Init,
                                            plan.primaryOp, init));
  for (Value result : plan.primaryOp->getResults())
    plan.operands.push_back(makeOperandPlan(
        plan, CudaTileOperandRole::Result, plan.primaryOp, result));
}

static void populateDispatchTensorOperandPlans(Operation *innerModule,
                                               CudaTileKernelPlan &plan) {
  innerModule->walk([&](IREE::TensorExt::DispatchTensorLoadOp op) {
    CudaTileOperandPlan operand =
        makeOperandPlan(plan, CudaTileOperandRole::Input, op.getOperation(),
                        op.getResult());
    operand.isDispatchLoad = true;
    operand.binding = findBindingForValue(plan, op.getSource());
    operand.physicalShape = getStaticShapeFromType(op.getSource().getType());
    operand.offsets = copyI64Array(op.getStaticOffsets());
    operand.sizes = copyI64Array(op.getStaticSizes());
    operand.strides = copyI64Array(op.getStaticStrides());
    plan.operands.push_back(std::move(operand));
  });

  innerModule->walk([&](IREE::TensorExt::DispatchTensorStoreOp op) {
    CudaTileOperandPlan operand =
        makeOperandPlan(plan, CudaTileOperandRole::Output, op.getOperation(),
                        op.getTarget());
    operand.logicalShape = getStaticShapeFromType(op.getValue().getType());
    operand.isDispatchStore = true;
    operand.binding = findBindingForValue(plan, op.getTarget());
    operand.physicalShape = getStaticShapeFromType(op.getTarget().getType());
    operand.offsets = copyI64Array(op.getStaticOffsets());
    operand.sizes = copyI64Array(op.getStaticSizes());
    operand.strides = copyI64Array(op.getStaticStrides());
    plan.operands.push_back(std::move(operand));
  });
}

static void incrementSemanticCount(CudaTileKernelPlan &plan,
                                   CudaTileSemanticKind semanticKind);
static CudaTileLoweringStrategy
selectLoweringStrategy(StringRef kernelClass,
                       CudaTileSemanticKind semanticKind,
                       const CudaTileConvPlan &convPlan);

static void populateFusedOpPlans(Operation *innerModule,
                                 CudaTileKernelPlan &plan) {
  innerModule->walk([&](linalg::LinalgOp linalgOp) {
    CudaTileConvPlan opConvPlan;
    if (auto genOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
      opConvPlan = extractCudaTileConvPlan(genOp);
      if (!opConvPlan)
        opConvPlan = extractPoolingPlan(genOp);
    }

    CudaTileSemanticKind semanticKind = classifyLinalgOp(linalgOp, opConvPlan);
    incrementSemanticCount(plan, semanticKind);
    if (semanticKind == CudaTileSemanticKind::Unknown)
      return;

    CudaTileFusedOpPlan fusedOp;
    fusedOp.op = linalgOp.getOperation();
    fusedOp.semanticKind = semanticKind;
    StringRef kernelClass;
    if (auto classAttr =
            fusedOp.op->getAttrOfType<StringAttr>("cuda_tile.kernel_class")) {
      kernelClass = classAttr.getValue();
      fusedOp.kind = getCudaTileKernelKind(classAttr.getValue());
    } else {
      fusedOp.kind = getKernelKindForSemantic(semanticKind);
      kernelClass = stringifyCudaTileKernelKind(fusedOp.kind);
    }
    fusedOp.conv = opConvPlan;
    fusedOp.loweringStrategy =
        selectLoweringStrategy(kernelClass, semanticKind, fusedOp.conv);
    fusedOp.reductionDims = extractTensorReductionDims(linalgOp);
    plan.fusedOps.push_back(std::move(fusedOp));
  });
}

static bool isCudaTilePrimaryCandidate(CudaTileLoweringStrategy strategy) {
  return strategy == CudaTileLoweringStrategy::Matmul ||
         strategy == CudaTileLoweringStrategy::PointwiseConvAsMatmul ||
         strategy == CudaTileLoweringStrategy::DirectConv2D ||
         strategy == CudaTileLoweringStrategy::Pooling;
}

static void refreshPrimaryFacts(CudaTileKernelPlan &plan) {
  plan.srcShape.clear();
  plan.dstShape.clear();
  plan.elementType = {};
  plan.conv = {};

  if (!plan.primaryOp)
    return;

  if (auto classAttr =
          plan.primaryOp->getAttrOfType<StringAttr>("cuda_tile.kernel_class")) {
    plan.kernelClass = classAttr.getValue().str();
  }

  extractPrimaryShapeAndType(plan);
  if (auto genOp = dyn_cast<linalg::GenericOp>(plan.primaryOp)) {
    plan.conv = extractCudaTileConvPlan(genOp);
    if (!plan.conv)
      plan.conv = extractPoolingPlan(genOp);
  }
}

static void promotePrimaryFromFusedOps(CudaTileKernelPlan &plan) {
  for (const CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
    if (!isCudaTilePrimaryCandidate(fusedOp.loweringStrategy))
      continue;

    plan.primaryOp = fusedOp.op;
    plan.kind = fusedOp.kind;
    plan.semanticKind = fusedOp.semanticKind;
    plan.loweringStrategy = fusedOp.loweringStrategy;
    plan.conv = fusedOp.conv;
    switch (fusedOp.loweringStrategy) {
    case CudaTileLoweringStrategy::DirectConv2D:
      plan.kernelClass = "conv";
      break;
    case CudaTileLoweringStrategy::Pooling:
      plan.kernelClass = "pooling";
      break;
    default:
      plan.kernelClass = "matmul";
      break;
    }

    refreshPrimaryFacts(plan);
    return;
  }
}

static CudaTileFusedOpPlan *findFusedOpPlan(CudaTileKernelPlan &plan,
                                            Operation *op) {
  for (CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
    if (fusedOp.op == op)
      return &fusedOp;
  }
  return nullptr;
}

static void markPrologueInput(CudaTileFusedOpPlan &fusedOp,
                              int64_t primaryInputIndex) {
  if (fusedOp.role == CudaTileFusedOpRole::Primary)
    return;
  fusedOp.role = CudaTileFusedOpRole::Prologue;
  if (fusedOp.primaryInputIndex < 0) {
    fusedOp.primaryInputIndex = primaryInputIndex;
  } else if (fusedOp.primaryInputIndex != primaryInputIndex) {
    fusedOp.primaryInputIndex = -1;
  }
}

static void markPrologueChain(CudaTileKernelPlan &plan, Value value,
                              int64_t primaryInputIndex,
                              llvm::DenseSet<Operation *> &visited) {
  Operation *op = value.getDefiningOp();
  if (!op || !visited.insert(op).second)
    return;

  CudaTileFusedOpPlan *fusedOp = findFusedOpPlan(plan, op);
  if (!fusedOp || fusedOp->op == plan.primaryOp ||
      fusedOp->role == CudaTileFusedOpRole::Epilogue) {
    return;
  }

  markPrologueInput(*fusedOp, primaryInputIndex);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    for (Value input : linalgOp.getDpsInputs())
      markPrologueChain(plan, input, primaryInputIndex, visited);
  }
}

static void markEpilogueChain(CudaTileKernelPlan &plan, Operation *op,
                              llvm::DenseSet<Operation *> &visited) {
  if (!op || !visited.insert(op).second)
    return;

  CudaTileFusedOpPlan *fusedOp = findFusedOpPlan(plan, op);
  if (!fusedOp || fusedOp->op == plan.primaryOp ||
      fusedOp->role == CudaTileFusedOpRole::Prologue) {
    return;
  }

  fusedOp->role = CudaTileFusedOpRole::Epilogue;
  fusedOp->primaryInputIndex = -1;
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers())
      markEpilogueChain(plan, user, visited);
  }
}

static void classifyFusedOpRoles(CudaTileKernelPlan &plan) {
  for (CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
    fusedOp.role = CudaTileFusedOpRole::Unknown;
    fusedOp.primaryInputIndex = -1;
  }

  if (!plan.primaryOp)
    return;

  if (CudaTileFusedOpPlan *primaryFusedOp =
          findFusedOpPlan(plan, plan.primaryOp)) {
    primaryFusedOp->role = CudaTileFusedOpRole::Primary;
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(plan.primaryOp)) {
    auto dpsInputs = linalgOp.getDpsInputs();
    for (auto [index, input] : llvm::enumerate(dpsInputs)) {
      llvm::DenseSet<Operation *> visited;
      markPrologueChain(plan, input, index, visited);
    }
  }

  for (Value result : plan.primaryOp->getResults()) {
    llvm::DenseSet<Operation *> visited;
    for (Operation *user : result.getUsers())
      markEpilogueChain(plan, user, visited);
  }
}

static void incrementSemanticCount(CudaTileKernelPlan &plan,
                                   CudaTileSemanticKind semanticKind) {
  switch (semanticKind) {
  case CudaTileSemanticKind::Map:
    plan.mapOpCount++;
    break;
  case CudaTileSemanticKind::Reduction:
    plan.reductionOpCount++;
    break;
  case CudaTileSemanticKind::Contraction:
    plan.contractionOpCount++;
    break;
  case CudaTileSemanticKind::WindowedReduction:
    plan.windowedReductionOpCount++;
    break;
  default:
    break;
  }
}

static CudaTileSemanticKind getCountedDispatchSemanticKind(
    const CudaTileKernelPlan &plan, CudaTileSemanticKind fallback) {
  if (plan.contractionOpCount > 0)
    return CudaTileSemanticKind::Contraction;
  if (plan.windowedReductionOpCount > 0)
    return CudaTileSemanticKind::WindowedReduction;
  if (plan.reductionOpCount > 0 && plan.mapOpCount > 0)
    return CudaTileSemanticKind::FusedReductionElementwise;
  if (plan.reductionOpCount > 0)
    return CudaTileSemanticKind::Reduction;
  if (plan.mapOpCount > 0)
    return CudaTileSemanticKind::Map;
  return fallback;
}

static CudaTileLoweringStrategy
selectLoweringStrategy(StringRef kernelClass,
                       CudaTileSemanticKind semanticKind,
                       const CudaTileConvPlan &convPlan) {
  if (convPlan.mode == CudaTileConvLoweringMode::DirectConv2D)
    return CudaTileLoweringStrategy::DirectConv2D;
  if (convPlan.mode == CudaTileConvLoweringMode::PointwiseMatmul)
    return CudaTileLoweringStrategy::PointwiseConvAsMatmul;
  if (convPlan.mode == CudaTileConvLoweringMode::Pooling &&
      convPlan.spatialRank == 2)
    return CudaTileLoweringStrategy::Pooling;

  if (kernelClass == "copy")
    return CudaTileLoweringStrategy::Copy;
  if (kernelClass == "extract_slice")
    return CudaTileLoweringStrategy::ExtractSlice;
  if (kernelClass == "insert_slice")
    return CudaTileLoweringStrategy::InsertSlice;
  if (kernelClass == "transpose")
    return CudaTileLoweringStrategy::Transpose;
  if (kernelClass == "broadcast")
    return CudaTileLoweringStrategy::Broadcast;
  if (kernelClass == "collapse_shape" || kernelClass == "expand_shape")
    return CudaTileLoweringStrategy::ReshapeCopy;

  switch (semanticKind) {
  case CudaTileSemanticKind::DataMovement:
    return CudaTileLoweringStrategy::Copy;
  case CudaTileSemanticKind::Map:
    return CudaTileLoweringStrategy::Elementwise;
  case CudaTileSemanticKind::Reduction:
    return CudaTileLoweringStrategy::Reduction;
  case CudaTileSemanticKind::Contraction:
    return CudaTileLoweringStrategy::Matmul;
  case CudaTileSemanticKind::FusedReductionElementwise:
    return CudaTileLoweringStrategy::FusedGeneric;
  case CudaTileSemanticKind::WindowedReduction:
  case CudaTileSemanticKind::ControlFlow:
  case CudaTileSemanticKind::Unknown:
    return CudaTileLoweringStrategy::Unsupported;
  }
  return CudaTileLoweringStrategy::Unsupported;
}

static CudaTileLoweringStrategy
selectLoweringStrategy(const CudaTileKernelPlan &plan) {
  return selectLoweringStrategy(plan.kernelClass, plan.semanticKind,
                                plan.conv);
}

} // namespace

CudaTileKernelKind getCudaTileKernelKind(StringRef kernelClass) {
  return llvm::StringSwitch<CudaTileKernelKind>(kernelClass)
      .Case("copy", CudaTileKernelKind::Copy)
      .Case("extract_slice", CudaTileKernelKind::ExtractSlice)
      .Case("insert_slice", CudaTileKernelKind::InsertSlice)
      .Case("transpose", CudaTileKernelKind::Transpose)
      .Case("broadcast", CudaTileKernelKind::Broadcast)
      .Case("elementwise", CudaTileKernelKind::Elementwise)
      .Case("reduce", CudaTileKernelKind::Reduction)
      .Case("matmul", CudaTileKernelKind::Matmul)
      .Case("conv", CudaTileKernelKind::Conv)
      .Case("conv2d", CudaTileKernelKind::Conv)
      .Case("fused_reduction_elementwise",
            CudaTileKernelKind::FusedReductionElementwise)
      .Case("scf", CudaTileKernelKind::SCF)
      .Case("generic", CudaTileKernelKind::Generic)
      .Default(CudaTileKernelKind::Unsupported);
}

StringRef stringifyCudaTileKernelKind(CudaTileKernelKind kind) {
  switch (kind) {
  case CudaTileKernelKind::Unsupported:
    return "unsupported";
  case CudaTileKernelKind::Generic:
    return "generic";
  case CudaTileKernelKind::Copy:
    return "copy";
  case CudaTileKernelKind::ExtractSlice:
    return "extract_slice";
  case CudaTileKernelKind::InsertSlice:
    return "insert_slice";
  case CudaTileKernelKind::Transpose:
    return "transpose";
  case CudaTileKernelKind::Broadcast:
    return "broadcast";
  case CudaTileKernelKind::Elementwise:
    return "elementwise";
  case CudaTileKernelKind::Reduction:
    return "reduction";
  case CudaTileKernelKind::Matmul:
    return "matmul";
  case CudaTileKernelKind::Conv:
    return "conv";
  case CudaTileKernelKind::FusedReductionElementwise:
    return "fused_reduction_elementwise";
  case CudaTileKernelKind::SCF:
    return "scf";
  }
  return "unsupported";
}

CudaTileSemanticKind getCudaTileSemanticKind(StringRef kernelClass) {
  return llvm::StringSwitch<CudaTileSemanticKind>(kernelClass)
      .Case("copy", CudaTileSemanticKind::DataMovement)
      .Case("extract_slice", CudaTileSemanticKind::DataMovement)
      .Case("insert_slice", CudaTileSemanticKind::DataMovement)
      .Case("transpose", CudaTileSemanticKind::DataMovement)
      .Case("broadcast", CudaTileSemanticKind::DataMovement)
      .Case("collapse_shape", CudaTileSemanticKind::DataMovement)
      .Case("expand_shape", CudaTileSemanticKind::DataMovement)
      .Case("elementwise", CudaTileSemanticKind::Map)
      .Case("reduce", CudaTileSemanticKind::Reduction)
      .Case("matmul", CudaTileSemanticKind::Contraction)
      .Case("conv", CudaTileSemanticKind::WindowedReduction)
      .Case("conv2d", CudaTileSemanticKind::WindowedReduction)
      .Case("fused_reduction_elementwise",
            CudaTileSemanticKind::FusedReductionElementwise)
      .Case("scf", CudaTileSemanticKind::ControlFlow)
      .Default(CudaTileSemanticKind::Unknown);
}

StringRef stringifyCudaTileSemanticKind(CudaTileSemanticKind kind) {
  switch (kind) {
  case CudaTileSemanticKind::Unknown:
    return "unknown";
  case CudaTileSemanticKind::DataMovement:
    return "data_movement";
  case CudaTileSemanticKind::Map:
    return "map";
  case CudaTileSemanticKind::Reduction:
    return "reduction";
  case CudaTileSemanticKind::Contraction:
    return "contraction";
  case CudaTileSemanticKind::WindowedReduction:
    return "windowed_reduction";
  case CudaTileSemanticKind::FusedReductionElementwise:
    return "fused_reduction_elementwise";
  case CudaTileSemanticKind::ControlFlow:
    return "control_flow";
  }
  return "unknown";
}

StringRef stringifyCudaTileScheduleSource(CudaTileScheduleSource source) {
  switch (source) {
  case CudaTileScheduleSource::CommandLine:
    return "command_line";
  case CudaTileScheduleSource::IREECodegen:
    return "iree_codegen";
  case CudaTileScheduleSource::IREEGPU:
    return "iree_gpu";
  case CudaTileScheduleSource::CudaTileHint:
    return "cuda_tile_hint";
  }
  return "command_line";
}

StringRef stringifyCudaTileConvLoweringMode(CudaTileConvLoweringMode mode) {
  switch (mode) {
  case CudaTileConvLoweringMode::NotConv:
    return "not_conv";
  case CudaTileConvLoweringMode::PointwiseMatmul:
    return "pointwise_matmul";
  case CudaTileConvLoweringMode::DirectConv2D:
    return "direct_conv2d";
  case CudaTileConvLoweringMode::Pooling:
    return "pooling";
  }
  return "not_conv";
}

StringRef stringifyCudaTileOperandRole(CudaTileOperandRole role) {
  switch (role) {
  case CudaTileOperandRole::Unknown:
    return "unknown";
  case CudaTileOperandRole::Input:
    return "input";
  case CudaTileOperandRole::Init:
    return "init";
  case CudaTileOperandRole::Output:
    return "output";
  case CudaTileOperandRole::Result:
    return "result";
  }
  return "unknown";
}

StringRef stringifyCudaTileFusedOpRole(CudaTileFusedOpRole role) {
  switch (role) {
  case CudaTileFusedOpRole::Unknown:
    return "unknown";
  case CudaTileFusedOpRole::Primary:
    return "primary";
  case CudaTileFusedOpRole::Prologue:
    return "prologue";
  case CudaTileFusedOpRole::Epilogue:
    return "epilogue";
  }
  return "unknown";
}

StringRef stringifyCudaTileLoweringStrategy(CudaTileLoweringStrategy strategy) {
  switch (strategy) {
  case CudaTileLoweringStrategy::Unsupported:
    return "unsupported";
  case CudaTileLoweringStrategy::Copy:
    return "copy";
  case CudaTileLoweringStrategy::ExtractSlice:
    return "extract_slice";
  case CudaTileLoweringStrategy::InsertSlice:
    return "insert_slice";
  case CudaTileLoweringStrategy::Transpose:
    return "transpose";
  case CudaTileLoweringStrategy::Broadcast:
    return "broadcast";
  case CudaTileLoweringStrategy::ReshapeCopy:
    return "reshape_copy";
  case CudaTileLoweringStrategy::Elementwise:
    return "elementwise";
  case CudaTileLoweringStrategy::Reduction:
    return "reduction";
  case CudaTileLoweringStrategy::FusedGeneric:
    return "fused_generic";
  case CudaTileLoweringStrategy::Matmul:
    return "matmul";
  case CudaTileLoweringStrategy::PointwiseConvAsMatmul:
    return "pointwise_conv_as_matmul";
  case CudaTileLoweringStrategy::DirectConv2D:
    return "direct_conv2d";
  case CudaTileLoweringStrategy::Pooling:
    return "pooling";
  }
  return "unsupported";
}

CudaTileConvPlan extractCudaTileConvPlan(linalg::GenericOp genOp) {
  CudaTileConvPlan plan;
  if (!isMultiplyAddGeneric(genOp))
    return plan;

  auto maps = genOp.getIndexingMapsArray();
  if (maps.size() < 3)
    return plan;

  auto inputShape = getStaticShapeFromType(genOp.getDpsInputs()[0].getType());
  auto filterShape = getStaticShapeFromType(genOp.getDpsInputs()[1].getType());
  llvm::SmallVector<int64_t> outputShape;
  for (auto result : genOp.getResults()) {
    outputShape = getStaticShapeFromType(result.getType());
    if (!outputShape.empty())
      break;
  }
  if (outputShape.empty() && !genOp.getDpsInits().empty())
    outputShape = getStaticShapeFromType(genOp.getDpsInits()[0].getType());
  if (inputShape.empty() || filterShape.empty() || outputShape.empty())
    return plan;
  if (filterShape.size() < 3)
    return plan;

  int64_t spatialRank = static_cast<int64_t>(filterShape.size()) - 2;
  bool hasBatchDim =
      static_cast<int64_t>(outputShape.size()) == spatialRank + 2;
  bool droppedBatchDim =
      static_cast<int64_t>(outputShape.size()) == spatialRank + 1;
  if (!hasBatchDim && !droppedBatchDim)
    return plan;
  if (static_cast<int64_t>(inputShape.size()) !=
          (hasBatchDim ? spatialRank + 2 : spatialRank + 1) ||
      static_cast<int64_t>(filterShape.size()) != spatialRank + 2)
    return plan;

  AffineMap inputMap = maps[0];
  AffineMap filterMap = maps[1];
  AffineMap outputMap = maps[2];
  if (outputMap.getNumResults() != outputShape.size() ||
      filterMap.getNumResults() != filterShape.size() ||
      inputMap.getNumResults() != inputShape.size())
    return plan;

  llvm::SmallVector<int64_t> outputLoops;
  for (AffineExpr expr : outputMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return plan;
    outputLoops.push_back(dimExpr.getPosition());
  }

  llvm::SmallVector<int64_t> filterLoops;
  filterLoops.reserve(filterShape.size());
  for (auto [expr, extent] : llvm::zip(filterMap.getResults(), filterShape)) {
    int64_t loopPos = -1;
    if (!getLoopPositionOrUnitConstant(expr, extent, loopPos))
      return plan;
    filterLoops.push_back(loopPos);
  }

  int64_t batchLoop = hasBatchDim ? outputLoops.front() : -1;

  // Find output channel loop: the loop that appears in both filter and output
  // (excluding batch). This is layout-agnostic — works for both NHWC and NCHW.
  int64_t outChannelLoop = -1;
  for (int64_t fl : filterLoops) {
    if (fl < 0)
      continue;
    if (hasBatchDim && fl == batchLoop)
      continue;
    for (int64_t ol : outputLoops) {
      if (fl == ol) {
        outChannelLoop = fl;
        break;
      }
    }
    if (outChannelLoop >= 0)
      break;
  }
  if (outChannelLoop < 0)
    return plan;

  // Determine layout from output channel position.
  // NHWC: channel is last → outputLoops.back() == outChannelLoop
  // NCHW: channel is first (no batch) or second (with batch)
  int64_t outChannelOutputIdx = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(outputLoops.size()); ++i) {
    if (outputLoops[i] == outChannelLoop) {
      outChannelOutputIdx = i;
      break;
    }
  }
  bool isNCHW = false;
  if (outChannelOutputIdx == static_cast<int64_t>(outputLoops.size()) - 1)
    isNCHW = false;
  else if (outChannelOutputIdx == (hasBatchDim ? 1 : 0))
    isNCHW = true;
  else
    return plan;

  // Extract spatial output loops (everything that isn't batch or channel).
  llvm::SmallVector<int64_t> outputSpatialLoops;
  for (int64_t i = 0; i < static_cast<int64_t>(outputLoops.size()); ++i) {
    if (outputLoops[i] == batchLoop || outputLoops[i] == outChannelLoop)
      continue;
    outputSpatialLoops.push_back(outputLoops[i]);
  }
  if (static_cast<int64_t>(outputSpatialLoops.size()) != spatialRank)
    return plan;

  // Categorize filter loops into kernel spatial loops vs input channel.
  // Input channel appears as a standalone AffineDimExpr in the input map;
  // kernel spatial loops appear only in additive expressions (d_oh + d_kh).
  llvm::SmallVector<int64_t> kernelLoops;
  int64_t inputChannelLoop = -1;
  for (int64_t fl : filterLoops) {
    if (fl < 0 || fl == outChannelLoop)
      continue;
    bool isDirectInInput = false;
    for (unsigned r = 0; r < inputMap.getNumResults(); ++r) {
      auto dimExpr = dyn_cast<AffineDimExpr>(inputMap.getResult(r));
      if (dimExpr &&
          dimExpr.getPosition() == static_cast<unsigned>(fl)) {
        isDirectInInput = true;
        break;
      }
    }
    if (isDirectInInput && inputChannelLoop < 0)
      inputChannelLoop = fl;
    else
      kernelLoops.push_back(fl);
  }
  if (inputChannelLoop < 0 ||
      static_cast<int64_t>(kernelLoops.size()) != spatialRank)
    return plan;

  // Find input spatial dim positions by matching input map results that
  // contain additive expressions involving outputSpatialLoops and kernelLoops.
  llvm::SmallVector<int64_t> inputSpatialPositions;
  for (int64_t i = 0; i < spatialRank; ++i) {
    for (unsigned r = 0; r < inputMap.getNumResults(); ++r) {
      llvm::SmallVector<int64_t> coeffs(genOp.getNumLoops(), 0);
      int64_t constant = 0;
      if (!collectAffineCoefficients(inputMap.getResult(r), coeffs, constant))
        continue;
      if (constant != 0)
        continue;
      if (outputSpatialLoops[i] >= 0 &&
          coeffs[outputSpatialLoops[i]] > 0 && kernelLoops[i] >= 0 &&
          coeffs[kernelLoops[i]] > 0) {
        inputSpatialPositions.push_back(r);
        break;
      }
    }
  }
  if (static_cast<int64_t>(inputSpatialPositions.size()) != spatialRank)
    return plan;

  // Extract strides and dilations from the input spatial expressions.
  llvm::SmallVector<int64_t> strides(spatialRank, 1);
  llvm::SmallVector<int64_t> dilations(spatialRank, 1);
  for (int64_t i = 0; i < spatialRank; ++i) {
    llvm::SmallVector<int64_t> coeffs(genOp.getNumLoops(), 0);
    int64_t constant = 0;
    if (!collectAffineCoefficients(inputMap.getResult(inputSpatialPositions[i]),
                                   coeffs, constant) ||
        constant != 0)
      return plan;

    for (int64_t d = 0; d < static_cast<int64_t>(coeffs.size()); ++d) {
      bool isOutputLoop =
          outputSpatialLoops[i] >= 0 && d == outputSpatialLoops[i];
      bool isKernelLoop = kernelLoops[i] >= 0 && d == kernelLoops[i];
      if (!isOutputLoop && !isKernelLoop && coeffs[d] != 0)
        return plan;
    }

    int64_t stride =
        outputSpatialLoops[i] >= 0 ? coeffs[outputSpatialLoops[i]] : 1;
    int64_t dilation =
        kernelLoops[i] >= 0 ? coeffs[kernelLoops[i]] : 1;
    if (stride <= 0 || dilation <= 0)
      return plan;

    strides[i] = stride;
    dilations[i] = dilation;
  }

  // Validate batch dimension in input map.
  if (hasBatchDim) {
    int64_t inputBatchLoop = -1;
    if (!getLoopPositionOrUnitConstant(inputMap.getResult(0), inputShape[0],
                                       inputBatchLoop))
      return plan;
    if (batchLoop >= 0) {
      if (inputBatchLoop != batchLoop)
        return plan;
    } else if (inputBatchLoop >= 0) {
      batchLoop = inputBatchLoop;
    } else {
      return plan;
    }
  }

  // Validate input channel dimension in input map.
  int64_t inputChannelPos =
      isNCHW ? (hasBatchDim ? 1 : 0) : (hasBatchDim ? spatialRank + 1
                                                     : spatialRank);
  auto inputChannelExpr = dyn_cast<AffineDimExpr>(
      inputMap.getResult(inputChannelPos));
  if (!inputChannelExpr || inputChannelExpr.getPosition() !=
                               static_cast<unsigned>(inputChannelLoop))
    return plan;

  plan.spatialRank = spatialRank;
  plan.strides = std::move(strides);
  plan.dilations = std::move(dilations);
  plan.isNCHW = isNCHW;

  // Determine which filter dims are spatial (for isPointwise check).
  // kernelLoops contains the iteration-space positions of kernel spatial dims.
  // Map them back to filter tensor dims.
  llvm::SmallVector<int64_t> filterSpatialExtents;
  for (int64_t kl : kernelLoops) {
    for (int64_t fi = 0; fi < static_cast<int64_t>(filterLoops.size()); ++fi) {
      if (filterLoops[fi] == kl) {
        filterSpatialExtents.push_back(filterShape[fi]);
        break;
      }
    }
  }
  plan.isPointwise =
      llvm::all_of(filterSpatialExtents,
                   [](int64_t dim) { return dim == 1; });

  // Normalize shapes to NHWC order: [N, spatial..., C].
  // The kernel generator always reads shapes in NHWC format.
  if (isNCHW) {
    // Input: NCHW → NHWC (or CHW → 1,H,W,C if no batch)
    llvm::SmallVector<int64_t> nhwcInput;
    if (hasBatchDim) {
      nhwcInput.push_back(inputShape[0]); // N
      for (int64_t i = 2; i < static_cast<int64_t>(inputShape.size()); ++i)
        nhwcInput.push_back(inputShape[i]); // H, W
      nhwcInput.push_back(inputShape[1]); // C
    } else {
      nhwcInput.push_back(1); // N
      for (int64_t i = 1; i < static_cast<int64_t>(inputShape.size()); ++i)
        nhwcInput.push_back(inputShape[i]); // H, W
      nhwcInput.push_back(inputShape[0]); // C
    }
    plan.inputShape = std::move(nhwcInput);

    // Output: NCHW → NHWC (or CHW → 1,H,W,C if no batch)
    llvm::SmallVector<int64_t> nhwcOutput;
    if (hasBatchDim) {
      nhwcOutput.push_back(outputShape[0]); // N
      for (int64_t i = 2; i < static_cast<int64_t>(outputShape.size()); ++i)
        nhwcOutput.push_back(outputShape[i]); // OH, OW
      nhwcOutput.push_back(outputShape[1]); // C
    } else {
      nhwcOutput.push_back(1); // N
      for (int64_t i = 1; i < static_cast<int64_t>(outputShape.size()); ++i)
        nhwcOutput.push_back(outputShape[i]); // OH, OW
      nhwcOutput.push_back(outputShape[0]); // C
    }
    plan.outputShape = std::move(nhwcOutput);

    // Filter: FCHW (OC,IC,KH,KW) → HWIO (KH,KW,IC,OC)
    llvm::SmallVector<int64_t> hwioFilter;
    for (int64_t i = 2; i < static_cast<int64_t>(filterShape.size()); ++i)
      hwioFilter.push_back(filterShape[i]); // KH, KW
    hwioFilter.push_back(filterShape[1]); // IC
    hwioFilter.push_back(filterShape[0]); // OC
    plan.filterShape = std::move(hwioFilter);
  } else {
    plan.filterShape = llvm::SmallVector<int64_t>(filterShape);
    if (hasBatchDim) {
      plan.inputShape = llvm::SmallVector<int64_t>(inputShape);
      plan.outputShape = llvm::SmallVector<int64_t>(outputShape);
    } else {
      plan.inputShape.push_back(1);
      plan.inputShape.append(inputShape.begin(), inputShape.end());
      plan.outputShape.push_back(1);
      plan.outputShape.append(outputShape.begin(), outputShape.end());
    }
  }

  // Store physical shapes (actual memory layout) for stride computation.
  if (isNCHW) {
    if (hasBatchDim) {
      plan.physInputShape = llvm::SmallVector<int64_t>(inputShape);
      plan.physOutputShape = llvm::SmallVector<int64_t>(outputShape);
    } else {
      plan.physInputShape.push_back(1);
      plan.physInputShape.append(inputShape.begin(), inputShape.end());
      plan.physOutputShape.push_back(1);
      plan.physOutputShape.append(outputShape.begin(), outputShape.end());
    }
    plan.physFilterShape = llvm::SmallVector<int64_t>(filterShape);
  } else {
    plan.physInputShape = plan.inputShape;
    plan.physFilterShape = plan.filterShape;
    plan.physOutputShape = plan.outputShape;
  }

  if (plan.isPointwise && hasUnitSteps(plan.strides) &&
      hasUnitSteps(plan.dilations)) {
    plan.mode = CudaTileConvLoweringMode::PointwiseMatmul;
  } else if (plan.spatialRank == 2) {
    plan.mode = CudaTileConvLoweringMode::DirectConv2D;
  }

  return plan;
}

CudaTileConvPlan extractPoolingPlan(linalg::GenericOp genOp) {
  CudaTileConvPlan plan;

  if (genOp.getNumDpsInputs() < 1 || genOp.getNumDpsInits() < 1)
    return plan;
  if (genOp.getNumReductionLoops() < 1)
    return plan;

  // Pooling body: single combiner op applied to (accumulator, input).
  Block &body = genOp.getRegion().front();
  std::string combiner;
  for (auto &op : body.without_terminator()) {
    if (isa<arith::MaximumFOp, arith::MaxNumFOp>(&op))
      combiner = "maxf";
    else if (isa<arith::AddFOp>(&op))
      combiner = "addf";
    else if (isa<arith::MinimumFOp, arith::MinNumFOp>(&op))
      combiner = "minf";
  }
  if (combiner.empty())
    return plan;

  auto maps = genOp.getIndexingMapsArray();
  if (maps.size() < 2)
    return plan;

  AffineMap inputMap = maps[0];
  AffineMap outputMap = maps.back();

  auto inputShape = getStaticShapeFromType(genOp.getDpsInputs()[0].getType());
  llvm::SmallVector<int64_t> outputShape;
  for (auto result : genOp.getResults()) {
    outputShape = getStaticShapeFromType(result.getType());
    if (!outputShape.empty())
      break;
  }
  if (outputShape.empty() && !genOp.getDpsInits().empty())
    outputShape = getStaticShapeFromType(genOp.getDpsInits()[0].getType());
  if (inputShape.empty() || outputShape.empty())
    return plan;

  // Extract output loop positions.
  llvm::SmallVector<int64_t> outputLoops;
  for (AffineExpr expr : outputMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return plan;
    outputLoops.push_back(dimExpr.getPosition());
  }

  auto iterTypes = genOp.getIteratorTypesArray();
  int64_t numParallel = 0;
  int64_t numReduction = 0;
  for (auto it : iterTypes) {
    if (it == mlir::utils::IteratorType::parallel)
      numParallel++;
    else
      numReduction++;
  }
  int64_t spatialRank = numReduction;
  if (spatialRank < 1 || spatialRank > 3)
    return plan;

  // Find spatial output loops and channel loop.
  // Pooling has the channel dimension as a pass-through parallel dim.
  // NCHW: output = [C, OH, OW], channel is first
  // NHWC: output = [OH, OW, C], channel is last
  // In both cases, the channel dim does NOT appear in additive input map exprs.
  llvm::SmallVector<int64_t> spatialOutputLoops;
  llvm::SmallVector<int64_t> reductionLoops;
  for (int64_t d = 0; d < static_cast<int64_t>(iterTypes.size()); ++d) {
    if (iterTypes[d] == mlir::utils::IteratorType::reduction)
      reductionLoops.push_back(d);
  }
  // Identify spatial output dims: those that appear in input map additive exprs
  // with a reduction dim.
  llvm::SmallVector<int64_t> outputSpatialPositions;
  for (auto [outPos, ol] : llvm::enumerate(outputLoops)) {
    bool isSpatial = false;
    for (unsigned r = 0; r < inputMap.getNumResults(); ++r) {
      llvm::SmallVector<int64_t> coeffs(genOp.getNumLoops(), 0);
      int64_t constant = 0;
      if (!collectAffineCoefficients(inputMap.getResult(r), coeffs, constant))
        continue;
      if (coeffs[ol] > 0) {
        for (int64_t rl : reductionLoops) {
          if (coeffs[rl] > 0) {
            isSpatial = true;
            break;
          }
        }
      }
      if (isSpatial)
        break;
    }
    if (isSpatial) {
      spatialOutputLoops.push_back(ol);
      outputSpatialPositions.push_back(outPos);
    }
  }

  if (static_cast<int64_t>(spatialOutputLoops.size()) != spatialRank)
    return plan;

  // Determine layout. The non-spatial output dims are batch/channel.
  // If channel comes before spatial → NCHW; after → NHWC.
  int64_t channelOutputIdx = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(outputLoops.size()); ++i) {
    bool isSpatial = llvm::is_contained(spatialOutputLoops, outputLoops[i]);
    if (!isSpatial && outputLoops[i] != outputLoops.front()) {
      channelOutputIdx = i;
    }
  }
  bool hasBatchDim =
      static_cast<int64_t>(outputLoops.size()) > spatialRank + 1;
  bool isNCHW = false;
  if (channelOutputIdx >= 0) {
    int64_t firstSpatialIdx = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(outputLoops.size()); ++i) {
      if (llvm::is_contained(spatialOutputLoops, outputLoops[i])) {
        firstSpatialIdx = i;
        break;
      }
    }
    isNCHW = channelOutputIdx < firstSpatialIdx;
  } else if (static_cast<int64_t>(outputLoops.size()) == spatialRank + 1) {
    isNCHW = true;
  }

  // Pair spatial output loops with reduction (kernel) loops via input map.
  llvm::SmallVector<int64_t> pairedKernelLoops;
  llvm::SmallVector<int64_t> inputSpatialPositions;
  llvm::SmallVector<int64_t> strides(spatialRank, 1);
  llvm::SmallVector<int64_t> dilations(spatialRank, 1);

  for (int64_t i = 0; i < spatialRank; ++i) {
    for (unsigned r = 0; r < inputMap.getNumResults(); ++r) {
      llvm::SmallVector<int64_t> coeffs(genOp.getNumLoops(), 0);
      int64_t constant = 0;
      if (!collectAffineCoefficients(inputMap.getResult(r), coeffs, constant))
        continue;
      if (constant != 0 || coeffs[spatialOutputLoops[i]] <= 0)
        continue;
      for (int64_t rl : reductionLoops) {
        if (coeffs[rl] > 0) {
          pairedKernelLoops.push_back(rl);
          inputSpatialPositions.push_back(r);
          strides[i] = coeffs[spatialOutputLoops[i]];
          dilations[i] = coeffs[rl];
          break;
        }
      }
      if (static_cast<int64_t>(pairedKernelLoops.size()) > i)
        break;
    }
  }
  if (static_cast<int64_t>(pairedKernelLoops.size()) != spatialRank)
    return plan;

  // Extract window shape from the window/kernel tensor input (second input).
  llvm::SmallVector<int64_t> windowShape;
  if (genOp.getNumDpsInputs() >= 2) {
    auto wShape =
        getStaticShapeFromType(genOp.getDpsInputs()[1].getType());
    windowShape = llvm::SmallVector<int64_t>(wShape);
  }
  if (windowShape.empty()) {
    for (int64_t i = 0; i < spatialRank; ++i) {
      int64_t inputExtent = inputShape[inputSpatialPositions[i]];
      int64_t outputExtent = outputShape[outputSpatialPositions[i]];
      int64_t remainder =
          inputExtent - (outputExtent - 1) * strides[i] - 1;
      if (dilations[i] <= 0 || remainder < 0 ||
          remainder % dilations[i] != 0)
        return plan;
      int64_t windowExtent = remainder / dilations[i] + 1;
      if (windowExtent <= 0)
        return plan;
      windowShape.push_back(windowExtent);
    }
  }
  if (static_cast<int64_t>(windowShape.size()) != spatialRank)
    return plan;

  plan.spatialRank = spatialRank;
  plan.strides = std::move(strides);
  plan.dilations = std::move(dilations);
  plan.windowShape = std::move(windowShape);
  plan.combiner = combiner;
  plan.isNCHW = isNCHW;
  plan.mode = CudaTileConvLoweringMode::Pooling;

  // Normalize shapes to NHWC order.
  if (isNCHW) {
    llvm::SmallVector<int64_t> nhwcIn, nhwcOut;
    if (hasBatchDim) {
      nhwcIn.push_back(inputShape[0]);
      for (int64_t i = 2; i < static_cast<int64_t>(inputShape.size()); ++i)
        nhwcIn.push_back(inputShape[i]);
      nhwcIn.push_back(inputShape[1]);
      nhwcOut.push_back(outputShape[0]);
      for (int64_t i = 2; i < static_cast<int64_t>(outputShape.size()); ++i)
        nhwcOut.push_back(outputShape[i]);
      nhwcOut.push_back(outputShape[1]);
    } else {
      nhwcIn.push_back(1);
      for (int64_t i = 1; i < static_cast<int64_t>(inputShape.size()); ++i)
        nhwcIn.push_back(inputShape[i]);
      nhwcIn.push_back(inputShape[0]);
      nhwcOut.push_back(1);
      for (int64_t i = 1; i < static_cast<int64_t>(outputShape.size()); ++i)
        nhwcOut.push_back(outputShape[i]);
      nhwcOut.push_back(outputShape[0]);
    }
    plan.inputShape = std::move(nhwcIn);
    plan.outputShape = std::move(nhwcOut);
    if (hasBatchDim) {
      plan.physInputShape = llvm::SmallVector<int64_t>(inputShape);
      plan.physOutputShape = llvm::SmallVector<int64_t>(outputShape);
    } else {
      plan.physInputShape.push_back(1);
      plan.physInputShape.append(inputShape.begin(), inputShape.end());
      plan.physOutputShape.push_back(1);
      plan.physOutputShape.append(outputShape.begin(), outputShape.end());
    }
  } else {
    if (hasBatchDim) {
      plan.inputShape = llvm::SmallVector<int64_t>(inputShape);
      plan.outputShape = llvm::SmallVector<int64_t>(outputShape);
    } else {
      plan.inputShape.push_back(1);
      plan.inputShape.append(inputShape.begin(), inputShape.end());
      plan.outputShape.push_back(1);
      plan.outputShape.append(outputShape.begin(), outputShape.end());
    }
    plan.physInputShape = plan.inputShape;
    plan.physOutputShape = plan.outputShape;
  }

  return plan;
}

void printCudaTileKernelPlan(const CudaTileKernelPlan &plan,
                             llvm::raw_ostream &os) {
  os << "cuda_tile.kernel_plan {\n";
  os << "  kernel_class = \"" << plan.kernelClass << "\"\n";
  os << "  kind = " << stringifyCudaTileKernelKind(plan.kind) << "\n";
  os << "  semantic = " << stringifyCudaTileSemanticKind(plan.semanticKind)
     << "\n";
  os << "  lowering_strategy = "
     << stringifyCudaTileLoweringStrategy(plan.loweringStrategy) << "\n";
  os << "  primary_op = ";
  printOpName(plan.primaryOp, os);
  os << "\n";
  os << "  tagged_op_count = " << plan.taggedOpCount << "\n";
  os << "  generic_op_count = " << plan.genericOpCount << "\n";
  os << "  op_counts = {map = " << plan.mapOpCount
     << ", reduction = " << plan.reductionOpCount
     << ", contraction = " << plan.contractionOpCount
     << ", windowed_reduction = " << plan.windowedReductionOpCount << "}\n";

  os << "  fused_ops = [\n";
  for (const CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
    os << "    {op = ";
    printOpName(fusedOp.op, os);
    os << ", kind = " << stringifyCudaTileKernelKind(fusedOp.kind)
       << ", semantic = "
       << stringifyCudaTileSemanticKind(fusedOp.semanticKind)
       << ", lowering_strategy = "
       << stringifyCudaTileLoweringStrategy(fusedOp.loweringStrategy)
       << ", role = " << stringifyCudaTileFusedOpRole(fusedOp.role)
       << ", primary_input_index = " << fusedOp.primaryInputIndex
       << ", reduction_dims = ";
    printI64Array(fusedOp.reductionDims, os);
    os << ", conv_mode = "
       << stringifyCudaTileConvLoweringMode(fusedOp.conv.mode) << "}\n";
  }
  os << "  ]\n";

  os << "  shapes = {src = ";
  printI64Array(plan.srcShape, os);
  os << ", dst = ";
  printI64Array(plan.dstShape, os);
  os << ", element_type = ";
  if (plan.elementType)
    plan.elementType.print(os);
  else
    os << "<none>";
  os << "}\n";

  os << "  loops = {parallel = ";
  printI64Array(plan.parallelLoopDims, os);
  os << ", reduction = ";
  printI64Array(plan.reductionLoopDims, os);
  os << ", tensor_reduction_dims = ";
  printI64Array(plan.reductionDims, os);
  os << "}\n";

  os << "  conv = {mode = "
     << stringifyCudaTileConvLoweringMode(plan.conv.mode)
     << ", spatial_rank = " << plan.conv.spatialRank << ", input_shape = ";
  printI64Array(plan.conv.inputShape, os);
  os << ", filter_shape = ";
  printI64Array(plan.conv.filterShape, os);
  os << ", output_shape = ";
  printI64Array(plan.conv.outputShape, os);
  os << ", strides = ";
  printI64Array(plan.conv.strides, os);
  os << ", dilations = ";
  printI64Array(plan.conv.dilations, os);
  os << ", window_shape = ";
  printI64Array(plan.conv.windowShape, os);
  os << ", combiner = \"" << plan.conv.combiner << "\"";
  os << ", is_nchw = ";
  printBool(plan.conv.isNCHW, os);
  os << ", is_pointwise = ";
  printBool(plan.conv.isPointwise, os);
  os << "}\n";

  const CudaTileContractionPlan &contraction = plan.contraction;
  os << "  contraction = {valid = ";
  printBool(contraction.isValid, os);
  os << ", m = " << contraction.m << ", n = " << contraction.n
     << ", k = " << contraction.k << ", batch = " << contraction.batch
     << ", lhs_shape = ";
  printI64Array(contraction.lhsShape, os);
  os << ", rhs_shape = ";
  printI64Array(contraction.rhsShape, os);
  os << ", result_shape = ";
  printI64Array(contraction.resultShape, os);
  os << ", lhs_binding = " << contraction.lhsBinding
     << ", rhs_binding = " << contraction.rhsBinding
     << ", result_binding = " << contraction.resultBinding
     << ", constant_rhs = ";
  printBool(static_cast<bool>(contraction.constantRhs), os);
  os << ", rhs_transposed = ";
  printBool(contraction.rhsTransposed, os);
  os << ", rank_flattening = ";
  printBool(contraction.hasRankFlattening, os);
  os << ", can_use_mma = ";
  printBool(contraction.canUseMma, os);
  os << ", has_schedule_tiles = ";
  printBool(contraction.hasScheduleTiles, os);
  os << ", schedule_tiles = [";
  if (contraction.hasScheduleTiles)
    os << contraction.tileM << ", " << contraction.tileN << ", "
       << contraction.tileK;
  os << "], schedule_source = "
     << stringifyCudaTileScheduleSource(contraction.scheduleSource) << "}\n";

  const CudaTileSchedulePlan &schedule = plan.schedule;
  os << "  schedule = {source = "
     << stringifyCudaTileScheduleSource(schedule.source)
     << ", fallback_tiles = [" << schedule.tileM << ", " << schedule.tileN
     << ", " << schedule.tileK << "], workgroup_tiles = ";
  printI64Array(schedule.workgroupTileSizes, os);
  os << ", workgroup_size = ";
  printI64Array(schedule.workgroupSize, os);
  os << ", subgroup_size = " << schedule.subgroupSize
     << ", has_lowering_config = ";
  printBool(schedule.hasLoweringConfig, os);
  os << ", has_iree_gpu_lowering_config = ";
  printBool(schedule.hasIREEGPULoweringConfig, os);
  os << ", has_translation_info = ";
  printBool(schedule.hasTranslationInfo, os);
  os << ", mma_kind = ";
  printOptionalAttr(schedule.mmaKind, os);
  os << "}\n";

  os << "  bindings = [\n";
  for (const CudaTileBindingPlan &binding : plan.bindingShapes) {
    os << "    {binding = " << binding.binding << ", shape = ";
    printI64Array(binding.shape, os);
    os << ", byte_offset = ";
    if (binding.hasStaticByteOffset)
      os << binding.byteOffset;
    else
      os << "<dynamic>";
    os << "}\n";
  }
  os << "  ]\n";

  os << "  operands = [\n";
  for (const CudaTileOperandPlan &operand : plan.operands) {
    os << "    {role = " << stringifyCudaTileOperandRole(operand.role)
       << ", owner = ";
    printOpName(operand.owner, os);
    os << ", binding = " << operand.binding << ", logical_shape = ";
    printI64Array(operand.logicalShape, os);
    os << ", physical_shape = ";
    printI64Array(operand.physicalShape, os);
    os << ", offsets = ";
    printI64Array(operand.offsets, os);
    os << ", sizes = ";
    printI64Array(operand.sizes, os);
    os << ", strides = ";
    printI64Array(operand.strides, os);
    os << ", dispatch_load = ";
    printBool(operand.isDispatchLoad, os);
    os << ", dispatch_store = ";
    printBool(operand.isDispatchStore, os);
    os << ", constant = ";
    printBool(operand.isConstant, os);
    os << "}\n";
  }
  os << "  ]\n";
  os << "}\n";
}

CudaTileKernelPlan extractCudaTileKernelPlan(Operation *innerModule,
                                             const CudaTileOptions &options) {
  CudaTileKernelPlan plan;
  plan.schedule.tileM = options.tileM;
  plan.schedule.tileN = options.tileN;
  plan.schedule.tileK = options.tileK;

  innerModule->walk([&](Operation *op) {
    if (op->hasAttr("cuda_tile.kernel_class")) {
      plan.primaryOp = op;
      plan.taggedOpCount++;
      plan.taggedOps.push_back(op);
    }
  });

  if (!plan.primaryOp) {
    Operation *fallbackOp = nullptr;
    innerModule->walk([&](linalg::LinalgOp op) {
      if (plan.primaryOp)
        return;
      if (isa<linalg::FillOp>(op.getOperation())) {
        if (!fallbackOp)
          fallbackOp = op;
        return;
      }
      plan.primaryOp = op;
    });
    if (!plan.primaryOp)
      plan.primaryOp = fallbackOp;
  }

  innerModule->walk([&](linalg::GenericOp) { plan.genericOpCount++; });

  innerModule->walk([&](Operation *op) {
    if (auto subspan = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(op)) {
      CudaTileBindingPlan binding;
      binding.binding = subspan.getBinding().getSExtValue();
      binding.memref = op->getResult(0);
      binding.shape = getStaticShapeFromType(op->getResult(0).getType());
      std::optional<int64_t> byteOffset =
          getStaticIndexValue(subspan.getByteOffset());
      if (byteOffset) {
        binding.byteOffset = *byteOffset;
      } else {
        binding.hasStaticByteOffset = false;
      }
      plan.bindingShapes.push_back(std::move(binding));
    }
  });

  innerModule->walk([&](IREE::TensorExt::DispatchTensorLoadOp op) {
    if (plan.singleLoadOp)
      plan.sawMultipleLoads = true;
    else
      plan.singleLoadOp = op;
  });
  innerModule->walk([&](IREE::TensorExt::DispatchTensorStoreOp op) {
    if (plan.singleStoreOp)
      plan.sawMultipleStores = true;
    else
      plan.singleStoreOp = op;
  });

  populateDispatchTensorOperandPlans(innerModule, plan);

  if (!plan.primaryOp) {
    innerModule->walk([&](Operation *op) {
      if (!plan.copyFallbackShape.empty())
        return;
      for (Value result : op->getResults()) {
        plan.copyFallbackShape = getStaticShapeFromType(result.getType());
        plan.copyFallbackElementType = getElementType(result.getType());
        if (!plan.copyFallbackShape.empty() && plan.copyFallbackElementType)
          return;
      }
    });
    plan.kind = CudaTileKernelKind::Copy;
    plan.kernelClass = "copy";
    plan.semanticKind = CudaTileSemanticKind::DataMovement;
    plan.loweringStrategy = CudaTileLoweringStrategy::Copy;
    return plan;
  }

  if (auto classAttr =
          plan.primaryOp->getAttrOfType<StringAttr>("cuda_tile.kernel_class")) {
    plan.kernelClass = classAttr.getValue().str();
  }
  plan.kind = getCudaTileKernelKind(plan.kernelClass);
  extractPrimaryShapeAndType(plan);
  if (auto genOp = dyn_cast<linalg::GenericOp>(plan.primaryOp)) {
    plan.conv = extractCudaTileConvPlan(genOp);
    if (!plan.conv)
      plan.conv = extractPoolingPlan(genOp);
  }
  plan.semanticKind = getCudaTileSemanticKind(plan.kernelClass);

  populateFusedOpPlans(innerModule, plan);
  promotePrimaryFromFusedOps(plan);
  classifyFusedOpRoles(plan);
  populatePrimaryOperandPlans(plan);

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(plan.primaryOp)) {
    if (plan.semanticKind == CudaTileSemanticKind::Unknown)
      plan.semanticKind = classifyLinalgOp(linalgOp, plan.conv);
    recordLoopDims(linalgOp, plan);
    plan.reductionDims = extractTensorReductionDims(linalgOp);
  }

  plan.semanticKind =
      getCountedDispatchSemanticKind(plan, plan.semanticKind);
  populateSchedulePlan(innerModule, plan);
  populateContractionPlan(plan);
  plan.loweringStrategy = selectLoweringStrategy(plan);
  return plan;
}

} // namespace mlir::iree_compiler::IREE::HAL
