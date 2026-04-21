// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MERLIN_COMPILER_PLUGINS_TARGET_CUDATILE_KERNEL_PLAN_H_
#define MERLIN_COMPILER_PLUGINS_TARGET_CUDATILE_KERNEL_PLAN_H_

#include <cstdint>
#include <string>

#include "compiler/plugins/target/CudaTile/CudaTileOptions.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::IREE::HAL {

enum class CudaTileKernelKind : uint8_t {
  Unsupported,
  Generic,
  Copy,
  ExtractSlice,
  InsertSlice,
  Transpose,
  Broadcast,
  Elementwise,
  Reduction,
  Matmul,
  Conv,
  FusedReductionElementwise,
  SCF,
};

enum class CudaTileSemanticKind : uint8_t {
  Unknown,
  DataMovement,
  Map,
  Reduction,
  Contraction,
  WindowedReduction,
  FusedReductionElementwise,
  ControlFlow,
};

enum class CudaTileConvLoweringMode : uint8_t {
  NotConv,
  PointwiseMatmul,
  DirectConv2D,
};

enum class CudaTileOperandRole : uint8_t {
  Unknown,
  Input,
  Init,
  Output,
  Result,
};

enum class CudaTileScheduleSource : uint8_t {
  CommandLine,
  IREECodegen,
  IREEGPU,
  CudaTileHint,
};

enum class CudaTileLoweringStrategy : uint8_t {
  Unsupported,
  Copy,
  ExtractSlice,
  InsertSlice,
  Transpose,
  Broadcast,
  ReshapeCopy,
  Elementwise,
  Reduction,
  FusedGeneric,
  Matmul,
  PointwiseConvAsMatmul,
  DirectConv2D,
};

struct CudaTileConvPlan {
  CudaTileConvLoweringMode mode = CudaTileConvLoweringMode::NotConv;
  int64_t spatialRank = 0;
  llvm::SmallVector<int64_t> inputShape;
  llvm::SmallVector<int64_t> filterShape;
  llvm::SmallVector<int64_t> outputShape;
  llvm::SmallVector<int64_t> strides;
  llvm::SmallVector<int64_t> dilations;
  bool isPointwise = false;

  explicit operator bool() const {
    return mode != CudaTileConvLoweringMode::NotConv;
  }
};

struct CudaTileBindingPlan {
  int64_t binding = -1;
  llvm::SmallVector<int64_t> shape;
  Value memref;
};

struct CudaTileOperandPlan {
  CudaTileOperandRole role = CudaTileOperandRole::Unknown;
  Operation *owner = nullptr;
  Value value;
  int64_t binding = -1;

  llvm::SmallVector<int64_t> logicalShape;
  llvm::SmallVector<int64_t> physicalShape;
  llvm::SmallVector<int64_t> offsets;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> strides;

  bool isDispatchLoad = false;
  bool isDispatchStore = false;
  bool isConstant = false;
};

struct CudaTileFusedOpPlan {
  Operation *op = nullptr;
  CudaTileKernelKind kind = CudaTileKernelKind::Unsupported;
  CudaTileSemanticKind semanticKind = CudaTileSemanticKind::Unknown;
  CudaTileLoweringStrategy loweringStrategy =
      CudaTileLoweringStrategy::Unsupported;
  CudaTileConvPlan conv;
  llvm::SmallVector<int64_t> reductionDims;
};

struct CudaTileContractionPlan {
  bool isValid = false;
  Operation *op = nullptr;

  llvm::SmallVector<int64_t> lhsShape;
  llvm::SmallVector<int64_t> rhsShape;
  llvm::SmallVector<int64_t> resultShape;

  int64_t m = 1;
  int64_t n = 1;
  int64_t k = 1;
  int64_t batch = 1;

  int64_t lhsBinding = -1;
  int64_t rhsBinding = -1;
  int64_t resultBinding = -1;
  Value constantRhs;

  bool lhsTransposed = false;
  bool rhsTransposed = false;
  bool hasRankFlattening = false;
  bool canUseMma = false;

  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  bool hasScheduleTiles = false;
  CudaTileScheduleSource scheduleSource = CudaTileScheduleSource::CommandLine;

  bool hasSlicedLhs = false;
  llvm::SmallVector<int64_t> slicedLhsSourceShape;
  llvm::SmallVector<int64_t> slicedLhsOffsets;
  llvm::SmallVector<int64_t> slicedLhsSizes;
  llvm::SmallVector<int64_t> slicedLhsStrides;
};

struct CudaTileSchedulePlan {
  CudaTileScheduleSource source = CudaTileScheduleSource::CommandLine;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  llvm::SmallVector<int64_t, 3> gridDims;

  llvm::SmallVector<int64_t> workgroupTileSizes;
  llvm::SmallVector<llvm::SmallVector<int64_t>, 4> tilingLevelSizes;
  llvm::SmallVector<int64_t, 3> workgroupSize;
  int64_t subgroupSize = 0;

  bool hasLoweringConfig = false;
  bool hasIREEGPULoweringConfig = false;
  bool hasTranslationInfo = false;

  Attribute target;
  Attribute loweringConfig;
  Attribute translationInfo;
  Attribute mmaKind;
};

struct CudaTileAsyncPlan {
  bool hasHints = false;
  bool allowTma = false;
  bool useAsyncCopies = false;
  int64_t pipelineDepth = 0;
};

struct CudaTileKernelPlan {
  CudaTileKernelKind kind = CudaTileKernelKind::Unsupported;
  CudaTileSemanticKind semanticKind = CudaTileSemanticKind::Unknown;
  CudaTileLoweringStrategy loweringStrategy =
      CudaTileLoweringStrategy::Unsupported;
  Operation *primaryOp = nullptr;
  std::string kernelClass = "generic";

  int taggedOpCount = 0;
  int genericOpCount = 0;
  int mapOpCount = 0;
  int reductionOpCount = 0;
  int contractionOpCount = 0;
  int windowedReductionOpCount = 0;
  llvm::SmallVector<Operation *> taggedOps;
  llvm::SmallVector<CudaTileFusedOpPlan, 4> fusedOps;
  llvm::SmallVector<int64_t> parallelLoopDims;
  llvm::SmallVector<int64_t> reductionLoopDims;
  llvm::SmallVector<int64_t> reductionDims;

  llvm::SmallVector<int64_t> srcShape;
  llvm::SmallVector<int64_t> dstShape;
  Type elementType;

  llvm::SmallVector<CudaTileBindingPlan> bindingShapes;
  llvm::SmallVector<CudaTileOperandPlan, 8> operands;
  CudaTileConvPlan conv;
  CudaTileContractionPlan contraction;
  CudaTileSchedulePlan schedule;
  CudaTileAsyncPlan asyncPlan;

  // Pure data-movement fallback facts for dispatches with no linalg op.
  IREE::TensorExt::DispatchTensorLoadOp singleLoadOp;
  IREE::TensorExt::DispatchTensorStoreOp singleStoreOp;
  bool sawMultipleLoads = false;
  bool sawMultipleStores = false;
  llvm::SmallVector<int64_t> copyFallbackShape;
  Type copyFallbackElementType;
};

CudaTileKernelKind getCudaTileKernelKind(StringRef kernelClass);

StringRef stringifyCudaTileKernelKind(CudaTileKernelKind kind);

CudaTileSemanticKind getCudaTileSemanticKind(StringRef kernelClass);

StringRef stringifyCudaTileSemanticKind(CudaTileSemanticKind kind);

StringRef stringifyCudaTileOperandRole(CudaTileOperandRole role);

StringRef stringifyCudaTileScheduleSource(CudaTileScheduleSource source);

StringRef stringifyCudaTileConvLoweringMode(CudaTileConvLoweringMode mode);

StringRef stringifyCudaTileLoweringStrategy(CudaTileLoweringStrategy strategy);

CudaTileConvPlan extractCudaTileConvPlan(linalg::GenericOp genOp);

CudaTileKernelPlan extractCudaTileKernelPlan(Operation *innerModule,
                                             const CudaTileOptions &options);

void printCudaTileKernelPlan(const CudaTileKernelPlan &plan,
                             llvm::raw_ostream &os);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // MERLIN_COMPILER_PLUGINS_TARGET_CUDATILE_KERNEL_PLAN_H_
