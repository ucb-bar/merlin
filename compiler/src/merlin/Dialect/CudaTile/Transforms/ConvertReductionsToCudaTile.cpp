// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 3: Classify reduction ops for cuda_tile codegen.
//
// Runs at BOTH preprocessing (named linalg.reduce) and translation time
// (linalg.generic with reduction iterators). Sets classification attrs only
// (kernel_class, combiner). Layout metadata (shapes, reduce_dims) is
// extracted at translation time from the actual inner module IR.

#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"
#include "compiler/src/merlin/Dialect/CudaTile/Utils/OpMapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler::CudaTile {

#define GEN_PASS_DEF_CUDATILECONVERTREDUCTIONSPASS
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"

namespace {

using cuda_tile::matchReduceCombiner;

/// Tag a named linalg.reduce with classification attrs.
static void classifyReduce(linalg::ReduceOp op) {
  StringRef combiner = matchReduceCombiner(op.getCombiner());
  if (combiner.empty())
    return;
  auto ctx = op->getContext();
  op->setAttr("cuda_tile.kernel_class", StringAttr::get(ctx, "reduce"));
  op->setAttr("cuda_tile.combiner", StringAttr::get(ctx, combiner));
}

/// Tag a linalg.generic with reduction iterators.
static void classifyGenericReduction(linalg::GenericOp genericOp) {
  if (genericOp->hasAttr("cuda_tile.kernel_class"))
    return;
  if (genericOp.getNumReductionLoops() == 0)
    return;
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return;

  StringRef combiner = matchReduceCombiner(genericOp.getRegion());
  if (combiner.empty())
    return;

  auto ctx = genericOp->getContext();
  genericOp->setAttr("cuda_tile.kernel_class",
                     StringAttr::get(ctx, "reduce"));
  genericOp->setAttr("cuda_tile.combiner",
                     StringAttr::get(ctx, combiner));
}

struct ConvertReductionsPass
    : impl::CudaTileConvertReductionsPassBase<ConvertReductionsPass> {

  ConvertReductionsPass() = default;
  explicit ConvertReductionsPass(const CudaTileTransformOptions &options)
      : options(options) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    // Named linalg.reduce (present at preprocessing time).
    funcOp->walk([&](linalg::ReduceOp op) { classifyReduce(op); });
    // Fallback: linalg.generic with reduction iterators (translation time).
    funcOp->walk([&](linalg::GenericOp op) { classifyGenericReduction(op); });
  }

  CudaTileTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertReductionsToCudaTilePass(
    const CudaTileTransformOptions &options) {
  return std::make_unique<ConvertReductionsPass>(options);
}

} // namespace mlir::iree_compiler::CudaTile
