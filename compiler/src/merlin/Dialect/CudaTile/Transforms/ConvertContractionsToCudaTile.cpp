// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 4: Classify contraction ops for cuda_tile codegen.
//
// Detects both named ops (linalg.matmul) at preprocessing and
// generic contractions (linalg.generic with mulf+addf body, reduction
// iterator) at translation time. This handles conv2d after IREE's
// im2col or direct lowering to generic contractions.

#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler::CudaTile {

#define GEN_PASS_DEF_CUDATILECONVERTCONTRACTIONSPASS
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"

namespace {

/// Classification only — layout metadata extracted at translation time.
static void classifyMatmul(Operation *op) {
  op->setAttr("cuda_tile.kernel_class",
              StringAttr::get(op->getContext(), "matmul"));
}

/// Detect a matmul-like contraction in linalg.generic:
/// - 2 inputs, 1 output
/// - At least 1 reduction dimension
/// - Body is: mulf(in0, in1) → addf(acc, product) → yield
static bool isGenericContraction(linalg::GenericOp genericOp) {
  if (genericOp->hasAttr("cuda_tile.kernel_class"))
    return false;
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
    return false;
  if (genericOp.getNumReductionLoops() == 0)
    return false;

  // Check body: expect mulf + addf + yield
  Block &body = genericOp.getRegion().front();
  auto ops = body.without_terminator();
  int opCount = 0;
  bool hasMul = false, hasAdd = false;
  for (auto &op : ops) {
    if (isa<arith::MulFOp, arith::MulIOp>(&op))
      hasMul = true;
    else if (isa<arith::AddFOp, arith::AddIOp>(&op))
      hasAdd = true;
    opCount++;
  }
  // Standard contraction: exactly mulf + addf
  return opCount == 2 && hasMul && hasAdd;
}

struct ConvertContractionsPass
    : impl::CudaTileConvertContractionsPassBase<ConvertContractionsPass> {

  ConvertContractionsPass() = default;
  explicit ConvertContractionsPass(const CudaTileTransformOptions &options)
      : options(options) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    funcOp->walk([&](Operation *op) {
      // Named ops (preprocessing time).
      if (isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op)) {
        classifyMatmul(op);
        return;
      }
      // Generic contractions (translation time — conv2d lowered to generic).
      if (auto genericOp = llvm::dyn_cast<linalg::GenericOp>(op)) {
        if (isGenericContraction(genericOp))
          classifyMatmul(op);
      }
    });
  }

  CudaTileTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertContractionsToCudaTilePass(
    const CudaTileTransformOptions &options) {
  return std::make_unique<ConvertContractionsPass>(options);
}

} // namespace mlir::iree_compiler::CudaTile
