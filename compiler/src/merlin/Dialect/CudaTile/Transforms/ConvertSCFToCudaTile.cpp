// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 5: Convert standalone SCF ops to cuda_tile IR.
//
// Handles: scf.for, scf.if, scf.while.
//
// Mappings:
//   scf.for   → cuda_tile.for (bounds, step, iter_values)
//   scf.if    → cuda_tile.if
//   scf.while → cuda_tile.loop + cuda_tile.if / cuda_tile.break
//   scf.yield → cuda_tile.continue / cuda_tile.yield
//
// Most SCF is already generated directly in Phase 4 (the K-loop).
// This pass handles standalone SCF surviving into the codegen path.

#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler::CudaTile {

#define GEN_PASS_DEF_CUDATILECONVERTSCFPASS
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"

namespace {

/// Check if an scf.for is NOT already part of a cuda_tile contraction K-loop
/// (those are generated directly and don't need this pass).
static bool isStandaloneSCF(Operation *op) {
  return !op->hasAttr("cuda_tile.kernel_class");
}

struct ConvertSCFPass
    : impl::CudaTileConvertSCFPassBase<ConvertSCFPass> {

  ConvertSCFPass() = default;
  explicit ConvertSCFPass(const CudaTileTransformOptions &options)
      : options(options) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    funcOp->walk([&](Operation *op) {
      if ((isa<scf::ForOp>(op) || isa<scf::IfOp>(op) ||
           isa<scf::WhileOp>(op)) &&
          isStandaloneSCF(op)) {
        op->setAttr("cuda_tile.kernel_class",
                     StringAttr::get(op->getContext(), "scf"));
      }
    });
  }

  CudaTileTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertSCFToCudaTilePass(const CudaTileTransformOptions &options) {
  return std::make_unique<ConvertSCFPass>(options);
}

} // namespace mlir::iree_compiler::CudaTile
