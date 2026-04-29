// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 2: Convert elementwise ops to cuda_tile IR.
//
// Handles: arith.addf/mulf/subf/divf/negf, math.exp/log/sqrt/rsqrt/sin/cos/
//          tanh/ceil/floor/fma, arith.cmpf/cmpi/select, type conversions
//          (arith.extf/trunci/sitofp/fptosi).
//
// Detects chains of elementwise ops and fuses into single kernel.

#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"
#include "compiler/src/merlin/Dialect/CudaTile/Utils/OpMapping.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler::CudaTile {

#define GEN_PASS_DEF_CUDATILECONVERTELEMENTWISEPASS
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"

namespace {

using cuda_tile::mapArithToCudaTile;
using cuda_tile::mapMathToCudaTile;

/// Check if an operation is a supported elementwise op.
static bool isSupportedElementwiseOp(Operation *op) {
  return !mapArithToCudaTile(op).empty() || !mapMathToCudaTile(op).empty() ||
         isa<arith::SelectOp, arith::CmpFOp, arith::CmpIOp, arith::ExtFOp,
             arith::TruncFOp, arith::SIToFPOp, arith::FPToSIOp,
             arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp, math::FmaOp>(
             op);
}

struct ConvertElementwisePass
    : impl::CudaTileConvertElementwisePassBase<ConvertElementwisePass> {

  ConvertElementwisePass() = default;
  explicit ConvertElementwisePass(const CudaTileTransformOptions &options)
      : options(options) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    // Tag linalg.generic ops that have elementwise bodies.
    funcOp->walk([&](Operation *rawOp) {
      auto genericOp = llvm::dyn_cast<linalg::GenericOp>(rawOp);
      if (!genericOp)
        return;
      {
      // Skip ops already tagged by earlier passes (e.g. data movement).
      if (rawOp->hasAttr("cuda_tile.kernel_class"))
        return;
      // Check if all iterator types are parallel (elementwise).
      // getNumReductionLoops() > 0 means it has reduction dims (not elementwise).
      if (genericOp.getNumReductionLoops() > 0)
        return;

      // Collect all compute ops in the body (in order).
      // A body may have multiple ops chained (e.g., subf → exp for softmax).
      SmallVector<StringRef> opNames;
      for (auto &op : genericOp.getBody()->without_terminator()) {
        StringRef name = mapArithToCudaTile(&op);
        if (name.empty())
          name = mapMathToCudaTile(&op);
        if (!name.empty())
          opNames.push_back(name);
      }

      if (!opNames.empty()) {
        // Store as semicolon-separated list: "subf;exp".
        std::string joined;
        for (size_t i = 0; i < opNames.size(); ++i) {
          if (i > 0)
            joined += ";";
          joined += opNames[i];
        }
        genericOp->setAttr("cuda_tile.kernel_class",
                           StringAttr::get(genericOp->getContext(),
                                           "elementwise"));
        genericOp->setAttr("cuda_tile.op_name",
                           StringAttr::get(genericOp->getContext(),
                                           joined));
      }
      }
    });
  }

  CudaTileTransformOptions options;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertElementwiseToCudaTilePass(
    const CudaTileTransformOptions &options) {
  return std::make_unique<ConvertElementwisePass>(options);
}

} // namespace mlir::iree_compiler::CudaTile
