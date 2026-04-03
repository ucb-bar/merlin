// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MERLIN_COMPILER_DIALECT_CUDATILE_TRANSFORMS_PASSES_H_
#define MERLIN_COMPILER_DIALECT_CUDATILE_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::CudaTile {

struct CudaTileTransformOptions {
  // Default tile dimensions for kernel partitioning.
  int64_t tileM = 128;
  int64_t tileN = 128;
  int64_t tileK = 32;

  // Enable individual op class conversions.
  bool enableDataMovement = true;
  bool enableElementwise = true;
  bool enableReductions = true;
  bool enableContractions = true;
  bool enableSCF = true;
};

// Phase 1: Data movement (copy, transpose, reshape, broadcast).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertDataMovementToCudaTilePass(
    const CudaTileTransformOptions &options = {});

// Phase 2: Elementwise (arith, math, type conversions).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertElementwiseToCudaTilePass(
    const CudaTileTransformOptions &options = {});

// Phase 3: Reductions (sum, max, min, mul).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertReductionsToCudaTilePass(
    const CudaTileTransformOptions &options = {});

// Phase 4: Contractions (matmul, batch_matmul, conv2d).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertContractionsToCudaTilePass(
    const CudaTileTransformOptions &options = {});

// Phase 5: SCF/loops (for, if, while).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertSCFToCudaTilePass(const CudaTileTransformOptions &options = {});

void registerCudaTilePasses();

#define GEN_PASS_DECL
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"

} // namespace mlir::iree_compiler::CudaTile

#endif // MERLIN_COMPILER_DIALECT_CUDATILE_TRANSFORMS_PASSES_H_
