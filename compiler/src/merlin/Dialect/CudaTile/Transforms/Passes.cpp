// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"

namespace mlir::iree_compiler::CudaTile {

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h.inc"
} // namespace

void registerCudaTilePasses() { registerPasses(); }

} // namespace mlir::iree_compiler::CudaTile
