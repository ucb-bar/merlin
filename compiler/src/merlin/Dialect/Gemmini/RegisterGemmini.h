// Copyright 2026 UCB-BAR
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MERLIN_DIALECT_GEMMINI_REGISTERGEMMINI_H
#define MERLIN_DIALECT_GEMMINI_REGISTERGEMMINI_H

#include <memory>
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace merlin {
namespace gemmini {

/// Register the Gemmini dialect into the given registry.
void registerGemminiDialect(mlir::DialectRegistry &registry);

/// Register all Gemmini passes with the MLIR pass registry.
void registerGemminiPasses();

/// Create the pass that converts linalg ops to Gemmini dialect ops.
std::unique_ptr<mlir::Pass> createConvertLinalgToGemminiPass();

/// Create the pass that lowers Gemmini ops to LLVM intrinsics.
std::unique_ptr<mlir::Pass> createLowerGemminiPass();

} // namespace gemmini
} // namespace merlin

#endif // MERLIN_DIALECT_GEMMINI_REGISTERGEMMINI_H