// Copyright 2026 UCB-BAR
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "merlin/Dialect/Gemmini/RegisterGemmini.h"
#include "merlin/Dialect/Gemmini/IR/GemminiDialect.h"

using namespace mlir;

// Forward declarations for pass factory/registration functions
// (defined in Transforms/*.cpp under namespace mlir::merlin)
namespace mlir {
namespace merlin {
std::unique_ptr<Pass> createLowerLinalgToGemminiPass();
std::unique_ptr<Pass> createLowerGemminiPass();
void registerLowerLinalgToGemminiPass();
void registerLowerGemminiPass();
void registerGemminiIRDumpsPass();
} // namespace merlin
} // namespace mlir

namespace merlin {
namespace gemmini {

void registerGemminiDialect(DialectRegistry &registry) {
  // GemminiDialect lives in ::merlin::gemmini (from generated .inc files)
  registry.insert<GemminiDialect>();
}

void registerGemminiPasses() {
  mlir::merlin::registerLowerLinalgToGemminiPass();
  mlir::merlin::registerLowerGemminiPass();
  mlir::merlin::registerGemminiIRDumpsPass();
}

std::unique_ptr<Pass> createConvertLinalgToGemminiPass() {
  return mlir::merlin::createLowerLinalgToGemminiPass();
}

std::unique_ptr<Pass> createLowerGemminiPass() {
  return mlir::merlin::createLowerGemminiPass();
}

} // namespace gemmini
} // namespace merlin