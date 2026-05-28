//===- GemminiToLLVMIRTranslation.h - Gemmini to LLVM IR ------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//
//
// Provides registration hooks that wire the MLIR `gemmini.intr.*` ops into
// `mlir-translate --mlir-to-llvmir`. The conversion table itself is generated
// by `mlir-tblgen -gen-llvmir-conversions GemminiOps.td` and consumed by the
// `convertOperation` method in GemminiToLLVMIRTranslation.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef MERLIN_TARGET_LLVMIR_DIALECT_GEMMINI_GEMMINITOLLVMIRTRANSLATION_H
#define MERLIN_TARGET_LLVMIR_DIALECT_GEMMINI_GEMMINITOLLVMIRTRANSLATION_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace merlin {

/// Registers the LLVMTranslationDialectInterface for the Gemmini dialect into
/// `registry` so MLIR translation modules see Gemmini ops as translatable.
void registerGemminiDialectTranslation(mlir::DialectRegistry &registry);

/// Convenience overload that registers into an existing context.
void registerGemminiDialectTranslation(mlir::MLIRContext &context);

} // namespace merlin

#endif // MERLIN_TARGET_LLVMIR_DIALECT_GEMMINI_GEMMINITOLLVMIRTRANSLATION_H
