// Copyright 2026 UCB-BAR
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Merlin Gemmini Includes
#include "merlin/Dialect/Gemmini/RegisterGemmini.h"
#include "merlin/Dialect/Gemmini/Transforms/Transforms.h"
#include "merlin/Dialect/Gemmini/IR/GemminiDialect.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// 1. Define Compiler Options
// These flags allow you to configure the hardware parameters from the command line.
struct GemminiOptions {
  int64_t dim = 16;
  int64_t addrLen = 32;
  int64_t accRows = 64;
  int64_t bankRows = 64;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("Merlin Gemmini Options");
    
    binder.opt<int64_t>("merlin-gemmini-dim", dim,
                        llvm::cl::desc("Gemmini systolic array dimension"),
                        llvm::cl::cat(category));
                        
    binder.opt<int64_t>("merlin-gemmini-addr-len", addrLen,
                        llvm::cl::desc("Gemmini address length"),
                        llvm::cl::cat(category));
                        
    binder.opt<int64_t>("merlin-gemmini-acc-rows", accRows,
                        llvm::cl::desc("Gemmini accumulator rows"),
                        llvm::cl::cat(category));
                        
    binder.opt<int64_t>("merlin-gemmini-bank-rows", bankRows,
                        llvm::cl::desc("Gemmini bank rows"),
                        llvm::cl::cat(category));
  }
};

// 2. Define a Pass Wrapper for Legalization
// This wraps the populateGemminiLegalizeForLLVMExportPatterns function 
// so it can be added to the pass pipeline.
class GemminiLegalizeForLLVMExportPass
    : public PassWrapper<GemminiLegalizeForLLVMExportPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GemminiLegalizeForLLVMExportPass)

  GemminiLegalizeForLLVMExportPass(const GemminiOptions &opts) : options(opts) {}

  StringRef getArgument() const override { return "merlin-gemmini-legalize-for-llvm"; }
  StringRef getDescription() const override { return "Legalize Gemmini dialect to LLVM intrinsics"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<merlin::gemmini::GemminiDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    LLVMTypeConverter typeConverter(context);
    RewritePatternSet patterns(context);

    // TODO: You may want to expose these sizes via options as well if they vary.
    size_t sizeOfElemT = 1;
    size_t sizeOfAccT = 4;

    // Populate the patterns from LegalizeForLLVMExport.cpp
    mlir::populateGemminiLegalizeForLLVMExportPatterns(
        typeConverter, patterns, options.dim, options.addrLen, 
        options.accRows, options.bankRows, sizeOfElemT, sizeOfAccT);

    LLVMConversionTarget target(*context);
    mlir::configureGemminiLegalizeForExportTarget(target);
    target.addLegalDialect<LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  GemminiOptions options;
};

// 3. Define the Plugin Session
struct GemminiSession : public PluginSession<GemminiSession, GemminiOptions> {
  
  // Register the Gemmini Dialect and the LLVM IR Translation Interface
  // using the unified registration function we created.
  void onRegisterDialects(DialectRegistry &registry) override {
    merlin::gemmini::registerGemminiDialect(registry);
  }

  // Extend the compilation pipeline
  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    // 1. Add the pass to convert Linalg to Gemmini operations
    pm.addPass(merlin::gemmini::createConvertLinalgToGemminiPass());

    // 2. Add the pass to legalize Gemmini operations to LLVM intrinsics
    // This typically happens later, but adding it here ensures it runs 
    // within the IREE compilation flow if you aren't using a custom backend target.
    pm.addPass(std::make_unique<GemminiLegalizeForLLVMExportPass>(options));
  }
};

} // namespace

// 4. Register the Options and the Plugin Entry Point
IREE_DEFINE_COMPILER_OPTION_FLAGS(GemminiOptions);

extern "C" bool iree_register_compiler_plugin_merlin_gemmini(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<GemminiSession>("merlin_gemmini");
  return true;
}