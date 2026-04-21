// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/CudaTile/CudaTileOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::CudaTileOptions);

namespace mlir::iree_compiler {

void CudaTileOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("CUDA Tile HAL Target");

  binder.opt<std::string>(
      "iree-cuda-tile-sm-arch", smArch, llvm::cl::cat(category),
      llvm::cl::desc("CUDA SM architecture for cuda_tile backend "
                     "(e.g., 'sm_80', 'sm_86', 'sm_90')."));

  binder.opt<std::string>(
      "iree-cuda-tile-tileiras-path", tileirasPath, llvm::cl::cat(category),
      llvm::cl::desc("Path to the tileiras assembler tool. If empty, "
                     "searches PATH for 'tileiras'."));

  binder.opt<std::string>(
      "iree-cuda-tile-tileiras-params", tileirasParams, llvm::cl::cat(category),
      llvm::cl::desc("Additional parameters to pass to tileiras."));

  binder.opt<bool>(
      "iree-cuda-tile-enable-codegen", enableCodegen, llvm::cl::cat(category),
      llvm::cl::desc("Enable codegen path (linalg -> cuda_tile ops -> "
                     "tilebc -> cubin). Default: false (external only)."));

  binder.opt<int64_t>("iree-cuda-tile-tile-m", tileM, llvm::cl::cat(category),
                      llvm::cl::desc("Default tile M dimension."));

  binder.opt<int64_t>("iree-cuda-tile-tile-n", tileN, llvm::cl::cat(category),
                      llvm::cl::desc("Default tile N dimension."));

  binder.opt<int64_t>("iree-cuda-tile-tile-k", tileK, llvm::cl::cat(category),
                      llvm::cl::desc("Default tile K dimension."));

  binder.opt<std::string>(
      "iree-cuda-tile-dump-kernel-plan-to", dumpKernelPlanTo,
      llvm::cl::cat(category),
      llvm::cl::desc("Dump recognized cuda_tile kernel plans to the given "
                     "path. Use '-' for stdout."));
}

LogicalResult CudaTileOptions::verify(mlir::Builder &builder) const {
  if (smArch.empty()) {
    return emitError(builder.getUnknownLoc(),
                     "cuda_tile target requires --iree-cuda-tile-sm-arch");
  }
  if (smArch.substr(0, 3) != "sm_") {
    return emitError(builder.getUnknownLoc(),
                     "cuda_tile SM architecture must start with 'sm_', got '")
           << smArch << "'";
  }
  return success();
}

} // namespace mlir::iree_compiler
