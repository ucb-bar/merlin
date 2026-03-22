// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MERLIN_COMPILER_PLUGINS_TARGET_CUDATILE_OPTIONS_H_
#define MERLIN_COMPILER_PLUGINS_TARGET_CUDATILE_OPTIONS_H_

#include <string>

#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

struct CudaTileOptions {
  // SM architecture target (e.g., "sm_80", "sm_86", "sm_90").
  std::string smArch = "sm_86";

  // Path to tileiras assembler. If empty, searches PATH.
  std::string tileirasPath;

  // Additional parameters to pass to tileiras.
  std::string tileirasParams;

  // Default tile dimensions for cuda_tile kernel generation.
  int64_t tileM = 128;
  int64_t tileN = 128;
  int64_t tileK = 32;

  void bindOptions(OptionsBinder &binder);
  LogicalResult verify(mlir::Builder &builder) const;
  using FromFlags = OptionsFromFlags<CudaTileOptions>;
};

} // namespace mlir::iree_compiler

#endif // MERLIN_COMPILER_PLUGINS_TARGET_CUDATILE_OPTIONS_H_
