// Copyright 2026 UCB-BAR
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "merlin/Dialect/Gemmini/RegisterGemmini.h"

namespace mlir::iree_compiler {
namespace {

struct GemminiSession
    : public PluginSession<GemminiSession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    merlin::gemmini::registerGemminiPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    merlin::gemmini::registerGemminiDialect(registry);
  }
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_merlin_gemmini(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::GemminiSession>(
      "merlin_gemmini");
  return true;
}