// CudaTilePlugin.cpp — IREE compiler plugin that registers Merlin's
// cuda-tile passes.  Activation policy is DefaultActivated so the pass
// is always available on the command line (--merlin-linalg-to-cuda-tile-text).

#include "LinalgToCudaTileText.h"

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::iree_compiler {
namespace {

struct CudaTileSession
    : public PluginSession<CudaTileSession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() { registerLinalgToCudaTileTextPass(); }
};

} // namespace
} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_merlin_cuda_tile(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::CudaTileSession>(
      "merlin_cuda_tile");
  return true;
}
