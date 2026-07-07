#include "Backend.h"
#include "Dialects.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

int main(int argc, char **argv) {
  bool verify = false, convIface = false, convLlvm = false;
  std::string emitCb;
  std::string inputFile;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--verify-diagnostics") {
      verify = true;
    } else if (a == "--convert-iface-to-gemmini") {
      convIface = true;
    } else if (a == "--convert-gemmini-to-llvm-rocc") {
      convLlvm = true;
    } else if (a.rfind("--emit-command-buffer=", 0) == 0) {
      emitCb = a.substr(std::string("--emit-command-buffer=").size());
    } else if (a.rfind("--", 0) == 0) {
      // ignore other flags
    } else {
      inputFile = a;
    }
  }
  (void)verify;

  if (inputFile.empty()) {
    llvm::errs() << "error: no input file\n";
    return 2;
  }

  MLIRContext ctx;
  ctx.loadDialect<iface::IfaceDialect, gem::GemminiDialect, LLVM::LLVMDialect>();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFile, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse " << inputFile << "\n";
    return 1;
  }

  bool anyTransform = convIface || convLlvm || !emitCb.empty();
  if (!anyTransform) {
    // parse-only (--verify-diagnostics): success since it parsed.
    return 0;
  }

  if (convIface)
    if (failed(backend::convertIfaceToGemmini(*module)))
      return 1;
  if (!emitCb.empty())
    if (failed(backend::emitCommandBuffer(*module, emitCb)))
      return 1;
  if (convLlvm)
    if (failed(backend::lowerToLlvmRocc(*module)))
      return 1;

  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
