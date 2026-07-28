//===- iface_irdl_demo.cpp - register merlin_iface from IRDL (NO hand-ODS) ===//
// Proves: a C++ OOT tool registers the merlin_iface input dialect dynamically from
// merlin_iface.irdl.mlir via mlir::irdl::loadDialects(), then parses a capsule.
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
int main(int argc, char **argv) {
  if (argc < 3) { llvm::errs() << "usage: iface-irdl-demo <merlin_iface.irdl.mlir> <capsule.interface.mlir>\n"; return 2; }
  MLIRContext ctx;
  ctx.printOpOnDiagnostic(true);
  ctx.getDiagEngine().registerHandler([](Diagnostic &d){ llvm::errs()<<"DIAG: "<<d.str()<<"\n"; return success(); });
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  // 1) parse the IRDL spec, 2) dynamically register the merlin_iface dialect into ctx
  OwningOpRef<ModuleOp> irdlMod = parseSourceFile<ModuleOp>(argv[1], &ctx);
  if (!irdlMod) { llvm::errs() << "demo: failed to parse IRDL file\n"; return 1; }
  if (failed(irdl::loadDialects(*irdlMod))) { llvm::errs() << "demo: irdl::loadDialects failed\n"; return 1; }
  // 3) now the capsule (merlin_iface.*) parses with ZERO hand-written ODS dialect
  OwningOpRef<ModuleOp> cap = parseSourceFile<ModuleOp>(argv[2], &ctx);
  if (!cap) { llvm::errs() << "demo: failed to parse capsule via IRDL-registered dialect\n"; return 1; }
  if (failed(verify(*cap))) { llvm::errs() << "demo: capsule failed verify\n"; return 1; }
  llvm::outs() << "OK: merlin_iface registered from IRDL; capsule parsed+verified (no hand-ODS)\n";
  return 0;
}
