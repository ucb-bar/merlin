#ifndef IREE_RADIANCE_COMPILER_DIALECT_RADIANCE_TRANSFORMS_PASSES_H_
#define IREE_RADIANCE_COMPILER_DIALECT_RADIANCE_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Radiance {

struct LowerRadianceToLLVMOptions {
	bool emitLLVMIR = false;
	std::string emitLLVMIRPath;
	int64_t numWarps = 4;
};

// Pre-conversion pass: walks operand/result types of every op in the module
// and rewrites any memref bearing a #radiance.global / #radiance.shared
// address-space attribute into a memref with a plain integer memory space
// (1 / 3 respectively). This lets the standard MemRef-to-LLVM conversion
// pipeline handle the lowering without needing a custom type converter for
// our dialect-private addrspace attribute.
std::unique_ptr<Pass> createConvertRadianceAddrSpacesPass();

// Driver pass: finds func.func ops tagged with `radiance.kernel`, runs the
// standard MLIR-to-LLVM conversion pipeline (scf->cf->llvm,
// memref->llvm, arith->llvm, func->llvm, cf->llvm), then optionally
// translates the resulting llvm dialect to LLVM IR text and writes it to
// disk. Output consumed by kernels/core/precompile.py with source_lang=ll.
std::unique_ptr<Pass> createLowerRadianceToLLVMPass(
	const LowerRadianceToLLVMOptions &options);

void registerRadiancePasses();

} // namespace mlir::iree_compiler::Radiance

#endif // IREE_RADIANCE_COMPILER_DIALECT_RADIANCE_TRANSFORMS_PASSES_H_
