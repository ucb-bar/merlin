// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// LowerRadianceToLLVMPass — finds func.func ops tagged with
// `radiance.kernel`, runs the standard MLIR-to-LLVM conversion pipeline
// on the module, and (optionally) writes the resulting LLVM IR to disk
// as a text .ll file.
//
// The output .ll is consumable by kernels/core/precompile.py with
// source_lang=ll: precompile.py calls llvm-muon clang -c on it to
// produce a .o, which the link step pulls into kernel.radiance.elf
// alongside the C++ wrapper from kernel_phase2.cpp.j2.
//
// Phase 2.6c first cut: relies on standard upstream conversion patterns
// (scf→cf, memref→llvm, arith→llvm, func→llvm, cf→llvm) plus the
// pre-pass ConvertRadianceAddrSpaces to translate dialect-private
// addrspace attrs to integer addrspaces. No custom op patterns yet —
// SIMT primitives (vx_bar, vx_split, etc.) will appear as inline_asm
// ops in a follow-up.

#include "compiler/src/merlin/Dialect/Radiance/IR/RadianceAttrs.h"
#include "compiler/src/merlin/Dialect/Radiance/Transforms/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::iree_compiler::Radiance;

namespace {

class LowerRadianceToLLVMPass
	: public PassWrapper<LowerRadianceToLLVMPass, OperationPass<ModuleOp>> {
  public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRadianceToLLVMPass)

	LowerRadianceToLLVMPass() = default;
	LowerRadianceToLLVMPass(const LowerRadianceToLLVMPass &) = default;
	explicit LowerRadianceToLLVMPass(const LowerRadianceToLLVMOptions &opts)
		: options(opts) {}

	StringRef getArgument() const final {
		return "radiance-lower-to-llvm";
	}
	StringRef getDescription() const final {
		return "Lower radiance-tagged func.func to LLVM dialect and "
			   "optionally emit LLVM IR text.";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<LLVM::LLVMDialect>();
	}

	void runOnOperation() override {
		ModuleOp module = getOperation();

		// Bail if no function in this module is tagged.
		bool anyKernel = false;
		module.walk([&](func::FuncOp fn) {
			if (fn->hasAttr(getKernelAttrName())) {
				anyKernel = true;
			}
		});
		if (!anyKernel)
			return;

		// Apply optional symbol rename: `radiance.entry_symbol` overrides
		// the function's MLIR name so the emitted LLVM IR symbol matches
		// the manifest's entry_symbol field exactly (extern-C linkable).
		module.walk([&](func::FuncOp fn) {
			if (auto sym =
					fn->getAttrOfType<StringAttr>(getEntrySymbolAttrName())) {
				fn.setSymName(sym.getValue());
			}
			// Drop our marker attrs so they don't survive into LLVM IR
			// metadata (they'd be unknown to llvm-muon clang).
			fn->removeAttr(getKernelAttrName());
			fn->removeAttr(getEntrySymbolAttrName());
			fn->removeAttr(getNumWarpsAttrName());
		});

		// Run the standard upstream conversion pipeline. We use a nested
		// PassManager so the conversion failure surfaces with proper
		// diagnostics. None of these passes know about our dialect; the
		// preceding ConvertRadianceAddrSpacesPass took care of the
		// translation from #radiance.global → integer addrspace.
		PassManager pm(&getContext());
		pm.addPass(createSCFToControlFlowPass());
		pm.addPass(createArithToLLVMConversionPass());
		pm.addPass(createFinalizeMemRefToLLVMConversionPass());
		pm.addPass(createConvertFuncToLLVMPass());
		pm.addPass(createConvertControlFlowToLLVMPass());
		pm.addPass(createReconcileUnrealizedCastsPass());
		if (failed(pm.run(module))) {
			module.emitError("radiance-lower-to-llvm: standard MLIR-to-LLVM "
							 "conversion failed");
			signalPassFailure();
			return;
		}

		if (options.emitLLVMIR && !options.emitLLVMIRPath.empty()) {
			// Translate the LLVM dialect to LLVM IR and write text.
			mlir::registerLLVMDialectTranslation(*module->getContext());
			llvm::LLVMContext llvmContext;
			auto llvmModule =
				mlir::translateModuleToLLVMIR(module, llvmContext);
			if (!llvmModule) {
				module.emitError(
					"radiance-lower-to-llvm: translateModuleToLLVMIR failed");
				signalPassFailure();
				return;
			}

			std::error_code ec;
			llvm::raw_fd_ostream out(
				options.emitLLVMIRPath, ec, llvm::sys::fs::OF_Text);
			if (ec) {
				module.emitError("radiance-lower-to-llvm: cannot open ")
					<< options.emitLLVMIRPath << ": " << ec.message();
				signalPassFailure();
				return;
			}
			llvmModule->print(out, /*AssemblyAnnotationWriter=*/nullptr);
			out.flush();
		}

		// After emission, erase all function-like ops in the module so
		// subsequent IREE passes don't try to interpret the LLVM dialect.
		// The .ll file we just wrote is the Radiance plugin's only
		// output; the iree-compile invocation that drove us only consumes
		// the side-effect file. Leave the module empty to avoid
		// "cannot determine function type" errors from IREE's input
		// converter.
		SmallVector<Operation *> toErase;
		module.walk([&](Operation *op) {
			if (op == module.getOperation())
				return;
			if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
				toErase.push_back(op);
			}
		});
		for (Operation *op : toErase)
			op->erase();
	}

  private:
	LowerRadianceToLLVMOptions options;
};

} // namespace

namespace mlir::iree_compiler::Radiance {

std::unique_ptr<Pass> createLowerRadianceToLLVMPass(
	const LowerRadianceToLLVMOptions &options) {
	return std::make_unique<LowerRadianceToLLVMPass>(options);
}

void registerRadiancePasses() {
	// Register passes via the create-function pattern so we don't need the
	// concrete pass classes (which live in their own TUs and would
	// otherwise need pulling into a shared header).
	mlir::registerPass([]() -> std::unique_ptr<Pass> {
		return createConvertRadianceAddrSpacesPass();
	});
	mlir::registerPass([]() -> std::unique_ptr<Pass> {
		LowerRadianceToLLVMOptions opts;
		return createLowerRadianceToLLVMPass(opts);
	});
}

} // namespace mlir::iree_compiler::Radiance
