#ifndef IREE_RADIANCE_COMPILER_PLUGIN_RADIANCEOPTIONS_H_
#define IREE_RADIANCE_COMPILER_PLUGIN_RADIANCEOPTIONS_H_

#include <string>

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

// Options for the Merlin Radiance/Muon target plugin.
//
// Phase 2.6 lays the plugin scaffolding: registering the option flags so
// `iree-compile` accepts them, and a no-op session (no passes). The
// dialect + lowering passes follow in 2.6b/2.6c, but the user-facing
// option surface is fixed here so downstream tooling (compile.py) can
// stabilize against it.
struct RadianceOptions {
	// Master enable. Required to activate any Radiance compile-time
	// behavior. Mirrors the gemmini plugin's `enable` toggle.
	bool enable = false;

	// Number of warps per Muon threadblock. Must match the
	// mu_schedule(fn, args, num_warps) call in the wrapper template.
	int64_t numWarps = 4;

	// Path where the lowering pass should drop the emitted LLVM IR
	// text (kernel body) when --iree-radiance-emit-llvm-ir=true. Empty
	// disables emission.
	std::string emitLLVMIRPath;

	// When true, emit the radiance dialect → LLVM dialect lowering as a
	// .ll file on disk (consumed downstream by kernels/core/precompile.py
	// with source_lang=ll). When false, the pass is a no-op.
	bool emitLLVMIR = false;

	void bindOptions(OptionsBinder &binder);
	using FromFlags = OptionsFromFlags<RadianceOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_RADIANCE_COMPILER_PLUGIN_RADIANCEOPTIONS_H_
