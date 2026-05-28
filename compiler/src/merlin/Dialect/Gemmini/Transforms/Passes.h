#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_H_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_H_

#include <memory>

#include "compiler/src/merlin/Dialect/Gemmini/IR/GemminiAttrs.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Gemmini {

// MX (microscaling) format selection for mxGemmini. Mapped per
// `third_party/gemmini-mx/src/main/scala/gemmini/MxParameters.scala:124-130`:
//   Disabled = 0  → no MX bits set in CONFIG_EX (vanilla Gemmini)
//   Fp4      = 0  (E2M2)
//   Fp6_0    = 1  (E2M4)
//   Fp8_0    = 2  (E4M4)
//   Fp6_1    = 3  (E3M3)
//   Fp8_1    = 4  (E5M3)
// Note: `Disabled` and `Fp4` share encoding 0; the dialect distinguishes
// them via the separate `commandIssue` / `useLut` flags.
enum class MxFormat : int64_t {
	Disabled = -1,
	Fp4 = 0,
	Fp6_0 = 1,
	Fp8_0 = 2,
	Fp6_1 = 3,
	Fp8_1 = 4,
};

// How RoCC commands reach the Gemmini hardware. Two paths:
//   - RoCC: emit custom-3 instructions; consumed by a Rocket core with
//     a RoCC-attached Gemmini (Spike, gemmini_u250 / FireSim
//     LeanGemminiRocketConfig, MxGemminiRocketConfig).
//   - MMIO: emit volatile stores to (mmioBase + 0x10/0x18/0x00) for
//     rs1/rs2/encoded-instruction-word; consumed by a Rocket core
//     without RoCC where Gemmini sits as a cluster-side MMIO peripheral
//     (RadianceGemminiOnlyConfig).
enum class CommandIssue {
	RoCC,
	MMIO,
};

// Hardware-specific Gemmini accelerator parameters. Defaults match the
// Spike libgemmini.so build configured by chipyard's
// `gemmini-rocc-tests/include/gemmini_params.h` (DIM=16, ADDR_LEN=32,
// BANK_NUM=4, BANK_ROWS=4096, ACC_ROWS=1024, elem_t=int8_t, acc_t=int32_t).
// Overriding these via plugin flags retargets the lowering for variants
// (e.g. mxGemmini fp8 with different scratchpad sizes), as long as a
// matching libgemmini build / RTL is used at run time.
struct GemminiTargetConfig {
	// Systolic array side length (elements per row of one tile).
	int64_t dim = 16;
	// Address-space bit-width. Controls the rs2 encoding of MVIN/MVOUT/etc.
	// MUST match libgemmini's runtime `addr_len`.
	int64_t addrLen = 32;
	// Accumulator row count.
	int64_t accRows = 1024;
	// Scratchpad rows per bank.
	int64_t bankRows = 4096;
	// Scratchpad bank count.
	int64_t bankNum = 4;
	// Element bit width (lhs/rhs of matmul). 8 for int8, 32 for fp32, etc.
	int64_t elemBits = 8;
	// Accumulator element bit width. 32 for int32 / fp32.
	int64_t accBits = 32;

	// MX-format selection for mxGemmini. Default Disabled keeps the
	// CONFIG_EX rs1 [15:10] / [5] bits at zero, matching Phase 1-4
	// vanilla-Gemmini behavior byte-identically.
	MxFormat mxFormat = MxFormat::Disabled;

	// Command issue path (see CommandIssue above). Default RoCC keeps the
	// Phase 1-4 lowering unchanged; MMIO triggers the
	// gemmini-lower-intr-to-mmio pass to replace gemmini.intr.* ops with
	// volatile stores before LLVM-IR translation.
	CommandIssue commandIssue = CommandIssue::RoCC;

	// MMIO base address for the gemmini control window. Used only when
	// commandIssue == MMIO. Default 0x40084000 matches
	// RadianceGemminiOnlyConfig's cluster-0 GEMMINI_CTRL (verified
	// against matmul_ws_mx_generic.c:28).
	int64_t mmioBase = 0x40084000;

	// Phase-8 LOOP_WS lowering. When true, gemmini.tile_matmul lowers to
	// a single LOOP_WS sequence (5 configs + 6 LOOP_WS_* + FLUSH = 12
	// commands per matmul) instead of the per-tile MVIN/PRELOAD/COMPUTE/
	// MVOUT expansion (~56 commands per 16x64x64 matmul). The
	// PluginRegistration auto-enables this when commandIssue == MMIO so the
	// dispatch ELF doesn't overrun gemmini's MMIO command queue (avoids the
	// GemminiTile.scala:446 backpressure assertion). Default false
	// preserves Phase 1-7 byte-identical behavior on RoCC/Spike.
	bool useLoopWs = false;

	// Opt-in: when true, emit volatile stores of binding pointers + matmul
	// operand pointers to fixed DRAM trace regions (defined in
	// runtime/.../merlin_debug_addresses.h) so the runtime's
	// MERLIN_DISPATCH_DEBUG-built loader can read them back. Off in
	// production; adds a handful of stores per matmul dispatch when on.
	bool dispatchDebug = false;

	// Helpers.
	int64_t elemSizeBytes() const {
		return elemBits / 8;
	}
	int64_t accSizeBytes() const {
		return accBits / 8;
	}
};

struct GemminiTransformOptions {
	bool enableMatmul = true;
	bool enableFP8Matmul = false;
	bool enableConv2D = true;
	bool enableRequantize = true;
	bool enableClamp = true;

	Dataflow defaultDataflow = Dataflow::OutputStationary;

	int64_t tileM = 16;
	int64_t tileN = 16;
	int64_t tileK = 16;

	// Hardware descriptor (DIM, addrLen, scratchpad sizes, dtype widths).
	GemminiTargetConfig target;
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createConvertToGemminiPass(
	const GemminiTransformOptions &options = {});
// createGemminiPreprocessPass() is auto-declared by GEN_PASS_DECL_*
// from Passes.td (no-arg, returns std::unique_ptr<mlir::Pass>).
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createGemminiCanonicalizeFuncPass();
std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerToISAPass(
	const GemminiTransformOptions &options = {});
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLowerGemminiToIREEPass();
// Phase-4 extra: options-taking variants of the tablegen-generated factories
// for the passes whose lowering depends on the target descriptor. The no-arg
// tablegen-generated factories below (declared by GEN_PASS_DECL) remain
// available — they construct the pass with `GemminiTransformOptions{}` (Spike
// int8 16x16 defaults), which matches the documented out-of-the-box behavior.
std::unique_ptr<Pass> createGemminiLegalizeForLLVMExportPassWithOptions(
	const GemminiTransformOptions &options);
std::unique_ptr<Pass> createGemminiLowerTileToISAPassWithOptions(
	const GemminiTransformOptions &options);
std::unique_ptr<Pass> createGemminiAttachCompilationInfoPassWithOptions(
	const GemminiTransformOptions &options);
// Note: createGemminiLowerTileToISAPass() /
// createGemminiLegalizeForLLVMExportPass() /
// createGemminiAttachCompilationInfoPass() are auto-generated by tablegen with
// `std::unique_ptr<mlir::Pass>` return type — they're declared inside
// `Passes.h.inc` via GEN_PASS_DECL, so we don't redeclare them here.

void registerGemminiPasses();

#define GEN_PASS_DECL
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

} // namespace mlir::iree_compiler::Gemmini

#endif // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_H_
