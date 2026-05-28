#include "compiler/plugins/target/Gemmini/GemminiOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::GemminiOptions);

namespace mlir::iree_compiler {

void GemminiOptions::bindOptions(OptionsBinder &binder) {
	static llvm::cl::OptionCategory category("IREE Gemmini plugin options");

	binder.opt<bool>("iree-gemmini-enable", enable,
		llvm::cl::desc(
			"Enables the Gemmini post-global-optimization pipeline."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-lower-back-to-iree", lowerBackToIREE,
		llvm::cl::desc(
			"Lower recovered gemmini.* ops back to ordinary IREE/MLIR IR "
			"before dispatch creation."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-matmul", enableMatmul,
		llvm::cl::desc("Recover Gemmini matmul patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-fp8-matmul", enableFP8Matmul,
		llvm::cl::desc("Enable recovery of FP8 matmul patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-conv2d", enableConv2D,
		llvm::cl::desc("Recover Gemmini conv2d patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-requantize", enableRequantize,
		llvm::cl::desc("Recover Gemmini requantize patterns."),
		llvm::cl::cat(category));

	binder.opt<bool>("iree-gemmini-enable-clamp", enableClamp,
		llvm::cl::desc("Recover Gemmini clamp patterns."),
		llvm::cl::cat(category));

	binder.opt<std::string>("iree-gemmini-dataflow", dataflow,
		llvm::cl::desc("Default Gemmini dataflow mode: os or ws."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-tile-m", tileM,
		llvm::cl::desc("Default Gemmini tile size M."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-tile-n", tileN,
		llvm::cl::desc("Default Gemmini tile size N."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-tile-k", tileK,
		llvm::cl::desc("Default Gemmini tile size K."),
		llvm::cl::cat(category));

	// Hardware descriptor (Phase 4). All defaults match Spike libgemmini.
	binder.opt<int64_t>("iree-gemmini-dim", dim,
		llvm::cl::desc("Systolic array side length (DIM). Default 16 (Spike "
					   "libgemmini)."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-addr-len", addrLen,
		llvm::cl::desc("Address-space bit-width for MVIN/MVOUT rs2 encoding "
					   "(ADDR_LEN). MUST match libgemmini's runtime addr_len. "
					   "Default 32."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-acc-rows", accRows,
		llvm::cl::desc("Accumulator row count (ACC_ROWS). Default 1024."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-bank-rows", bankRows,
		llvm::cl::desc("Scratchpad rows per bank (BANK_ROWS). Default 4096."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-bank-num", bankNum,
		llvm::cl::desc("Scratchpad bank count (BANK_NUM). Default 4."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-elem-bits", elemBits,
		llvm::cl::desc("Element bit width (lhs/rhs of matmul). Default 8 "
					   "(int8). Use 32 for fp32, 16 for fp16/bf16, etc."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-acc-bits", accBits,
		llvm::cl::desc(
			"Accumulator element bit width. Default 32 (int32 / fp32)."),
		llvm::cl::cat(category));

	// mxGemmini selection (Phase 5).
	binder.opt<std::string>("iree-gemmini-mx-format", mxFormat,
		llvm::cl::desc("MX (microscaling) format for mxGemmini. One of: "
					   "disabled (default, vanilla Gemmini), fp4 (E2M2), "
					   "fp6_0 (E2M4), fp6_1 (E3M3), fp8_0 (E4M4), fp8_1 "
					   "(E5M3). Sets CONFIG_EX rs1 [15:10] / [5] bits."),
		llvm::cl::cat(category));

	binder.opt<std::string>("iree-gemmini-command-issue", commandIssue,
		llvm::cl::desc(
			"How RoCC commands reach Gemmini hardware. 'rocc' (default) "
			"emits custom-3 instructions for a Rocket+RoCC system. 'mmio' "
			"emits volatile stores to (mmioBase + 0x10/0x18/0x00) for "
			"systems where Gemmini sits as a cluster-side MMIO peripheral "
			"(e.g. RadianceGemminiOnlyConfig)."),
		llvm::cl::cat(category));

	binder.opt<int64_t>("iree-gemmini-mmio-base", mmioBase,
		llvm::cl::desc("MMIO base address for the gemmini control window. "
					   "Used only when --iree-gemmini-command-issue=mmio. "
					   "Default 0x40084000 matches "
					   "RadianceGemminiOnlyConfig's cluster-0 GEMMINI_CTRL."),
		llvm::cl::cat(category));

	// Phase-8 LOOP_WS lowering toggle.
	binder.opt<std::string>("iree-gemmini-use-loop-ws", useLoopWs,
		llvm::cl::desc(
			"Use LOOP_WS lowering for gemmini.tile_matmul. One of: "
			"'auto' (default; follows --iree-gemmini-command-issue), "
			"'true' (always emit ~12-command LOOP_WS sequence per matmul), "
			"'false' (always emit per-tile MVIN/PRELOAD/COMPUTE/MVOUT). "
			"Required for the MMIO command-issue path to avoid the "
			"GemminiTile.scala:446 backpressure assertion."),
		llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
