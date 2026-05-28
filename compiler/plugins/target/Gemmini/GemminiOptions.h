#ifndef IREE_GEMMINI_COMPILER_PLUGIN_GEMMINIOPTIONS_H_
#define IREE_GEMMINI_COMPILER_PLUGIN_GEMMINIOPTIONS_H_

#include <string>

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

struct GemminiOptions {
	bool enable = false;

	bool lowerBackToIREE = true;

	bool enableMatmul = true;
	bool enableFP8Matmul = false;
	bool enableConv2D = true;
	bool enableRequantize = true;
	bool enableClamp = true;

	std::string dataflow = "os";

	int64_t tileM = 16;
	int64_t tileN = 16;
	int64_t tileK = 16;

	// GemminiTargetConfig (Phase 4). Defaults match the Spike libgemmini
	// build: DIM=16, ADDR_LEN=32, ACC_ROWS=1024, BANK_ROWS=4096, BANK_NUM=4,
	// elem_t=int8_t (8 bits), acc_t=int32_t (32 bits). Override via the
	// CLI flags below to retarget for a Gemmini variant with different
	// systolic-array dim, scratchpad, or dtype widths (e.g. mxGemmini fp8).
	int64_t dim = 16;
	int64_t addrLen = 32;
	int64_t accRows = 1024;
	int64_t bankRows = 4096;
	int64_t bankNum = 4;
	int64_t elemBits = 8;
	int64_t accBits = 32;

	// mxGemmini selection (Phase 5). All defaults preserve Phase 1-4
	// vanilla-Gemmini RoCC behavior byte-identically.
	//   mxFormat: "disabled" | "fp4" | "fp6_0" | "fp6_1" | "fp8_0" | "fp8_1"
	//   commandIssue: "rocc" (custom-3 instructions, default) | "mmio"
	//                 (volatile stores to mmioBase, used by
	//                 RadianceGemminiOnlyConfig)
	//   mmioBase: address of GEMMINI_CTRL when commandIssue=mmio.
	std::string mxFormat = "disabled";
	std::string commandIssue = "rocc";
	int64_t mmioBase = 0x40084000;
	// Phase-8 LOOP_WS lowering. Tri-state: "auto" (default) follows the
	// command-issue path — true when commandIssue=mmio, false otherwise —
	// "true" / "false" force the explicit setting independently of
	// commandIssue. Mapped through to GemminiTargetConfig::useLoopWs in
	// PluginRegistration.cpp.
	std::string useLoopWs = "auto";

	void bindOptions(OptionsBinder &binder);
	using FromFlags = OptionsFromFlags<GemminiOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_GEMMINI_COMPILER_PLUGIN_GEMMINIOPTIONS_H_
