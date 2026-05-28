//===- AttachCompilationInfo.cpp - Pin gemmini codegen pipeline ----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//
//
// Walks `linalg.matmul` operations in a module and attaches an
// `iree_codegen.compilation_info` attribute pointing at a custom textual
// pipeline (`iree_codegen.pass_pipeline`) that runs the gemmini recovery
// and ISA-tier lowering inside the dispatch executable.
//
// This is the IREE-blessed plugin extension point for codegen pipelines:
// `MaterializeUserConfigsPass` propagates the `compilation_info`'s
// `translation_info` to the dispatch func, and
// `LLVMCPULowerExecutableTargetPass` calls
// `PipelineAttrInterface::buildPipeline` which parses our textual pipeline
// string and runs the gemmini passes against the dispatch body.
//
// Net effect: `linalg.matmul` reaches the dispatch (via IREE's standard
// outliner), our pipeline runs on it (recovering `gemmini.matmul_tile`,
// bufferizing, lowering to `gemmini.tile_matmul`, then to
// `gemmini.intr.*`), and the GemminiToLLVMIR translation interface
// produces `llvm.intr.riscv.*` calls that the RISC-V backend lowers to
// custom-3 RoCC opcodes.
//
//===----------------------------------------------------------------------===//

#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.h"
#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::Gemmini {

// 2026-05-26 Opt#E: enable native tile_conv path. AttachCompilationInfo
// now tags conv-shaped linalg.generic ops with the same textual codegen
// pipeline as matmuls; LowerBufferizedLinalgConvToTileConv in
// LowerTileToISA.cpp then fires inside the dispatch executable to rewrite
// the bufferized 6-loop conv (plus its bias-add + rescale chain) into a
// single gemmini.tile_conv. Requires `--iree-global-opt-convert-conv2d-to-
// img2col` to be REMOVED from the preprocessing pipeline so convs survive
// to this point as linalg.generic 6-loop ops.
static constexpr bool kAttachOnConv2D = true;

static bool looksLikeConv2DGeneric(linalg::GenericOp op) {
	auto iters = op.getIteratorTypesArray();
	if (iters.size() != 6)
		return false;
	int nParallel = 0, nReduction = 0;
	for (auto kind : iters) {
		if (kind == utils::IteratorType::parallel)
			++nParallel;
		else if (kind == utils::IteratorType::reduction)
			++nReduction;
	}
	if (nParallel != 3 || nReduction != 3)
		return false;
	auto maps = op.getIndexingMapsArray();
	if (maps.size() != 3 && maps.size() != 5)
		return false;
	if (maps.back().getNumResults() != 3)
		return false;
	return true;
}

// Opt#G — yolov8n's `matmul_like_*` 4D-QDQ form. 1×1 convs that have been
// img2col'd (so the kh/kw dims are gone, leaving 3 parallel iters for
// f/h/w and 1 reduction for c) but NOT yet flattened to a 2D matmul. 5
// ins (feat, weight, zp_feat, zp_weight, [optional bias]) with the
// canonical extsi/subi-zp/extsi/subi-zp/muli/addi/yield body.
//
// Without attaching the gemmini pipeline here, these dispatches stay on
// scalar CPU codegen at ~50–170M cycles each (matmul_like_32x6400x32
// alone is 167M).
static bool looksLikeMatmulLikeGeneric(linalg::GenericOp op) {
	auto iters = op.getIteratorTypesArray();
	if (iters.size() != 4)
		return false;
	int nParallel = 0, nReduction = 0;
	for (auto kind : iters) {
		if (kind == utils::IteratorType::parallel)
			++nParallel;
		else if (kind == utils::IteratorType::reduction)
			++nReduction;
	}
	if (nParallel != 3 || nReduction != 1)
		return false;
	auto maps = op.getIndexingMapsArray();
	// 5 maps: feat, weight, zp_feat, zp_weight, out. (Also accept 3 maps in
	// case IREE has already folded the zeros.)
	if (maps.size() != 3 && maps.size() != 5)
		return false;
	if (maps.back().getNumResults() != 3)
		return false;
	// 2D weight + 3D feature map — matmul-like shape.
	if (op.getNumDpsInputs() < 2)
		return false;
	auto featTy = dyn_cast<RankedTensorType>(op.getDpsInputs()[0].getType());
	auto wTy = dyn_cast<RankedTensorType>(op.getDpsInputs()[1].getType());
	if (!featTy || !wTy)
		return false;
	if (featTy.getRank() != 3 || wTy.getRank() != 2)
		return false;
	if (!featTy.getElementType().isSignlessInteger(8))
		return false;
	if (!wTy.getElementType().isSignlessInteger(8))
		return false;
	return true;
}

// Textual pipeline that runs inside the dispatch executable. The string is
// parsed by `PassPipelineAttr::buildPipeline` via MLIR's
// `parsePassPipeline`, and `LLVMCPULowerExecutableTargetPass` invokes that
// against a `func::FuncOp`-rooted `OpPassManager`. So the pipeline string
// must NOT wrap with `builtin.module(func.func(...))` — the passes here
// directly receive a func-level OpPassManager.
//
// Order matters:
//   1. IREE bufferization (linalg.matmul tensor → memref).
//   2. Lower the bufferized linalg.matmul down to memref ISA-tier ops
//      (gemmini.tile_matmul). Skipping the gemmini.* tensor-tier — those
//      exist only for the host-IR debug path (lowerBackToIREE=true) and
//      don't implement BufferizableOpInterface.
//   3. Strip the HAL descriptor_type memory-space (so the LLVM type
//      converter can lower memrefs to plain integer address spaces — same
//      placement as IREE's own LLVMCPU codegen pipeline,
//      LLVMCPU/Passes.cpp:638).
//   4. Legalize the tile-level memref ops onto `gemmini.intr.*` RoCC
//      intrinsic ops. The Phase-4 hardware descriptor (DIM, ADDR_LEN, …)
//      is formatted into the pass-options braces of this step.
//      The IREE LLVM-CPU codegen pipeline then runs its standard
//      `convert-to-llvm` after our pipeline returns, and the
//      `gemmini.intr.*` ops travel through the GemminiToLLVMIR translation
//      interface to become `llvm.intr.riscv.*` calls.
static std::string buildGemminiDispatchPipeline(int64_t dim, int64_t addrLen,
	int64_t accRows, int64_t bankRows, int64_t elemBits, int64_t accBits,
	int64_t mxFormat, llvm::StringRef commandIssue, int64_t mmioBase,
	bool loopWs, bool dispatchDebug) {
	std::string result;
	llvm::raw_string_ostream os(result);
	os << "iree-codegen-llvmcpu-bufferization-pipeline,"
		  "gemmini-lower-tile-to-isa,"
		  "canonicalize,"
		  "cse,"
		  "iree-codegen-erase-hal-descriptor-type-from-memref,"
		  "merlin-gemmini-legalize-for-llvm-export{dim="
	   << dim << " addr-len=" << addrLen << " acc-rows=" << accRows
	   << " bank-rows=" << bankRows << " elem-bits=" << elemBits
	   << " acc-bits=" << accBits << " mx-format=" << mxFormat
	   << " command-issue=" << commandIssue
	   << " loop-ws=" << (loopWs ? "true" : "false")
	   << " dispatch-debug=" << (dispatchDebug ? "true" : "false") << "},";
	// Only insert the MMIO step when the user opts in. The default
	// (commandIssue="rocc") preserves Phase 1-4 behavior byte-identically:
	// gemmini.intr.* ops survive into the LLVM-IR translation interface
	// where they become llvm.intr.riscv.* (custom-3 RoCC instructions).
	if (commandIssue.equals_insensitive("mmio")) {
		os << "gemmini-lower-intr-to-mmio{mmio-base=" << mmioBase << "},";
	}
	os << "canonicalize,cse";
	return result;
}

namespace {

// Build a minimal `iree_codegen.compilation_info` pinning the textual pipeline
// above. The lowering config carries a `distribution` tile-sizes entry that
// caps the per-workgroup M dim at `dim` (=16). This makes IREE's dispatch
// distributor split a matmul with M > dim into ceil(M/dim) workgroups, each
// computing a 16-row slice of the output.
//
// Why this is needed: the plugin's `spTiledMatmulWs` codegen (called from
// `tiledMatmulOuter` once per outer (i0,j0,k0) iteration) emits, *for each
// inner tile triple*, a block of [MVIN-A x tileI*tileK, MVIN-B x tileK*tileJ,
// MVIN-D x tileI*tileJ, COMPUTEs x tileI*tileJ*tileK, MVOUTs x tileI*tileJ].
// When tileI > 1 (e.g., dronet conv1's M=3136 picks tileI=16 via the autotile
// loop), one inner-tile triple issues 16x more compute commands than the
// accumulator's drain rate can absorb between MVOUTs, and the Gemmini RoCC
// command queue deadlocks on FireSimGemminiAndOPUShuttleConfig. mlp_wide is
// unaffected because all its matmuls have M = dim = 16 -> tileI = 1.
//
// Capping the workgroup at M=dim forces every dispatch to enter the
// `spTiledMatmulWs(tileI=1, ...)` path that mlp_wide already exercises
// successfully. The IREE dispatch distributor splits the matmul across
// ceil(M/dim) workgroups; the plugin sees each as a clean M=dim sub-matmul.
static IREE::Codegen::CompilationInfoAttr buildCompilationInfo(
	MLIRContext *ctx, StringRef pipelineString, int64_t dim) {
	Attribute pipelineAttr =
		IREE::Codegen::PassPipelineAttr::get(ctx, pipelineString);
	// Use the auto-generated AttrDef::get that takes the pipelineAttr as a
	// generic Attribute (so we can plug in our PipelineAttrInterface impl
	// rather than the DispatchLoweringPassPipeline enum).
	auto translationInfo =
		IREE::Codegen::TranslationInfoAttr::get(ctx, pipelineAttr,
			/*codegenSpec=*/SymbolRefAttr(),
			/*workgroupSize=*/ArrayRef<int64_t>{},
			/*subgroupSize=*/int64_t(0),
			/*configuration=*/DictionaryAttr());
	// Workgroup distribution tile sizes: [M=dim, N=0 (no tile), K=0 (no tile)].
	// The "distribution" key is the workgroup-level tiling level recognised by
	// `IREE::CPU::LoweringConfigAttr::getWorkgroupTileSizes()` (see
	// `IREECPUAttrs.cpp:216` -> `TilingLevel::DistributionTiles`).
	auto distTilingLevel = IREE::Codegen::LoweringConfigTilingLevelAttr::get(
		ctx, /*sizes=*/ArrayRef<int64_t>{dim, 0, 0},
		/*interchange=*/ArrayRef<int64_t>{},
		/*scalableFlags=*/ArrayRef<bool>{});
	SmallVector<NamedAttribute> items;
	items.emplace_back(StringAttr::get(ctx, "distribution"), distTilingLevel);
	auto loweringConfig = IREE::CPU::LoweringConfigAttr::get(ctx, items);
	return IREE::Codegen::CompilationInfoAttr::get(
		ctx, loweringConfig, translationInfo);
}

#define GEN_PASS_DEF_GEMMINIATTACHCOMPILATIONINFOPASS
#include "compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h.inc"

struct GemminiAttachCompilationInfoPass final
	: impl::GemminiAttachCompilationInfoPassBase<
		  GemminiAttachCompilationInfoPass> {
	using Base = impl::GemminiAttachCompilationInfoPassBase<
		GemminiAttachCompilationInfoPass>;
	using Base::Base;

	explicit GemminiAttachCompilationInfoPass(
		const GemminiTransformOptions &transformOpts) {
		this->dim = transformOpts.target.dim;
		this->addrLen = transformOpts.target.addrLen;
		this->accRows = transformOpts.target.accRows;
		this->bankRows = transformOpts.target.bankRows;
		this->elemBits = transformOpts.target.elemBits;
		this->accBits = transformOpts.target.accBits;
		this->mxFormat = static_cast<int64_t>(transformOpts.target.mxFormat);
		this->commandIssue =
			(transformOpts.target.commandIssue == CommandIssue::MMIO) ? "mmio"
																	  : "rocc";
		this->mmioBase = transformOpts.target.mmioBase;
		this->loopWs = transformOpts.target.useLoopWs;
		this->dispatchDebug = transformOpts.target.dispatchDebug;
	}

	void runOnOperation() override {
		FunctionOpInterface func = getOperation();
		std::string pipelineString =
			buildGemminiDispatchPipeline(this->dim.getValue(),
				this->addrLen.getValue(), this->accRows.getValue(),
				this->bankRows.getValue(), this->elemBits.getValue(),
				this->accBits.getValue(), this->mxFormat.getValue(),
				this->commandIssue.getValue(), this->mmioBase.getValue(),
				this->loopWs.getValue(), this->dispatchDebug.getValue());
		IREE::Codegen::CompilationInfoAttr info = buildCompilationInfo(
			&getContext(), pipelineString, this->dim.getValue());
		// Attach to named linalg.matmul ops AND to linalg.generic ops that
		// implement ContractionOpInterface (matmul-shaped generics produced
		// by IREE's GlobalOptimization pipeline before generalization is
		// disabled). Other linalg.generic ops (e.g. fills, elementwise)
		// don't trigger our pipeline.
		func.walk([&](Operation *op) {
			if (isa<linalg::MatmulOp>(op)) {
				setCompilationInfo(op, info);
				return;
			}
			if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
				// Matmul-shaped generic (M×K · K×N).
				if (linalg::isaContractionOpInterface(generic)) {
					setCompilationInfo(op, info);
					return;
				}
				// Conv2D-shaped generic. ConvertToGemmini will
				// rewrite it to gemmini.conv2d → tile_matmul →
				// loop_conv_ws RoCC opcodes inside the dispatch.
				// MaterializeUserConfigsPass propagates this
				// compilation_info to the dispatch func.
				if (kAttachOnConv2D && looksLikeConv2DGeneric(generic)) {
					setCompilationInfo(op, info);
					return;
				}
				// Opt#G — yolov8n matmul_like_* 4D-QDQ form (img2col'd
				// 1×1 convs). The inside-dispatch pipeline plus the
				// `CanonicalizeSwappedMatmulLikeGeneric` pattern in
				// LowerTileToISA take it from there.
				if (looksLikeMatmulLikeGeneric(generic)) {
					setCompilationInfo(op, info);
					return;
				}
			}
		});
	}
};

} // namespace

std::unique_ptr<Pass> createGemminiAttachCompilationInfoPass() {
	return std::make_unique<GemminiAttachCompilationInfoPass>();
}

std::unique_ptr<Pass> createGemminiAttachCompilationInfoPassWithOptions(
	const GemminiTransformOptions &options) {
	return std::make_unique<GemminiAttachCompilationInfoPass>(options);
}

} // namespace mlir::iree_compiler::Gemmini
