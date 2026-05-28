#ifndef IREE_MERLIN_COMPILER_DIALECT_QNN_TRANSFORMS_PASSES_H_
#define IREE_MERLIN_COMPILER_DIALECT_QNN_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::QNN {

// Pattern-matches recognized linalg conv/elementwise/etc patterns and rewrites
// them as `qnn.*` ops. Mirrors the Python recognizers in
// `kernels/qnn/recognizers/` 1:1 — each `try_recognize` returning a
// `QnnGraphDesc` becomes one C++ rewrite pattern. See the rosy-sundae plan
// (Phase 2) for context.
std::unique_ptr<Pass> createConvertLinalgToQNNPass();

// NHWC layout legalization: insert `linalg.transpose` at the boundaries of any
// linalg conv/pool op whose input is NCHW so subsequent QNN conversion sees
// the NHWC form its verifier requires. Replaces the upstream IREE
// `iree-preprocessing-convert-conv-to-channels-last` pass which produces
// invalid IR for `linalg.conv_2d_nchw_fchw_q`.
std::unique_ptr<Pass> createLegalizeLayoutToNHWCPass();

// QDQ → quant.uniform fold (Phase 2 / scalable quant): walks QDQ-decomposed
// linalg.generic chains, extracts scale/zp from body constants, rewrites
// producer/consumer tensor types to wrap with `quant.uniform<i8:f32, scale,
// zp>`. Universal across pt2e / TFLite / ONNX-QDQ frontends.
std::unique_ptr<Pass> createRewriteQDQToQuantUniformPass();

// NCHW→NHWC activation-tensor rewrite at module level. For yolov8n int8,
// every cross-dispatch activation comes in as `<C, H, W>` (rank-3,
// N=1-elided). HTA refuses `qnn.transpose` ops, so our conv emitters
// internally wrap the conv with transposes when source is CHW. This pass
// rewrites those activation tensors to `<H, W, C>` form so the in-conv
// transposes are unnecessary. Inserts boundary NCHW↔NHWC transposes at
// the function entry/exit to preserve user-visible signature. See
// `docs/dev_blog/2026-05-11-nhwc-binding-rewrite.md`.
std::unique_ptr<Pass> createRewriteToNHWCBindingsPass();

// Y-Ph7d follow-up: lower the `unrealized_conversion_cast` ops left by
// RewriteToNHWCBindings to explicit `linalg.transpose` ops where the
// types are layout permutations of each other, or DCE when they're
// identity. Required to make the NHWC pipeline produce valid IR for
// downstream codegen.
std::unique_ptr<Pass> createLowerNHWCCastsToTransposesPass();

// Within-body QDQ-roundtrip folder: walks each `linalg.generic` region and
// rewrites the `mulf(sitofp(fptosi(min(max(addf(roundeven(divf(x, s)),
// zp), -128), 127))), s)` chain back to `x` (when zp=0 and x is in the
// valid quant range, this is identity to within roundeven noise). Required
// to simplify yolov8n int8 dispatch bodies whose ONNX-QDQ export inserts
// dequant-requant cycles between every quant op, producing 4-stage fused
// bodies that no single pattern can match. After folding, the body
// collapses to standard `conv-bias-(activation)-quantize` form.
std::unique_ptr<Pass> createFoldBodyQDQRoundtripPass();

// Schedule-driven re-quantization (Phase A2 of the heterogeneous-scheduling
// pipeline). Reads the placement_requant.json sidecar emitted by
// XPU-RT/scripts/heterogeneous_loop.py and inserts quant.qcast/dcast
// round-trip pairs around source-level anchor ops whose corresponding
// dispatch's placement requires a different dtype than the source IR.
std::unique_ptr<Pass> createApplyPlacementRequantizationPass(
	StringRef sidecar_path = "");

// Y-Ph7h: inline constant-initialized `util.global` tensors. For each
// `util.global private` whose `initial_value` is a dense ElementsAttr,
// replace every `util.global.load` of it with an `arith.constant` at the
// load site, then erase the global. Required for QNN HTA conv weights:
// HTA's Conv2d validator demands `QNN_TENSOR_TYPE_STATIC` weights, which
// `SerializeGraph::extractConstantBytes` only produces when the conv
// weight operand is an inline `arith.constant`. Without this pass IREE's
// frontend hoists weights into `util.global` with `inlining_policy =
// #util.inline.never`, leaving them as `dispatch.tensor.load` bindings
// (APP_WRITE at runtime) which HTA rejects at `graphAddNode rc=6000`.
// Must run BEFORE dispatch creation so the materialized constants live
// in func scope where IREE's dispatch-creation clone-into-region step
// can pull them into the dispatch body.
std::unique_ptr<Pass> createInlineConstantUtilGlobalsPass();

void registerQNNPasses();

} // namespace mlir::iree_compiler::QNN

#endif // IREE_MERLIN_COMPILER_DIALECT_QNN_TRANSFORMS_PASSES_H_
