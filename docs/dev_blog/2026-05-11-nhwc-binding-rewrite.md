# NHWC binding rewrite: design + plan

**Status:** in_progress (Y-Ph7a)
**Goal:** rewrite every cross-dispatch activation tensor from `<C, H, W>` (NCHW collapsed) to `<H, W, C>` (NHWC) so that QNN_HTA accepts the conv-bearing dispatches as `qnn.transpose`-free graphs.

## Why this is necessary

QNN HTA's op set (per `HtaOpDefSupplement.html`) does **not** include a `Transpose` op (only `TransposeConv2d`, which is the strided-conv variant). Our `ConvertLinalgToQNN` patterns emit `qnn.transpose` whenever the source activation arrives in CHW form but the conv internally operates NHWC. After the [[FoldBodyQDQRoundtripPass]] and the preprocessing-phase legalize-to-NHWC pass, the dispatch bodies already operate in NHWC throughout — but the **dispatch input/output bindings remain in CHW** because they reflect the model's source NCHW layout. So the lowering pattern has to insert a transpose between binding-load and conv to bridge the layouts.

Current matrix coverage on yolov8n int8 (156 dispatches):
- 1 real PASS on HTA (the one stem fixture authored NHWC)
- 42 dispatches blocked by `qnn.transpose` from this exact mismatch
- 54 blocked by f32 boundaries (unfixable at compiler level)
- 14 blocked by SDK rejection of MatMul (unfixable at compiler level)
- 12 pattern-coverage gaps (residual SiLU long tail)

Eliminating the cross-dispatch transposes lifts ~40-60 dispatches to HTA.

## Survey of the IR after global-opt

Rank-3 activation tensors (the dispatch-level form after IREE's N=1 batch elision) — these are the rewrite targets:

| Shape | Count | Layout assignment |
|---|---:|---|
| `<64x20x20xi8>` | 80 | `[C, H, W]` → rewrite to `<20x20x64xi8>` |
| `<128x10x10xi8>` | 64 | `[C, H, W]` → `<10x10x128xi8>` |
| `<32x40x40xi8>` | 60 | `[C, H, W]` → `<40x40x32xi8>` |
| `<64x22x22xi8>` | 48 | `[C, H, W]` → `<22x22x64xi8>` |
| `<32x42x42xi8>` | 32 | `[C, H, W]` → `<42x42x32xi8>` |
| `<16x80x80xi8>` | 30 | `[C, H, W]` → `<80x80x16xi8>` |
| `<64x42x42xi8>` | 26 | `[C, H, W]` → `<42x42x64xi8>` |
| `<128x12x12xi8>` | 22 | `[C, H, W]` → `<12x12x128xi8>` |
| `<128x22x22xi8>` | 20 | `[C, H, W]` → `<22x22x128xi8>` |
| `<64x40x40xi8>` | 12 | `[C, H, W]` → `<40x40x64xi8>` |

Rank-4 i8 tensors are weights in HWIO form (`<KH, KW, IC, OC>`). **Don't rewrite** — already correct layout.

Rank-4 f32 tensors are model-input/output boundaries (the "f32 boundary" dispatches we can't fix at compile level anyway):
- 1 × `<1, 3, 320, 320>` (network input)
- Several × `<1, C, H, W>` between Q/DQ boundary dispatches

## Pass design

### Where to run

**Phase: post-flow-dispatch-creation, pre-stream.** At this point each dispatch has clear input/output `flow.dispatch.tensor` types and the bindings are still expressed as flow operations (easier to rewrite than after stream encoding).

In QNN target plugin: add `extendFlowPassPipeline` (or similar hook) that registers the pass between dispatch creation and stream.

### Pass structure

1. **First walk: layout assignment.** Visit every `flow.dispatch.workgroups` op. For each operand/result `flow.dispatch.tensor`:
   - If rank-3 and dim-0 is plausibly a channel count (heuristic: it appears as the broadcast axis in any consumer's bias-broadcast indexing) → assign `NHWC_collapsed` (permute `[1, 2, 0]`).
   - If rank-4 with shape `<N, C, H, W>` where N=1 → assign `NHWC_full` (permute `[0, 2, 3, 1]`).
   - Else → leave alone (weight/scalar/already-permuted).

2. **Second walk: type rewrite via MLIR's TypeConverter.**
   - Convert all `flow.dispatch.tensor` types to permuted form.
   - Update all `iree_tensor_ext.dispatch.tensor.load/store` to use the new shapes.

3. **Third walk: in-dispatch indexing-map rewrite.** For each `linalg.generic` whose operand types were just permuted:
   - Compute the inverse permutation on the affine indexing map.
   - Replace with the rewritten map.
   - Validate the map still type-checks against the new tensor shape.

4. **Fourth walk: boundary transpose insertion.** At module level:
   - Find the model entry — the very first `flow.dispatch` that consumes the function arguments.
   - Insert a single `linalg.transpose` (NCHW→NHWC) before the dispatch's input.
   - Find the model exit — the last `flow.dispatch` whose results are returned by the function.
   - Insert a single `linalg.transpose` (NHWC→NCHW) on the output.
   - These two transposes (and only these two) survive in the final IR.

## Edge cases (Y-Ph7f)

- **`tensor.concat axis`:** if axis was 1 (C in NCHW), remap to last axis (C in NHWC).
- **`tensor.collapse_shape` / `tensor.expand_shape`:** reassoc indices change when reshaping between rank-4 NCHW and rank-3 spatial-flattened forms. Most yolov8 reshapes don't span channel axis, so identity preserves correctness.
- **Residual connections:** a residual `addf` between two convs at different depths uses the same activation type at both ends. As long as the pass assigns the same layout to both, no special handling needed.
- **`linalg.broadcast bias`:** the bias broadcast `dimensions` attribute must remap from `[0, 1, 2]` (broadcast across NHW with C as the rank-1 source) to `[0, 1, 2]` (broadcast across NHW with C at axis 3). Actually since the bias source is rank-1 and target is rank-4, the dimensions list lists which output dims to broadcast over — in NCHW it's `[0, 2, 3]` (N, H, W), in NHWC it's `[0, 1, 2]` (N, H, W). Different list.

## Risk profile

| Risk | Likelihood | Mitigation |
|---|---|---|
| Numerical drift from wrong indexing-map rewrite | High | Y-Ph7g compares output bytes to CPU ref |
| Concat axis bug | High | Add `LayoutClass` audit pass that asserts axis-remap correctness |
| Constants (weights) get permuted by accident | Low | Filter on rank + only-activation-uses |
| IREE's downstream optimizations break on the new shapes | Medium | Test each phase post-rewrite with `--compile-to=<phase>` |

## Acceptance criterion

`compile_dispatch_matrix.py --source yolov8n.q.int8.mlir --targets qnn_hta` shows:
- ≥ 40/156 dispatches with `real=true` HTA feasibility
- 0 regressions on fp32 yolov8 (still 70/92 GPU)
- 0 regressions on baseline fixtures (i8out NHWC stem still passes)

If numerical correctness fails Y-Ph7g, the layout pass is logically correct but plumbing is wrong — debug via per-dispatch isolation tests.
