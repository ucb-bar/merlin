# TVM TensorIR operation support

Current implementation audited: TVM `64bed0cb9c876b89707e69584909abe5e13b0384`.
Read [ARCHITECTURE.md](ARCHITECTURE.md) for the top-down path, remaining custom
integration and paper constraints. This file is the canonical ops document;
[the prior audit](https://github.com/ucb-bar/merlin/blob/9ab339ca2/examples/gemmini/comparisons/tvm/TENSORIR_OP_AUDIT.md)
retains the historical device-ops proposals and measurements. Superseded packing,
bias-folding and XGB descriptions there are not current implementation claims.

Only declarations in the [8ec99308 software specification](https://github.com/ucb-bar/merlin/blob/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/target/software-spec.yaml)
are candidate device contracts (SHA256 `769b77782ea3d7b9bc35c0f21d2797ce09697a0f2aed52c7b8a1dde10a7de9f4`).
The checkout-default spec differs; never substitute it. Similar names or passing
small tests cannot enlarge a declaration's dtype, rank, bound or rounding domain.
The audit is traceability, **not a complete production spec-admission gate**.

## From mathematical TensorIR to primitive commands

For an admitted `C[i,j] += int32(A[i,k])*int32(B[k,j])`, a traced schedule tiles
loops, creates operand/accumulator caches, selects layouts, blockizes and matches
registered TensorIntrin descriptions. The implementation replaces the matched
region with external calls; physical memory lowering supplies concrete SRAM rows.
The C runtime encodes primitive Gemmini commands, with ordinary software loops.
No per-ATen kernel or hardware loop FSM is required.

| TensorIR region | Intrinsic/runtime mapping | Handling beyond a single math instruction |
| --- | --- | --- |
| A/B operand tile copy | `load_a`/`load_b` → `tvm_gemmini_dma_a`/`dma_b` | Strides, alignment, reindex/packing, host pointers and scratchpad allocation. |
| Explicit zero accumulator initialization | `init` → `tvm_gemmini_zero_acc` | Initialization remains explicit; no first-K overwrite optimization. |
| Explicit INT32 seed already in the input reduction | Bias-init TensorIntrin → unscaled full-width accumulator DMA | Shape/seed/bound checks. Post-reduction bias is not automatically folded into this form. |
| INT8 × INT8 → INT32 contraction tile | `compute`/`compute_ws` → primitive preload/compute sequence | OS/WS configuration, legal partial sums and software tile loops. Convolution/batch layouts need explicit scheduling. |
| Raw INT32 accumulator readout | `store` → `tvm_gemmini_dma_out` | Full-width accumulator exit, byte stride, destination ownership and completion. |
| Proved scaled/RNE/saturated readout, optional ReLU | Epilogue TensorIntrin → configured narrow readout | Exact selected scale/order proof. Nonuniform scales can retain the original host epilogue; no scale averaging/specialization pass. |
| Static materialized movement | Copy/region/byte TensorIntrins → DMA | Selector and alias proofs, dtype-bit preservation, fill/gather work and memory hazards. Views can disappear without device execution. |
| Declared bounded affine/residual/sum construction | Family-specific TensorIntrin → multiple load/compute/readout commands | Composite proof, scratch storage and extra commands must be accounted for; not a native affine/sum opcode. |
| Pooled readout | Pooled-mvout implementation with explicit gathered window where used | A standalone matcher is not proof of an admitted producer/readout composition. Host gather/padding work counts. |

These are semantic-block mappings, not a promise of one hardware instruction per
source op or TensorIntrin. Copies default to host unless opted in or selected by
a measured database record; `relu_i8`/`affine_i8` have separate arithmetic handling.
The active FireSim sweep tunes contractions only; generic backend availability
of other family rules does not imply that those families were tuned.

## Exact spec families and remaining admission requirements

All candidate rows below require selected hardware facts, program-specific
arithmetic/composition proofs and instruction/execution evidence. None is an
unconditional hardware-qualified whitelist. Component decomposition is absent.

| Exact YAML declaration (`operations.` prefix) | Current implementation candidate | Strict limit / missing evidence |
| --- | --- | --- |
| `contraction` | i8*i8 -> i32; split/cache/blockize/tensorize; OS/WS primitive compute | Every retained PE partial <=524287 and accumulator partial <=2147483647, including seed; all logical ranks/layout/tails and raw-exit audit required. |
| `elementwise_map` | Producer readout: explicit f32 product, RNE, saturation, optional ReLU | Selected scheme and multiplier bits, scalar command partition, nonnegative scale/order proof, pinned RNE oracle; threshold proof alone does not discharge all emission audits. |
| `elementwise_map_standalone` | Same-shape one/two widened byte terms with signed-byte coefficients and constant i32 bias | Bound all prefixes. Full-domain wrapping i32 add/sub/scalar-add is outside bounded-exact admission; standalone full-i32 readout needs its own bound/identity proof. |
| `elementwise_map_after_operand_sum` | Exact byte sum followed by a single proved narrow readout | Use this exact rounding contract; no bias epilogue; per-scale exhaustive contract and emission evidence remain required. |
| `elementwise_map_after_per_operand_sum` | Two independently rounded/saturated byte operands, exact i32 sum, proved narrow readout | Rank 2 only, no broadcast/bias/pool/raw exit. Screen all 65536 pairs for EVERY scale triple; current two fixtures do not admit arbitrary triples or ranks. |
| `pooling_on_readout` | Existing standalone matcher gathers a window, widens to accumulator, and emits pooled mvout | Declaration is fused readout, not blanket standalone maxpool. Require selected composition/producer proof and complete region audit. Gathered padding is host work, not an implicit padded-pool capability. |
| `reduction_standalone` | Bounded widened-i8 sum; already-present selected reciprocal/readout for window mean | Resolve standalone hardware facts. Mesh-against-ones must separately satisfy contraction rank/bounds; arbitrary f32/i64 reduction is not admitted by similarity. |
| `movement` | Materialized static copies, broadcasts, uniform fills and selectors; legal views can be elided | Resolve actual standalone movement forms/facts and each selector/alias proof. i32/f32 accumulator-carrier copies are not the F1 int8-transfer implementation. |
| `movement_bitwise` | Opaque byte-copy TensorIntrin for bool/i16/f16/bf16/i64/f64; existing i8 DMA | Explicit declared dtype/rank, closed selectors, LE byte storage, disjointness, unit scale, no pooling, nonzero address, bounded rows/bytes. complex<f32> is declared but not implemented. |
| `component_linear` | No general signed-radix-256 component lowering in this TVM branch | Do not relabel byte affine as full integer component support. Requires WS-only component proof and exact host recombination. |
| `component_bilinear` | No general multi-limb bilinear lowering in this TVM branch | Requires WS-only component proof, raw full-i32 exits, bounded partials and exact host recombination; no float/int64 arithmetic inference. |


## Scope restrictions and the 51 original fixtures

Current fixture classifications are 5 array, 0 full-i32 accumulator arithmetic,
1 narrow ReLU, 21 movement, 4 movement-plus-host, 15 host and 5 elided/ABI-copy.
These classify tested typed fixtures, not all overloads or whole frameworks.
Full-domain INT32 add/sub/scalar-add and standalone unbounded INT32 readout
retain host semantics; bounded byte affine is a different program/domain.

Contraction and pooled-readout result ranks are 2–4. Standalone movement,
affine and sum matchers stop at rank five; per-operand rounded sums require rank
two. Rank-five 3D convolution remains host. Byte sum checks include
`128*K <= INT32_MAX`, even for an INT64 output ABI. These restrictions do not
by themselves prove every scale, PE prefix, selector or composition obligation.
Opaque movement supports bool/i16/f16/bf16/i64/f64 bytes alongside existing
byte/word carriers; that does not implement floating or wide-integer arithmetic.
Declared complex storage and general component-linear/bilinear lowering are absent.

The following exact original fixture names remain classified. “Candidate” names
only a possible device subregion, not a whole-op guarantee. Outside-domain
fixtures preserve their original inputs, expected outputs and host arithmetic.

| Historical fixture | Exact declaration candidate | Strict disposition |
| --- | --- | --- |
| `mm` | `contraction` | device_subregion_candidate |
| `bmm` | `contraction` | device_subregion_candidate |
| `addmm` | `contraction` | device_subregion_candidate |
| `convolution` | `contraction` | device_subregion_candidate |
| `convolution_backward` | `movement` | host_arithmetic_with_movement_candidate |
| `add.Tensor` | `elementwise_map_standalone` | outside_declared_exact_domain |
| `sub.Tensor` | None for this typed fixture | host_only_for_baseline |
| `add.Scalar` | `elementwise_map_standalone` | outside_declared_exact_domain |
| `sub.Scalar` | `elementwise_map_standalone` | outside_declared_exact_domain |
| `mul.Scalar` | None for this typed fixture | host_only_for_baseline |
| `div.Scalar` | None for this typed fixture | host_only_for_baseline |
| `neg` | None for this typed fixture | host_only_for_baseline |
| `relu` | `elementwise_map_standalone` | device_subregion_candidate |
| `mean.dim` | None for this typed fixture | host_only_for_baseline |
| `avg_pool1d` | `movement` | host_arithmetic_with_movement_candidate |
| `avg_pool2d` | `movement` | host_arithmetic_with_movement_candidate |
| `avg_pool3d` | `movement` | host_arithmetic_with_movement_candidate |
| `adaptive_avg_pool1d` | None for this typed fixture | host_only_for_baseline |
| `_adaptive_avg_pool2d` | None for this typed fixture | host_only_for_baseline |
| `_adaptive_avg_pool3d` | None for this typed fixture | host_only_for_baseline |
| `avg_pool2d_backward` | None for this typed fixture | host_only_for_baseline |
| `_adaptive_avg_pool2d_backward` | None for this typed fixture | host_only_for_baseline |
| `col2im` | None for this typed fixture | host_only_for_baseline |
| `clone` | `movement` | elided_or_host_abi_copy |
| `repeat` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `cat` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `slice_scatter` | `movement` | movement_facts_required |
| `select_scatter` | `movement` | movement_facts_required |
| `fill.Scalar` | `movement` | elided_or_host_abi_copy |
| `full` | `movement` | elided_or_host_abi_copy |
| `full_like` | `movement` | elided_or_host_abi_copy |
| `constant_pad_nd` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `reflection_pad1d` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `reflection_pad2d` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `reflection_pad3d` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `replication_pad2d` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `replication_pad3d` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `_to_copy` | `elementwise_map_standalone` | widening_candidate |
| `as_strided` | `movement` | movement_facts_required |
| `diagonal` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `permute` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `scalar_tensor` | `movement` | elided_or_host_abi_copy |
| `select.int` | `movement` | movement_facts_required |
| `slice.Tensor` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `split_with_sizes` | `movement` | movement_facts_required |
| `squeeze.dim` | `movement` | movement_facts_required |
| `squeeze.dims` | `movement` | movement_facts_required |
| `unsqueeze` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `view` | `movement`, `movement_bitwise` | movement_or_view_candidate |
| `mean.default` | None for this typed fixture | host_only_for_baseline |
| `sum.dim_IntList` | `reduction_standalone`, `contraction` | device_subregion_candidate |


## Evidence and reproduction

[Current local verification](../../../../out/artifacts/verification/gemmini-replay-defaults/v1/verification_gemmini-replay-defaults_v1_20261028T040938Z_9794880/README.md)
contains numerical/replay/control tests, including all 51 original inventory
subtests and the new reuse default. This is host/emulator evidence, not new
FireSim cycles. [The older 71-case diagnostic Spike release](../../../../out/artifacts/handoff/gemmini-simplified-tvm/v1/handoff_gemmini-simplified-tvm_v1_20261028T021958Z_dd079d3/README.md)
binds TVM `619c7c202`; it is not execution evidence for changed schedules or the
unavailable spec-pinned oracle. No frozen source/result identity is relabeled.

`audit_software_spec.py --spec PINNED_YAML --tvm TVM_CHECKOUT --output NEW_JSON`
checks the exact specification hash and accounts for all 51 fixture names.
It emits declaration/source identities and outstanding obligations, with
`execution_admission: false`. It does not prove compilation, launch safety or
correctness of arbitrary matched programs. Resolve the listed proofs and inspect
actual placement before admitting benchmark ELFs; eligible device work must not
silently become an unreported CPU baseline. See [FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md)
for remaining complete-call builders and remote qualification requirements.

## Current baseline Spike evidence

[SPIKE_HANDOFF_EVIDENCE.md](SPIKE_HANDOFF_EVIDENCE.md) records the fresh 74-case
operator suite (51 original, 20 device controls and three reuse/tail controls),
all 16 complete kernel cases and ResNet checks. The repaired diagnostic oracle
is distinct from the missing spec-pinned provider. Passing finite fixtures does
not turn candidate families into a fully proved hardware whitelist.
