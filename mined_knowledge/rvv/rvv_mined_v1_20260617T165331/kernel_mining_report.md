# Kernel mining report

## Corpus summary
- **Total kernels indexed:** 675
- **openblas:** 130
    - ops: gemm×14, pack×12, gemv×10, trmm×9, amax×8, amin×8, scal×6, symv×6
- **xnnpack:** 545
    - ops: other×222, gemm×70, vcvt×25, dwconv×23, vadd×20, vmul×20, vclamp×16, dwconv2d×16

## Motifs
| motif | kernels | sources | verdict |
|---|---:|---|:--|
| intrinsic_lowering | 628 | openblas, xnnpack | structural (no policy) |
| vector_length_polymorphic | 620 | openblas, xnnpack | ✅ policy |
| vl_polymorphic_tail | 620 | openblas, xnnpack | ✅ policy |
| lmul_grouping | 496 | openblas, xnnpack | ✅ policy |
| accumulator_lifetime | 376 | openblas, xnnpack | structural (no policy) |
| epilogue_before_commit | 308 | openblas, xnnpack | structural (no policy) |
| tiling_blocking | 295 | openblas, xnnpack | structural (no policy) |
| int8_widening_mac | 77 | xnnpack | ✅ policy |
| vector_reduction | 75 | openblas, xnnpack | ✅ policy |
| accumulator_commit | 73 | xnnpack | ✅ policy |
| packed_rhs | 62 | openblas, xnnpack | ✅ policy |
| requant_narrowing | 46 | xnnpack | ✅ policy |
| scalar_broadcast_fma | 37 | openblas, xnnpack | ✅ policy |
| reused_packed_rhs | 34 | xnnpack | structural (no policy) |

_Promotion gate: ≥2 sources OR ≥10 kernels. 'structural' motifs clear the gate but are intentionally not mapped to a policy (too ubiquitous to be actionable)._

### Promoted abstraction candidates
- **resident_packed_tensor** (memory_state) — immutable RHS/weight is packed once and reused across a region; keep it resident to avoid repeated pack/load.
    - interface_features: resident_pack, resident_tensor_type, evict
    - evidence: openblas_rvv_gemm, openblas_rvv_trmm, xnnpack_rvv_gemm
- **accumulator_commit** (memory_state) — on a contraction op the accumulator stays live across a bias/requant/activation epilogue; commit to memory only after the epilogue to avoid extra writes.
    - interface_features: accumulator_type, commit, keep_accumulator_live
    - evidence: xnnpack_rvv_dwconv, xnnpack_rvv_gemm

### Promoted policy rules
- **packed_rhs_policy** — when: rhs_reuse_count >= 2, rhs_mutable false
    - actions: preserve_packed_rhs_layout, hoist_pack, consider_resident_packed_tensor
    - evidence: openblas_rvv_gemm, openblas_rvv_trmm, xnnpack_rvv_gemm
- **accumulator_commit_policy** — when: op gemm|matmul|conv, has_epilogue true, accumulator_live_across_epilogue true
    - actions: keep_accumulator_resident, fuse_epilogue_before_commit, single_commit_store
    - evidence: xnnpack_rvv_dwconv, xnnpack_rvv_gemm
- **vl_agnostic_loop_policy** — when: target_has_scalable_vectors true
    - actions: emit_vl_agnostic_loop, use_predicated_or_vl_tail, avoid_fixed_width_assumptions
    - evidence: openblas_rvv_amax, openblas_rvv_amin, openblas_rvv_asum, openblas_rvv_axpby, openblas_rvv_axpy, openblas_rvv_copy, openblas_rvv_dot, openblas_rvv_gemm, openblas_rvv_gemv, openblas_rvv_hemv, openblas_rvv_max, openblas_rvv_min, openblas_rvv_nrm2, openblas_rvv_pack, openblas_rvv_rot, openblas_rvv_scal, openblas_rvv_sum, openblas_rvv_swap, openblas_rvv_symv, openblas_rvv_transpose, openblas_rvv_trmm, xnnpack_rvv_avgpool, xnnpack_rvv_dwconv, xnnpack_rvv_dwconv2d, xnnpack_rvv_gemm, xnnpack_rvv_maxpool, xnnpack_rvv_other, xnnpack_rvv_rdsum, xnnpack_rvv_rminmax, xnnpack_rvv_rsum, xnnpack_rvv_transpose, xnnpack_rvv_vadd, xnnpack_rvv_vclamp, xnnpack_rvv_vcvt, xnnpack_rvv_vdiv, xnnpack_rvv_velu, xnnpack_rvv_vexp, xnnpack_rvv_vgelu, xnnpack_rvv_vmul, xnnpack_rvv_vmulcaddc, xnnpack_rvv_vrsqrt, xnnpack_rvv_vsigmoid, xnnpack_rvv_vsqrt, xnnpack_rvv_vsub, xnnpack_rvv_vtanh
- **lmul_grouping_policy** — when: target_has_scalable_vectors true, op gemm|matmul|conv|dot, dtype f32|i8|bf16
    - actions: prefer_high_lmul, set_vector_group_m4_or_m8
    - evidence: openblas_rvv_amax, openblas_rvv_amin, openblas_rvv_asum, openblas_rvv_axpby, openblas_rvv_axpy, openblas_rvv_copy, openblas_rvv_dot, openblas_rvv_gemm, openblas_rvv_gemv, openblas_rvv_hemv, openblas_rvv_max, openblas_rvv_min, openblas_rvv_nrm2, openblas_rvv_rot, openblas_rvv_scal, openblas_rvv_sum, openblas_rvv_swap, openblas_rvv_symv, openblas_rvv_transpose, openblas_rvv_trmm, xnnpack_rvv_avgpool, xnnpack_rvv_dwconv, xnnpack_rvv_dwconv2d, xnnpack_rvv_gemm, xnnpack_rvv_maxpool, xnnpack_rvv_other, xnnpack_rvv_rdsum, xnnpack_rvv_rminmax, xnnpack_rvv_rsum, xnnpack_rvv_transpose, xnnpack_rvv_vadd, xnnpack_rvv_vclamp, xnnpack_rvv_vcvt, xnnpack_rvv_vdiv, xnnpack_rvv_velu, xnnpack_rvv_vexp, xnnpack_rvv_vgelu, xnnpack_rvv_vmul, xnnpack_rvv_vmulcaddc, xnnpack_rvv_vrsqrt, xnnpack_rvv_vsigmoid, xnnpack_rvv_vsqrt, xnnpack_rvv_vsub, xnnpack_rvv_vtanh
- **fma_broadcast_policy** — when: op gemm|matmul, rhs_reuse_count >= 1
    - actions: emit_scalar_broadcast_fma, fuse_multiply_add, register_block_rhs
    - evidence: openblas_rvv_gemm, openblas_rvv_gemv, openblas_rvv_trmm, xnnpack_rvv_gemm
- **int8_widening_policy** — when: dtype i8, op gemm|matmul|conv
    - actions: use_vwmacc_widening, i32_accumulator
    - evidence: xnnpack_rvv_dwconv, xnnpack_rvv_gemm, xnnpack_rvv_other, xnnpack_rvv_rdsum, xnnpack_rvv_rsum, xnnpack_rvv_vcvt, xnnpack_rvv_vmul
- **vl_tail_policy** — when: target_has_scalable_vectors true
    - actions: emit_vsetvl_loop, vl_or_mask_tail
    - evidence: openblas_rvv_amax, openblas_rvv_amin, openblas_rvv_asum, openblas_rvv_axpby, openblas_rvv_axpy, openblas_rvv_copy, openblas_rvv_dot, openblas_rvv_gemm, openblas_rvv_gemv, openblas_rvv_hemv, openblas_rvv_max, openblas_rvv_min, openblas_rvv_nrm2, openblas_rvv_pack, openblas_rvv_rot, openblas_rvv_scal, openblas_rvv_sum, openblas_rvv_swap, openblas_rvv_symv, openblas_rvv_transpose, openblas_rvv_trmm, xnnpack_rvv_avgpool, xnnpack_rvv_dwconv, xnnpack_rvv_dwconv2d, xnnpack_rvv_gemm, xnnpack_rvv_maxpool, xnnpack_rvv_other, xnnpack_rvv_rdsum, xnnpack_rvv_rminmax, xnnpack_rvv_rsum, xnnpack_rvv_transpose, xnnpack_rvv_vadd, xnnpack_rvv_vclamp, xnnpack_rvv_vcvt, xnnpack_rvv_vdiv, xnnpack_rvv_velu, xnnpack_rvv_vexp, xnnpack_rvv_vgelu, xnnpack_rvv_vmul, xnnpack_rvv_vmulcaddc, xnnpack_rvv_vrsqrt, xnnpack_rvv_vsigmoid, xnnpack_rvv_vsqrt, xnnpack_rvv_vsub, xnnpack_rvv_vtanh
- **vector_reduction_policy** — when: op softmax|layernorm|rmsnorm|dot|reduce, target_has_scalable_vectors true
    - actions: emit_vector_reduction_tree, use_vredsum_or_vfredusum
    - evidence: openblas_rvv_amax, openblas_rvv_amin, openblas_rvv_asum, openblas_rvv_axpy, openblas_rvv_dot, openblas_rvv_gemv, openblas_rvv_hemv, openblas_rvv_max, openblas_rvv_min, openblas_rvv_nrm2, openblas_rvv_sum, openblas_rvv_symv, openblas_rvv_trmm, xnnpack_rvv_other, xnnpack_rvv_rminmax, xnnpack_rvv_rsum
- **requant_narrowing_policy** — when: dtype i8, has_epilogue true
    - actions: fuse_requant_narrowing_store, emit_vnclip_then_vse8
    - evidence: xnnpack_rvv_dwconv, xnnpack_rvv_gemm, xnnpack_rvv_other, xnnpack_rvv_vcvt, xnnpack_rvv_vmul

### Interface candidates (L5) — exposed via the 4 lowering variants
- **resident_packed_tensor** — ops: resident_pack, matmul_resident, evict; types: resident_tensor
    - compiler must prove: rhs_immutable, reuse_count_above_threshold, capacity_fit_or_eviction_inserted, consumers_accept_packed_layout
    - hardware must provide: resident_storage, packed_tensor_handle, validity_until_eviction
    - runtime must provide: persistent_handle_lifetime, command_ordering, invalidation_protocol
    - lowering variants: baseline, software_visible, hardware_managed, oracle
- **accumulator_commit** — ops: accumulator, commit; types: accumulator
    - compiler must prove: epilogue_consumes_accumulator_immediately, no_intervening_user_visible_materialization, output_dtype_and_layout_known
    - hardware must provide: accumulator_state, commit_epilogue_path
    - runtime must provide: command_ordering
    - lowering variants: baseline, software_visible, hardware_managed, oracle

### Runtime candidates (L7)
- _(none)_

### Dialect requirements (L6 — input to TargetGen, status `proposed`)
- **resident_packed_tensor** @ toy_npu — ops: resident_pack, matmul_resident, evict; types: resident_tensor; verifiers: capacity_constraint, lifetime_constraint, layout_constraint
- **accumulator_commit** @ toy_npu — ops: accumulator, commit; types: accumulator; verifiers: no_intervening_materialization, output_dtype_known, epilogue_adjacency

### LLVM requirements (L8)
- All 2 emitted with `requires_llvm_fork: false` — no machine-code change is justified until Stage F (target lowering) and Stage G (exploitability) pass. Recorded fork triggers name what *would* justify one.

## Actionability scorecard

| policy | kernels | sources | op families | Stage-D | regime sweep | drives | falsifier | next step |
|---|---:|---:|---:|---|---|---|---|---|
| packed_rhs_policy | 62 | 2 | 2 | repeated:holds; no:correctly_silent; capacity:holds | fires 16/20, controls silent | `merlin.schedule.hoist_pack` → `merlin.interface.resident_pack` | no_reuse_matmul; mutable-RHS control | Stage F: lower `resident_packed_tensor` per dialect requirement (toy_npu) |
| accumulator_commit_policy | 73 | 1 | 2 | matmul:holds; no:correctly_silent | shape-independent | `merlin.interface.accumulator` + `commit` | no_reuse_matmul (no epilogue) | Stage F: lower `accumulator_commit` per dialect requirement (toy_npu) |
| vl_agnostic_loop_policy | 620 | 2 | 43 | no workload mapped | shape-independent | `merlin.schedule` VL-polymorphic loop emission | fixed-width-only target | measure on real shapes (no HW/SW interface needed) |
| lmul_grouping_policy | 496 | 2 | 42 | no workload mapped | shape-independent | — | — | measure on real shapes (no HW/SW interface needed) |
| fma_broadcast_policy | 37 | 2 | 3 | no workload mapped | fires 0/20, controls silent | — | — | measure on real shapes (no HW/SW interface needed) |
| int8_widening_policy | 77 | 1 | 7 | no workload mapped | shape-independent | — | — | measure on real shapes (no HW/SW interface needed) |
| vl_tail_policy | 620 | 2 | 43 | no workload mapped | shape-independent | — | — | measure on real shapes (no HW/SW interface needed) |
| vector_reduction_policy | 75 | 2 | 16 | no workload mapped | shape-independent | — | — | measure on real shapes (no HW/SW interface needed) |
| requant_narrowing_policy | 46 | 1 | 5 | no workload mapped | shape-independent | — | — | measure on real shapes (no HW/SW interface needed) |

## L2 memory roles (example)
`xnnpack_rvv_gemm` — op_sequence ['matmul', 'clamp']:
- **rhs**: streaming, immutable=True, measured reuse_count=4, packed_once=False
- **acc**: accumulator, widening=False, materialized_before_epilogue=False
- **lhs**: streaming_activation  ·  **output**: committed_output

## Held-out validation (Stage D — symbolic, no execution)
- **packed_rhs_policy** — repeated_rhs_matmul: **holds**, no_reuse_matmul: **correctly_silent**, capacity_stress_reuse: **holds**
    - capacity @ 65536B: footprint 131072B → OVERFLOW
    - capacity @ 131072B: footprint 131072B → fits
    - capacity @ 262144B: footprint 131072B → fits
    - regime matrix (fires 16/20 cells; negative controls — mutable_rhs: **correctly_silent**, no_reuse: **correctly_silent**):

| reuse | K=64 | K=72 (tail) | K=1024 | K=1032 (tail) |
|---:|---|---|---|---|
| 1 | · silent | · silent | · silent | · silent |
| 2 | ✓ fires | ✓ fires | ✓ fires | ✓ fires |
| 4 | ✓ fires | ✓ fires | ✓ fires | ✓ fires |
| 8 | ✓ fires | ✓ fires | ✓ fires | ✓ fires |
| 16 | ✓ fires | ✓ fires | ✓ fires | ✓ fires |

- **accumulator_commit_policy** — matmul_bias_requant_relu: **holds**, no_reuse_matmul: **correctly_silent**
    - regime matrix: shape-independent (`when` references no shape facts)
- **vl_agnostic_loop_policy** — _no benchmark workload mapped_
    - regime matrix: shape-independent (`when` references no shape facts)
- **lmul_grouping_policy** — _no benchmark workload mapped_
    - regime matrix: shape-independent (`when` references no shape facts)
- **fma_broadcast_policy** — _no benchmark workload mapped_
    - regime matrix (fires 0/20 cells; negative controls — mutable_rhs: **LEAK(n/a)**, no_reuse: **LEAK(n/a)**):

| reuse | K=64 | K=72 (tail) | K=1024 | K=1032 (tail) |
|---:|---|---|---|---|
| 1 | n/a | n/a | n/a | n/a |
| 2 | n/a | n/a | n/a | n/a |
| 4 | n/a | n/a | n/a | n/a |
| 8 | n/a | n/a | n/a | n/a |
| 16 | n/a | n/a | n/a | n/a |

- **int8_widening_policy** — _no benchmark workload mapped_
    - regime matrix: shape-independent (`when` references no shape facts)
- **vl_tail_policy** — _no benchmark workload mapped_
    - regime matrix: shape-independent (`when` references no shape facts)
- **vector_reduction_policy** — _no benchmark workload mapped_
    - regime matrix: shape-independent (`when` references no shape facts)
- **requant_narrowing_policy** — _no benchmark workload mapped_
    - regime matrix: shape-independent (`when` references no shape facts)

## Consistency invariants
- ✅ subset: reused_packed_rhs ⊆ packed_rhs (0 violations)
- ✅ subset: accumulator_commit ⊆ accumulator_lifetime ∩ epilogue_before_commit (0 violations)
- ✅ target-restricted motifs fire only on their targets (0 violations)
- ✅ motif table equals recount (0 violations)
- ✅ promoted evidence ids exist in corpus (0 violations)
- ✅ many_small_dispatches implies dispatch metrics (0 violations)
- ✅ no motif fired on an unexpected op family

## Caveats (read before trusting any policy)
- Motifs are *decisions* extracted by deterministic markers, not measured speedups. **No kernel was executed or timed.** Policies are validated only by symbolic match against the benchmark workloads (positive fires / negative control silent).
- **Autocomp:** shapes/dtypes are parsed from the `void test(...)` C signature; the Autocomp `score` is recorded in metadata only and is NOT treated as correctness.
- **Autocomp:** ~1700 of the 2637 manifest entries are 0-byte dedup placeholders and are skipped; counts reflect real, non-empty kernels only.
- **Triton / triton-cpu:** the mined corpora are tutorial + shipped-kernel trees — pedagogical but real optimization decisions; verbatim copies across the two repos are deduplicated by content hash before counting.
- **OpenBLAS:** BLAS1/2 kernels are precision-generic via `DOUBLE` macros, so their dtype is recorded as `unknown`; scalar fallback files are skipped.
- Plots visualize evidence *frequency*, never speedup.
- A promoted motif is a *policy candidate*, not a proven compiler abstraction. Promotion to a dialect op/type requires held-out-shape and target-lowering validation (later sessions). This report does **not** claim automatic abstraction discovery.
