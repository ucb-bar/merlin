# RVV kernel-mining -> compiler-improvement: evidence report

_Auditable chain from curated RVV kernels to certified compiler changes. Generated from versioned on-disk artifacts (deterministic; not session-dependent)._

## 1. Mining provenance

- Mined artifact: `mined_knowledge/rvv/mining_rvv_v3_20260619T092631`
- run_id: `mining_rvv_v3_20260619T092631`
- target: `rvv`
- op: `matmul`
- created: `20260619T092631`
- mined_from: `mined_knowledge/rvv/rvv_mined_v1_20260617T165331`
- baseline_run: `hand_v0_matmul_f32_64x64x64`
- n_divergences: `6`
- n_actions: `5`
- n_unrouted: `1`
- `openblas_index`: 130 kernel records
- `xnnpack_index`: 545 kernel records

## 2. Abstracted policies (mined evidence -> reusable abstraction)

| policy | #kernels | sources | actions |
|---|---|---|---|
| `packed_rhs_policy` | 62 | openblas,xnnpack | preserve_packed_rhs_layout; hoist_pack; consider_resident_packed_tensor |
| `accumulator_commit_policy` | 73 | xnnpack | keep_accumulator_resident; fuse_epilogue_before_commit; single_commit_store |
| `vl_agnostic_loop_policy` | 620 | openblas,xnnpack | emit_vl_agnostic_loop; use_predicated_or_vl_tail; avoid_fixed_width_assumptions |
| `lmul_grouping_policy` | 496 | openblas,xnnpack | prefer_high_lmul; set_vector_group_m4_or_m8 |
| `fma_broadcast_policy` | 37 | openblas,xnnpack | emit_scalar_broadcast_fma; fuse_multiply_add; register_block_rhs |
| `int8_widening_policy` | 77 | xnnpack | use_vwmacc_widening; i32_accumulator |
| `vl_tail_policy` | 620 | openblas,xnnpack | emit_vsetvl_loop; vl_or_mask_tail |
| `vector_reduction_policy` | 75 | openblas,xnnpack | emit_vector_reduction_tree; use_vredsum_or_vfredusum |
| `requant_narrowing_policy` | 46 | xnnpack | fuse_requant_narrowing_store; emit_vnclip_then_vse8 |

Each policy is justified by named kernels (the `evidence:` list in `policy_rules.yaml`) and only promoted at >=2 sources or >=min_kernels — so an abstraction can always be traced back to the curated kernels that motivated it.

## 3. CCA divergences -> typed CompilerActions (this run)

The deterministic comparator (`cca_compare`) diffs the expert CCA (built from the mined policies) against ours (decoded from the frozen baseline object), then `action_catalog` routes each populated divergence to a *typed* `CompilerAction`. Emitted directly from this run's `divergences.yaml` / `actions.yaml`.

| divergence axis | expert | ours | -> action class | target seam | forkable now |
|---|---|---|---|---|---|
| `compute.contraction_form` | `fused_fma` | `mul_add` | **PASS** | `impr_features:fused_vfmacc_contraction` | yes |
| `compute.widening` | `True` | `False` | **KNOB** | `schedule:dtype_strategy=int8_w8a8 (vwmacc datapath)` | yes |
| `compute.reduction_form` | `vredsum_tree` | `none` | _unrouted_ | — | — |
| `compute.epilogue` | `requant_narrow` | `none` | **PASS** | `pass:fuse-requant-narrowing-store` | NO (work-item) |
| `vector.lmul` | `4.0` | `2.0` | **KNOB** | `schedule:vector_sizes (widen N to raise LMUL)` | yes |
| `vector.vl_strategy` | `vsetvl_loop` | `vsetivli_fixed` | **PASS** | `pass:vl-polymorphic-tail (emit vsetvli loop)` | NO (work-item) |

- **`compute.contraction_form`** (PASS, forkable) — form a real vector.contract -> outerproduct(kind=add) -> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction outerproduct + lower_outerproduct), instead of separate vfmul.vv+vfadd.vv _Expected:_ vfmacc replaces vfmul+vfadd; MEASURED 7.9x faster on K1 silicon (64^3 f32 matmul, N=5, cos=1.0) vs the frozen baseline _Evidence:_ openblas_rvv_gemm, openblas_rvv_gemv, openblas_rvv_trmm, xnnpack_rvv_gemm.
- **`compute.widening`** (KNOB, forkable) — route the i8 matmul through the widening vwmacc i8xi8->i32 datapath _Expected:_ i32-accumulating widening MAC instead of dequantize-to-f32 _Evidence:_ xnnpack_rvv_dwconv, xnnpack_rvv_gemm, xnnpack_rvv_other, xnnpack_rvv_rdsum, xnnpack_rvv_rsum, xnnpack_rvv_vcvt, xnnpack_rvv_vmul.
- **`compute.epilogue`** (PASS, deferred work-item) — fuse the requantize + narrowing (vnclip/vfncvt) into the store epilogue _Expected:_ single narrowing store; no separate requant pass over the tile _Evidence:_ xnnpack_rvv_dwconv, xnnpack_rvv_gemm, xnnpack_rvv_other, xnnpack_rvv_vcvt, xnnpack_rvv_vmul.
- **`vector.lmul`** (KNOB, forkable) — widen the N tile/vector so the emitted vector group uses a higher LMUL _Expected:_ larger vector groups -> fewer vset/loop iterations per output tile _Evidence:_ openblas_rvv_amax, openblas_rvv_amin, openblas_rvv_asum, openblas_rvv_axpby, openblas_rvv_axpy, openblas_rvv_copy, openblas_rvv_dot, openblas_rvv_gemm.
- **`vector.vl_strategy`** (PASS, deferred work-item) — emit a VL-agnostic vsetvli loop with mask/vl tail instead of fixed vsetivli unrolling (matches the expert vl_agnostic_loop_policy) _Expected:_ one kernel handles any VLEN; smaller code; no fixed-width tail waste _Evidence:_ openblas_rvv_amax, openblas_rvv_amin, openblas_rvv_asum, openblas_rvv_axpby, openblas_rvv_axpy, openblas_rvv_copy, openblas_rvv_dot, openblas_rvv_gemm.

**Unrouted divergences** (surfaced, never silently dropped — no typed action registered for them yet): `compute.reduction_form` (expert=`vredsum_tree`).

### Honest catalog reconciliation

The `action_catalog` (rvv) routes **9** divergence axes: `compute.accumulator_resident`, `compute.activation_vectorization`, `compute.contraction_form`, `compute.epilogue`, `compute.mr_adapts_to_m`, `compute.nr_is_vsetvlmax`, `compute.widening`, `vector.lmul`, `vector.vl_strategy`.

This deterministic **matmul** mining run emits typed actions for **5** of them: `compute.contraction_form`, `compute.epilogue`, `compute.widening`, `vector.lmul`, `vector.vl_strategy`.

It does **not** emit: `compute.accumulator_resident`, `compute.activation_vectorization`, `compute.mr_adapts_to_m`, `compute.nr_is_vsetvlmax`. This is a structural property of the pipeline, not an omission — and is stated honestly here:

- `compute.accumulator_resident`, `compute.nr_is_vsetvlmax` — the mined policies DO set these on the *expert* CCA (`accumulator_commit_policy`, `vl_agnostic_loop_policy`), but the frozen baseline object decodes them as `null` (the lifter does not observe a definitive value in the baseline asm). `cca_compare` only diffs facets populated on BOTH sides, so no divergence — hence no action — is emitted, even though the catalog would route one.
- `compute.mr_adapts_to_m`, `compute.activation_vectorization` — these axes have **no CCA facet field** and **no policy** in the mine driver's `expert_cca_from_policies`; they arise from *non-matmul* divergences (M=1 token-decode matmul tail; the GELU/sigmoid scalar-libm-vs-vectorized-poly activation gap). The catalog routes them (the compiler features `accumulator_resident_mtail` / `vectorized_transcendental_activation` exist and are certified), but a matmul-op CCA mining run structurally cannot mint them. Re-running with `--op activation`/`--op conv` does **not** change this: the expert CCA is built from the same `policy_rules.yaml` and the baseline glob decodes the same matmul object, so the emitted divergence/action set is identical (verified). Surfacing these would require either a CCA facet + policy for them, or mining an activation/M=1 baseline object — a deferred pipeline extension, not something to fabricate into this run.

## 3b. Legacy motif -> knob gap-router (superseded by §3; kept for continuity)

| divergence axis | policy | lever | forkable now | note |
|---|---|---|---|---|
| `lmul_class` | lmul_grouping_policy | knob | yes | widen N tile/vector x2 to push vector grouping toward higher LMUL |
| `lmul_class` | lmul_grouping_policy | knob | yes | widen N tile/vector x4 |
| `fma_form` | fma_broadcast_policy | knob | yes | try outerproduct contraction lowering (NOTE: proven no-op; kept so the beam records it as  |
| `fma_form` | fma_broadcast_policy | llvm_requirement | NO (work-item) | RECOVER FUSED vfmacc: inject fast-math `contract` at MLIR emission so clang fuses fmul+fad |
| `vl_strategy` | vl_tail_policy | llvm_requirement | NO (work-item) | expert uses vsetvl-loop (VL-polymorphic); we emit vsetivli (fixed immediate). Needs a scal |
| `int_widening` | int8_widening_policy | knob | yes | route i8 matmul through the vwmacc integer datapath (passes_quant_int) |

`knob` = expressible in the transform schedule today (tile/vector size, LMUL, lowering pattern). `lowering_pattern`/`llvm_requirement` = a deferred compiler work-item the router surfaces but does not pretend is a one-flag fix.

## 4. Certified experiments (baseline vs fork — measured, gated)

| run | workload | gate | vfmacc | total vf | cycles | ladder |
|---|---|---|---|---|---|---|
| `hand_v0_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 0 | 8 | 27118799 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |
| `impr_rvv_v1_20260618T170618_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 0 | 60 | 9071855 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |
| `impr_rvv_v2_20260618T170747_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 0 | 60 | 9071855 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |
| `impr_rvv_v3_20260618T172539_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 0 | 8 | 24156367 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |
| `impr_rvv_v4_20260618T172740_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 0 | 8 | 27118799 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |
| `impr_rvv_v5_20260618T174246_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 8065 | 8192 | 135574 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |
| `rvv_tuned_v1_d1_vfmacc_outerproduct_matmul_f32_64x64x64` | matmul_f32_64x64x64 | pass | 0 | 8 | 27118799 | K0=pass,K1=pass,K2=pass,K3=pass,K4=pass,K5=not_run,K6=not_run |

- **matmul_f32_64x64x64**: `impr_rvv_v1_20260618T170618_matmul_f32_64x64x64` vs baseline -> vfmacc 0→0, correctness ok — **changed**
- **matmul_f32_64x64x64**: `impr_rvv_v2_20260618T170747_matmul_f32_64x64x64` vs baseline -> vfmacc 0→0, correctness ok — **changed**
- **matmul_f32_64x64x64**: `impr_rvv_v3_20260618T172539_matmul_f32_64x64x64` vs baseline -> vfmacc 0→0, correctness ok — **changed**
- **matmul_f32_64x64x64**: `impr_rvv_v4_20260618T172740_matmul_f32_64x64x64` vs baseline -> vfmacc 0→0, correctness ok — **no-op (histogram unchanged)**
- **matmul_f32_64x64x64**: `impr_rvv_v5_20260618T174246_matmul_f32_64x64x64` vs baseline -> vfmacc 0→8065, correctness ok — **CLOSED gap (vfmacc emitted)**
- **matmul_f32_64x64x64**: `rvv_tuned_v1_d1_vfmacc_outerproduct_matmul_f32_64x64x64` vs baseline -> vfmacc 0→0, correctness ok — **no-op (histogram unchanged)**

## 6. Measured fork attempts (asm re-decoded — incl. honest no-ops)

| run | vfmacc | vfmul | vfadd | dominant vtype |
|---|---|---|---|---|
| `hand_v0_matmul_f32_64x64x64` | 0 | 4 | 4 | e32m2tama |
| `impr_rvv_v1_20260618T170618_matmul_f32_64x64x64` | 0 | 32 | 4 | e32m2tamu |
| `impr_rvv_v2_20260618T170747_matmul_f32_64x64x64` | 0 | 32 | 4 | e32m2tamu |
| `impr_rvv_v3_20260618T172539_matmul_f32_64x64x64` | 0 | 4 | 4 | e32m4tama |
| `impr_rvv_v4_20260618T172740_matmul_f32_64x64x64` | 0 | 4 | 4 | e32m2tama |
| `impr_rvv_v5_20260618T174246_matmul_f32_64x64x64` | 8065 | 0 | 0 | e32m8tama |

The fused-`vfmacc` story (the loop measuring its way to a real fix): forks v1–v4 (outerproduct; K=4 tile; +`-ffp-contract=fast`; +`-ffast-math`) all decode to `vfmacc=0` — knobs/flags can't fuse the baseline's K=1-tiled contraction, so the action was demoted to a deferred PASS. The PASS was then implemented (`vectorize_children` -> `vector.contract` -> outerproduct -> `vector.fma` -> `vfmacc`): **v5 certifies correct on spike AND decodes to `vfmacc>0, vfmul=0, vfadd=0` — gap CLOSED**, and the action re-promoted to forkable.

## 7. Fold-in status

- **Forkable wins** (a fork beat baseline, gate ok): promote the knob into the default schedule (`pipeline.RVV_TRANSFORM_SCHEDULE`) via a human-reviewed PR with this evidence bundle attached.
- **Deferred work-items** (router lever != `knob`): tracked compiler features (e.g. the fused-`vfmacc` recovery needs a vectorize-structure change so a `vector.contract` forms — empirically confirmed here that the `outerproduct` lowering strategy alone is a no-op).
