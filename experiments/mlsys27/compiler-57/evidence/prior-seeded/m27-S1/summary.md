# Evaluation m27-S1 (corpus fin6w2-m27)

Candidate package-id f430113415f19be7 (compiler.py ); corpus template fin6w2-m27; evaluated 20261105T073453Z; wall 60 min.

**VERDICT: NOT PROVEN: device 56/57 proven (weak lowering 56, full definition 56, corpus-clean 57); host 0/136 proven; corpus scope: full. failing device ops: 1; failing host ops: 136**

Timing: regrade_s=3632, total_s=3655, candidate_package_id=f430113415f19be7, compiler_py_sha12=, regrade_cpus=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, harness_cpus=20,21,23,24

## 1. Native regrade (Spike L0-L2, process-pool; models/L3/RTL/FireSim not run)

Scope: the whole graded roster

* core: 318/321 pass  fail=2  unavailable=1
* scalability: 111/111 pass  
* total: 429/432 pass

### By capsule class (group/kind/semantic family)

| class | n | pass | fail | skipped | unavailable | n/a |
|---|---|---|---|---|---|---|
| core/isa/contraction | 19 | 19 | 0 | 0 | 0 | 0 |
| core/isa/elementwise_map | 24 | 24 | 0 | 0 | 0 | 0 |
| core/isa/movement | 88 | 88 | 0 | 0 | 0 | 0 |
| core/isa/reduction | 31 | 31 | 0 | 0 | 0 | 0 |
| core/layer/contraction | 135 | 133 | 2 | 0 | 0 | 0 |
| core/layer/elementwise_map | 9 | 9 | 0 | 0 | 0 | 0 |
| core/layer/movement | 13 | 13 | 0 | 0 | 0 | 0 |
| core/model/contraction | 1 | 0 | 0 | 0 | 1 | 0 |
| core/model_slice/contraction | 1 | 1 | 0 | 0 | 0 | 0 |
| scalability/model_slice/contraction | 111 | 111 | 0 | 0 | 0 | 0 |

### Failure categories (grader's own: category / plane / stage / tier)

| group | category | plane | stage | tier | n |
|---|---|---|---|---|---|
| core | MUST_REFUSE_VIOLATED | expected_refusal | post_tier | - | 2 |

## 2. 193-overload lowering harness (FULL definition for the 57 device ops)

* DEVICE57: weak lowering (RoCC retired + exact) 56/57; **meets full definition (numerical + host_compute + trace) 56/57**
* HOST136: 0/136 correct host execution with zero accelerator opcodes
* device ops proven (full definition AND all graded corpus capsules of the op pass, >=1 graded): 56/57

## 3. Decision table: 57 device ops

| op | lowers on device | numerical | host_compute | trace | full | corpus (own-program capsules) pass/graded (credited) | proven | fix class |
|---|---|---|---|---|---|---|---|---|
| _adaptive_avg_pool2d.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| _adaptive_avg_pool2d_backward.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| _adaptive_avg_pool3d.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| _to_copy.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| adaptive_avg_pool1d.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| add.Scalar | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| add.Tensor | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| addmm.default | lowered-on-device | pass | pass | pass | YES | 1/1 (1; tier tier1) | YES |  |
| alias.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| as_strided.default | lowered-on-device | pass | pass | pass | YES | 6/6 (6; tier tier1) | YES |  |
| avg_pool1d.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| avg_pool2d.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| avg_pool2d_backward.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| avg_pool3d.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| bmm.default | lowered-on-device | pass | pass | pass | YES | 1/1 (1; tier tier1) | YES |  |
| cat.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| clone.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| col2im.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| constant_pad_nd.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| convolution.default | lowered-on-device | pass | pass | pass | YES | 41/41 (55; tier tier1) | YES |  |
| convolution_backward.default | lowered-on-device | pass | pass | pass | YES | 4/4 (4; tier tier1) | YES |  |
| copy.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| cumsum.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| diagonal.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| div.Scalar | wrong | fail | fail | pass | NO | 3/3 (3; tier tier1) | NO | compiler authoring gap (wrong result) |
| expand.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| fill.Scalar | lowered-on-device | pass | pass | pass | YES | 7/7 (7; tier tier1) | YES |  |
| flip.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| full.default | lowered-on-device | pass | pass | pass | YES | 7/7 (7; tier tier1) | YES |  |
| full_like.default | lowered-on-device | pass | pass | pass | YES | 7/7 (7; tier tier1) | YES |  |
| mean.default | lowered-on-device | pass | pass | pass | YES | 2/2 (2; tier tier1) | YES |  |
| mean.dim | lowered-on-device | pass | pass | pass | YES | 4/4 (4; tier tier1) | YES |  |
| mm.default | lowered-on-device | pass | pass | pass | YES | 127/127 (175; tier tier1) | YES |  |
| mul.Scalar | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| neg.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| permute.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| reflection_pad1d.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| reflection_pad2d.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| reflection_pad3d.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| relu.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| repeat.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| replication_pad2d.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| replication_pad3d.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| scalar_tensor.default | lowered-on-device | pass | pass | pass | YES | 1/1 (1; tier tier1) | YES |  |
| select.int | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| select_scatter.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| slice.Tensor | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| slice_scatter.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| split_with_sizes.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| squeeze.dim | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| squeeze.dims | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| sub.Scalar | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| sub.Tensor | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| sum.dim_IntList | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| unsqueeze.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| upsample_nearest2d.vec | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
| view.default | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |

## 4. Host ops (136): zero accelerator opcodes + correct output

0/136 pass; 136 failing.

| result | reason | fix class | n | ops |
|---|---|---|---|---|
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) | compiler authoring gap (accelerator opcodes in a host op, or | 34 | _cdist_forward.default, _pdist_forward.default, abs.default, amin.default, ceil.default, clamp.Tensor, clamp.default, cos.default, div.Tensor, exp.default, floor.default, fmod.Tensor, hardtanh.default, log.default, log2.default, maximum.default, minimum.default, mul.Tensor, native_layer_norm.default, pow.Scalar, pow.Tensor_Scalar, pow.Tensor_Tensor, prod.default, reciprocal.default, remainder.Tensor, round.default, rsqrt.default, sigmoid.default, sign.default, sin.default, sqrt.default, tanh.default, trunc.default, where.self |
| declined | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed  | compiler authoring gap (host op not executed: declined-objec | 34 | _embedding_bag.default, _native_batch_norm_legit.default, _native_batch_norm_legit.no_stats, arange.start_step, bitwise_and.Scalar, bitwise_and.Tensor, bitwise_not.default, bitwise_or.Scalar, bitwise_or.Tensor, bitwise_xor.Scalar, bitwise_xor.Tensor, div.Scalar_mode, div.Tensor_mode, eq.Scalar, eq.Tensor, ge.Scalar, ge.Tensor, gt.Scalar, gt.Tensor, isinf.default, isnan.default, le.Scalar, le.Tensor, leaky_relu.default, logical_and.default, logical_not.default, logical_or.default, logical_xor.default, lt.Scalar, lt.Tensor, ne.Scalar, ne.Tensor, sort.default, topk.default |
| declined | NotImplementedError: no lowering for HOST_LINALG | compiler authoring gap (host op not executed: NotImplemented | 21 | acos.default, acosh.default, asin.default, asinh.default, atan.default, atan2.default, atanh.default, cosh.default, elu.default, erf.default, expm1.default, gelu.default, log10.default, log1p.default, scatter.src, scatter.value, scatter_add.default, scatter_reduce.two, sinh.default, tan.default, upsample_bilinear2d.vec |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) | compiler authoring gap (accelerator opcodes in a host op, or | 7 | amax.default, prod.dim_int, sym_is_contiguous.default, sym_numel.default, sym_size.int, sym_storage_offset.default, sym_stride.int |
| declined | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload res | compiler authoring gap (host op not executed: declined-objec | 6 | argmax.default, argmin.default, max.dim, max_pool2d_with_indices.default, max_pool3d_with_indices.default, min.dim |
| crashed | parse: ValueError: payload/full-source operation order disagreement | compiler authoring gap (host op not executed: parse: ValueEr | 5 | _local_scalar_dense.default, grid_sampler_2d.default, masked_scatter.default, nonzero.default, randn.default |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 10, static 10) | compiler authoring gap (accelerator opcodes in a host op, or | 4 | _log_softmax.default, _softmax.default, fmod.Scalar, remainder.Scalar |
| declined | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: captured extraction  | compiler authoring gap (host op not executed: declined-objec | 4 | embedding.default, gather.default, index.Tensor, index_select.default |
| declined | declined-object: {"reason": "direct argument returns lack a source-operation result owner"} | compiler authoring gap (host op not executed: declined-objec | 3 | _native_batch_norm_legit_no_training.default, empty.memory_format, empty_strided.default |
| declined | declined-object: {"reason": "linalg operations not yet lowered: linalg.reduce: unimplemented typed o | compiler authoring gap (host op not executed: declined-objec | 3 | any.default, any.dim, any.dims |
| declined | declined-object: {"reason": "linalg operations not yet lowered: scf.for: unimplemented typed operati | compiler authoring gap (host op not executed: declined-objec | 3 | embedding_dense_backward.default, index_put.default, max_pool2d_with_indices_backward.default |
| declined | declined-object: {"reason": "linalg operations not yet lowered: ml_program.global_load: unimplemente | compiler authoring gap (host op not executed: declined-objec | 3 | native_dropout.default, rand.default, randperm.default |
| declined | declined-object: {"reason": "repeated returns lack independent source-result writers"} | compiler authoring gap (host op not executed: declined-objec | 2 | atan2.out, resize_.default |
| crashed | abi: input arg_N: N case elements vs command buffer [N, N, N] | compiler authoring gap (host op not executed: abi: input arg | 1 | _fft_c2r.default |
| crashed | built: TypeError: float() argument must be a string or a real number, not 'dict' | compiler authoring gap (host op not executed: built: TypeErr | 1 | _fft_r2c.default |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 74, static 50) | compiler authoring gap (accelerator opcodes in a host op, or | 1 | native_group_norm.default |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 202, static 162) | compiler authoring gap (accelerator opcodes in a host op, or | 1 | native_group_norm_backward.default |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 78, static 78) | compiler authoring gap (accelerator opcodes in a host op, or | 1 | native_layer_norm_backward.default |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 18, static 18) | compiler authoring gap (accelerator opcodes in a host op, or | 1 | var.correction |
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 14, static 14) | compiler authoring gap (accelerator opcodes in a host op, or | 1 | var.dim |

## 5. Failing corpus capsules (first 80)

| group | capsule | outcome | category | plane | tier | fix class |
|---|---|---|---|---|---|---|
| core | SY_composition_contraction_readout_q_pow2_tensor_os_bound_outside | fail | MUST_REFUSE_VIOLATED | expected_refusal |  | compiler authoring gap |
| core | SY_composition_contraction_readout_q_pow2_tensor_os_runtime | fail | MUST_REFUSE_VIOLATED | expected_refusal |  | compiler authoring gap |
| core | SY_micro_model | unavailable | NOT_RUN_IS_NOT_PASS | candidate_model_execution |  | grader issue (not a pass, not a compiler fault: tier skipped/unavailable) |

Files: table.csv (per-op decision table), capsules.csv (per-capsule), summary-v2.tsv (grader rows).
