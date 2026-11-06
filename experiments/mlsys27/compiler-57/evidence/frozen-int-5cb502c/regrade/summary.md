# Evaluation m27-int-5cb502c (corpus fin6w2-m27b)

Candidate package-id a0ea14e44c6ab2da (compiler.py ); corpus template fin6w2-m27b; evaluated 20261106T032117Z; wall 204 min.

**VERDICT: NOT PROVEN: device 57/57 proven (weak lowering 57, full definition 57, corpus-clean 57); host 0/136 proven; corpus scope: full. failing device ops: 0; failing host ops: 136**

Timing: regrade_s=12251, total_s=12255, candidate_package_id=a0ea14e44c6ab2da, compiler_py_sha12=, regrade_cpus=36,37,38,39,40,41,42,43,44,45,46,47, harness_cpus=32,33,34,35

## 1. Native regrade (Spike L0-L2, process-pool; models/L3/RTL/FireSim not run)

Scope: the whole graded roster

* core: 320/321 pass  unavailable=1
* scalability: 111/111 pass  
* total: 431/432 pass

### By capsule class (group/kind/semantic family)

| class | n | pass | fail | skipped | unavailable | n/a |
|---|---|---|---|---|---|---|
| core/isa/contraction | 19 | 19 | 0 | 0 | 0 | 0 |
| core/isa/elementwise_map | 24 | 24 | 0 | 0 | 0 | 0 |
| core/isa/movement | 88 | 88 | 0 | 0 | 0 | 0 |
| core/isa/reduction | 31 | 31 | 0 | 0 | 0 | 0 |
| core/layer/contraction | 135 | 135 | 0 | 0 | 0 | 0 |
| core/layer/elementwise_map | 9 | 9 | 0 | 0 | 0 | 0 |
| core/layer/movement | 13 | 13 | 0 | 0 | 0 | 0 |
| core/model/contraction | 1 | 0 | 0 | 0 | 1 | 0 |
| core/model_slice/contraction | 1 | 1 | 0 | 0 | 0 | 0 |
| scalability/model_slice/contraction | 111 | 111 | 0 | 0 | 0 | 0 |

### Failure categories (grader's own: category / plane / stage / tier)

| group | category | plane | stage | tier | n |
|---|---|---|---|---|---|
| (no failures) | | | | | |

## 2. 193-overload lowering harness (FULL definition for the 57 device ops)

* DEVICE57: weak lowering (RoCC retired + exact) 57/57; **meets full definition (numerical + host_compute + trace) 57/57**
* HOST136: 0/136 correct host execution with zero accelerator opcodes
* device ops proven (full definition AND all graded corpus capsules of the op pass, >=1 graded): 57/57

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
| div.Scalar | lowered-on-device | pass | pass | pass | YES | 3/3 (3; tier tier1) | YES |  |
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
| wrong | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) | compiler authoring gap (accelerator opcodes in a host op, or | 22 | _cdist_forward.default, _pdist_forward.default, cos.default, elu.default, erf.default, exp.default, expm1.default, gelu.default, log.default, log10.default, log1p.default, log2.default, mul.Tensor, native_layer_norm.default, pow.Scalar, pow.Tensor_Scalar, pow.Tensor_Tensor, rsqrt.default, sigmoid.default, sin.default, sqrt.default, tanh.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf" | compiler authoring gap (host op not executed: declined-objec | 14 | eq.Scalar, eq.Tensor, ge.Scalar, ge.Tensor, gt.Scalar, gt.Tensor, isnan.default, le.Scalar, le.Tensor, leaky_relu.default, lt.Scalar, lt.Tensor, ne.Scalar, ne.Tensor |
| declined | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer  | compiler authoring gap (host op not executed: declined-objec | 11 | _log_softmax.default, amax.default, amin.default, any.default, any.dim, any.dims, native_group_norm_backward.default, prod.default, prod.dim_int, var.correction, var.dim |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.selec | compiler authoring gap (host op not executed: declined-objec | 11 | clamp.Tensor, clamp.default, div.Scalar_mode, div.Tensor_mode, grid_sampler_2d.default, hardtanh.default, native_layer_norm_backward.default, remainder.Scalar, remainder.Tensor, sign.default, where.self |
| declined | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} | compiler authoring gap (host op not executed: declined-objec | 7 | argmax.default, argmin.default, max.dim, max_pool2d_with_indices.default, max_pool3d_with_indices.default, min.dim, native_group_norm.default |
| declined | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code | compiler authoring gap (host op not executed: declined-objec | 6 | _local_scalar_dense.default, sym_is_contiguous.default, sym_numel.default, sym_size.int, sym_storage_offset.default, sym_stride.int |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} | compiler authoring gap (host op not executed: declined-objec | 5 | _embedding_bag.default, scatter.src, scatter.value, scatter_add.default, scatter_reduce.two |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori" | compiler authoring gap (host op not executed: declined-objec | 5 | bitwise_not.default, bitwise_xor.Scalar, bitwise_xor.Tensor, logical_not.default, logical_xor.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extr | compiler authoring gap (host op not executed: declined-objec | 5 | embedding.default, gather.default, index.Tensor, index_select.default, upsample_bilinear2d.vec |
| declined | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} | compiler authoring gap (host op not executed: declined-objec | 5 | embedding_dense_backward.default, index_put.default, masked_scatter.default, max_pool2d_with_indices_backward.default, nonzero.default |
| declined | declined-object: {"op": "builtin.unregistered", "reason": "no authored lowering for builtin.unregist | compiler authoring gap (host op not executed: declined-objec | 4 | native_dropout.default, rand.default, randn.default, randperm.default |
| declined | declined-object: {"op": "tensor.empty", "reason": "tensor read before complete source production"} | compiler authoring gap (host op not executed: declined-objec | 3 | _native_batch_norm_legit_no_training.default, empty.memory_format, empty_strided.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation linalg.inde | compiler authoring gap (host op not executed: declined-objec | 3 | arange.start_step, sort.default, topk.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.andi" | compiler authoring gap (host op not executed: declined-objec | 3 | bitwise_and.Scalar, bitwise_and.Tensor, logical_and.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.ori"} | compiler authoring gap (host op not executed: declined-objec | 3 | bitwise_or.Scalar, bitwise_or.Tensor, logical_or.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "Parallel runtime product is excluded from the s | compiler authoring gap (host op not executed: declined-objec | 2 | _fft_c2r.default, _fft_r2c.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.absf"} | compiler authoring gap (host op not executed: declined-objec | 2 | abs.default, isinf.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atan2" | compiler authoring gap (host op not executed: declined-objec | 2 | atan2.default, atan2.out |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.divf" | compiler authoring gap (host op not executed: declined-objec | 2 | div.Tensor, reciprocal.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation builtin.unr | compiler authoring gap (host op not executed: declined-objec | 2 | fmod.Scalar, fmod.Tensor |
| crashed | emit:                          Operation llvm.fptrunc is not registered | compiler authoring gap (host op not executed: emit:          | 1 | _native_batch_norm_legit.default |
| crashed | emit:                         Operation llvm.fptrunc is not registered | compiler authoring gap (host op not executed: emit:          | 1 | _native_batch_norm_legit.no_stats |
| declined | declined-object: {"op": "tensor.expand_shape", "reason": "Selected normalization island retains IEEE | compiler authoring gap (host op not executed: declined-objec | 1 | _softmax.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.acos"} | compiler authoring gap (host op not executed: declined-objec | 1 | acos.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.acosh" | compiler authoring gap (host op not executed: declined-objec | 1 | acosh.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.asin"} | compiler authoring gap (host op not executed: declined-objec | 1 | asin.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.asinh" | compiler authoring gap (host op not executed: declined-objec | 1 | asinh.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atan"} | compiler authoring gap (host op not executed: declined-objec | 1 | atan.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atanh" | compiler authoring gap (host op not executed: declined-objec | 1 | atanh.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.ceil"} | compiler authoring gap (host op not executed: declined-objec | 1 | ceil.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.cosh"} | compiler authoring gap (host op not executed: declined-objec | 1 | cosh.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.floor" | compiler authoring gap (host op not executed: declined-objec | 1 | floor.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "initializer is not a uniform literal"} | compiler authoring gap (host op not executed: declined-objec | 1 | maximum.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.minim | compiler authoring gap (host op not executed: declined-objec | 1 | minimum.default |
| wrong | N kernel outputs, N reference outputs | compiler authoring gap (accelerator opcodes in a host op, or | 1 | resize_.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.rounde | compiler authoring gap (host op not executed: declined-objec | 1 | round.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.sinh"} | compiler authoring gap (host op not executed: declined-objec | 1 | sinh.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.tan"} | compiler authoring gap (host op not executed: declined-objec | 1 | tan.default |
| declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.trunc" | compiler authoring gap (host op not executed: declined-objec | 1 | trunc.default |

## 5. Failing corpus capsules (first 80)

| group | capsule | outcome | category | plane | tier | fix class |
|---|---|---|---|---|---|---|
| core | SY_micro_model | unavailable | NOT_RUN_IS_NOT_PASS | candidate_model_execution |  | grader issue (not a pass, not a compiler fault: tier skipped/unavailable) |

Files: table.csv (per-op decision table), capsules.csv (per-capsule), summary-v2.tsv (grader rows).
