# RESULT-193 (m27-int-5cb502c): authored backend vs all 193 core-ATen overloads, canonical case; package file-set id a0ea14e44c6ab2da

**DEVICE57 (30 exact movement + 27 int8-quantized, ALL57 of tooling/exercised/exercised.py): 57 / 57 lowered on device (accelerator instructions retired, exact output).  HOST136 (100 host-only + 36 declared host-only): 0 / 136 ran on the host path correctly with zero accelerator opcodes.**

Verification instrument (never an author input). Route per case: parse -> lower_interface_to_target -> emit_target_bundle -> kernel ABI v2 -> clang-23 -> Spike with -l RoCC (0x7b) retirement count (tools/run_case.py). Inputs: DEVICE57 are handed the op-capsule group's int8/integer form (forms93, from Merlin's core_aten_op_capsules registry, exact integer reference) (operator decision 2026-11-04: every DEVICE57 op is graded on a spec-declared form; 47 through the op-capsule registry's builder lanes, cumsum/diagonal/select/split_with_sizes/convolution_backward/upsample_nearest2d/full/full_like/scalar_tensor through the spec form table (tools/gen_spec_forms.py, lane SPEC_FORM), and div.Scalar through its spec_int8 form: the f32 capture compared EXACTLY with the software spec's declared quantized model, tools/gen_spec_int8_form.py); HOST136 are handed Merlin's own corpus_193 canonical f32 capture (f32 rounding tolerance for floats, exact for ints). Column f32_capture_result in the CSV gives the outcome of the corpus_193 f32 capture for every overload (what Merlin's own backend consumes). Classes: lowered-on-device / lowered-on-host / declined / wrong / crashed. A 'wrong' on HOST136 includes any accelerator opcode.

## Summary

| class | n | lowered-on-device | lowered-on-host | declined | wrong | crashed | other |
|---|---|---|---|---|---|---|---|
| DEVICE57 | 57 | 57 | 0 | 0 | 0 | 0 | 0 |
| HOST136 | 136 | 0 | 0 | 111 | 23 | 2 | 0 |

(DEVICE57 'lowered-on-host (device expected)' is counted under other.)

## Declines / failures by reason

| class | result | reason | n | overloads |
|---|---|---|---|---|
| HOST136 | wrong | HOST-ONLY overload emitted accelerator opcodes (retired N, static N) | 22 | _cdist_forward.default, _pdist_forward.default, cos.default, elu.default, erf.default, exp.default, expm1.default, gelu.default, log.default, log10.default, log1p.default, log2.default, mul.Tensor, native_layer_norm.default, pow.Scalar, pow.Tensor_Scalar, pow.Tensor_Tensor, rsqrt.default, sigmoid.default, sin.default, sqrt.default, tanh.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} | 14 | eq.Scalar, eq.Tensor, ge.Scalar, ge.Tensor, gt.Scalar, gt.Tensor, isnan.default, le.Scalar, le.Tensor, leaky_relu.default, lt.Scalar, lt.Tensor, ne.Scalar, ne.Tensor |
| HOST136 | declined | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} | 11 | _log_softmax.default, amax.default, amin.default, any.default, any.dim, any.dims, native_group_norm_backward.default, prod.default, prod.dim_int, var.correction, var.dim |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} | 11 | clamp.Tensor, clamp.default, div.Scalar_mode, div.Tensor_mode, grid_sampler_2d.default, hardtanh.default, native_layer_norm_backward.default, remainder.Scalar, remainder.Tensor, sign.default, where.self |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} | 7 | argmax.default, argmin.default, max.dim, max_pool2d_with_indices.default, max_pool3d_with_indices.default, min.dim, native_group_norm.default |
| HOST136 | declined | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integ | 6 | _local_scalar_dense.default, sym_is_contiguous.default, sym_numel.default, sym_size.int, sym_storage_offset.default, sym_stride.int |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} | 5 | _embedding_bag.default, scatter.src, scatter.value, scatter_add.default, scatter_reduce.two |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori"} | 5 | bitwise_not.default, bitwise_xor.Scalar, bitwise_xor.Tensor, logical_not.default, logical_xor.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extract"} | 5 | embedding.default, gather.default, index.Tensor, index_select.default, upsample_bilinear2d.vec |
| HOST136 | declined | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} | 5 | embedding_dense_backward.default, index_put.default, masked_scatter.default, max_pool2d_with_indices_backward.default, nonzero.default |
| HOST136 | declined | declined-object: {"op": "builtin.unregistered", "reason": "no authored lowering for builtin.unregistered"} | 4 | native_dropout.default, rand.default, randn.default, randperm.default |
| HOST136 | declined | declined-object: {"op": "tensor.empty", "reason": "tensor read before complete source production"} | 3 | _native_batch_norm_legit_no_training.default, empty.memory_format, empty_strided.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation linalg.index"} | 3 | arange.start_step, sort.default, topk.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.andi"} | 3 | bitwise_and.Scalar, bitwise_and.Tensor, logical_and.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.ori"} | 3 | bitwise_or.Scalar, bitwise_or.Tensor, logical_or.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "Parallel runtime product is excluded from the selected na | 2 | _fft_c2r.default, _fft_r2c.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.absf"} | 2 | abs.default, isinf.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atanN"} | 2 | atan2.default, atan2.out |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.divf"} | 2 | div.Tensor, reciprocal.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation builtin.unregistered" | 2 | fmod.Scalar, fmod.Tensor |
| HOST136 | crashed | emit:                          Operation llvm.fptrunc is not registered | 1 | _native_batch_norm_legit.default |
| HOST136 | crashed | emit:                         Operation llvm.fptrunc is not registered | 1 | _native_batch_norm_legit.no_stats |
| HOST136 | declined | declined-object: {"op": "tensor.expand_shape", "reason": "Selected normalization island retains IEEE arithmeti | 1 | _softmax.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.acos"} | 1 | acos.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.acosh"} | 1 | acosh.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.asin"} | 1 | asin.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.asinh"} | 1 | asinh.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atan"} | 1 | atan.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atanh"} | 1 | atanh.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.ceil"} | 1 | ceil.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.cosh"} | 1 | cosh.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.floor"} | 1 | floor.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "initializer is not a uniform literal"} | 1 | maximum.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.minimumf"} | 1 | minimum.default |
| HOST136 | wrong | N kernel outputs, N reference outputs | 1 | resize_.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.roundeven"} | 1 | round.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.sinh"} | 1 | sinh.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.tan"} | 1 | tan.default |
| HOST136 | declined | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.trunc"} | 1 | trunc.default |

## Per overload

| # | overload | class | input | result | device insns | reason |
|---|---|---|---|---|---|---|
| 000 | _adaptive_avg_pool2d.default | DEVICE57 | int8 form | lowered-on-device | 82 |  |
| 001 | _adaptive_avg_pool2d_backward.default | DEVICE57 | int8 form | lowered-on-device | 162 |  |
| 002 | _adaptive_avg_pool3d.default | DEVICE57 | int8 form | lowered-on-device | 322 |  |
| 014 | _to_copy.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 018 | adaptive_avg_pool1d.default | DEVICE57 | int8 form | lowered-on-device | 98 |  |
| 019 | add.Scalar | DEVICE57 | int8 form | lowered-on-device | 230 |  |
| 020 | add.Tensor | DEVICE57 | int8 form | lowered-on-device | 130 |  |
| 021 | addmm.default | DEVICE57 | int8 form | lowered-on-device | 19 |  |
| 022 | alias.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 031 | as_strided.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 038 | avg_pool1d.default | DEVICE57 | int8 form | lowered-on-device | 122 |  |
| 039 | avg_pool2d.default | DEVICE57 | int8 form | lowered-on-device | 150 |  |
| 040 | avg_pool2d_backward.default | DEVICE57 | int8 form | lowered-on-device | 386 |  |
| 041 | avg_pool3d.default | DEVICE57 | int8 form | lowered-on-device | 162 |  |
| 049 | bmm.default | DEVICE57 | int8 form | lowered-on-device | 32 |  |
| 050 | cat.default | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 054 | clone.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 055 | col2im.default | DEVICE57 | int8 form | lowered-on-device | 70 |  |
| 056 | constant_pad_nd.default | DEVICE57 | int8 form | lowered-on-device | 12 |  |
| 057 | convolution.default | DEVICE57 | int8 form | lowered-on-device | 819 |  |
| 058 | convolution_backward.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 980 |  |
| 059 | copy.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 062 | cumsum.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 339 |  |
| 063 | diagonal.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 064 | div.Scalar | DEVICE57 | spec_int8 form (f32 capture, exact vs th | lowered-on-device | 3338 |  |
| 077 | expand.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 079 | fill.Scalar | DEVICE57 | int8 form | lowered-on-device | 198 |  |
| 080 | flip.default | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 084 | full.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 085 | full_like.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 118 | mean.default | DEVICE57 | int8 form | lowered-on-device | 70 |  |
| 119 | mean.dim | DEVICE57 | int8 form | lowered-on-device | 218 |  |
| 122 | mm.default | DEVICE57 | int8 form | lowered-on-device | 17 |  |
| 123 | mul.Scalar | DEVICE57 | int8 form | lowered-on-device | 178 |  |
| 132 | neg.default | DEVICE57 | int8 form | lowered-on-device | 178 |  |
| 134 | permute.default | DEVICE57 | int8 form | lowered-on-device | 260 |  |
| 144 | reflection_pad1d.default | DEVICE57 | int8 form | lowered-on-device | 14 |  |
| 145 | reflection_pad2d.default | DEVICE57 | int8 form | lowered-on-device | 38 |  |
| 146 | reflection_pad3d.default | DEVICE57 | int8 form | lowered-on-device | 122 |  |
| 147 | relu.default | DEVICE57 | int8 form | lowered-on-device | 98 |  |
| 150 | repeat.default | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 151 | replication_pad2d.default | DEVICE57 | int8 form | lowered-on-device | 38 |  |
| 152 | replication_pad3d.default | DEVICE57 | int8 form | lowered-on-device | 122 |  |
| 156 | scalar_tensor.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 161 | select.int | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 10 |  |
| 162 | select_scatter.default | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 167 | slice.Tensor | DEVICE57 | int8 form | lowered-on-device | 8 |  |
| 168 | slice_scatter.default | DEVICE57 | int8 form | lowered-on-device | 12 |  |
| 170 | split_with_sizes.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 14 |  |
| 172 | squeeze.dim | DEVICE57 | int8 form | lowered-on-device | 98 |  |
| 173 | squeeze.dims | DEVICE57 | int8 form | lowered-on-device | 98 |  |
| 174 | sub.Scalar | DEVICE57 | int8 form | lowered-on-device | 230 |  |
| 175 | sub.Tensor | DEVICE57 | int8 form | lowered-on-device | 210 |  |
| 176 | sum.dim_IntList | DEVICE57 | int8 form | lowered-on-device | 42 |  |
| 186 | unsqueeze.default | DEVICE57 | int8 form | lowered-on-device | 98 |  |
| 188 | upsample_nearest2d.vec | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 20 |  |
| 191 | view.default | DEVICE57 | int8 form | lowered-on-device | 206 |  |
| 003 | _cdist_forward.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 004 | _embedding_bag.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} |
| 005 | _fft_c2r.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "Parallel runtime product is excluded from the selected native byte  |
| 006 | _fft_r2c.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "Parallel runtime product is excluded from the selected native byte  |
| 007 | _local_scalar_dense.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integer output_ |
| 008 | _log_softmax.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 009 | _native_batch_norm_legit.default | HOST136 | f32 capture | crashed |  | emit:                          Operation llvm.fptrunc is not registered |
| 010 | _native_batch_norm_legit.no_stats | HOST136 | f32 capture | crashed |  | emit:                         Operation llvm.fptrunc is not registered |
| 011 | _native_batch_norm_legit_no_training.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.empty", "reason": "tensor read before complete source production"} |
| 012 | _pdist_forward.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 013 | _softmax.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.expand_shape", "reason": "Selected normalization island retains IEEE arithmetic on the h |
| 015 | abs.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.absf"} |
| 016 | acos.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.acos"} |
| 017 | acosh.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.acosh"} |
| 023 | amax.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 024 | amin.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 025 | any.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 026 | any.dim | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 027 | any.dims | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 028 | arange.start_step | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation linalg.index"} |
| 029 | argmax.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 030 | argmin.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 032 | asin.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.asin"} |
| 033 | asinh.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.asinh"} |
| 034 | atan.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atan"} |
| 035 | atan2.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atan2"} |
| 036 | atan2.out | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atan2"} |
| 037 | atanh.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.atanh"} |
| 042 | bitwise_and.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.andi"} |
| 043 | bitwise_and.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.andi"} |
| 044 | bitwise_not.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori"} |
| 045 | bitwise_or.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.ori"} |
| 046 | bitwise_or.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.ori"} |
| 047 | bitwise_xor.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori"} |
| 048 | bitwise_xor.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori"} |
| 051 | ceil.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.ceil"} |
| 052 | clamp.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 053 | clamp.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 060 | cos.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 061 | cosh.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.cosh"} |
| 065 | div.Scalar_mode | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 066 | div.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.divf"} |
| 067 | div.Tensor_mode | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 068 | elu.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 069 | embedding.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extract"} |
| 070 | embedding_dense_backward.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} |
| 071 | empty.memory_format | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.empty", "reason": "tensor read before complete source production"} |
| 072 | empty_strided.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.empty", "reason": "tensor read before complete source production"} |
| 073 | eq.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 074 | eq.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 075 | erf.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 076 | exp.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 078 | expm1.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 081 | floor.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.floor"} |
| 082 | fmod.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation builtin.unregistered"} |
| 083 | fmod.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation builtin.unregistered"} |
| 086 | gather.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extract"} |
| 087 | ge.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 088 | ge.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 089 | gelu.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 090 | grid_sampler_2d.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 091 | gt.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 092 | gt.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 093 | hardtanh.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 094 | index.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extract"} |
| 095 | index_put.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} |
| 096 | index_select.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extract"} |
| 097 | isinf.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.absf"} |
| 098 | isnan.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 099 | le.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 100 | le.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 101 | leaky_relu.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 102 | log.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 103 | log10.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 104 | log1p.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 105 | log2.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 106 | logical_and.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.andi"} |
| 107 | logical_not.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori"} |
| 108 | logical_or.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.ori"} |
| 109 | logical_xor.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.xori"} |
| 110 | lt.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 111 | lt.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 112 | masked_scatter.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} |
| 113 | max.dim | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 114 | max_pool2d_with_indices.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 115 | max_pool2d_with_indices_backward.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} |
| 116 | max_pool3d_with_indices.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 117 | maximum.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "initializer is not a uniform literal"} |
| 120 | min.dim | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 121 | minimum.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.minimumf"} |
| 124 | mul.Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 125 | native_dropout.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "builtin.unregistered", "reason": "no authored lowering for builtin.unregistered"} |
| 126 | native_group_norm.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "generic must yield one tensor"} |
| 127 | native_group_norm_backward.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 128 | native_layer_norm.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 129 | native_layer_norm_backward.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 130 | ne.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 131 | ne.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.cmpf"} |
| 133 | nonzero.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "scf.for", "reason": "no authored lowering for scf.for"} |
| 135 | pow.Scalar | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 136 | pow.Tensor_Scalar | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 137 | pow.Tensor_Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 138 | prod.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 139 | prod.dim_int | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 140 | rand.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "builtin.unregistered", "reason": "no authored lowering for builtin.unregistered"} |
| 141 | randn.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "builtin.unregistered", "reason": "no authored lowering for builtin.unregistered"} |
| 142 | randperm.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "builtin.unregistered", "reason": "no authored lowering for builtin.unregistered"} |
| 143 | reciprocal.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.divf"} |
| 148 | remainder.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 149 | remainder.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 153 | resize_.default | HOST136 | f32 capture | wrong | 14 | N kernel outputs, N reference outputs |
| 154 | round.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.roundeven"} |
| 155 | rsqrt.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 157 | scatter.src | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} |
| 158 | scatter.value | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} |
| 159 | scatter_add.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} |
| 160 | scatter_reduce.two | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation scf.for"} |
| 163 | sigmoid.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 164 | sign.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |
| 165 | sin.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 166 | sinh.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.sinh"} |
| 169 | sort.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation linalg.index"} |
| 171 | sqrt.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 177 | sym_is_contiguous.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integer output_ |
| 178 | sym_numel.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integer output_ |
| 179 | sym_size.int | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integer output_ |
| 180 | sym_storage_offset.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integer output_ |
| 181 | sym_stride.int | HOST136 | f32 capture | declined |  | declined-object: {"op": "tensor.splat", "reason": "Selected FILL ABI accepts only integer value_code and integer output_ |
| 182 | tan.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.tan"} |
| 183 | tanh.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 184 | topk.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation linalg.index"} |
| 185 | trunc.default | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation math.trunc"} |
| 187 | upsample_bilinear2d.vec | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation tensor.extract"} |
| 189 | var.correction | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 190 | var.dim | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.reduce", "reason": "named reduction requires an exact typed integer sum"} |
| 192 | where.self | HOST136 | f32 capture | declined |  | declined-object: {"op": "linalg.generic", "reason": "unsupported scalar affine operation arith.select"} |

## FULL definition ('supported' = numerical AND host-compute AND instruction-trace)

**DEVICE57: 57/57 weak pass (RoCC retired + exact output) -> 57/57 meets the full definition.**  HOST136: n/a (kept as zero accelerator opcodes + correct output: 0/136); host-compute and trace checks do not apply.

Checks (the grader's own functions, run unmodified in the FIN6yn neutral release namespace: operand-selection sidecar e5f2486b, observer active): host-compute = offload_census._source_host_compute -> host_compute_audit.audit_group, budget 1.0 host arithmetic ops/element (also 0.25 float/convert ops/element, 0.5 scalar-payload/element-byte), category host_compute_in_accelerator_group; instruction-trace = rocc.decode.decode_module(lowered LLVM) -> trace_check.check(trace, capsule.expected, cb, address_model=pointer_args, subsumes, target, lowered_module) with the op-capsule registry's own `expected` modes (the grader's trace is the static decode of the emitted .insn stream, not the Spike log). Numerical = this harness's exact (device57) comparison on Spike.

| step | n |
|---|---|
| DEVICE57 total | 57 |
| weak pass (RoCC insns retired + exact output) | 57 |
|   of which host_compute pass | 57 |
|   of which host_compute FAIL (incl. census silent) | 0 |
|   of which host_compute unknown | 0 |
|   of which trace pass | 57 |
|   of which trace unavailable_fields/protocol | 0 |
|   of which trace fail | 0 |
| meets full definition (strict: host_compute must be 'pass') | 57 |
| meets full definition if declared_host (not judged, veto does not fire) is accepted | 57 |

### DEVICE57 per overload (full definition)

| # | overload | numerical | host_compute | host ops/elem (budget 1.0) | float ops/elem | trace | trace reason | meets full definition |
|---|---|---|---|---|---|---|---|---|
| 000 | _adaptive_avg_pool2d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 001 | _adaptive_avg_pool2d_backward.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 002 | _adaptive_avg_pool3d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 014 | _to_copy.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 018 | adaptive_avg_pool1d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 019 | add.Scalar | pass | pass | 0.2917 | 0.0 | pass |  | YES |
| 020 | add.Tensor | pass | pass | 0.5 | 0.0 | pass |  | YES |
| 021 | addmm.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 022 | alias.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 031 | as_strided.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 038 | avg_pool1d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 039 | avg_pool2d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 040 | avg_pool2d_backward.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 041 | avg_pool3d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 049 | bmm.default | pass | pass | 0.0781 | 0.0 | pass |  | YES |
| 050 | cat.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 054 | clone.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 055 | col2im.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 056 | constant_pad_nd.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 057 | convolution.default | pass | pass | 0.5938 | 0.0 | pass |  | YES |
| 058 | convolution_backward.default | pass | pass | 0.9508 | 0.0 | pass |  | YES |
| 059 | copy.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 062 | cumsum.default | pass | pass | 0.0861 | 0.0 | pass |  | YES |
| 063 | diagonal.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 064 | div.Scalar | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 077 | expand.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 079 | fill.Scalar | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 080 | flip.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 084 | full.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 085 | full_like.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 118 | mean.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 119 | mean.dim | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 122 | mm.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 123 | mul.Scalar | pass | pass | 0.3125 | 0.0 | pass |  | YES |
| 132 | neg.default | pass | pass | 0.3125 | 0.0 | pass |  | YES |
| 134 | permute.default | pass | pass | 0.4062 | 0.0 | pass |  | YES |
| 144 | reflection_pad1d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 145 | reflection_pad2d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 146 | reflection_pad3d.default | pass | pass | 0.48 | 0.0 | pass |  | YES |
| 147 | relu.default | pass | pass | 0.375 | 0.0 | pass |  | YES |
| 150 | repeat.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 151 | replication_pad2d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 152 | replication_pad3d.default | pass | pass | 0.456 | 0.0 | pass |  | YES |
| 156 | scalar_tensor.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 161 | select.int | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 162 | select_scatter.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 167 | slice.Tensor | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 168 | slice_scatter.default | pass | pass | 1.0 | 0.0 | pass |  | YES |
| 170 | split_with_sizes.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 172 | squeeze.dim | pass | pass | 0.375 | 0.0 | pass |  | YES |
| 173 | squeeze.dims | pass | pass | 0.375 | 0.0 | pass |  | YES |
| 174 | sub.Scalar | pass | pass | 0.2917 | 0.0 | pass |  | YES |
| 175 | sub.Tensor | pass | pass | 0.4375 | 0.0 | pass |  | YES |
| 176 | sum.dim_IntList | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 186 | unsqueeze.default | pass | pass | 0.375 | 0.0 | pass |  | YES |
| 188 | upsample_nearest2d.vec | pass | pass | 0.0293 | 0.0 | pass |  | YES |
| 191 | view.default | pass | pass | 0.1328 | 0.0 | pass |  | YES |

## Harness controls

* ctl-movement (correct i8 movement program, correct reference): expect outcome ok with device insns > 0: got ok / 6.
* ctl-movement-neg (same, one reference element corrupted): expect wrong: got wrong.
