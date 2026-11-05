# RESULT-193 (m27-S1): authored backend vs all 193 core-ATen overloads, canonical case; package file-set id f430113415f19be7

**DEVICE57 (30 exact movement + 27 int8-quantized, ALL57 of tooling/exercised/exercised.py): 56 / 57 lowered on device (accelerator instructions retired, exact output).  HOST136 (100 host-only + 36 declared host-only): 0 / 136 ran on the host path correctly with zero accelerator opcodes.**

Verification instrument (never an author input). Route per case: parse -> lower_interface_to_target -> emit_target_bundle -> kernel ABI v2 -> clang-23 -> Spike with -l RoCC (0x7b) retirement count (tools/run_case.py). Inputs: DEVICE57 are handed the op-capsule group's int8/integer form (forms93, from Merlin's core_aten_op_capsules registry, exact integer reference) (operator decision 2026-11-04: every DEVICE57 op is graded on a spec-declared form; 47 through the op-capsule registry's builder lanes, cumsum/diagonal/select/split_with_sizes/convolution_backward/upsample_nearest2d/full/full_like/scalar_tensor through the spec form table (tools/gen_spec_forms.py, lane SPEC_FORM), and div.Scalar through its spec_int8 form: the f32 capture compared EXACTLY with the software spec's declared quantized model, tools/gen_spec_int8_form.py); HOST136 are handed Merlin's own corpus_193 canonical f32 capture (f32 rounding tolerance for floats, exact for ints). Column f32_capture_result in the CSV gives the outcome of the corpus_193 f32 capture for every overload (what Merlin's own backend consumes). Classes: lowered-on-device / lowered-on-host / declined / wrong / crashed. A 'wrong' on HOST136 includes any accelerator opcode.

## Summary

| class | n | lowered-on-device | lowered-on-host | declined | wrong | crashed | other |
|---|---|---|---|---|---|---|---|
| DEVICE57 | 57 | 56 | 0 | 0 | 1 | 0 | 0 |
| HOST136 | 136 | 0 | 0 | 79 | 50 | 7 | 0 |

(DEVICE57 'lowered-on-host (device expected)' is counted under other.)

## Declines / failures by reason

| class | result | reason | n | overloads |
|---|---|---|---|---|
| DEVICE57 | wrong | outN[fN]: max_abs_err=N.N | 1 | div.Scalar |
| HOST136 | wrong | HOST-ONLY overload emitted accelerator opcodes (retired N, static N) | 50 | _cdist_forward.default, _log_softmax.default, _pdist_forward.default, _softmax.default, abs.default, amax.default, amin.default, ceil.default, clamp.Tensor, clamp.default, cos.default, div.Tensor, exp.default, floor.default, fmod.Scalar, fmod.Tensor, hardtanh.default, log.default, log2.default, maximum.default, minimum.default, mul.Tensor, native_group_norm.default, native_group_norm_backward.default, native_layer_norm.default, native_layer_norm_backward.default, pow.Scalar, pow.Tensor_Scalar, pow.Tensor_Tensor, prod.default, prod.dim_int, reciprocal.default, remainder.Scalar, remainder.Tensor, round.default, rsqrt.default, sigmoid.default, sign.default, sin.default, sqrt.default, sym_is_contiguous.default, sym_numel.default, sym_size.int, sym_storage_offset.default, sym_stride.int, tanh.default, trunc.default, var.correction, var.dim, where.self |
| HOST136 | declined | linalg.generic (no exact command-buffer lowering) | 44 | _embedding_bag.default, _native_batch_norm_legit.default, _native_batch_norm_legit.no_stats, arange.start_step, argmax.default, argmin.default, bitwise_and.Scalar, bitwise_and.Tensor, bitwise_not.default, bitwise_or.Scalar, bitwise_or.Tensor, bitwise_xor.Scalar, bitwise_xor.Tensor, div.Scalar_mode, div.Tensor_mode, embedding.default, eq.Scalar, eq.Tensor, gather.default, ge.Scalar, ge.Tensor, gt.Scalar, gt.Tensor, index.Tensor, index_select.default, isinf.default, isnan.default, le.Scalar, le.Tensor, leaky_relu.default, logical_and.default, logical_not.default, logical_or.default, logical_xor.default, lt.Scalar, lt.Tensor, max.dim, max_pool2d_with_indices.default, max_pool3d_with_indices.default, min.dim, ne.Scalar, ne.Tensor, sort.default, topk.default |
| HOST136 | declined | NotImplementedError: no lowering for HOST_LINALG | 21 | acos.default, acosh.default, asin.default, asinh.default, atan.default, atan2.default, atanh.default, cosh.default, elu.default, erf.default, expm1.default, gelu.default, log10.default, log1p.default, scatter.src, scatter.value, scatter_add.default, scatter_reduce.two, sinh.default, tan.default, upsample_bilinear2d.vec |
| HOST136 | crashed | parse: ValueError: payload/full-source operation order disagreement | 5 | _local_scalar_dense.default, grid_sampler_2d.default, masked_scatter.default, nonzero.default, randn.default |
| HOST136 | declined | declined-object: {"reason": "direct argument returns lack a source-operation result owner"} | 3 | _native_batch_norm_legit_no_training.default, empty.memory_format, empty_strided.default |
| HOST136 | declined | linalg.reduce (no exact command-buffer lowering) | 3 | any.default, any.dim, any.dims |
| HOST136 | declined | scf.for (no exact command-buffer lowering) | 3 | embedding_dense_backward.default, index_put.default, max_pool2d_with_indices_backward.default |
| HOST136 | declined | ml_program.global_load (no exact command-buffer lowering) | 3 | native_dropout.default, rand.default, randperm.default |
| HOST136 | declined | declined-object: {"reason": "repeated returns lack independent source-result writers"} | 2 | atan2.out, resize_.default |
| HOST136 | crashed | abi: input arg_N: N case elements vs command buffer [N, N, N] | 1 | _fft_c2r.default |
| HOST136 | crashed | built: TypeError: float() argument must be a string or a real number, not 'dict' | 1 | _fft_r2c.default |

## Per overload

| # | overload | class | input | result | device insns | reason |
|---|---|---|---|---|---|---|
| 000 | _adaptive_avg_pool2d.default | DEVICE57 | int8 form | lowered-on-device | 166 |  |
| 001 | _adaptive_avg_pool2d_backward.default | DEVICE57 | int8 form | lowered-on-device | 186 |  |
| 002 | _adaptive_avg_pool3d.default | DEVICE57 | int8 form | lowered-on-device | 330 |  |
| 014 | _to_copy.default | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 018 | adaptive_avg_pool1d.default | DEVICE57 | int8 form | lowered-on-device | 106 |  |
| 019 | add.Scalar | DEVICE57 | int8 form | lowered-on-device | 118 |  |
| 020 | add.Tensor | DEVICE57 | int8 form | lowered-on-device | 338 |  |
| 021 | addmm.default | DEVICE57 | int8 form | lowered-on-device | 17 |  |
| 022 | alias.default | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 031 | as_strided.default | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 038 | avg_pool1d.default | DEVICE57 | int8 form | lowered-on-device | 134 |  |
| 039 | avg_pool2d.default | DEVICE57 | int8 form | lowered-on-device | 214 |  |
| 040 | avg_pool2d_backward.default | DEVICE57 | int8 form | lowered-on-device | 418 |  |
| 041 | avg_pool3d.default | DEVICE57 | int8 form | lowered-on-device | 166 |  |
| 049 | bmm.default | DEVICE57 | int8 form | lowered-on-device | 28 |  |
| 050 | cat.default | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 054 | clone.default | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 055 | col2im.default | DEVICE57 | int8 form | lowered-on-device | 74 |  |
| 056 | constant_pad_nd.default | DEVICE57 | int8 form | lowered-on-device | 34 |  |
| 057 | convolution.default | DEVICE57 | int8 form | lowered-on-device | 599 |  |
| 058 | convolution_backward.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 945 |  |
| 059 | copy.default | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 062 | cumsum.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 11 |  |
| 063 | diagonal.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 077 | expand.default | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 079 | fill.Scalar | DEVICE57 | int8 form | lowered-on-device | 6 |  |
| 080 | flip.default | DEVICE57 | int8 form | lowered-on-device | 26 |  |
| 084 | full.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 085 | full_like.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 118 | mean.default | DEVICE57 | int8 form | lowered-on-device | 19 |  |
| 119 | mean.dim | DEVICE57 | int8 form | lowered-on-device | 78 |  |
| 122 | mm.default | DEVICE57 | int8 form | lowered-on-device | 15 |  |
| 123 | mul.Scalar | DEVICE57 | int8 form | lowered-on-device | 101 |  |
| 132 | neg.default | DEVICE57 | int8 form | lowered-on-device | 165 |  |
| 134 | permute.default | DEVICE57 | int8 form | lowered-on-device | 258 |  |
| 144 | reflection_pad1d.default | DEVICE57 | int8 form | lowered-on-device | 14 |  |
| 145 | reflection_pad2d.default | DEVICE57 | int8 form | lowered-on-device | 62 |  |
| 146 | reflection_pad3d.default | DEVICE57 | int8 form | lowered-on-device | 302 |  |
| 147 | relu.default | DEVICE57 | int8 form | lowered-on-device | 165 |  |
| 150 | repeat.default | DEVICE57 | int8 form | lowered-on-device | 18 |  |
| 151 | replication_pad2d.default | DEVICE57 | int8 form | lowered-on-device | 62 |  |
| 152 | replication_pad3d.default | DEVICE57 | int8 form | lowered-on-device | 302 |  |
| 156 | scalar_tensor.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 6 |  |
| 161 | select.int | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 10 |  |
| 162 | select_scatter.default | DEVICE57 | int8 form | lowered-on-device | 14 |  |
| 167 | slice.Tensor | DEVICE57 | int8 form | lowered-on-device | 10 |  |
| 168 | slice_scatter.default | DEVICE57 | int8 form | lowered-on-device | 14 |  |
| 170 | split_with_sizes.default | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 66 |  |
| 172 | squeeze.dim | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 173 | squeeze.dims | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 174 | sub.Scalar | DEVICE57 | int8 form | lowered-on-device | 118 |  |
| 175 | sub.Tensor | DEVICE57 | int8 form | lowered-on-device | 338 |  |
| 176 | sum.dim_IntList | DEVICE57 | int8 form | lowered-on-device | 15 |  |
| 186 | unsqueeze.default | DEVICE57 | int8 form | lowered-on-device | 66 |  |
| 188 | upsample_nearest2d.vec | DEVICE57 | spec form (corpus builder at the declare | lowered-on-device | 8194 |  |
| 191 | view.default | DEVICE57 | int8 form | lowered-on-device | 386 |  |
| 064 | div.Scalar | DEVICE57 | spec_int8 form (f32 capture, exact vs th | wrong | 10 | out0[f32]: max_abs_err=0.00624999 |
| 003 | _cdist_forward.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 004 | _embedding_bag.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 005 | _fft_c2r.default | HOST136 | f32 capture | crashed |  | abi: input arg_N: N case elements vs command buffer [N, N, N] |
| 006 | _fft_r2c.default | HOST136 | f32 capture | crashed | 102 | built: TypeError: float() argument must be a string or a real number, not 'dict' |
| 007 | _local_scalar_dense.default | HOST136 | f32 capture | crashed |  | parse: ValueError: payload/full-source operation order disagreement |
| 008 | _log_softmax.default | HOST136 | f32 capture | wrong | 10 | HOST-ONLY overload emitted accelerator opcodes (retired 10, static 10) |
| 009 | _native_batch_norm_legit.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 010 | _native_batch_norm_legit.no_stats | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 011 | _native_batch_norm_legit_no_training.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "direct argument returns lack a source-operation result owner"} |
| 012 | _pdist_forward.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 013 | _softmax.default | HOST136 | f32 capture | wrong | 10 | HOST-ONLY overload emitted accelerator opcodes (retired 10, static 10) |
| 015 | abs.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 016 | acos.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 017 | acosh.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 023 | amax.default | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 024 | amin.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 025 | any.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.reduce: unimplemented typed operation linalg.redu |
| 026 | any.dim | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.reduce: unimplemented typed operation linalg.redu |
| 027 | any.dims | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.reduce: unimplemented typed operation linalg.redu |
| 028 | arange.start_step | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 029 | argmax.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload results require a dedic |
| 030 | argmin.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload results require a dedic |
| 032 | asin.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 033 | asinh.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 034 | atan.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 035 | atan2.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 036 | atan2.out | HOST136 | f32 capture | declined |  | declined-object: {"reason": "repeated returns lack independent source-result writers"} |
| 037 | atanh.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 042 | bitwise_and.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 043 | bitwise_and.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 044 | bitwise_not.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 045 | bitwise_or.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 046 | bitwise_or.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 047 | bitwise_xor.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 048 | bitwise_xor.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 051 | ceil.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 052 | clamp.Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 053 | clamp.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 060 | cos.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 061 | cosh.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 065 | div.Scalar_mode | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 066 | div.Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 067 | div.Tensor_mode | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 068 | elu.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 069 | embedding.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: captured extraction body contains unsupp |
| 070 | embedding_dense_backward.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: scf.for: unimplemented typed operation scf.for"} |
| 071 | empty.memory_format | HOST136 | f32 capture | declined |  | declined-object: {"reason": "direct argument returns lack a source-operation result owner"} |
| 072 | empty_strided.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "direct argument returns lack a source-operation result owner"} |
| 073 | eq.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 074 | eq.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 075 | erf.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 076 | exp.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 078 | expm1.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 081 | floor.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 082 | fmod.Scalar | HOST136 | f32 capture | wrong | 10 | HOST-ONLY overload emitted accelerator opcodes (retired 10, static 10) |
| 083 | fmod.Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 086 | gather.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: captured extraction body contains unsupp |
| 087 | ge.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 088 | ge.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 089 | gelu.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 090 | grid_sampler_2d.default | HOST136 | f32 capture | crashed |  | parse: ValueError: payload/full-source operation order disagreement |
| 091 | gt.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 092 | gt.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 093 | hardtanh.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 094 | index.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: captured extraction body contains unsupp |
| 095 | index_put.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: scf.for: unimplemented typed operation scf.for"} |
| 096 | index_select.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: captured extraction body contains unsupp |
| 097 | isinf.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 098 | isnan.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 099 | le.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 100 | le.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 101 | leaky_relu.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 102 | log.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 103 | log10.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 104 | log1p.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 105 | log2.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 106 | logical_and.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 107 | logical_not.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 108 | logical_or.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 109 | logical_xor.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 110 | lt.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 111 | lt.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 112 | masked_scatter.default | HOST136 | f32 capture | crashed |  | parse: ValueError: payload/full-source operation order disagreement |
| 113 | max.dim | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload results require a dedic |
| 114 | max_pool2d_with_indices.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload results require a dedic |
| 115 | max_pool2d_with_indices_backward.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: scf.for: unimplemented typed operation scf.for"} |
| 116 | max_pool3d_with_indices.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload results require a dedic |
| 117 | maximum.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 120 | min.dim | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: multiple payload results require a dedic |
| 121 | minimum.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 124 | mul.Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 125 | native_dropout.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: ml_program.global_load: unimplemented typed operation ml |
| 126 | native_group_norm.default | HOST136 | f32 capture | wrong | 74 | HOST-ONLY overload emitted accelerator opcodes (retired 74, static 50) |
| 127 | native_group_norm_backward.default | HOST136 | f32 capture | wrong | 202 | HOST-ONLY overload emitted accelerator opcodes (retired 202, static 162) |
| 128 | native_layer_norm.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 129 | native_layer_norm_backward.default | HOST136 | f32 capture | wrong | 78 | HOST-ONLY overload emitted accelerator opcodes (retired 78, static 78) |
| 130 | ne.Scalar | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 131 | ne.Tensor | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 133 | nonzero.default | HOST136 | f32 capture | crashed |  | parse: ValueError: payload/full-source operation order disagreement |
| 135 | pow.Scalar | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 136 | pow.Tensor_Scalar | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 137 | pow.Tensor_Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 138 | prod.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 139 | prod.dim_int | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 140 | rand.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: ml_program.global_load: unimplemented typed operation ml |
| 141 | randn.default | HOST136 | f32 capture | crashed |  | parse: ValueError: payload/full-source operation order disagreement |
| 142 | randperm.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: ml_program.global_load: unimplemented typed operation ml |
| 143 | reciprocal.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 148 | remainder.Scalar | HOST136 | f32 capture | wrong | 10 | HOST-ONLY overload emitted accelerator opcodes (retired 10, static 10) |
| 149 | remainder.Tensor | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 153 | resize_.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "repeated returns lack independent source-result writers"} |
| 154 | round.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 155 | rsqrt.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 157 | scatter.src | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 158 | scatter.value | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 159 | scatter_add.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 160 | scatter_reduce.two | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 163 | sigmoid.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 164 | sign.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 165 | sin.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 166 | sinh.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 169 | sort.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 171 | sqrt.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 177 | sym_is_contiguous.default | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 178 | sym_numel.default | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 179 | sym_size.int | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 180 | sym_storage_offset.default | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 181 | sym_stride.int | HOST136 | f32 capture | wrong | 6 | HOST-ONLY overload emitted accelerator opcodes (retired 6, static 6) |
| 182 | tan.default | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 183 | tanh.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 184 | topk.default | HOST136 | f32 capture | declined |  | declined-object: {"reason": "linalg operations not yet lowered: linalg.generic: unimplemented typed operation linalg.gen |
| 185 | trunc.default | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |
| 187 | upsample_bilinear2d.vec | HOST136 | f32 capture | declined |  | NotImplementedError: no lowering for HOST_LINALG |
| 189 | var.correction | HOST136 | f32 capture | wrong | 18 | HOST-ONLY overload emitted accelerator opcodes (retired 18, static 18) |
| 190 | var.dim | HOST136 | f32 capture | wrong | 14 | HOST-ONLY overload emitted accelerator opcodes (retired 14, static 14) |
| 192 | where.self | HOST136 | f32 capture | wrong | 2 | HOST-ONLY overload emitted accelerator opcodes (retired 2, static 2) |

## FULL definition ('supported' = numerical AND host-compute AND instruction-trace)

**DEVICE57: 56/57 weak pass (RoCC retired + exact output) -> 56/57 meets the full definition.**  HOST136: n/a (kept as zero accelerator opcodes + correct output: 0/136); host-compute and trace checks do not apply.

Checks (the grader's own functions, run unmodified in the FIN6yn neutral release namespace: operand-selection sidecar e5f2486b, observer active): host-compute = offload_census._source_host_compute -> host_compute_audit.audit_group, budget 1.0 host arithmetic ops/element (also 0.25 float/convert ops/element, 0.5 scalar-payload/element-byte), category host_compute_in_accelerator_group; instruction-trace = rocc.decode.decode_module(lowered LLVM) -> trace_check.check(trace, capsule.expected, cb, address_model=pointer_args, subsumes, target, lowered_module) with the op-capsule registry's own `expected` modes (the grader's trace is the static decode of the emitted .insn stream, not the Spike log). Numerical = this harness's exact (device57) comparison on Spike.

| step | n |
|---|---|
| DEVICE57 total | 57 |
| weak pass (RoCC insns retired + exact output) | 56 |
|   of which host_compute pass | 56 |
|   of which host_compute FAIL (incl. census silent) | 0 |
|   of which host_compute unknown | 0 |
|   of which trace pass | 56 |
|   of which trace unavailable_fields/protocol | 0 |
|   of which trace fail | 0 |
| meets full definition (strict: host_compute must be 'pass') | 56 |
| meets full definition if declared_host (not judged, veto does not fire) is accepted | 56 |

### DEVICE57 per overload (full definition)

| # | overload | numerical | host_compute | host ops/elem (budget 1.0) | float ops/elem | trace | trace reason | meets full definition |
|---|---|---|---|---|---|---|---|---|
| 000 | _adaptive_avg_pool2d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 001 | _adaptive_avg_pool2d_backward.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 002 | _adaptive_avg_pool3d.default | pass | pass | 0.1875 | 0.0 | pass |  | YES |
| 014 | _to_copy.default | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 018 | adaptive_avg_pool1d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 019 | add.Scalar | pass | pass | 0.625 | 0.0 | pass |  | YES |
| 020 | add.Tensor | pass | pass | 0.375 | 0.0 | pass |  | YES |
| 021 | addmm.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 022 | alias.default | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 031 | as_strided.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 038 | avg_pool1d.default | pass | pass | 0.1667 | 0.0 | pass |  | YES |
| 039 | avg_pool2d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 040 | avg_pool2d_backward.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 041 | avg_pool3d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 049 | bmm.default | pass | pass | 0.0156 | 0.0 | pass |  | YES |
| 050 | cat.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 054 | clone.default | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 055 | col2im.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 056 | constant_pad_nd.default | pass | pass | 0.7 | 0.0 | pass |  | YES |
| 057 | convolution.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 058 | convolution_backward.default | pass | pass | 0.1061 | 0.0 | pass |  | YES |
| 059 | copy.default | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 062 | cumsum.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 063 | diagonal.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 077 | expand.default | pass | pass | 0.6667 | 0.0 | pass |  | YES |
| 079 | fill.Scalar | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 080 | flip.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 084 | full.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 085 | full_like.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 118 | mean.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 119 | mean.dim | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 122 | mm.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 123 | mul.Scalar | pass | pass | 0.5 | 0.0 | pass |  | YES |
| 132 | neg.default | pass | pass | 0.6875 | 0.0 | pass |  | YES |
| 134 | permute.default | pass | pass | 0.1875 | 0.0 | pass |  | YES |
| 144 | reflection_pad1d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 145 | reflection_pad2d.default | pass | pass | 0.96 | 0.0 | pass |  | YES |
| 146 | reflection_pad3d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 147 | relu.default | pass | pass | 0.6875 | 0.0 | pass |  | YES |
| 150 | repeat.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 151 | replication_pad2d.default | pass | pass | 0.84 | 0.0 | pass |  | YES |
| 152 | replication_pad3d.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 156 | scalar_tensor.default | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 161 | select.int | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 162 | select_scatter.default | pass | pass | 1.0 | 0.0 | pass |  | YES |
| 167 | slice.Tensor | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 168 | slice_scatter.default | pass | pass | 0.8 | 0.0 | pass |  | YES |
| 170 | split_with_sizes.default | pass | pass | 0.2383 | 0.0 | pass |  | YES |
| 172 | squeeze.dim | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 173 | squeeze.dims | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 174 | sub.Scalar | pass | pass | 0.625 | 0.0 | pass |  | YES |
| 175 | sub.Tensor | pass | pass | 0.375 | 0.0 | pass |  | YES |
| 176 | sum.dim_IntList | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 186 | unsqueeze.default | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 188 | upsample_nearest2d.vec | pass | pass | 0.0 | 0.0 | pass |  | YES |
| 191 | view.default | pass | pass | 0.25 | 0.0 | pass |  | YES |
| 064 | div.Scalar | fail | fail | 7.5 | 0.5 | pass | placed on accelerator and the host still works at element scale: 7.50 host arithmetic operations per element ( | NO |

## Harness controls

* ctl-movement (correct i8 movement program, correct reference): expect outcome ok with device insns > 0: got ok / 66.
* ctl-movement-neg (same, one reference element corrupted): expect wrong: got wrong.
