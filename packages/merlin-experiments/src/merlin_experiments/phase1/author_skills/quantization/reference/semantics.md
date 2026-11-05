# Integer pipeline semantics, as the library defines them

Citations read `PATH:LINE text`: PATH is relative to the editable library
(`submission/compiler_library/` in the workspace), LINE holds `text`. The reference arithmetic in
`merlin/runtime/tensor.py` and `merlin/runtime/commandbuffer.py` is the arithmetic the case
declarations name; the proofs in `merlin/llvmlower/requantization.py` and
`merlin/llvmlower/integer_readout.py` model the same functions over the whole i32 domain.

## Where the numbers live

| What | Where to read it |
|---|---|
| One case's operands, epilogue order, scales, output dtype, tolerance | its `capsule.yaml` (`operation.attributes`, `inputs`, `numeric_policy`); `quant_params.py case` |
| Per-column scales | `acc_scale_by_column` (one f32 per output column); exclusive with `acc_scale` (`merlin/runtime/commandbuffer.py:270 def declared_acc_scale_by_column`) |
| Residual add multipliers and allowed distance | `lhs_scale`, `rhs_scale`, `bound_lsb` (`merlin/runtime/commandbuffer.py:228 def residual_add_params`) |
| Partial-sum width actually available | `integer_partial_sum_bound` in `capsule.yaml` (`bound` vs `signed_positive_limit`) |
| Model quantization scheme | `operation.attributes.quant_scheme` and the captured IR's `quant_ext.*` ops; `quant_params.py ir` |
| What the accelerator's readout can apply, in order | `output_epilogue.output_path` in `target_contract.yaml`, vocabulary `merlin/targetgen/output_path.py:29 READOUT_KINDS`; `quant_params.py target` |
| How every model was quantized for this target | `quantization.methodology` in `public_declarations/software-spec.yaml`, realized by `merlin/targetgen/quant_methodology.py:243 def derive` |

Per tensor vs per channel: a scale that is one number (a splat constant, `acc_scale`) is per tensor;
a vector over the output channels (`acc_scale_by_column`, a `tensor<Nxf32>` operand) is per channel.
The readout may hold one scale per group of columns (`granularity: per_command`,
`columns_per_scale`); per-channel source scales then need one readout setting per run of equal
scales (`merlin/llvmlower/integer_readout.py:183 def partition_column_scales`).

## The pipeline, stage by stage

1. **Contraction.** `acc = sum_k a[i8] * w[i8]` in i32 from zero
   (`merlin/runtime/tensor.py:130 def matmul`). Products lie in [-16256, 16384]; K terms must fit the
   partial-sum width at every prefix
   (`merlin/llvmlower/integer_producer_range.py:12 class IntegerSumProductsRange`). A conversion of
   `acc` to f32 is exact only while |acc| <= 2**24
   (`merlin/llvmlower/integer_producer_range.py:107 def require_exact_binary_conversion`).
2. **Bias.** One i32 per output column added to the accumulator
   (`merlin/runtime/tensor.py:154 def add_bias`); the stage names its tensor in the attributes or
   operands (`merlin/runtime/commandbuffer.py:307 def bias_tensor_name`). A captured float bias
   becomes the integer code `round_half_even(b / (s_x * s_w))`
   (`merlin/targetgen/quant_methodology.py:402 round_half_even(b / (s_x * s_w))`).
3. **Scale (`acc_scale`).** `f32(f32(acc) * f32(scale))`, then round half to even
   (`merlin/runtime/tensor.py:193 prod = f32(f32(float(x)) * s)`, ties at
   `merlin/runtime/tensor.py:178 def requant_acc_scale`). Per column:
   `merlin/runtime/tensor.py:206 def requant_acc_scale_by_column`. No stage default exists: a
   declared stage without its multiplier is refused
   (`merlin/runtime/commandbuffer.py:253 def declared_acc_scale`).
4. **Several source scales.** A captured chain multiplies in order, each product rounded to f32
   (`merlin/llvmlower/requantization.py:22 value = f32(value * scale)`). Folding them into one
   scale is legal only when it changes no output
   (`merlin/llvmlower/requantization.py:31 def prove_scale`); otherwise synthesize a store scale
   (`merlin/llvmlower/requantization.py:140 def synthesize_store_scale`) or bound the error
   (`merlin/llvmlower/requantization.py:76 def prove_scale_bound`). With a float bias between the
   scales, the source order is
   `merlin/llvmlower/requantization.py:256 Source order is f32(acc), sequential f32 multiplications, f32 bias add,`
   and `merlin/llvmlower/requantization.py:253 def synthesize_bias` finds the integer biases.
5. **Rectifier.** `max(y, 0)` (`merlin/runtime/tensor.py:255 def relu`). With saturation it makes
   the low clamp 0 (`merlin/llvmlower/requantization.py:23 low = 0 if relu else -128`).
6. **Saturation.** Clamp to the output dtype, then store
   (`merlin/runtime/tensor.py:31 def _i8_clamp`, `merlin/runtime/tensor.py:381 def to_i8`). The
   proofs saturate before rounding and round half to even
   (`merlin/llvmlower/requantization.py:28 return max(low, min(127, round(value)))`).
7. **Shift requant.** A different operation: `(y + 2**(s-1)) >> s`, ties round up
   (`merlin/runtime/tensor.py:175 out = [(x + half) >> shift for x in self.data]`).
8. **Residual add.** Both operands scaled to the output grid, added in f32, rounded once
   (`merlin/runtime/tensor.py:248 total = f32(f32(f32(float(a)) * sa) + f32(f32(float(b)) * sb))`).
   A device that rounds each operand first differs by at most `bound_lsb` steps; ReLU is its only
   stage (`merlin/runtime/commandbuffer.py:225 RESIDUAL_ADD_STAGES`).
9. **Quantize / dequantize in captured IR.**
   `clamp(roundeven(x * f32(1/scale)) + zp, qmin, qmax)`, reciprocal first
   (`merlin/llvmlower/passes_xdsl.py:140 clamp(roundeven(x * float32(1 / scale)) + zp, qmin, qmax)`).
   The library's strict chain reader accepts zero point 0 and [-128, 127] only
   (`merlin/llvmlower/captured_requant.py:46 nonzero output zero point`).
10. **Dynamic activations.** Where an activation has no captured scale, the integer pass uses
    `s = max(max|x|/127, 1e-5)` per row, zero point 0
    (`merlin/llvmlower/passes_quant_int.py:9 s = max(max|x|/127, 1e-5)`).

## Which contractions are int8

A captured contraction is integer when an input traces through layout-only views to a
`quant_ext.dequantize*` (`merlin/llvmlower/quant_scope.py:143 def captured_quantized_contraction`).
Under `merlin/llvmlower/quant_scope.py:17 FEATURE = "respect_captured_quantization_scope"` the runner
prints `[quant_scope] respected captured quantization evidence` and only contractions are made
integer; nonlinear approximations stay off. f32 remains for the float island
(`merlin/targetgen/quant_methodology.py:50 ISLAND_OPS`,
`merlin/targetgen/quant_methodology.py:52 ISLAND_DTYPE = "fp32"`). Low-precision float element
types parse after `merlin/xdsl_dialects/fp8.py:160 def register_fp8_types`.

## Readout on the device

The output path's `readout` list is the order the narrowing readout applies its stages; `exits`
says whether a raw accumulator or a narrowed value leaves the unit
(`merlin/targetgen/output_path.py:32 EXITS`). A full-width exit applies no stage:
`merlin/verify/epilogue_applicability.py:6 hardware simply discarded the activation, because on that target the full-width readout writes the`.
The fusion pass tries cuts longest first
(`merlin/xdsl_dialects/lowering/output_fusion.py:49 CUTS = ("pool", "requant", "accumulator", "bare")`)
and takes one only when every output threshold of the source and the device coincide.
