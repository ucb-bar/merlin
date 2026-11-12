# Host-control and math probes

This authored PyTorch input exercises scalar negation, selection, sine/cosine,
integer conversion/addition/comparison and position-mask construction. Its small
fixed inputs are independent of headline validation models and checkpoints.
It is not a generated capsule, compiler implementation or host-support declaration.

Capture one case with the existing worker and a selected model2MLIR interpreter:

```bash
M2M_HOST_PROBE_CASE=integer_to_float "$CAPTURE_PYTHON" \
  src/merlin/development/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/host_control_math/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$MERLIN_OUT_ROOT/artifacts/probes/host-control-math/integer-to-float/capture"
```

The other selectors are `negate`, `select`, `sine`, `cosine`, `integer_add`,
`integer_compare`, `integer_tensor_compare`, `position_mask`, `boolean_prefix_sum`,
`boolean_sum`, `bucketize_left`, `bucketize_right` and `identity_alias`. Record the selected case and input bytes
before execution; an unknown case is refused. Keep generated MLIR, external tensor
sidecars, frontend trace and qualification records under the configured output root.

The original selectors have one finite FP32 output containing every checked element. Integer
cases include signed endpoints and FP32 conversion boundaries. The addition case
exposes all 64 result bits through four unsigned 16-bit words, each exactly
representable in FP32; it therefore does not hide low-bit errors through rounding.
The conversion case checks the actual rounded FP32 result. Neither establishes
exhaustive input coverage. The positional case is bounded and does not use any
validation-model sequence length.

The selection and position-mask comparisons operate on rank-4 tensors before
producing their outputs. Inspect the actual frontend/MLIR operation rank, not
just the final result rank: a reshape after a comparison would not exercise a
rank-4 comparison. Changed probe inputs need fresh capture and qualification;
old receipts retain their original source digest.

`integer_compare` tests scalar equality/order; `integer_tensor_compare` tests
two independent rank-1 i64 inputs with equal, less-than and greater-than pairs,
including signed endpoints. Scalar comparisons do not qualify tensor comparisons.
Both return exact FP32 0/1 values, retaining the Boolean-to-float lowering check.

The Boolean reduction probes use rank-2 inputs with all-false, all-true and
mixed rows. Prefix sums include the current element; row sums count true values.
Both expose every exact small integer result as FP32. Bucketization uses two
independent inputs: rank-2 values and sorted rank-1 boundaries. The left/right
cases include values below, above, between and exactly equal to boundaries, so
they distinguish lower-bound from upper-bound search. Neither uses model shapes,
checkpoints or a handwritten search/reduction implementation.

`identity_alias` returns its input through PyTorch's alias operator. Capture may
erase that alias to a direct MLIR argument return, with no `linalg` operations.
Native execution still needs to populate the caller's output correctly; merely
linking an empty compute body does not establish that ABI behavior. The sample
also retains a negative-zero value for raw-bit comparison.

`f32_eq_scalar_rank4` is one ordered FP32 equality comparison against a scalar,
returning raw Boolean values for independent NaN, infinities, signed zeros and
finite inputs at a small neutral rank-four shape. `f32_amax_rank4_axis3` reduces
one axis to a rank-three FP32 result and includes NaN and infinities. Its
`f32_amax_rank4_axis3_finite` sibling uses the same operation on finite inputs
with both signed-zero orders, so the native checker can run while retaining a
strict numerical counterexample. These are separate source/body and native
probes, not a host admission. The selected MLIR `arith.maximumf` gives positive
zero priority over negative zero, whereas PyTorch CPU `amax` may retain the
first zero sign; a strict mismatch must remain a failed numerical check.

Use the existing `merlin.verification.scalar_host` command with an explicit
scalar package, board catalog and pinned DTS. That checker requires bit-identical
saved-capture outputs: a mismatch must remain a mismatch. Sine/cosine library
implementations may differ; do not loosen the checker or substitute easy inputs
after observing a failure. Any approximate contract needs separately selected
tolerances and qualified evidence.

Passing finite probes supports review of their exact signatures only. It does not
approve BF16, arbitrary host fallback, a full model, accelerator execution, or all
values/shapes. Accelerator-eligible computation must still be routed to the device.

## Typed, ordered and boundary probes

Additional independent selectors are `sine_rank4`, `cosine_rank4`,
`arange_f32_exact`, `arange_f32_fractional`, `boolean_not`, `boolean_and`,
`boolean_select`, `boolean_mul_lhs_singleton`, `boolean_mul_rhs_singleton`,
`tensor_not_equal`, `tensor_bitwise_xor`, `min_values`, `f32_cumsum_rows`,
`f32_cumsum_columns`, `f32_cumsum_signed_zero`, `bucketize_left_nan` and
`bucketize_right_nan`. The tensor comparison, XOR and Boolean multiplication
cases return raw Boolean outputs; the other selectors return FP32 outputs.
Capture and qualify the declared dtype without converting raw Boolean results
to FP32. All two-input cases use independently populated tensors.

The cumulative sums expose loss of a small addend if a lowering rounds each
intermediate to FP32, along with mixed signed-zero inputs. Minimum reduction
includes ties and both zero signs. Bucketization includes NaN and infinities as
inputs and returns finite insertion positions; equal boundaries distinguish
left from right insertion. Trigonometric cases retain nontrivial inputs at rank
four. Floating ranges exercise both binary-exact and fractional step sizes.
Passing the source-fixture tests does not qualify their native lowerings; keep
the exact saved-capture gate and retain any numerical or unsupported-operation
failure as a failure.

## Independent pointwise rank/tail matrix

For an exact one-primitive result, select
`M2M_HOST_PROBE_CASE=matrix_<operation>_r<rank>` where rank is 1, 2, 3 or 4.
The neutral output shapes are the suffixes of `(2, 2, 3, 7)`; the final
extent of seven provides a nontrivial tail without a validation-derived size.
Allowed operations are `f32_add/sub/mul/neg/le/select/nonzero`,
`i64_add/sub/mul/le/to_f32`, `i1_and/xor/not/to_f32/to_i64`, and
`i1_mul_lhs_projected`/`i1_mul_rhs_projected`. A selector names one original
PyTorch operation and one raw f32, i64 or i1 result tensor. Two-input cases
use independently populated operands. The i64 inputs include both signed
endpoints and wrapping arithmetic; f32 comparison includes equal values and
both zero signs. Boolean projection cases use a singleton first axis on only
one operand. An invalid operation or rank refuses before capture.

Use the same capture and scalar-host qualification commands as above, with the
case name as a separate immutable output directory. Record the complete
selected case string alongside the loader, input/golden, MLIR, ELF and tool
hashes: ordinary capture receipts do not independently attest the environment
variable selecting a case. Do not reinterpret older captures after changing
the loader. Native qualification must compare every raw output bit; i64 must
not pass through FP32. The matrix excludes compound arithmetic, reductions,
transcendentals, GELU, FP32 scan, nonfinite behavior and any host admission.
