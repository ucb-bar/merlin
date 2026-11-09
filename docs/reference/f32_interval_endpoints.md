---
title: Source-rounded interval endpoints
kind: reference
status: draft
owner: core
last_verified: 2026-10-05
code_refs:
  - src/merlin/runtime/native/c/f32_interval_endpoint.h
  - src/merlin/runtime/native/c/prepared_bit_polynomial.h
  - src/merlin/runtime/native/c/prepared_dot_norms.h
---

# Source-rounded interval endpoints

`f32_interval_endpoint.h` provides optional finite binary32 interval arithmetic,
a structural enclosure for a parameterized bit-polynomial expression, and
explicit BF16 endpoint policies. It changes no compiler or runtime default.

Callers must first establish the `ordered_fma_bounds.h` environment contract:
round-to-nearest-even, gradual underflow, stable rounding mode, nontrapping and
unobserved exception flags. Arithmetic must not reassociate; multiplication and
addition remain separate except explicit source `fmaf`. A source expression
and every operand/product bound must be independently established by the caller.

The polynomial helper follows the specified multiply, floor, Horner-FMA,
subtraction, final FMA, integer conversion and bitcast. It splits integer floor
segments rather than assuming polynomial monotonicity. Unsupported conversion
ranges, nonfinite values and excessive segment spans refuse certification.

`merlin_interval_bf16_endpoint` accepts only a single finite BF16 output bin.
`merlin_interval_bf16_bounded_endpoint` accepts a caller-selected budget of zero
or one BF16 step. The latter requires the complete source interval to round into
at most two adjacent finite same-sign BF16 words. The caller's deterministic
reconstructed center must itself round into that interval. This proves at most
one adjacent-word difference from the original source result; it does not prove
a whole-model accuracy criterion. Wider intervals, nonfinite values, zero
crossings and unsupported budgets require original-source fallback.

These are numerical primitives, not an attention provider or a dispatch pass.
Source grouping, masks, reduction order, live outputs and storage remain caller
obligations. A model's existing independent correctness gate still governs any
explicit bounded-policy experiment. No device timing is implied by certification.

## Prepared polynomial plans

`prepared_bit_polynomial.h` adds an optional immutable prepared value. Preparation
copies and validates the source coefficients once, then encloses each Horner
stage over the complete fractional domain `[0, 1]`. A strictly positive or
negative intermediate admits the corresponding product extrema directly;
unknown signs use the original corner evaluator. Application checks the actual
fractional domain and retains the original path at zero endpoints. It therefore
preserves the original interval endpoints and refusal behavior, including raw
signed-zero endpoint handling; it does not tighten a bound or change a policy.

The caller owns the prepared value and must not mutate it between preparation
and use. The existing RNE, gradual-underflow, stable-FENV and unobserved-exception
contract applies to both phases. No model, source identifier or measured output
participates in preparation. Invalid source plans remain invalid, and unsupported
per-call domains retain the original refusal. Existing entry points and their
emitted bodies remain unchanged; production callers must explicitly opt in.

Independent tests compare raw endpoint words across different coefficient sign
patterns, floor and cutoff boundaries, zero, random intervals and refusal cases.
Preparation is included in timing when reporting a complete invocation.

## Prepared dot-product norms

`prepared_dot_norms.h` computes conservative row L1, maximum and L2 norms from
caller-proved upper bounds on absolute source values. Accumulation and squared
magnitudes round upward. A proposed square-root upper bound is independently
checked by a downward-rounded square against the accumulated upper bound; a
failed or overflowing L2 preparation retains the valid L1/maximum alternative.
The product helper chooses the smaller of Hölder and Cauchy upper bounds.

Callers prepare each immutable row once. Representation residuals can use the
same API, with independently enclosed absolute errors. This is a source-bound
magnitude certificate, not a numerical approximation policy. It does not alter
source reduction order, dot centers, endpoint bins or compiler defaults.
Independent tests compare bounds with exact rational sums and squared norms,
including zero, subnormal inputs, overflow, invalid values and unrelated shapes.
