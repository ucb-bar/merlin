---
title: Ordered FMA certificates
kind: reference
status: current
owner: core
last_verified: 2026-10-06
related: [verified_compiler_roadmap]
code_refs:
  - merlin/runtime/c/ordered_fma_bounds.h
  - merlin/runtime/c/fma_product_norms.h
  - merlin/runtime/c/bf16_radix_pack.h
  - merlin/tests/runtime/test_ordered_fma_bounds.py
  - merlin/tests/runtime/test_bf16_radix_pack.py
  - merlin/runtime/c/monotone_bit_polynomial.h
  - merlin/tests/runtime/test_monotone_bit_polynomial.py
  - merlin/runtime/c/bf16_quant_frontier.h
  - merlin/tests/runtime/test_bf16_quant_frontier.py
  - merlin/runtime/c/positive_scalar_interval.h
  - merlin/tests/runtime/test_positive_scalar_interval.py
  - merlin/tests/runtime/test_outward_numeric_capability.py
  - merlin/runtime/c/f32_floor_bits.h
  - merlin/tests/runtime/test_f32_floor_bits.py
  - src/merlin/llvmlower/source_numeric_capability.py
  - merlin/runtime/c/source_f32_math.h
  - merlin/tests/runtime/test_source_numeric_capability.py
---

# Ordered FMA certificates

`merlin/runtime/c/ordered_fma_bounds.h` provides optional arithmetic certificates.
It does not select a compiler transform or change any model accuracy gate.
The source operation is zero-seeded increasing-K binary32 FMA, with round to
nearest even and gradual underflow. An OOT backend can supply reconstructed
product summaries from exact integer contractions; Merlin owns the numerical
certificate, host metadata, and source replay policy. Device schedules, layouts,
instruction encoding, and device cost facts remain in the OOT provider.

## Exact monotone polynomial source enclosure

The optional `monotone_bit_polynomial.h` helper copies the supplied source plan
and prepares an enclosure without changing the source polynomial. For the real
transform `E(s)=s-P(frac(s))`, a coefficient-derived upper bound `P'<1` proves
monotonicity within each integer interval. `P(1)>=P(0)` proves the jump at each
integer is also nondecreasing. Unsupported coefficients retain the checked
interval implementation.

The source scaling is binary32. Exact real `E` is compared to the actual rounded
fraction, three source Horner FMAs and final subtraction. Outward binary64
arithmetic prepares absolute rounding budgets, including gradual underflow;
power-of-two buckets bound the subtraction magnitude. For source endpoint
values `q_left` and `q_right` and budget `e`, every interior source `q` is enclosed
by `[q_left-2e, q_right+2e]`. Outward conversion to binary32 followed by the
original positive-multiplier FMA retains monotone rounding through the final
integer conversion and positive finite IEEE encoding. Point intervals execute
the original source expression exactly.

Callers must retain the immutable source plan, supported domain, stable round to
nearest even and gradual underflow. Arithmetic must be nontrapping and exception
flags unobserved. This helper does not authorize replacing a consumer, infer
alias permissions, select a target or enable any approximation policy. A wider
valid interval may increase selective source replay; numerical qualification and
complete measured cost govern any use.

An explicit `MERLIN_MONOTONE_F32_FLOOR` capability may replace the two endpoint
floor calls with an independently proved equivalent source-value operation.
The default remains `floorf`. The optional portable `f32_floor_bits.h` helper
clears the binary32 fractional significand and increments negative nonintegral
magnitudes before returning the exact original floor value. Magnitudes below
one, signed zeros and exponent carries are handled separately; all finite values
at least `2^23` are integral. Nonfinite inputs retain the source `floorf` call.
The caller proves the matching IEEE storage representation and unobserved,
nontrapping exceptions. No rounding mode, source coefficients, source FMA order
or enclosure budget changes. Complete generated-code cost still decides use.

## Exact BF16 quantized observations

`bf16_quant_frontier.h` accepts finite BF16 lower/upper endpoints and a candidate
inside every interval under an explicit source quantization plan. Row absolute
extrema bound the escaping BF16 scale. If that scale is unique, monotone source
reciprocal/product/BF16/RNE/clamp operations determine whether each integer bin
is unique. Otherwise only potential extrema are marked for source refinement.
Unsupported inputs or rounding modes refuse before pending outputs are written.
The reciprocal is prepared once for each row certificate.

The caller establishes complete typed consumer coverage and the actual source
arithmetic contract, stable RNE, nontrapping arithmetic and unobserved exception
flags. Input and scratch buffers must be valid and disjoint. This certificate
alone grants no permission to replace a producer with additional floating uses.
The optional coordinator owns no allocation or hidden state: callers supply
head/row/channel storage, row scratch, certification flags and refinement
callbacks. Reuse of certified rows requires row-local refinement and preservation
of those rows by every refresh. Failure returns control to the original source;
caller output publication must occur only after complete certification.

## Scalar source and outward arithmetic capabilities

`emit_source_numeric_capability` separately selects source FMA, object-copy and
finite-classification compiler builtins. Every choice is disabled by default.
Finite classification additionally requires standard classification semantics,
unobserved library interposition, nontrapping execution and unobserved exception
flags. Existing FMA or copy permission does not admit classification. The default
`MERLIN_SOURCE_ISFINITE` remains the ordinary C macro; an admitted component uses
`__builtin_isfinite` even with ordinary no-builtin compilation. It must preserve
the result for signed zeros, subnormals, infinities and quiet/signaling NaNs,
evaluate its operand once and preserve the rounding environment. The actual
provider compiler, flags, transitive headers, LLVM and object remain pinned;
static removal of a library call does not establish a cycle improvement.

Absolute-value builtins have a separate default-off choice and contract. Standard
`fabsf`/`fabs` semantics, unobserved interposition/errno/exception flags/NaN
payloads and nontrapping execution are required. Prior FMA, copy or classification
permission does not imply these obligations. Signed-zero inputs produce positive
zero and finite values retain their exact magnitude in every rounding mode.
Actual platform tests additionally compare the source library and selected
builtin over nonfinite representation boundaries; they do not grant a generic
NaN payload observation policy. The provider still closes the actual compiled
component and all original consumer observations before performance admission.

Min/max builtins are independently selected and require standard results,
unobserved library interposition/errno/exception flags, nontrapping execution,
and explicitly unobserved min/max signed-zero and NaN-payload distinctions.
Compiler min/max intrinsics can carry a signed-zero relaxation without global
fast math. The caller must close its actual typed consumer frontier before
using this capability. Defaults remain the original library operations; no
FMA/absolute-value/classification permission admits min/max. Differential tests
compare all other result bits over both precisions, nonfinite cases and rounding
modes, including a NaN paired with a number and single evaluation of operands.
This capability grants no other reassociation or relaxed accuracy threshold.

`positive_scalar_interval.h` specializes actual binary32 source multiplication
and FMA by a finite positive scalar. Monotonicity selects two endpoints without
enumerating repeated scalar corners. Multiplication retains the source multiply
and point signed zeros. Other scalar FMA factors retain the checked corner path.
These helpers require the existing source environment and exception contract;
unsupported or overflowing arithmetic refuses. The explicit nonnegative-scalar
variant also admits a zero factor while
retaining each endpoint's original source zero sign. A strictly positive RHS
interval needs two sign-selected original products. Neither specialization
replaces source multiplication with an FMA having an additional positive zero.
Sharing row-invariant source denominators or factors additionally requires
unchanged source operation order per result, immutable metadata and proved
nonoverlap with every output; arbitrary C pointers grant none of these facts.

The ordered-FMA header also accepts explicitly supplied upward-add,
downward-add and upward-multiply capabilities through `MERLIN_F64_OUTWARD_*`
macros. Each capability must enclose the exact real operation in the declared
direction. Source operations must still execute with admitted RNE and gradual
underflow, with nontrapping arithmetic and unobserved exception flags. The
default adjacency implementation is retained when capabilities are absent.
Provider ISA legality, numerical proof and cost are separate obligations.
Correctly directed rounding may give tighter bounds than adjacency arithmetic;
the resulting decisions require complete source/consumer qualification.

LLVM constrained floating-point rounding metadata describes an assumption about
the runtime environment; it does not request a different rounding instruction.
Using upward metadata while the actual source environment is RNE is invalid.
An explicit independent rounding implementation or correctly restored scope is
required before selecting such a capability. Tests use exact rational oracles
and check source rounding and sticky-flag restoration for a scoped provider.

## Summary contract

For consecutive reconstructed products, the caller provides:

- An outward interval containing their exact signed sum `T`.
- An upper bound `A` on the sum of their absolute values.
- An upper bound `R` on the sum of the absolute differences between original
  products and reconstructed products.
- Their actual consecutive source length, including a final short chunk.

These are mathematical summaries of the supplied operands. The helper checks
finite arithmetic and environment eligibility; it cannot establish how a caller
obtained its summaries. A rounded device result is insufficient without an
independent enclosure. In particular, binary64 recombination must be proven
exact or supplied as an outward interval. The operation must preserve source
order and zero initialization. Source partial reductions must be certified
separately and combined in the original source order.

## Signed-prefix bound

Let a chunk's total positive and negative variations be `P` and `N`.
Then `T=P-N`, `A=P+N`, and any internal chunk prefix lies between `-N` and
`P`, hence between `(T-A)/2` and `(T+A)/2`. If the incoming reconstructed
prefix is `S`, the magnitude of every internal reconstructed prefix is at most

```
|S + T/2| + A/2.
```

The implementation encloses `S` and `T` with outward binary64 intervals and
adds the cumulative absolute representation-error bound. This gives `M`, an
upper bound on the magnitude of every exact *original* prefix in the chunk.
For a single zero-seeded chunk the reconstructed term reduces to
`(|T|+A)/2`. Cancellation therefore helps without any extra partial readback.

## Rounding recurrence and binade refinement

Write `e_j` for the accumulated source rounding error after step `j`, and
`u=2^-24`. The exact input to the next source FMA differs from the exact
original prefix by the previous rounding error. Consequently,

```
e_j <= (1+u)*e_(j-1) + u*M + 2^-150.
```

For length `l` with `l*u<1`, `gamma_l=l*u/(1-l*u)` bounds
`(1+u)^l-1`. An outward coarse radius is therefore

```
(1+gamma_l)*incoming_error + gamma_l*M + l*2^-150/(1-l*u).
```

This also encloses every intermediate rounding error, because each term is
nonnegative and the length bound increases monotonically. Add it to `M` to
bound the magnitude `B` of every exact source FMA input. The largest binary32
rounding error anywhere in that range is at most

```
max(2^-150, 2^(floor(log2(B))-24)).
```

A second safe radius is the incoming error plus `l` times this absolute
half-ulp bound. Taking the smaller of the coarse and refined radii is sound.
Every binary64 arithmetic result in the certificate is widened toward the
required direction. Final interval endpoints are converted outward to binary32.
Possible overflow, nonfinite summaries, unsupported format, wrong rounding
mode, or flushed subnormals refuse the certificate.

The interval certifies a real value, not the sign bit of exact zero. Callers
must replay an interval containing zero when their output bit gate distinguishes
signed zeros. Equality of both nonzero BF16 endpoint conversions certifies
that the source conversion produces the same BF16 bits. No reference output
participates in this decision.

## Metadata-only absolute product bounds

`fma_product_norms.h` computes outward row and column metadata. For original
operands `a,b` and reconstructed operands `ar,br`, representation error obeys

```
sum |a*b-ar*br| <= sum |a-ar| * max |b| + sum |ar| * max |b-br|.
```

Three independently valid Holder bounds on the reconstructed absolute product
sum are available: row-L1 times column-Linf, row-Linf times column-L1, and the
square root of the product of the squared L2 norms. Their minimum is valid.
The last bound additionally requires correctly rounded binary64 square root.
Metadata reductions cost `O(M*K+N*K)` and pair formation costs `O(M*N)`; the
implementation never hides a host matrix contraction inside the certificate.
The squared L2 sums are converted to outward row/column Euclidean norms once;
pair formation multiplies these bounds. Square roots are therefore `O(M+N)`,
not `O(M*N)`. An initially zero state also uses the single-chunk algebra
directly, avoiding unnecessary interval additions to zero.
`merlin_fma_zero_chunk_gamma` is an optional cheaper alternative for exactly
one initially zero chunk. It retains the signed-prefix magnitude bound and
the coarse gamma radius, omitting binade refinement. Callers must measure the
tradeoff between certificate work and additional exact source replay.
`merlin_fma_zero_gamma_prepare` computes length-dependent gamma and gradual
underflow constants once; `merlin_fma_zero_gamma_apply` reuses that unmodified
plan for each output. The summaries must have the exact prepared source length.
The caller preserves RNE/gradual-underflow eligibility and does not observe
floating exception/errno side effects of repeated constant evaluation. As with
the supplied exact-sum summaries, the plan's construction and immutability are
semantic obligations, not an opaque authentication mechanism. Independent
tests verify identical endpoint bits to the scalar gamma helper across all
adversarial source cases and reject source-length mismatches. This moves
binary64 divisions outside the output loop without changing the certificate.

`merlin_fma_zero_chunk_half_ulp` is another optional one-chunk certificate that
avoids gamma divisions entirely. For a safe exact-prefix magnitude bound `M`,
choose a half-ulp bound `h`, set `E=n*h`, and explicitly check that the half-ulp
at the outward bound `M+E` is no larger than `h`. By induction, the rounding
error before every source step is at most `(j-1)*h`, so every exact FMA input
is bounded by `M+E`, and its rounding error is at most `h`. The final error
therefore cannot exceed `E`. The implementation starts with the binade of `M`,
enlarges it using the provisional radius, then performs the required final
consistency check. It refuses rather than assuming the check always succeeds.
All original eligibility, summary, overflow and endpoint checks still apply.
Omitting the minimum with the coarse gamma radius can cause additional replay;
actual full cost, rather than mathematical tightness alone, determines selection.

## Optional BF16 signed-radix packing

`bf16_radix_pack.h` supplies a general row-scaled signed radix128 codec, with
caller strides for inputs, reconstructed values, elements and digit planes.
It derives the exponent from the finite row maximum and sets
`step=2^(row_exponent+1-7*digits)`. It supports one through three digits; these
coefficients have at most21bits and are exactly binary32 representable.
BF16 exponent/mantissa decoding forms each quotient by integer shifts,
explicit nearest-even right shifting and saturation to `2^(7*digits)-1`.
Normal BF16 values are an8bit significand times a power of two; subnormals are
their7bit fraction times `2^-133`. Thus the exact quotient has at most8bits of
significance. If the corresponding floating divider quotient is normal, it is
exactly binary32 representable. If quotient underflow occurs, both that result
and the exact quotient are too small to round to a nonzero integer. This
preserves the original divider/lrintf nearest-even coefficient, including ties
and the one-digit saturation edge. Reconstructed integer-times-step values
remain exactly binary32 representable under the checked scale/coefficient range.

The codec checks finite BF16 encodings, supported format/RNE eligibility,
digit count, representable nonzero step and stride arithmetic. Unsupported
numeric contracts select the source path. Valid storage and nonoverlap remain
caller obligations; partial outputs after refusal must be discarded. No
target layout or instruction is embedded. Eight independent test cases compare
all65,280 finite BF16 patterns for each supported digit count against both the
original floating encoder and exact rational nearest-even rounding, plus
dynamic rows, signs, ties, tails, strided layouts, subnormals, saturation and
invalid numeric/compile contracts. This optional helper does not enable a
production optimization by itself.

## Qualification and use

Optional `MERLIN_F32_OUTWARD_FROM_F64_DOWN/UP` capabilities supply directed
binary64-to-binary32 bounds. The lower result is no greater than the exact input;
the upper result is no smaller. Defaults retain the RNE cast plus binary32
adjacency. The monotone source polynomial uses these bounds after its outward
source-q error arithmetic, retaining the original final source FMA. Directed
conversion can tighten the enclosure as well as reduce implementation work;
it does not alter source rounding or admit an approximation policy. Platform
instruction legality, stable source RNE, gradual underflow, nontrapping
arithmetic and unobserved exception flags remain explicit caller obligations.
Independent rational tests cover every binary32 exponent, boundary mantissas,
midpoints, signs, signed zeros, gradual underflow and overflow, including tight
directed bounds across all four native ambient rounding modes.

Thirty-eight native tests independently compare source FMA against an integer/rational
nearest-even oracle. They cover mixed-sign BF16 products, cancellation, bin
midpoints, gradual underflow, source tails, lossy reconstructed products,
outward Holder metadata, malformed summaries, overflow, and non-RNE refusal.

The runtime uses inline IEEE binary64 adjacency and exponent decoding to avoid
math-library calls for each bound operation. Their values match nextafter and
arithmetic half-ulp construction; floating exception and errno side effects
are deliberately absent. Adjacency tests audit edge cases and 2,000 independent
random patterns for each of binary32 and binary64; exponent tests cover every
binade through the source range.

Pinned attention diagnostics are retained under
`out/artifacts/probes/attention-prefix-guard-20261005`. Those observations cover
the original first vision attention operands across 12 heads, not the other
model layers or a complete SmolVLA execution. Device throughput and scalar
instruction counts cannot establish full-model cycles. Promotion requires
source-bound device execution, complete original accuracy gates, all executable
ELF instruction audits, and measured packing/certificate/readback/replay costs.
Expensive readback variants are retained as explicit tradeoffs, not selected
automatically. Production selection must depend on reduction semantics,
dimensions, layouts, numerical contracts, and hardware capabilities.
