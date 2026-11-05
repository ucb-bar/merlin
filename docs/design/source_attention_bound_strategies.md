---
title: "Source-attention bound strategies that were tried and not kept"
kind: design
status: current
owner: core
last_verified: 2026-10-08
related: [ordered_fma_certificates, exact-integer-radix-reconstruction, prepared_operand_owners, verified_compiler_roadmap]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - src/merlin/llvmlower/radix_integer_reconstruct.py
  - src/merlin/llvmlower/radius_stage_schedule.py
  - merlin/runtime/c/ordered_fma_bounds.h
  - merlin/runtime/c/fma_product_norms.h
  - merlin/runtime/c/separable_fma_radius.h
---

# Source-attention bound strategies that were tried and not kept

The source-attention frontier (`source_attention_frontier.py`) certifies a device result
against the original source arithmetic: an exact or bounded center, an outward radius for the
source's increasing-K binary32 FMA order, and replay of every output the bound cannot settle.
Every option below was an explicit, default-off variant of one of those three pieces. Each was
built and unit-tested between 2026-10-06 and 2026-10-07, and none is in the tree. This page
records what each one did, what was measured, and why it was dropped, so the same idea is not
rebuilt without the evidence. Source commits are named so the code can be recovered.

What decides whether a bound strategy is kept is the **complete cost**: the extra products,
plane conversion, readback, allocation and any change in replay, measured as one group against
the exact control. A tighter bound that costs more than the replay it saves is a loss. Several
entries below never reached that measurement.

## Exact absolute products (dropped: measured slower)

Source: `2f45db902`, `exact_absolute_products.py` and `exact_absolute_dot_bounds.h`.

The canonical encoder writes three signed seven-bit magnitude digits per coefficient, all with
the coefficient's sign. Taking each digit's magnitude therefore encodes the exact absolute
value, with no carry recoding. After the last signed product, the option converted the private
planes in place, ran the five product groups again, and reconstructed
`T = sum(|a[i] * b[i]|)` exactly in binary64. The radius became `gamma_K * T + eta_K`, which
encloses the source FMA reduction. Rows whose encoded value did not match the source kept the
checked norm path. The identity does not hold for arbitrary balanced-digit encodings.

Measured on the first complete-group screen, consumer outputs were preserved and replay fell,
but **retired instructions rose by 7.79%**. Hardware cycles were not measured. The option also
refused to compose with the softmax producer spans, whose successful-writer coverage did not
include the new producer. The metadata-only Holder bounds in `fma_product_norms.h` (row-L1 times
column-Linf, its transpose, and the L2 product, at `O(MK+NK+MN)`) bound the same quantity
without a second contraction.

## Coarse absolute upper bounds (dropped with its parent)

Source: `8fed8bd44` and `d80c90082`, stacked on the exact absolute products.

This replaced the five extra product groups with one. For an explicit precision of one to six
bits, each canonical coefficient `N` became `U = ceil(|N| / 2^s)` with `s = 21 - b`. One
degree-zero integer product gave `sum U_a U_b`, admitted statically when
`K * 2^(2b) <= INT32_MAX`. Scaling by `2^(2s)` and the row and column steps gave an upper bound
on `T`, not its exact value. The precision was a caller choice, never derived from model
identity or accuracy results.

It was never measured. It depends on the exact-absolute-products producer seam, which was
dropped, so it was dropped too.

## One-endpoint word polynomial enclosure (dropped: negative complete cost)

Source: `0da3f1e05`, `one_endpoint_word_polynomial.h`. Its own commit archives it as disabled
after a negative complete-cost experiment.

Inside one floor segment of the source exp polynomial, a prepared power of two bounds the
encoded derivative. The ordered binary32 word distance and the maximum ULP then bound the scaled
input distance, including subnormals and signed zero. One source evaluation, that distance and
the two-error rounding budget enclose every interior source word. A separately proved global
upper word caps the result. Floor crossings kept the existing two-endpoint enclosure.

It needs one source evaluation per interval instead of one per endpoint, but the enclosure is
wider, and a wider interval sends more outputs to source replay. The complete-group experiment
came out negative, which section timing alone would not have shown. The commit did not retain
the measured numbers.

## Binary32 separable source radius (abandoned prototype)

Source: `d65f4bf19`, `binary32_separable_radius.h`.

The separable radius (`separable_fma_radius.h`) shares the `gamma * L1` factor across columns
and evaluates each column's radius in binary64. This prototype evaluated it in binary32 with
explicit directed operations (FMA rounded up, adds rounded up and down) and exact
floor and ceiling casts of the binary64 inputs. A binary32 overflow fell back to the binary64
path rather than publishing an infinity.

It selected no policy and was never wired into the emitter or measured. The emitter has since
batched the binary64 radius over eight independent lanes (`radius_stage_schedule.py`), and the
prototype's text match targets the unbatched call that batching replaces.

## Source-word inverse cells (abandoned prototype)

Source: `d4ce91c35`, `source_word_cells.py`.

A bounded exhaustive builder for an immutable, monotone, finite source map. It enumerated the
complete key domain, stored the full positive binary32 result word (never a BF16 observation) at
each change point, and answered `enclose(low, high)` with the exact extremal words. Incomplete,
non-monotone, nonfinite and over-budget enumerations were refused, and the resource limits were
explicit caller policy. Binding the enumeration to the actual source implementation was left to
the caller.

No consumer was written, so nothing called it and nothing measured it.

## Scaled radix finish (not wired)

Source: `6ef5510e2`, `radix_finish_schedule.py`.

With integer reconstruction, the emitter converts the exact int64 center to binary64 and then
applies the row and column scales in a second loop. This stage matched that exact emitted
sequence and replaced it with one helper. The helper handles eight independent coordinates at a
time. Each keeps its own cast and both multiplications, with no reassociation or contraction.
It is exact by construction.

It was never given an emitter option, so the frontier could not select it, and it was never
measured. The tree has since added fused integer reconstruction (`c_fused_header` in
`radix_integer_reconstruct.py`), which removes the int64 buffer and emits a different producer.
The stage only recognizes the unfused one. A scaled finish now belongs on whichever
reconstruction form survives measurement.
