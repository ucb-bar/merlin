---
title: "Optional separable ordered-FMA radius"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs:
  - src/merlin/llvmlower/source_attention_frontier.py
  - merlin/runtime/c/separable_fma_radius.h
  - merlin/runtime/c/prepared_fma_product_bounds.h
---

# Optional separable ordered-FMA radius

`separable_source_radius=True` requires the existing exact reconstructed producer-domain contract. It is default-off. Unknown representation error, uncertain source operands, failed source/overflow admission and unsupported environments retain the complete existing checked bound.

Eligibility is a proof from private immutable data: every admitted RHS error norm and the current admitted LHS error norm must be exactly zero; the uncertain-position count must be zero. The reconstructed center is therefore the exact real dot of the source operands, under the separately retained signed-radix integer/binary64 proof. An arbitrary public center is never admitted by this API.

For a zero-seeded sequence of `k` binary32 RNE FMAs with gradual underflow, the existing prepared gamma theorem gives

`abs(ordered_source_dot - exact_real_dot) <= gamma_k * sum(abs(a_i*b_i)) + eta_k`.

The source L1 bound and each source column maximum give `sum(abs(a_i*b_ij)) <= L1(a)*max(abs(b_j))`. Prepare `R = upward(gamma_k*L1(a))` once per row; each cell needs only `upward(upward(R*column_maximum)+eta_k)`. Upward products enclose the real nonnegative product despite reassociation. Outward addition/subtraction around the exact center and outward binary32 conversion finish the enclosure.

Admission also proves `upward(global_absolute_bound + global_radius) < FLT_MAX`; this covers every source prefix and every final enclosure, so overflow cannot invalidate the theorem. All metadata and centers stay disjoint, immutable and live during the synchronous row use. The existing stable RNE, nontrapping/unobserved exception-flags and mathematical capability contracts still apply.

This bound is usually wider than the existing center-sensitive Holder/Cauchy bound, so it can increase exact replay. It changes no source arithmetic, consumer observations, masks, source fallback, or accuracy threshold. Source-derived complete-cost measurements, including all extra replay, determine usefulness; no workload-specific or empirical threshold selector is introduced. Native tests cover source-ordered FMA containment, exact cancellation, underflow, signed zero, nonzero errors, uncertainty, overflow and unsupported-environment refusal. Whole-source gates remain mandatory before promotion.

The Python frontier emitter installs this alternative only after product-domain
preparation. `separable_fma_radius.h` owns exact-column admission, row radius
preparation and the checked outward finish; `prepared_fma_product_bounds.h`
supplies the admitted product metadata. Exact conversion hooks, if selected,
retain their separate provider contract.
