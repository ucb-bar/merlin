---
title: "Prepared probability bins"
kind: reference
status: current
owner: core
last_verified: 2026-10-06
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs: [src/merlin/llvmlower, merlin/runtime/c]
---

# Prepared probability bins

`emit_source_attention_frontier(..., prepare_probability_bins=True)` snapshots
both rounded BF16 endpoint values before the probability ambiguity test. Exact
source replay refreshes the snapshot before publication. The original interval,
source denominator, replay decision and output semantics remain unchanged.

For finite ordered endpoints whose rounded bits agree, the source binary64
midpoint, binary32 conversion and BF16 conversion must produce that same bin by
monotonicity. Different signed-zero bits, nonfinite or unordered endpoints use
the original midpoint expression. This requires stable source rounding and
nontrapping, unobserved exception flags; it is not an approximate policy.

The option defaults off. The snapshot is a private value, never a cached pointer
or a lifetime claim about mutable interval storage. A changed refinement grammar
is refused. Tests cover all BF16 anchors and nearby binary32 tie values, both
endpoint orders, nonfinite values, signed zeros and four native rounding modes,
plus complete independent source-group mask/shape/refusal tests.

Target execution and complete-group cost qualification are separate from this
portable proof. No hardware performance claim follows from fewer conversions.

## Private exact source-point spans

`emit_source_attention_frontier(..., prepare_probability_points=True)` can share
one private source array across the value, lower-bound and upper-bound roles of
the following product call. It requires probability-bin preparation, complete
softmax producer spans and encoded-row preparation. The option defaults off.

Every successful probability store must have equal finite BF16 endpoint bins,
refreshed after exact source replay. Ambiguous bins, distinct signed zeros and
nonfinite values refuse before publication. The original BF16 query copy also
supplies an exact source value. This proves the source interval is a point;
radix encoding can still have representation error, and the original ordered
source-FMA bounds and consumer acceptance checks remain required.

The private span binds an immutable source base, element count and fresh local
epoch through one synchronous product call. Complete producer/store/gather
structure, disjoint output and plane storage, and retained original fallback
are provider obligations. No witness survives workspace reuse. The transform
specializes Merlin's owned source template; it does not admit arbitrary C or
caller-supplied point flags. Changed producer or consumer structure refuses.

Selected calls omit duplicate lower/upper gathers and supply exact-source
interval bounds to the encoder and product-bound helper. Allocated workspace
capacity remains unchanged. Tests cover every BF16 anchor and binary32 tie
neighborhood in four native rounding modes, stale/mismatched epochs, changed
producer grammar, unsupported plans, and the complete source executor's dirty
workspace, masked/strided input and refusal behavior. Whole-model and target
performance qualification remain separate.
