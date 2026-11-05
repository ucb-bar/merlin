---
title: Exact bounded host rounding
kind: reference
status: current
owner: core
last_verified: 2026-10-20
related: [architecture, lowering_pipeline]
code_refs: [src/merlin/llvmlower/late_quant_rne.py, src/merlin/compile/baremetal_model.py, src/merlin/llvmlower/bounded_rne_maps.py, src/merlin/llvmlower/bounded_rne_packet_llvm.py, src/merlin/llvmlower/bounded_rne_lanes.py, merlin/tests/ir/test_late_quant_rne.py, merlin/tests/ir/test_late_quant_rne_intrinsic.py, merlin/tests/runtime/test_bounded_roundeven_spike.py]
---

# Exact bounded host rounding

`merlin.llvmlower.late_quant_rne` recognizes a complete binary32 clamp,
truncate, fractional comparison, parity and signed-bump chain after ordinary
upstream lowering. It preserves tensor fusion and all other SSA uses. Recognition
uses explicit lexical tokens, typed instruction productions and function-local
SSA connections. Comments, quoted identifiers and split operand lines are parsed
structurally; unknown forms remain unchanged.

It also accepts the equivalent `fptosi(llvm.roundeven.f32(clamped_value))`
form emitted by automatic host rounding. Both literal clamps must still equal
the actual signed integer type's endpoints. Live uses of the rounding result
retain their original instructions. The report identifies this source form as
`bounded_roundeven_conversion`. Either clamp order and either literal-operand
position are accepted; defined finite integer results are identical.

The integer type supplies its signed saturation bounds. Supported widths 1–24
have exactly representable binary32 bounds and fit the emitted signed 32-bit
conversion. Fast flags, changed endpoints/predicates/parity, attached unknown
metadata, strict FP and constrained operations refuse. The build callback asks
LLVM's assembler to verify the complete original, selected and portable modules.

## Explicit host codegen policy

`rewrite(text)` leaves bytes unchanged. `host_isa="portable"` selects exact
`llvm.roundeven.f32` plus integer conversion for native verification.
`host_isa="rv64gc"` selects `fcvt.w.s ..., rne`; optional `combine_clamp=True`
emits the proved min/max and conversion together with an explicit temporary
register clobber. Selection depends on the declared CPU ISA, independently of
the accelerator. Unsupported policies fail explicitly. No accelerator naming,
encoding, model name or capture ID enters the shared implementation.

`merlin_host_llvm_transform(llvm_bin, host_isa=..., ...)` supplies the existing
normal builder's pre-object callback. It records the route, tools and selected
content hashes. Normal compiler flags, object generation, supplemental device
objects, linking and final build identity remain owned by the builder. The
optional `temporary_prefix` changes printer identity only and lets a legacy
provider preserve previously qualified LLVM bytes. OOT adapters delegate this
mechanism; legacy target relink/audit tools remain OOT.

`compile_saved_model(..., host_bounded_rne=True)` selects this existing callback
with combined clamps through the normal saved-model builder. The option defaults
off, requires a scalar package supporting RV64GC, and leaves captured numerical
scope and accelerator routing intact. The input receipt records the choice and
the output receipt records the actual source/selected LLVM identities. An
explicit selection without a recognized bounded chain refuses.

The default-off optional pass `host-bounded-rne` selects that same existing
saved-model option through `MERLIN_PASSES`. An explicit `-host-bounded-rne`
disables it, including when the direct option or its environment fallback is on.
The normal compilation recipe binds the original legalizer report to its actual
source and selected LLVM. Native adoption and every input-specific relink reopen
that strict recipe and require the current choice to match the original marker.
Keep the selected context inside the configured compiler environment for static
builds, revalidation and the entire numerical call. Cached ordinary builds also
bind the existing `MERLIN_*` environment identity, so default, off and on choices
cannot reuse a stale opposite selection.

## Source-defined numeric contract

The bounded truncation and integer-to-float conversion are exact. The fractional
subtraction is exact by Sterbenz's bound or subtraction from zero. Parity and
half comparisons therefore implement nearest, ties-even rounding independently
of the dynamic rounding mode. Explicit RNE does not modify `frm`.

Infinities clamp to the finite endpoints. NaN propagates through the original
LLVM min/max into poison from `fptosi`; no finite-input assumption or NaN output
promise is introduced. CPU numeric min/max may refine that previously undefined
integer result. The unconstrained source has no exception-flag contract; strict
or constrained modules remain unchanged.

Independent portable native tests compare source and selected code at every
i8/i16 half-step and both adjacent binary32 neighbors, random finite bit patterns,
signed zero, subnormals and infinities. Structural/refusal tests cover other uses,
multiple functions, alternative literal spellings and temporary-name collisions.
Physical hardware timing remains a separate qualification.

The standalone Spike regression now executes the actual RV64GC control,
conversion-only and combined-clamp loops under all five rounding modes, twice
per mode, with every output and both guards checked against independent NumPy
rounding. It covers every i8/i16 half-step and its adjacent binary32 values,
random non-NaN words, infinities, zeros and subnormals. Enable it with explicit
`MERLIN_TEST_GCC` and `MERLIN_TEST_SPIKE`. Its counters are local simulator
instruction proxies; physical cache and pipeline costs require separate timing.

### Why the integration gap escaped earlier checks

The earlier late matcher recognized the residual arithmetic chain. Automatic
host rounding replaced that chain with the equivalent standard intrinsic, so
the late matcher selected no routes. Each pass's correctness checks could pass
while their composition missed the optimization. The new regression runs
ordinary lowering with automatic rounding before checking late recognition.
Independent CPU execution then tests the actual emitted instruction sequence.
This is a host compiler composition gap; it requires no Phase 0 corpus change
and no Phase 1 or Phase 2 accelerator implementation.

## Explicit lane scheduling

`schedule_bounded_rne_maps(module, lanes=...)` keeps the original scalar graph
inside each lane. Static source index maps establish independent immutable
inputs; the carried tensor result and upstream bufferization retain alias and
lifetime authority. The default uses four lanes. Explicit widths through eight
have separate full packets and exact static tails, including a tail-only map.

`rewrite_packet_helpers(text, host_isa="rv64gc", max_lanes=8)` explicitly permits
pure return aggregates with five through eight independently proved results.
Its default lane budget remains four, preserving the prior refusal domain and
emitted bytes. The separate CPU helper emitter also supports explicit widths
through eight. Floating temporaries and clobbers cover every selected lane;
there is no automatic width selection or profitability claim.

The existing exception-flags-unobserved source policy and strict/constrained
refusals remain unchanged. Comparing final runtime rounding modes and flags
between schedules supplies additional operational evidence, without extending
the declared source policy. Compiled register pressure, spills, code footprint,
conversion/clamp latency and memory behavior require measured costs.
