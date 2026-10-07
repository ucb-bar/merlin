---
title: Exact bounded host rounding
kind: reference
status: current
owner: core
last_verified: 2026-10-06
related: [architecture, lowering_pipeline]
code_refs: [src/merlin/llvmlower/late_quant_rne.py, merlin/tests/ir/test_late_quant_rne.py]
---

# Exact bounded host rounding

`merlin.llvmlower.late_quant_rne` recognizes a complete binary32 clamp,
truncate, fractional comparison, parity and signed-bump chain after ordinary
upstream lowering. It preserves tensor fusion and all other SSA uses. Recognition
uses explicit lexical tokens, typed instruction productions and function-local
SSA connections. Comments, quoted identifiers and split operand lines are parsed
structurally; unknown forms remain unchanged.

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
Hardware qualification and five-mode CPU execution remain pinned experimental
receipts, rather than claims inferred from these unit tests.
