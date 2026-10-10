---
title: Conditional source value cones
kind: design
status: current
owner: merlin-core
last_verified: 2026-10-10
related:
  - docs/design/independent_rtl_intake.md
code_refs:
  - src/merlin/targetgen/rtl/hw_conditional_value_cones.py
  - src/merlin/targetgen/rtl/hw_value_bindings.py
  - src/merlin/targetgen/rtl/hw_combinational.py
  - merlin/tests/targetgen/test_hw_conditional_value_cones.py
---

# Conditional source value cones

`prepare_conditional_value_cones` reopens supplied source bytes through the
existing structural HW reader, follows exact named typed instance bindings,
and composes selected scalar sinks using the existing primitive reader and
known-bit evaluator. It accepts explicit `OriginalValueSelection` sinks,
`ConditionalValueCut` declarations, `ValueBindingLimits` and `EvaluationLimits`.
It supplies no target roles or default state values.

## Exact boundaries

A cut names a canonical original operation result by source SHA, occurrence,
module, operation ordinal, result slot and type. Its declared boundary must
match a genuine state result, memory-read result or opaque instance result
discovered in the selected cone. Every reachable boundary requires one exact
declaration. Missing, stale, duplicate and unused cut declarations refuse.
Unsupported logic cannot be hidden by declaring a cut. Noninteger and clock
values refuse, including clock operands selected directly.

Each original root input and declared cut gets a generated scalar input name
and an exact source identity in `inputs`. A case must name every such input
with an unsigned known-bit value of the original width. Each result is named
`value_N` in the caller's original sink order. Repeated callees retain separate
occurrence and state identities; equal widths never alias them.

## Bounded composition

The complete source lexical-width budget belongs to `ValueBindingLimits`.
`EvaluationLimits.scalar_bits` separately bounds every selected cone value.
Source bytes and syntax are bounded before parsing, and complete module,
operation, occurrence and port-binding bounds precede hierarchy expansion.
Node and bit-work budgets bound preparation. Whole case membership, input
values, output membership and aggregate case work are checked before results
are allocated. Dependency nodes are composed in evaluation order; structural
discovery order is not assumed to be topological.

In this API, `ValueBindingLimits.metadata_bytes` also bounds the complete
canonical JSON returned by `record()`, including original source membership,
input/output/cut identities and wrapper metadata. A structural attribute count
alone cannot satisfy that bound. The existing structural reader's budget and
record meaning remain unchanged.

The returned record retains complete structural frames, operation membership
in visited definitions, nested effects and original unknowns. Unselected
expressions are not evaluated or marked supported. Each primitive in the
selected cone is rechecked against its live original operation, operand order,
type and parameters. The record returns a fresh metadata copy and makes no
claim that source bytes were reopened again when that copy was requested.

## Conditional scope

Supplying a register result observes its expression uses; it never initializes
or advances that register. A selected next or reset operand is its raw source
value, not a post-event state. A supplied memory-read result is independent
known bits, not a read execution or initialized storage. An opaque result is
not evidence about its implementation. Reset priority, reachability, history,
collision validity, event acceptance, completion and effects need their own
premises. Whole-domain, source/SDK, runtime, physical, resource and admission
claims remain unknown. Existing structural and closed-module readers retain
their original meaning and frozen records.
