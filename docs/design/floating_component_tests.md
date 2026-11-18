---
title: Portable floating component tests and scheduling witnesses
kind: design
status: current
owner: compiler
last_verified: 2026-11-18
related: [static_scheduling, compiler_plane]
code_refs: [src/merlin/targetgen/component_program.py, packages/merlin-experiments/src/merlin_experiments/phase0/writer.py]
---

# Portable floating component tests and scheduling witnesses

## Scope

Extend the existing component program, capsule writer and explicit input palettes
to produce useful floating-point compiler tests. Keep integer programs and
historical reference engines unchanged. An explicitly selected floating schema
owns finite arithmetic, node rounding and conversion semantics. No architecture
name, geometry, timing constant or instruction encoding belongs in this machinery.

Source generation is separate from target admission. A generated program and
independent golden establish a test's semantics, not that a target implements it.
Selected hardware/software facts, independent runtime controls and frozen corpus
admission remain prerequisites for a compiler-authoring campaign.

## Source and numerical contract

The existing rank-two typed DAG supports matmul, add, copy, transpose, alias and
update. An opt-in floating contract adds ReLU and explicit conversion. Matmul
operands must match the selected operand type; its output is the selected
accumulator type. A dependent contraction must explicitly convert its input.
Arithmetic nodes declare their computation type and rounding; no target-wide
default silently chooses a vector operation's semantics.

The first floating version supports finite FP8 E4M3/E5M2, BF16, FP16 and FP32.
It preserves subnormals and rejects nonfinite inputs and overflowing intermediates.
It does not claim saturating conversion, flush-to-zero, approximate transcendental
functions or arbitrary hardware reduction trees. Those need separately selected
contracts. Ordered matmul rounds each product and each addition in the declared
accumulator format; reassociation and contraction into a different FMA are not
implicitly permitted. Zero-sign behavior is explicit in the reference.

Emit ordinary upstream tensor/Linalg/arithmetic IR. RNE uses floating operations
and explicit extensions/truncations. BF16 round-toward-zero after FP32 computation
uses explicit bit truncation, so the source itself carries the arithmetic rather
than relying on an unconsumed policy attribute. Unknown rounding/type combinations
refuse. Preserve the finite ordered contract on the generated capsule and source.

The independent evaluator consumes the typed program and canonical inputs, not
the emitted IR or candidate helpers. Exact rational scalar arithmetic followed by
explicit format rounding avoids host BLAS reduction order and model dependencies.
Format layout comes from the existing format registry. Every published output is
evaluated, including intermediates with further consumers. Existing integer
modular arithmetic and its bounds remain on their original path.

## Public test families

Generate eight public cases at one explicitly selected logical shape set. Shapes
are source choices, not inferred hardware tile boundaries. Existing
fact-derived tiling/capacity suites retain ownership of those boundaries.

1. Ordered contraction.
2. Contraction followed by ReLU.
3. Residual fan-out with a published intermediate and both consumers visible.
4. Independent contractions with both outputs visible.
5. Two contractions sharing a weight.
6. Dependent contractions with explicit narrowing.
7. Logical alias and update with a preserved pre-update copy.
8. Pointwise rounding and narrow/widen boundaries, with a transpose round trip.

All inputs come from existing explicit palettes, with signs, distinct operands
and boundary values where the family requires them. Counts describe source cases,
not pytest functions or claims of physical scheduling coverage. Private transfer
members must be selected separately by the existing cohort owner; their contents
are not copied into a public configuration.

Cancellation witnesses derive their separation from accumulator precision and
scale both factors to fit the operand exponent range. They must remain sensitive
to reassociation after that scaling. Conversion midpoints are used only when the
source format can represent them; matching source/destination precision exercises
exact conversion instead. All palettes must be representable in their actual
input storage before reference evaluation.

## Scheduling evidence

These source programs expose dependencies, independent work, reuse, mutation and
publication. They cannot force a particular register allocation or prove that a
DMA channel was reused. Physical scheduling obligations must be observed on the
actual emitted program by the independently selected runtime/provider.

Reuse existing timed/completion verifiers and positive/negative qualification
controls. Check shortened/removed delay, missing completion, resource contention,
premature reuse and final drain where applicable. Functional tests accept legal
serialization. An optimized result must match the same independent golden; only
a separately qualified timer can establish a performance improvement. Dynamic
schedulers do not inherit an explicit-delay requirement.

## Integration and refusal boundaries

Use the normal capsule builder and writer for floating programs and bind reference
implementation identity and canonical input provenance to the golden. Bound work
before materializing tensors. The optional diagnostic suite writer delegates to
those owners and produces a source-only manifest; it cannot manufacture reviewed
hardware facts, oracle tiers, hidden coverage or a Phase 1 release.

Keep old schema meanings. Existing integer and non-component float tests must
continue to pass. Exercise the same new families with contrasting operand and
accumulator formats. Verify generated IR with an upstream parser and, for a small
representative set, execute upstream lowering independently of the evaluator.

Additional format/shape variants are repository regression tests, not additional
cases charged to the OOT authoring agent.

Target-specific campaign work still includes reviewed operation placement,
compatible source/model pins, actual evaluator selection, structured ISA domains,
DMA/CFG integration, and whole-model host/device receipts. Test generation must
retain those gaps instead of declaring a campaign ready because eight files exist.
