---
title: Derive coverage axes from selected facts and source programs
kind: design
status: current
owner: core
last_verified: 2026-10-12
related: [generating_capsules, phase0_specification, capsule_phase_split, phase2_test_justification]
code_refs: [src/merlin/targetgen/conformance.py, src/merlin/targetgen/corpus_synth.py, packages/merlin-experiments/src/merlin/targetgen/store_probe.py]
---

# Derive coverage axes from selected facts and source programs

Phase 0 derives its requirement from selected hardware facts, reviewed software
semantics and independent public source programs. Semantic-family coverage alone
cannot establish shape, memory, composition or host/device coverage. Report each
axis and its witness separately.

## Shape and numerical constraints

Derive legal extents from the selected operand formats and hardware interfaces.
A block-scaled or packed representation can impose a granularity different from
the compute tile. Do not demand impossible tails or silently replace them with
convenient shapes.

Geometry distinguishes sub-tile work, aligned extents, valid partial tiles and
aspect ratios. Preserve broadcasting and contraction axes through views rather
than treating every dimension as occupied compute. Unsupported or conflicting
axis traces remain unresolved.

Rank and batch dimensions also belong to the requirement. A rank-two witness
cannot establish batched indexing, arbitrary strides or forwarding across graph
boundaries.

## Memory and composition

Use extracted storage, interface and access facts where available. Static tensor
bytes are logical demand; they do not establish physical capacity, simultaneous
residency, traffic or allocation lifetime. An independent capacity observation
must retain complete outputs, its selected source and the meaning of refusal.
A numerical failure is not a capacity boundary.

Connected source cuts preserve producer-consumer relationships, live outputs,
representations and required crossings. An isolated contraction cannot witness
fusion, transfer ownership or whole-model placement. Missing source semantics,
unknown placement or an unavailable interaction stays in the requirement's
uncovered denominator.

## Generation and evidence

Keep every generated, omitted, unsupported and failed member in the normal
manifest and provenance reports. Derive membership from the selected inputs;
counts and class labels are not functional or performance qualification.
Correctness and timing engines must resolve from the selected target policy,
and unknown observations cannot become successful empty coverage.

Use the [Phase 0 input guide](../guides/phase0_specification.md) to inspect the
source/fact closure and generated evidence. See [capsule generation](../guides/generating_capsules.md)
for the ordinary commands and [performance test justification](../guides/phase2_test_justification.md)
for comparators, falsifiers and representative selection.
