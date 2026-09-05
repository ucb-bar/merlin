---
title: "Design: which phase a capsule can serve, derived for every target"
kind: design
status: current
owner: core
last_verified: 2026-09-05
related: [perf_corpus_scope_gap, derived_capsule_axes, perf_phase2_wiring, dialect_test_bar]
code_refs:
  - merlin/python/merlin/targetgen/phase_policy.py
  - merlin/python/merlin/targetgen/cert_cost.py
  - merlin/contract/capsules/profiles/_perf.yaml
  - merlin/python/merlin/targetgen/corpus_synth.py
---

# Which phase a capsule can serve

The corpus grades two different things. Phase 1 asks *is this compiler correct*, which needs an
independent golden and a cycle-accurate oracle, and that bounds how **big** a member may be. Phase 2
asks *is this compiler fast*, which needs the member to carry work worth optimising and a lever that
reaches it, and that bounds how **small** a member may usefully be. The bounds point in opposite
directions, which is why one corpus cannot serve both by accident.

Until now the split was not derived anywhere. The functional corpus is synthesized from a conformance
requirement (`admitted ∩ observed`); the performance corpus is a hand-authored sweep template; and
nothing relates them. `merlin.targetgen.phase_policy` makes a capsule's phase a computed property of
the target and the capsule, the same way its tier already is.

## The two predicates

**`certifiable`** — can this member's answer be checked at full fidelity? It declares a cycle-accurate
tier; its largest operand is inside the measured range (`cert_cost.MEASURED_MAX_OPERAND_ELEMENTS`,
65,536); and its size is inside what a budget affords on **this target's own** certified runs.

**`priceable`** — can a performance claim about it be falsified? Its declared work must be countable
and non-zero. An unpriced member costs more than itself: a `None` price nulls every derived rate and
**disables the corpus-wide attainment stop condition for every other member**.

Work is derived from what the capsule declares, never from an operation-name allowlist — an allowlist
must be edited each time a family becomes priceable, and the edit is the thing that gets forgotten.
The derivation reads, in order: the `lhs`/`weight` attribute pair; a positional `arg_order`; a
convolution's declared window and `padding`/`stride`/`dilation` geometry; operand **roles** (a
weight-stationary member names one weight and several activations, and the reuse is the point, so its
work is the sum over the activations sharing that weight); two activations sharing a reduction axis (a
scores block contracts Q against K and neither is a parameter); and finally the semantic family, which
settles that a non-contracting family's multiply-accumulate work is **zero** — a true quantity, not a
missing price.

Hand-checked against the corpus: a 10×10 input with a 3×3 window, unit stride and no padding gives
8×8 output positions against a 36-tap 16-output weight, so 36,864 MACs; `attention_qk` at 16×32 by
16×32 gives 8,192. Both match the arithmetic done by hand.

## `UNKNOWN` is not `NO`, and the distinction is the point

`phase_of` has **five** outcomes, not four. `neither` says a capsule serves no phase, which is a
finding about the *corpus*. `undetermined` says a predicate could not be answered, which is a finding
about the *evidence*. The first is fixed by rewriting a capsule; the second by certifying something.

This was a real defect in the first version of this module, caught by running it: with no measured
certification history every target reported `both = 0`, which read as "no capsule serves both phases"
when it meant "we cannot tell". That is the failure this repo keeps re-encountering — a check that
could not run reporting a result — and it appeared here within an hour of writing a docstring warning
against it.

## Measured, all six targets

Budget 300 s, corpus as of the arm4 integration. `cert-fit` is the target's own measured
certification history, which is what makes the size predicate answerable at all.

| target | capsules | both | phase-1 only | phase-2 only | neither | undetermined | cert-fit |
|---|---|---|---|---|---|---|---|
| gemmini | 133 | **68** | 31 | 31 | 3 | 0 | n=24 |
| radiance | 144 | 0 | 0 | 86 | 58 | 0 | n=32 |
| atlas | 90 | 0 | 0 | 4 | 0 | **86** | none |
| mx_gemmini | 49 | 0 | 0 | 4 | 0 | **45** | none |
| saturn_opu | 47 | 0 | 0 | 9 | 0 | **38** | none |
| saturn_opu_rvv | 46 | 0 | 0 | 4 | 0 | **42** | none |
| **all** | **509** | 68 | 31 | 138 | 61 | **211** | |

Three findings, in the order they matter.

**1. 211 of 509 capsules — 41% — cannot have their phase decided at all**, because four of the six
targets have never certified anything. This is not a corpus defect and must not be reported as one:
sizing a capsule against a budget requires a measured history, and the honest answer without one is
that no size can be shown affordable. The remedy is to certify each target's existing corpus once,
after which its split becomes computable. Until then, any statement about those targets' phase
membership is an assertion.

**2. Radiance can certify nothing.** All 144 of its capsules declare `required_oracle_tiers`
topping out at `L2`, so the cycle-accurate predicate fails for every one — 86 are priceable and
therefore phase-2-only, 58 are neither. Radiance has the *larger* measured certification history of
the two targets that have one (n=32), so this is not missing evidence; it is a corpus that never asks
for the tier its evidence would support.

**3. Where the split IS decided, `both` is the majority.** On gemmini 68 of 133 capsules serve both
phases. That is the healthy state and it is what the cost arithmetic wants: certification cost is
dominated by a per-member floor rather than by member size, so a member serving both phases costs one
floor and yields two verdicts, while two disjoint corpora pay two floors for verdicts that never meet.

### Why the single-phase members are single-phase

Gemmini is the only target whose reasons are all decidable, and they fall into two clean groups.

**Phase-1 only is dominated by families that do not contract** — movement 9, elementwise_map 6,
reduction 5, softmax 3, normalization 2: 25 of 31. These carry zero multiply-accumulate work, so they
have no utilization to improve and cannot move a MAC-denominated objective. That is structural, not a
defect: those families belong to phase 1 by their nature, and admitting them to a performance corpus
would add members that cannot move the objective while still paying a full certification floor.

**Phase-2 only is entirely size** — either the member exceeds what the budget affords (852 elements
at 300 s on this target's fit) or its largest operand leaves the measured range (65,536). These are
exactly the large, representative, perf-facing members that must rest on a certified sibling through
`extends` rather than replace one. Nothing else makes a member phase-2 only, which is the result the
design predicted.

Four contractions remain unpriceable and are recorded rather than rounded: whole-model capsules, whose
declared input is the network's entry tensor rather than a contraction's operands, and a depthwise
convolution whose weight rank the geometry rule does not cover.

## What this does not yet do

The predicates cover certifiability and priceability. Two further phase-2 admission conditions from
the design are **not** implemented here and must not be read as passing: that a member's headroom
exceeds the cost model's band, and that a declared lever actually reaches it. The second is the one
that matters most — the measured case is a residency family owning 92.4% of a corpus's cycles whose
lever no available action touches — and it needs `perf.claim_reach` wired in. Until then a `priceable`
verdict says the member can be priced, not that optimising it is reachable.
