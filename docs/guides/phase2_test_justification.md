---
title: "Justifying Phase 2 tests from Phase 0 evidence"
kind: guide
status: current
last_verified: 2026-10-05
owner: experiments
related: [phase0_specification, generating_capsules, perf_phase2_wiring]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/test_justification.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/corpus.py
  - experiments/templates/phase0/performance.yaml
---

# Justifying Phase 2 tests from Phase 0 evidence

Phase 0 generates two distinct cohorts. The public functional cohort tests Phase 1 compiler
behavior. The `_perf` development cohort asks Phase 2 performance questions. A Phase 1 pass is
not a speedup, and a generated `_perf` capsule is not a measurement. The phase selections in the
corpus manifest make this boundary explicit.

For current generated corpora, Phase 2 discovery derives `merlin.phase2.test_justification.v1`
over **every** generated performance member before applying a requested subset selection. Freeze
rechecks the original members and evidence bytes, saves the receipt with the selected capsule
snapshot, and replay verifies its hashes. An incomplete corpus without `MANIFEST.yaml` cannot be
admitted. The receipt is generated under the run's output directory; it is not an authored test
definition.

Each row answers a concrete question:

| Question | Receipt field | What establishes it |
| --- | --- | --- |
| Why this target? | `hardware_basis` | Required traits and execution capabilities from the frozen Phase 0 performance facts; an unsatisfied gate refuses a generated member. |
| Why this workload shape? | `workload_need` | The capsule's shape census stamp and the selected application inventory identity. `off_census` and `not_established` remain visible. |
| What could be false? | `hypothesis` | The declared observation and falsifier from the shared performance family. |
| What is compared? | `matched_comparator` | Declared equal-demand fields and, when present, the candidate group members. Matching remains `planned_unverified` until the analyzer checks emitted programs and results. |
| What detects a spurious effect? | `negative_control` | The family's declared control. Its presence is a plan; the control is not established by generation. |
| How is it measured? | `measurement`, `correctness`, `repeat_dispersion` | Instrument, the concrete timing and correctness engines with their tiers, replicate plan and dispersion rule. A missing policy stays missing rather than becoming zero noise. |
| Can this member run? | `support` | The software screen; unknown support stays unverified and an explicit refusal stays unsupported. |

The receipt binds the shared template digest, raw RTL digest, derived performance facts, operation
accounting, evidence manifest, provenance manifest and each capsule's byte tree. It refuses a
changed comparator, control or evidence byte between discovery and freeze. Its status is always
`diagnostic_unmeasured`: it does not certify simulator fidelity, functional correctness, a
negative control, statistical significance or a speedup. Those claims require completed,
identity-matched correctness and timing cells and the family's declared analyzer. In particular,
instruction counts from a functional simulator do not substitute for cycle measurements from the
selected timing engine.

The shared performance template names oracle *tiers*, not simulators: its evidence fields carry
placeholders such as `$target_oracle:L2` and `$target_oracle:L3`. Phase 0 resolves them per target
from the capability contract or the recipe's explicit oracle selection and freezes the concrete
engine names, with the placeholders they came from, into each generated member. The receipt's
`measurement.timing_simulator` and `correctness.simulator` therefore name that target's engines. A
placeholder that does not resolve to a concrete simulator is a generation error for that member
(recorded under `families_not_generated.errors`); it never falls back to whichever simulator is
installed. Its form-performance `PW` members derive form classes and source-convolution windows
from the declared iteration workloads, then plan paired candidate and target-support reference
measurements. A generated pair, even with a declared acceptance analyzer and replicate band, is
still an unmeasured hypothesis; separate measured evidence must establish both arms on identical
demand and the selected timing oracle.

Read `test_justification.json` next to a frozen `performance_corpus_manifest.json`. First inspect
`families_not_generated`, then each member's workload and support status. A missing family may be
inapplicable, unimplemented or failed; it is not evidence that the compiler handles that lever.
Families admitted only by the selected frozen requirement (a captured multi-region scope chain or
a derived model form class) are listed as `skipped_inapplicable` when that requirement selects
nothing, and as `blocked_unimplemented` when the requirement lacks the scope they need. Families
the template declares but this cohort cannot measure, such as claims decided from the decoded
instruction stream rather than from cycles, are also listed as `blocked_unimplemented`.
For a performance conclusion, inspect the later measured result and its analyzer verdict as a
separate artifact. Do not promote a generated test or a diagnostic receipt to a measured claim.
