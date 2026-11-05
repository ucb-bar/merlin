---
title: Select prospective public source roles before Phase 2 evaluation
kind: guide
status: current
owner: experiments
last_verified: 2026-10-19
related: [phase2_test_justification, phase2_corpus_search, phase0_specification]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/public_source_roles.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/requirements.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/public_programs.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/writer.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/corpus.py
---

# Select prospective public source roles before Phase 2 evaluation

An unfrozen planning inventory does not select scored objectives. The original
source request, complete declaration and selected operator setup profile own the
prospective role. Keep historical inventory bytes, counts, failed generations,
source refusals and publication evidence unchanged.

`public_source_roles.select(path, sha256, operator_exclusion=selection)` reads an
exact mode0600 operator profile outside publication roots, reopens every original
source inventory, and retains its complete member set. The profile schema is
`merlin.phase0.prospective_public_source_roles.v1`; its scope is
`prospective_public_development_before_performance_evaluation`. Inventories name
original manifest paths and SHA256s. Each member records its inventory-qualified
identity, exact request and declaration digests, role, origin and reason.

An explicit `functional_guard`, `development_control` or `upper_geometry_control`
request remains a functional obligation. An explicit `development_performance`
request can be a prospective objective. Undeclared source intent requires a new
`prospective_operator_profile` decision and a nonempty recorded reason. It never
defaults to OBJECTIVE. Keep expensive, unsupported and failed members in the
selected roster; runtime results cannot select their roles or remove obligations.

Pass the live owner and the complete current label-to-member binding through
`requirements._materialized_iteration_capsules(source_role_selection=owner,
source_role_members=...)`. The ordinary source/capture/profile identity checks
precede role binding. Objectives keep `_perf` and the existing complete-call
OBJECTIVE metadata. Functional guards take the `model` route without a performance
block. Both retain the original signed-source attestation, complete references,
per-output numerical policy, actual required accelerator work, arithmetic and
live realization owner. Keeping the existing complete capture transport does not
turn a guard into a performance objective.

Use `public_source_roles.scoped_selection(owner)` around the original writer and
new formal corpus discovery/freeze. A role-bearing writer entry without the live
owner refuses. Publication carries `source-role-profile.json` and exact role,
request, loader, capture, receipt and capture-profile bindings. Durable verification
checks that custody; serialized profile records provide no live operator authority.
New formal freezes rejoin the live selection and require every selected objective,
including currently missing or refused ones. Functional obligations remain a
separate complete roster. Missing members refuse freezing instead of improving
the score through omission.

These checks do not establish compiler correctness, hardware support, numerical
equivalence, execution, qualified timing, objective weights or readiness. The
ordinary admission gates remain mandatory. Reusing an immutable historical capture
for a new publication does not rewrite its capture date or original role. Record
that chronology and perform all newly required checks before evaluation. Legacy
public-program selections and publications retain their original interpretation.
