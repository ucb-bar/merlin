---
title: Start capsule-qualified ordinary source development
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-20
related: [phase2_public_diagnostic_launch, phase2_component_search]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/capsule_ordinary_scope.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/authoring.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_policy.py
---

# Capsule-qualified ordinary source development

`authoring.run_stage` and `prepare_ordinary_search_policy` accept an explicit
`capsule_phase1_owner` and `required_cohort_declaration`. The default route still
requires the existing full formal qualification. The opt-in route requires the
exact original `capsule_iteration.FrozenQualifiedCapsulePhase1` object and reruns
its original verifier. A receipt dictionary, foreign callback, subclass, old
successful check, or renamed qualification scope cannot substitute for that owner.

This route is unscored compiler-source development. It keeps the complete original
functional capsule roster and separately pinned complete prospective cohort
visible. The declaration grants no qualification and does not replace the frozen
performance corpus, tiers, weights or objectives. Full-model validation, formal
grade and final-result qualification remain pending. The stage returns a
`capsule_development_checkpoint.json`, never a consumable performance candidate.

## Original launch API

Construct the owner inside the original sealed `frozen_python` / source-snapshot
lifetime required by `capsule_qualification_replay.verify`. Retain its exact
original qualification run, descriptor, compiler digest and qualification receipt.
The original V4 functional input snapshot, common-profile sources and certificate
metadata must independently reopen successfully. A new current diagnostic profile
cannot replace a historical profile in that proof.

```python
from pathlib import Path
from merlin_experiments.phase2 import authoring, capsule_iteration

phase1 = capsule_iteration.FrozenQualifiedCapsulePhase1(
    Path(original_run), Path(original_descriptor), original_compiler_sha256,
    Path(original_qualification), original_qualification_sha256,
)
# existing_stage_inputs retains all original source/resource/corpus/edit checks.
checkpoint = authoring.run_stage(
    **existing_stage_inputs,
    families="all", capsules="all", waive_functional_gate=(),
    search_configuration=original_search_configuration,
    capsule_phase1_owner=phase1,
    required_cohort_declaration={"path": str(original_public_cohort),
                                 "sha256": original_public_cohort_sha256},
    checked_source_diagnostic=original_live_checked_source_selection,
)
```

The owner run must equal the selected functional run and compiler; the descriptor
must have the selected exact digest. No new named waiver is introduced. The
required cohort declaration is a closed path/digest pin to the original public
planning data. Pending, refused and unselected members remain explicit obligations.
The normal original search configuration still binds the complete corpus,
original baseline, historical certificate and bounded resources. Construct it for
the new source epoch; do not reuse an expired launch clock.

For generated sources, pass the original live checked-source selector issued from
the same original compiler-pass selection. Its explicit native execution protocol
(e.g. the already implemented pointer-source protocol) passes unchanged through
`run_stage` → `ordinary_search_stage.prepare_round` →
`prepare_ordinary_search_policy`. No global program plan is fabricated. This does
not qualify a timing model or extend its domain. The existing optional archive
screen selection is separate and must pass its original gates. Joint-public native
exploration is not silently substituted for ordinary archive screening.

## Queue and finalization boundaries

Only revision, compile/Spike submission, polling, cancellation and existing search
cleanup/selection controls are exposed. RTL measurement, impact measurement and
calibration actions are absent. Pool admission and retained evaluation separately
reject RTL kinds; the original development evaluator also refuses RTL dispatch
before calling any execution service. Historical certificate validation runs
without a new simulator execution.

A selected compile/Spike job preserves the full original roster and records
unselected members as pending. The original checked-source callbacks, complete
numerical outputs and all current source/runtime/binding checks remain mandatory.
The existing coordinator's measured ranking still expects its original RTL kind,
which this scope cannot submit. Cheap observations therefore cannot become a
measured incumbent; ordinary provisional screening remains exploration only.

Each round needs an append-only broker receipt joined to an actual completed job
from the original live evaluation pool for the final source digest. Saved feedback
JSON alone cannot issue that receipt. Final source bytes, source-owner callbacks,
qualification evidence and cohort pin reopen on each scope validation. No automatic
RTL/full-model finalizer runs. The checkpoint declares `capsule_iteration_only`,
`final_results_qualified=false`, `external_full_models_complete=false` and no
promotion authority. Standard candidate validation rejects its distinct kind/schema.

Host tests establish these transport and authority boundaries. They do not establish
that a particular target program compiles, a timing domain is supported, or an
agentic target search has run. A fresh real launch still requires all original
owner, profile, corpus, host and resource preflights.
