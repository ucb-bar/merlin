---
title: Phase 2 setup and execution boundaries
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-19
related: [phase2_corpus_search, spike_timing, shared_pass_phase2, ../design/phase2_capacity_cohort]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_diagnostic_launch.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_policy.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/calibration_validation.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_capsule_feedback.py
  - src/merlin/transforms/pipeline.py
---

# Phase 2 setup and execution boundaries

Phase 2 learns compiler decisions from public development programs using an
explicitly qualified timing domain. Preparation, diagnostic execution and formal
optimization are distinct states. Passing one state's checks does not enter the
next state automatically.

## Prepare without changing Phase 1

Preserve the original compiler, profile, numerical policy, software contract,
hardware facts and qualification evidence. Copy the compiler into independent
Phase 2 baseline and candidate directories. Record both the original identity
and each comparison identity. A provisional Phase 1 package can exercise setup
and diagnostic tooling while its unfinished formal gates remain visible.

Select any additional reviewed compiler library explicitly for both arms. An
older compiler can use a new library without another Phase 1 authoring run;
the comparison needs fresh same-release functional checks. Historical profiles
retain their exact source membership and grants.

Phase 0 supplies source programs, references, numerical/readout boundaries and
coverage requirements. PyTorch capture is one source path. A shape declaration
or a generated reference is insufficient: the selected compiler must realize
accelerator work and independent execution must check complete outputs. Preserve
every refused member and its original role. Functional guards do not become
performance objectives merely because they execute successfully.

Build the public development spectrum from permitted source-derived operation
distributions, layouts and selected resource limits. Include independent axis
tails, skewed and short contractions, deep reductions, joint storage boundaries,
connected state and reuse controls, and supported attention arithmetic. Reduced
depth graphs check composition and parameter handling separately from individual
operators. Workloads inspected during development cannot later be held out.

## Qualify the timing domain before iteration

Reuse original public Phase 1 simulator records and compatible fresh Spike
observations, retaining executable, inputs, clock, hardware, memory model and
counter-window identities. Source variants and repeats stay in the same split.
RTL facts constrain resources; software and instruction semantics establish
dependencies, state, ownership and completion. Fitting prices cannot supply
missing dependencies or legal overlap.

Hold-out error, ordering, coverage and independent-group gates qualify only
their explicit domains. Capacity, attention or new instruction patterns require
compatible evidence before expansion. Unsupported predictions remain `UNKNOWN`.
Functional instruction counters, predicted target cycles, measured target cycles
and simulator walltime remain separate quantities.

The intended optimization loop uses qualified Spike predictions without a new
RTL-simulator run per candidate. Missing domain evidence is a setup dependency,
not permission to extrapolate. Full roofline attainment additionally requires
physical traffic, usable ports, feeding/packing costs and legal overlap; partial
bounds can guide diagnostics but cannot establish convergence.

## Probe the queue, then launch optimization

The public diagnostic launcher exercises the original retained-revision queue
with copied compiler arms, complete numerical checks and explicitly selected
native feedback. Its permanent diagnostic scope grants no incumbent promotion.
An unchanged control checks transport; a distinct candidate must change actual
emitted work before it demonstrates compiler iteration.

Formal search additionally freezes admitted objectives, comparison release,
edit authority, timing domain, resource limits and stopping policy. The author
selects permitted pass order, shape/resource heuristics and schedule alternatives,
and may change or introduce transformations within that authority. Checked
pipeline prerequisites and analysis invalidation constrain legal ordering.
Shared mechanisms are reusable authored algorithms parameterized by computation,
ISA effects and resources; availability does not qualify a target adapter.

Use the shared CPU admission pool for compiler, simulator and helper children.
Increase concurrency only after an actual larger allocation is granted. The
coordinator must not hold a child CPU lease while awaiting workers. Retain each
candidate's hypothesis, emitted changes, complete correctness, scoped feedback,
decision, costs and clean shutdown. A fixed number of sweeps is an estimate;
observed breakpoints and frozen stopping rules determine how much search suffices.
