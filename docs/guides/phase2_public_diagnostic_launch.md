---
title: Launch a public diagnostic retained queue
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-19
related: [phase2_component_search, spike_timing]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_diagnostic_launch.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_source_diagnostic.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_native_exploration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_fast_feedback.py
---

# Public diagnostic retained queue

`merlin-phase2-public-diagnostic` (or `python -m merlin_experiments.phase2.public_diagnostic_launch`)
executes one same-revision public source pair through the original retained queue.
This is an infrastructure diagnostic. It does not satisfy the formal Phase1 gate,
issue a performance cohort, accept an incumbent, or qualify timing.

The original operator supplies `--plan FILE --plan-sha256 SHA256 --output FRESH_DIRECTORY`.
The exact `merlin.public_diagnostic_launch.v1` plan selects public published source directories,
a real hash-only operator exclusion selection, a separate full planning inventory,
the immutable compiler inventory, an edit contract, ordinary contract source,
the target descriptor/source root/board selection, genuine compiler profile archive/workspace,
the exact selected shared library, and an original source-pinned admission factory.
Only executing the factory creates live diagnostic source owners; a saved successful
registry cannot issue them. The clock is an existing explicit infrastructure packet,
limited to 30 minutes and 600 CPU seconds with at least eight minutes for cleanup.
An original enforcing process supervisor must cover setup and the queue, using the
clock's CPU, RSS, file and aggregate storage limits; the plan itself is not hard enforcement.

The launcher copies the original compiler into separate baseline and candidate
packages, freezes original edit authority and a retained revision, and submits
`corpus-selected-compile-spike-feedback` outside its metadata CPU lease. Each arm
uses the original boxed compiler profile, complete source/native numerical checks
and a fresh original source/effect checker. Existing public declarations stay
immutable; the versioned host execution view requests L0–L3 while retaining L3 as
pending. Refusals and source-admission costs are retained in `status.json`.

The result is permanently `DIAGNOSTIC_ONLY`, with null measured cycles and UNKNOWN
timing. A one-member smoke roster does not replace the frozen workload inventory.
The Python `run_plan` API optionally exposes its actual live completed pool/job to
an original in-process consumer before cleanup; saved feedback does not recreate
that authority. An optional typed native exploration owner may be selected by an
original host when the corresponding policy API supports it; there is no JSON
native-qualification switch in this launcher.

For native predictions, the original host calls
`prepare_public_native_exploration(native_selection)` with the existing exact
`NativeArchiveScreenSelection`. Preparation reopens public archive eligibility,
native held gates, hardware correspondence and independently reviewed event
semantics. Pass the returned live owner as `native_exploration=` to
`run_plan` or `prepare_public_source_diagnostic`. Saved qualification records
cannot select it. The default remains the original diagnostic route.

This explicit selection uses diagnostic feedback version 2. After each original
complete L0/L1/L2/source observation, it runs the existing boxed plain and annotated
count observer on that same ELF. Its original numerical receipt is unchanged.
Unsupported source geometry, capacity, layouts or event activity remain UNKNOWN;
no extra process is started for a rejected source domain. The first version
admits only the existing checked direct signed integer contraction source domain.

Per-member residual intervals provide exploration priorities. Overlapping
intervals leave ordering UNKNOWN. Unselected obligations remain visible, with no
complete performance-objective claim. Predictions never populate measured cycles,
roofline attainment, a measured incumbent or plateau evidence. Existing formal
search and final qualification retain their original gates. Count observation is
additional evaluation work inside the same parent deadline and resource envelope.

To inspect setup separately from execution, run:

```sh
merlin-phase2-public-diagnostic --check-plan --plan PLAN.json --plan-sha256 SHA256
```

This reuses the same plan/schema, declared metadata leaf-pin and existing clock
validation, prints selected public paths/identities and remaining packet time,
and returns `NOT_EXECUTED` with no authority. It creates no output directory and
never reconstructs or executes an admission factory, verifies a profile, freezes
a compiler, captures a source, or submits a queue job. Public source/reference/
private-exclusion checks, compiler inventories, original factory source closure,
actual resource admission, correctness and timing remain explicitly deferred.
The private exclusion selection's path/hash are omitted from the report. An
expired packet refuses; setup never creates or resets a clock. Omit
`--check-plan` and supply `--output FRESH_DIRECTORY` for the existing execution
route under its original enforcing supervisor.
