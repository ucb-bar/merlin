---
title: Consolidating performance development for integration
kind: guide
status: current
owner: experiments
last_verified: 2026-10-21
related: [phase2_corpus_search, optional_passes, shared_compiler_passes, shared_phase1_scheduling, spike_timing, phase2_shape_spectrum]
code_refs:
  - src/merlin/common/artifacts.py
  - src/merlin/llvmlower/optional_passes.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_session.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_target.py
---

# Consolidating performance development for integration

Prepare a contribution branch in each editable repository from the selected
integration tip, using the workspace task manager for isolation. Merge useful
development history without changing the integration branch. Record starting
tips, source tips, merge bases and resulting merge commits. Audit helper
branches for patch-equivalent and superseded changes before cherry-picking.
Review both repositories together when the host/compiler boundary crosses
Merlin and an out-of-tree target repository.

## Preserve implementation and evidence separately

Merlin owns reusable host generation, packing, graph transforms, scheduling
algorithms and runtime infrastructure. The target repository owns dialects,
instruction encoding, layouts, resource facts and device implementation.
Preserve an immutable original compiler seed separately from candidate edits.
A historical compiler snapshot needs a byte inventory and original provenance;
it is neither a new default provider nor a compiler answer for fresh authoring.

Keep point-in-time results, integration briefs, generated receipts, logs and
machine-specific reproduction under the configured output root, created through
`merlin.common.artifacts.new_product`. Use a small index of receipt paths,
hashes, sizes, scopes and archive successors. Do not duplicate large traces,
model tensors, binaries or grading outputs into source control. A receipt
bundle must disclose whether it includes referenced objects transitively.

## Review an optimization claim

Check the actual source, host, pass selection, emitted IR/object/ELF and
execution identities. A requested option or a pass report alone does not prove
that work changed. Compare numerical outputs under unchanged gates, then
commands, requested bytes, useful/padded work and resolved estimated cycles.
Retain a no-op or regression as measured negative evidence.

Use the same host and timing scope for a device comparison, and the same
device objects for a host comparison when isolating that change. Do not
combine unrelated ratios. Keep individual kernels, captured graph cuts,
complete model chains, original qualification and physical validation as
distinct scopes. A partial model latency cannot establish a complete result.

Functional simulator instruction counts, event-model estimated target cycles
and measured physical cycles have different authorities. Bind the timing
model and fixed roofline policy to every result. Nominal peak assumptions do
not establish an attainable or physical ceiling. Report unresolved timing
obligations without relabeling instruction counts as cycles.

## Make the next session actionable

The brief should name existing APIs and limits, explain which changes are
implemented, list failures with evidence, and separate future proposals from
validated behavior. Include a first measurable iteration, correctness and
performance gates, fixed time/storage/resource budgets, independent calibration
and held-validation requirements, and original final qualification obligations.

Pass ordering uses declared prerequisites and analysis invalidations in
`PipelineSession`; schedule lowering uses `compile_schedule` with selected
encoding/effect contracts. Named host options use the optional-pass registry.
None of these interfaces grants arbitrary asynchronous overlap or unknown
operation semantics. The next author must demonstrate real backend consumption.

Run focused tests for affected behavior after the merge, including overlapping
integration edits. Select the test environment explicitly so shared editable
installs cannot silently import another checkout. Retain test failures and
fixes, report skips, regenerate the docs index, and run layout/access checks.
Finally verify that both selected integration tips remain ancestors of the
paired contribution branches and report the clean branch heads. Publishing
or launching a new campaign is a separate user-authorized action.
