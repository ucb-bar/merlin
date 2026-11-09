---
title: Reviewing compiler optimization changes
kind: guide
status: current
owner: experiments
last_verified: 2026-11-09
related: [phase2_corpus_search, optional_passes, shared_compiler_passes, shared_phase1_scheduling, spike_timing, phase2_shape_spectrum]
code_refs:
  - src/merlin/common/artifacts.py
  - src/merlin/llvmlower/optional_passes.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/compiler/schedule/schedule_pipeline.py
  - src/merlin_experiments/phase2/search_session.py
  - src/merlin_experiments/phase2/search_target.py
---

# Reviewing compiler optimization changes

A compiler change needs an inspectable implementation, unchanged correctness
requirements and measurements bound to the code that ran. Review the generic
compiler and out-of-tree target changes together when a transformation crosses
their interface.

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


## Check the implementation and release

Pass ordering uses declared prerequisites and analysis invalidations in
`PipelineSession`. Schedule lowering uses `compile_schedule` with explicitly
selected encoding and effect contracts. Named host options use the optional-pass
registry. An advertised API is not evidence of backend consumption. Check that
the actual emitted product contains the intended transformation.

Run the affected behavior tests, including negative applicability cases. Select
the environment explicitly so an editable install cannot import another checkout.
Record failures and skips, regenerate code-derived documentation, and run source
layout and access checks. A release contribution should identify the code change,
its applicability, numerical policy, measured scope and known limits. Keep dated
results and deployment-specific records in the result artifact, not this guide.

The finite schedule entrypoint is now
`merlin.compiler.schedule.schedule_pipeline.compile_schedule`; its historical
`compile.scheduling` name preserves compatibility. Open that canonical owner to
follow selected encoding/effect contracts through final independently decoded
bytes. Source-IR `PipelineSession` and optional lowering switches are separate
interfaces with their own real callers.
