---
title: Reuse public Phase 1 timing archives in ordinary Phase 2 search
kind: guide
status: current
owner: experiments
last_verified: 2026-10-20
related: [phase2_corpus_search, spike_timing, phase2_capsule_feedback]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/archive_screen_selection.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/archive_screen_review.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/archive_spike_runtime.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/archive_count_observation.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_archive_screen.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_search_stage.py
  - packages/merlin-experiments/tests/test_ordinary_archive_screen.py
---

# Public archive screening

The original host can select an archive-calibrated empirical screen for the
ordinary retained-revision queue. This reuses original public hardware-cycle
labels and observes fresh candidate instruction activity on Spike. Installation
starts no new RTL simulator. Original Phase 1, corpus, compiler, numerical,
measured-incumbent and final gates remain required.

First use the `spike-timing archive-inspect`, `archive-prepare` and `archive-attach`
workflow described in [Spike timing](spike_timing.md). Keep original archive,
per-execution counter-review receipts, sources, inputs, simulator and hardware
identities. Missing observations remain declared and unavailable. Attachment
uses frozen plan order, so JSON object ordering cannot change the fitted dataset.
An empirical fit must pass its original source-grouped held gates. Its
coefficients do not qualify native completion, resource or overlap semantics.

Freeze an explicit `archive_spike_runtime.freeze_runtime(...)` selection. Supply
the original functional argv without an ELF, its ordered functional libraries,
the independent count observer/provider, model, source manifest, disassembler,
counter selection and original entry/completion symbols. Preserve ISA, memory,
privilege and instruction-budget arguments. Tool bytes and source pins are
checked; only execution leaves enter the count child's namespace.

Obtain the target's original live `timing_model_correspondence` capability
through its independent source-review route. Its exact five-field binding must
match hardware facts, observed software contract, count-feature schema, model
and ordered libraries. A copied success receipt or an SDK capability declaration
cannot replace that owner. Compiler numerical admission remains separate from
the count-feature schema identity.

Construct `ArchiveScreenSelection` on the original host with those owners,
explicit archive/binding/observations/screen paths and their expected identities.
`selection.preflight()` reopens the archive, reconstructs the dataset, reproduces
the fit and held gates, and derives training support for every raw activity
feature. This supplies no normal search or hardware-execution admission.

Pass `archive_screen_selection=selection` to `run_stage` alongside the ordinary
`search_configuration`. The original round installs it after build selection
and before queue admission. This is a typed host API; source configuration JSON
cannot instantiate capability owners. Default selection preserves existing
feedback and workflow versions.

Submit the normal complete `corpus-compile-spike-feedback` job. Every arm first
passes its original complete-output L0/L1/L2 screen and actual ELF-producer
checks. The fixed count observer then uses the original narrowed namespace,
child guardian, deadline and shared CPU/resource admission. Plain and annotated
functional consoles must match. Raw reports, counter reviews and process records
remain private. Unsupported tool closures or unresolved cleanup never fall
through to a raw host invocation. Use a selected namespace launcher supporting
the existing nested-user-namespace restrictions.

Explicit archive feedback uses `merlin.ordinary_fast_feedback.v2`. The complete
original roster stays present. Known source families and all raw training ranges
guard screening; unsupported capacity, attention and extrapolation domains stay
UNKNOWN. Screening estimates and residual intervals rank exploration only.
Same-workload comparisons share the bounded comparison budget; unrelated members
are never compared to manufacture a speedup. Saved observations are reopened
through their actual same-ELF invocation producers.

Each screening evidence cell includes `capsule_feedback` for both arms. It joins
original source work, emitted diagnostics, complete counts and correlated screen
terms with the selected facts' partial roofline. Logical payload and requested
bytes are separate from physical traffic. Packing, host, port and overlap costs
remain explicit when unresolved. Prediction-derived attainment is absent;
partial or refuted floors cannot establish convergence. See
[per-capsule feedback](phase2_capsule_feedback.md).

An original host may explicitly select additional public archives for a native
**diagnostic** through `NativeArchiveScreenSelection.additional_archives` and
`AdditionalArchiveEvidence`. Each archive retains its own original byte identity,
binding, reports, per-record counter receipt and source split. Original intake
reopens every component; the strict union rejects incompatible domains, duplicate
executions and source groups crossing splits. Missing observations remain missing.
This selects native exploration record version 4; it does not alter the formal
single-archive installation or infer a timing domain from a source inventory.

Statistical held gates and raw activity support are separate obligations. A
pooled statistical pass cannot override an unsupported instruction count, a
component's narrower evidence or missing target semantics. Preserve component
residual summaries and every refusal. A newly fitted model that fails live
selection remains unavailable to the queue; the earlier immutable model keeps
its own qualification and domain. Compiler candidates cannot edit either model.
