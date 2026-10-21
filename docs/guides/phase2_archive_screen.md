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

For a checked standalone pointer source, explicitly select
`source_domain_protocol="merlin.public_native_direct_pointer_domain.v1"` in
`prepare_public_native_exploration`. The original live pointer-source selector
and compiler/native correspondence capability are required. This selects native
record V5; it does not change the legacy plan reader. A current direct arm must
reopen its private complete numerical/source receipt and its actual consumed
source, command buffer, LLVM, object, ELF and producer chain. A generic plan
diagnostic with status UNKNOWN supplies no authority and cannot repair a failed
original producer join. The exact joint raw-dot semantics and geometry must
occur in eligible public TRAIN evidence; complete current counts must then pass
every original raw-activity range and native inference gate. A source match alone
does not qualify the prediction. Predictions remain unscored, measured cycles
stay null, and incomplete capacity/attention/state domains stay UNKNOWN.

## Request missing evidence without restarting compiler authoring

An incomplete timing domain does not by itself require a new Phase 1 authoring
run. First reconstruct compatible existing public observations with their
original source groups and receipts. A family label such as attention or
capacity does not establish coverage of a connected attention program or a
scratchpad/accumulator boundary.

For genuinely missing coverage, freeze a public measurement request separately
from the Phase 2 search. Include the source and input digests, exact arithmetic
and layouts, shape/resource regime, compiler and hardware identities, complete
independent references, execution boundary, counter-review requirements and
prospective source-group split. Preserve existing held sources and unavailable
records. Request measurements from the compatible Phase 1 producer; Phase 2
iterations continue to use Spike within their qualified domains.

New cycle labels cannot repair an undefined arithmetic contract, an unconsumed
pass, or missing instruction effects. Resolve those implementation/specification
requirements first. Refit only on declared training groups and rerun the full
unchanged held, activity-support and applicability gates before expanding the
predictor's domain. A measurement at one boundary supplies evidence for that
boundary, not every capacity, attention, state or overlap regime.

## Reopen a complete declared roster without launching measurements

`merlin-phase2-timing-readiness` (also available as
`python -m merlin_experiments.phase2.timing_domain_readiness`) reports source and
archive prerequisites. Its input schema is
`merlin.phase2.public_timing_readiness_inputs.v1`, with a pinned `cohort` and an
ordered `archives` list. Each archive entry pins `archive`, `binding` and
`observations`; every pin has an absolute canonical `path`, `sha256`, and optional
integer `bytes`. The cohort keeps its original `members`, each with `identity`,
`role`, and `source_requirements.parsed_operations.source` or an explicit original
source refusal. Source pins select actual typed bytes, never shape-axis ranges.

Pass `--inputs FILE --inputs-sha256 SHA256 --output FRESH_FILE`. The command uses
the original archive inspect/prepare/attach, source reader, exact joint TRAIN
profile checks and unchanged statistical/raw-activity gates. All cases, repeated
records, source groups, splits and missing reports remain visible. Original raw
report availability is separate from eligible dataset records: a missing repeat
cannot select its available partner as a faster representative. The primary
archive is the first declared entry, not a post-error selection. Incompatible
unions refuse; the command does not reinterpret families or resplit sources.
An optional pinned `calibration_plan` uses the existing calibration-validation
plan schema and retains every paired measurement obligation as pending.

Exit zero means the report was written. It issues no live qualification owner,
executes no compiler or simulator, fits no model, and grants no timing, formal
launch or promotion authority. A statistical preflight pass still requires the
original queue-selected runtime/Count/native-semantics owners and fresh complete
source, output and event observations before prediction.
