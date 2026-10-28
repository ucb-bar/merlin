---
title: Reference-guided kernel reconstruction
kind: guide
status: current
owner: mining
last_verified: 2026-10-27
related: [kernel_mining, rvv_kernel_mining_methodology, core_dialects, reference_guided_compilation]
code_refs: [src/merlin/kernels/reconstruction.py, src/merlin/kernels/kernel_evidence.py, packages/merlin-mining/src/merlin/mining/reconstruction_cli.py]
---

# Reference-guided kernel reconstruction

For the two-sided approach and a small example, see
[bottom-up lifting and top-down reconstruction](../design/reference_guided_compilation.md).

Reconstruct a possible compiler realization of an expert **semantic region**, using the
transformation library as the action space and the scheduling dialect as the executable
decision layer. Feed diagnosed gaps to the automatic compiler-learning loop. A reference
kernel may contain several projections, numerical epilogues, or connected operations;
match the complete region before comparing implementations.

This is an explicitly selected `reference_guided` workflow. It does not change an existing
campaign's reference selection, source inventory, or visibility. Private reference source,
traces, indexed results, and derived task cards belong only to the selected workflow.

## Install and inspect a corpus

Install the optional mining distribution with its `reconstruction` extra alongside core.
The extra includes xDSL, the solver used by the existing compact schedule verifier, and
qualified C AST parser versions. There is no download of reference kernels or target tools.

Select input files in a JSON manifest:

```json
{
  "schema": "merlin.kernel_reference_manifest.v1",
  "mode": "reference_guided",
  "library_version": 23,
  "references": [{
    "id": "selected-reference",
    "family": "shared_input_projections",
    "visibility": "operator_only",
    "format": "c_source",
    "artifact": {"path": "reference.zip", "sha256": "<archive SHA256>", "member": "kernel.c"},
    "symbols": ["selected_function"]
  }]
}
```

Paths resolve relative to the manifest. Archive members are read without extraction, after
checking the container's exact hash. Optional `member_sha256` checks the member separately.
Supported formats are `c_source`, `python_schedule`, `elf`, `disassembly`,
`normalized_commands`, and `semantic_region`. Visibility is explicit for every input.

```sh
merlin-kernel-reconstruct analyze --manifest selection.json --out "$TMPDIR/analysis"
merlin-kernel-reconstruct report --input "$TMPDIR/analysis/index.json" --out "$TMPDIR/report.md"
```

Analysis hashes selected inputs, observes source functions/loops/calls without importing
reference code, and selects one or two structurally diverse references per declared family.
Family labels are analysis metadata; they never select a compiler strategy. Native C parser
execution is isolated and bounded; parser failure becomes an actionable evidence gap.

An ELF requires its own selected ISA and operand projector. Static instruction identities
do not establish runtime operands. Native command capture establishes issue order, while
hardware completion and timing require separate evidence. Missing observations remain
unknown rather than becoming a zero count or an inferred performance advantage.

## Process a complete model listing

Use `merlin.perf.pc_instruction_listing` to scope a PC/word listing into call regions,
construct each region's CFG, and compare complete bodies with selected ELF symbols.
`normalized_text` and `lift_schedule` feed the existing disassembly analyzer,
`KernelEvidence`, `ScheduleModel` and CCA projector. A PC-unique listing establishes
instruction sites and control-flow edges; runtime trip counts, operand values and
completion timing require additional evidence.

Index the selected framework capture with
`merlin.capture.graph_regions.capture_graph_regions`. This retains every original call,
its SSA frontier, tensor types, prepared/quantized lineage and pinned numerical recipe.
`merlin.kernels.graph_binding.graph_binding_hypotheses` associates exact named bodies
with possible graph regions and bounded successor/fusion alternatives. These are
provenance hypotheses, including ambiguous names; they do not establish quantizer,
layout or fusion equality. Unmatched bodies retain an explicit source-binding gap.

Select one pinned `region_binding.factory` in the ordinary manifest. It returns
`{"regions": [...]}` with distinct `region_id` values, allowing all expert bodies,
source programs and orchestration gaps to traverse the existing CLI together. Checked
reconstruction jobs use the ordinary adapter and transformation-library search.
Source-only jobs instead select a `hypothesis(source=..., workspace=...)` callback:
it must return pinned emitted bytes and reanalysis naming their exact hash. Compilation
does not credit reference equivalence, numerical correctness, scheduling-dialect
consumption or performance. A whole graph cannot masquerade as an outlined kernel.

Every region retains its own evidence, source alternatives, status and learning tasks.
`packet-shards.json` indexes all bounded experiment packets when the task count exceeds
one packet's budget. Submit the selected shard through the existing reference-experiment
interface; source-region eligibility prevents unrelated capsules from satisfying it.
No kernel-specific human task cards or additional execution engine are needed.

## CCA, semantics, and executable schedules

The evidence model retains exact provenance and relationships that a summary cannot express:
which operand region is reused, which outputs share an input, where each output is quantized,
and which storage windows or dependencies are observed. Source SSA extraction preserves
shared values, nested scalar bodies, and external users automatically.

CCA projects this evidence into the existing compute, memory, dispatch, spatial, layout,
and communication vocabulary. Lift both reference and emitted candidate through the same
interfaces. Unknown CCA fields remain explicit; instruction order alone never establishes
physical overlap, safe buffer reuse, or timing.

The scheduling dialect expresses the executable reconstruction hypothesis. The supplied dense
adapter creates `schedule.plan`, round-trips its text, consumes its blocking/pipeline/loop
decisions through the existing checked library, and emits the resulting counted kernel.
The default counted-command output is a compiler artifact, not a machine execution result.
An independently selected target emitter supplies actual target instructions.

## Connect a target once, then run automatically

For ordinary source capture and dense reconstruction, select the reusable bindings once in
the manifest. No per-kernel semantic graph or schedule is required:

```json
{
  "capture_binding": {
    "factory": "merlin.kernels.reference_capture:capture_reference",
    "settings": {
      "header_root": "<selected original header tree>",
      "header": "<source-selected instruction header>",
      "backend_root": "<selected backend contract tree>",
      "sink_macro": "<selected low-level instruction macro>",
      "fence_macro": "<selected completion macro>"
    }
  },
  "target_binding": {
    "factory": "mlir_oot.reference_reconstruction:bind_dense",
    "backend_root": "<selected backend implementation tree>",
    "module_root": "<target OOT package root>",
    "sha256": "<factory source SHA256>"
  }
}
```

With these bindings, the normal `analyze` and `reconstruct` commands automatically use
`merlin.mining.reconstruction_reference:reference_job`. The capture derives source shapes,
math and numerical boundaries. The job creates frontend SSA, independently recognizes its
complete computation, derives goals from expert CCA, and runs existing target/library bindings.
Unsupported fusion or epilogues remain explicit gaps; raw accumulator output cannot stand in
for a reference's quantized output.

Native instruction counts and logical schedule-call counts have different boundaries. A target
macro can emit configuration plus several issue commands. Until the selected target supplies
an emitted-instruction projection, the logical count stays diagnostic, the shared dispatch
count remains unknown, and the experiment receives an adapter task.

Physical packing needs an exact logical-view witness before logical semantics can match.
For a column-panel view, the checked mapping is `wp[n//P,k,n%P] = rhs[k,n]` with complete
source domains and the inverse `n=panel*P+lane`. This establishes a permutation, not common
input bytes or a free preparation step. Packing/preparation and complete ROI timing remain
separate unknowns until independently measured.

Residency goals retain their scope. Reuse within a macro-panel does not establish that the
complete operand survived the expert's full reduction and output sweep. Stronger observed
requirements enter reconstruction explicitly; unknown lifetime scope remains a typed learning
requirement instead of making a weaker reconstruction successful.

An explicit operator adapter factory (`module:factory`) receives `(manifest, inventory)` and
returns a mapping with:

- `reference_region` and `source_region`: typed `SemanticRegion` objects or their records.
- `adapter`: an executable `ReconstructionAdapter`.
- `reference_cca`: an existing CCA object; supported observations automatically become goals.
- `goals`: optional additional mechanism goals supported by observations and transformation bindings.
- `learning`: keyword arguments for `run_learning`, when automatic gap repair is requested.

The adapter is a reusable backend/source binding, not a collection of per-kernel human choices.
Its numerical/layout projectors derive properties from current source and selected contracts.
Its target effects, ISA, resources, and emitter remain target-owned. The factory's source hash
is recorded, and the caller pins its complete dependency/environment identity for learning.

```sh
merlin-kernel-reconstruct reconstruct --manifest selection.json \
  --adapter selected_backend.reconstruction:factory --out "$TMPDIR/reconstruction"
merlin-kernel-reconstruct learn --manifest selection.json \
  --adapter selected_backend.reconstruction:factory --out "$TMPDIR/learning"
```

`compare_regions` checks graph topology, shared operands, ordered outputs, layouts, precision,
numerical contracts, aliases, and external observations. Missing semantics prevent reconstruction
from being credited as an equivalent region. Measurement compatibility separately checks
preparation, packing, conversion, synchronization, readout, input identity, and runtime scope.

## Bind and compose the transformation library

`TransformationBinding` connects actual `PassFamily` factories to required and established
mechanisms, existing verification hooks, and executable consumers. Reconstruction uses ordinary
`generate_candidates`/`explore_candidates` and `PipelineSession`; it does not maintain another
pass executor. Backward requirements guide search, while forward execution establishes whether
the returned artifact actually realizes them.

Bindings can include frontend fusion/grouping before lowering, schedule blocking and loop
transforms, residency and pipelining, and final target emission. Keep each pass at its supported
IR level. Explicit adapters bridge representation boundaries. Recompute invalidated analyses;
record actual intermediate IR and program hashes. `replay` checks those identities again.

Differentiate an order/parameter problem, missing adapter, composition gap, representation gap,
unavailable target law, and incomplete evidence. Bounded search exhaustion does not prove that
the library lacks an optimization. A declared provider is not evidence that its pass fired.

## Automatic gap repair and evaluation

`learning_tasks` converts reconstruction gaps into typed, machine-consumable tasks. The loop
selects the corresponding parameter/order, adapter, composition, representation, or evidence
route. It runs bounded proposals and feeds executable refusals into the next attempt without
per-case user input.

The production proposer uses the existing sandbox machinery and returns a multi-file
`CompilerPatch` under declared compiler surfaces. Candidate source is copied into an isolated
workspace; builds run in fresh processes. Patches cannot change evaluation, evidence, or learning
owners. Existing single-module pass proposals retain their behavior.

`LearningHooks` belong to the independent compiler/evaluation owner. They must reconstruct
through the actual candidate, verify source semantics, numerical results, storage, target effects,
scheduling-dialect consumption, emitted changes, mechanism achievement, and the frozen control.
The candidate cannot supply its own acceptance callbacks. Goldens and held-out inputs remain
outside proposer access.

Freeze disjoint training, validation, and held-out geometries. Only development refusals guide
proposals; evaluate held-out cases after freezing a candidate. Resume is pinned to source,
environment, task, surfaces, and partition identities. Held-out results never feed retraining.

The loop delegates evaluation to `evaluate_emitted` and ranks only eligible same-domain,
complete-ROI cycle results. Missing or provisional timing remains explicit. Instruction counts
and CCA similarity do not substitute for elapsed cycles. Private measurements do not update the
public physical cycle model through this workflow.

Default bounds are eight candidates, 4,096 search steps, three proposal attempts, and serial
evaluation. All truncation is reported. JSON records are the automatic experiment interface;
the Markdown report is a compact navigation aid. Reports default to private file permissions.

## Steer an existing author experiment

Inside an explicitly guided author session, the existing selfcheck client can select
an exact entry from the pinned reference map and compare freshly read machine views:

```sh
python agent_selfcheck.py --reference-analysis lift --reference-entry ENTRY --reference-limit 1
python agent_selfcheck.py --reference-analysis compare --reference-entry ENTRY \
  --candidate-path variants/NAME/build/CASE/kernel.dis --candidate-symbol SYMBOL
```

Line/PC locators select stripped listings; mapped function labels are annotations when
symbol headers are absent. Wordless disassembly retains scalar mnemonics and recovers
exact `.insn` words, without inventing missing encodings, register values or loop counts.
Source-only references need a machine view before emitted-code comparison. An empty
CCA comparison denominator means missing evidence. Declared semantic contracts can be
compared separately with `--reference-format semantic_region`; declarations alone do
not bind either artifact. Candidate `schedule_mlir` analysis verifies scheduling intent;
actual executable consumer receipts and emitted mechanism witnesses remain required.

Construction feedback joins this analysis to the selected operation/pass catalogs and
existing A/B tools, including explicit adapter gaps before correctness is complete.
Declared producer/pool chains reuse the composition discovery tool and packing route.
An imported helper, a changed setting and a matching instruction count cannot establish
mechanism realization, numerical equivalence or a speedup.

Reference comparisons can drive the existing author feedback and idle tuning loop. Generate
a bounded packet from a reconstruction index, the selected compiler/evaluator context, and
pinned structured experiment history:

```sh
merlin-kernel-reconstruct steer --input reconstruction/index.json \
  --context experiment-context.json --history history-selection.json --out "$TMPDIR/steering"
```

For automatic packet generation during `analyze`, `reconstruct`, or `learn`, select the
same inputs in the existing manifest. Paths are relative to the manifest:

```json
{
  "experiment_context": {"path": "experiment-context.json", "sha256": "<SHA256>"},
  "experiment_history": [{"path": "history-selection.json", "sha256": "<SHA256>"}]
}
```

The output contains `packet.json`, `selection.json`, an indexed report, and the selection's
SHA256. An explicit operator invocation enables it for a new experiment or continuation:

```sh
python -m merlin_experiments.phase1 <ordinary invocation arguments> \
  --reference-experiments "$TMPDIR/steering/selection.json" \
  --reference-experiments-sha256 <printed selection SHA256>
```

The existing checked launch configuration accepts the same selection:

```yaml
reference_experiments:
  path: "${PREP}/steering/selection.json"
  sha256: "<printed selection SHA256>"
```

Fresh preparation stages a compact `REFERENCE_EXPERIMENTS.json` and task instructions before
the ordinary treatment seal. The author can use these goals from its first turn, including
adapter/representation work without declared tuning knobs. Existing diagnostics, catalog
inspection, pass application and selfcheck clients supply the experiment workflow; no additional
agent or runner is needed. Resume verifies the originally served guidance instead of rebuilding it.

After a compiler edit, immutable guidance remains available as prior evidence and hypotheses.
The round brief directs the author to collect current source/emission evidence and continue
through ordinary selfchecks. Old exact requests and results cannot transfer to the new compiler.
Missing adapters stay construction tasks; they do not block independent work on other regions.

Absent selection preserves the existing treatment. Workspace files and ambient environment
variables cannot enable reference feedback. Resume retains the original selected digest;
changing reference inputs requires a separately recorded continuation. Select public or
explicitly reference-guided visibility before exposing tasks to an author. This does not
add reference material to an existing sealed campaign. Keep the selection directory host-private,
outside author mounts: it also holds the broker's small outcome journal. The author receives the
bounded feedback projection through existing clients.

Each task retains observed facts, uncertain hypotheses, prior outcomes, dependencies, and
acceptance checks. Missing semantic/effect adapters and source-attribution evidence take
different routes from performance experiments. A smaller instruction count, fewer drains,
or an imported pass cannot close a performance task. An unqualified historical regression
remains useful context but cannot suppress a new experiment as an exact duplicate.

Context identifies the compiler, source region, environment, reference and evaluator. For
Phase 1 feedback, `compiler_sha256` uses the existing broker submission digest; set
`compiler_identity_basis` to `phase1_submission_sha256`. Tuning domains come from the
compiler's actual declaration and settings through `workspace_tuning`; explicit mechanism
bindings connect them to task goals. Public capsule mappings supply the existing selfcheck
arguments and frozen training/validation/held-out membership. Names identify requests, never
compiler strategy. Keep evaluator inputs and held-out results outside proposal workspaces.

Controlled requests vary declared settings while retaining the other settings. The idle tuner
validates requests again against the current compiler declaration and submits eligible training
cases using its existing private-copy `tuning_overrides` transport. Its ordinary CPU, request,
numerical-baseline and timing-domain gates still apply. No variant copier or simulator scheduler
is introduced. Code/adapter tasks remain explicit author work when no executable knob is bound.
Generated selfcheck commands include `--tuning-overrides` for a changed setting and
`--reference-request` for the selected task. The latter proposes a result association;
the broker resolves it against the trusted packet and checks the actual execution context.

The actual selfcheck response and round brief carry a bounded next-experiment summary. The
broker supplies the graded compiler identity; source drift requests fresh evidence. Experiment
responses retain task/request identities alongside the original tuning observations. Completion
requires independently collected witnesses for the requested claim; author-supplied success
labels and a timing result alone cannot establish mechanism realization or generalization.

For a fresh compiler, select representative regions from a top-down semantics/coverage map
first. Mine ISA and expert mechanisms during bring-up, compare as soon as a region compiles,
then prioritize connected-region costs and shape policies as coverage grows. Keep mathematics,
mechanism reconstruction, correctness and performance selection separately observable.

## Qualification

Focused integration checks exercise real scheduling-dialect consumers, independent numerical
execution of emitted neutral commands on aligned/ragged/thin shapes, replay, CCA projection,
semantic-fusion mismatch rejection, and isolated proposal retries. A neutral-machine test
qualifies those compiler mechanisms, not target hardware performance. Physical performance
parity requires matched numerical and measurement scopes and actual timing observations.
