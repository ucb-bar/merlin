---
title: Defining and inspecting Phase 0 inputs
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-12
related: [generating_capsules, adding_a_target, software_instruction_semantics, component_phase0_diagnostics]
code_refs:
  - src/merlin/targetgen/corpus_instruction_roles.py
  - src/merlin/targetgen/software_spec.py
  - src/merlin/targetgen/instruction_semantics.py
  - src/merlin/targetgen/rtl/circt_introspect.py
  - src/merlin/targetgen/rtl/elaboration.py
  - src/merlin/targetgen/rtl/source_selection.py
  - src/merlin/targetgen/rtl/hw_memory_ports.py
  - src/merlin/targetgen/rtl/hw_hierarchy_bindings.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/hierarchical_memory_intake.py
  - src/merlin/targetgen/dialect_source_scope.py
  - src/merlin/targetgen/isa_mode_audit.py
  - src/merlin/targetgen/generate/typed_mlir.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/evidence.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/generation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/declared_run.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_source_performance.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/source_requirement_ledger.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/original_reference_roster.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/original_reference_flow.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/original_reference_requirements.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/original_reference_standard_ir.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/original_schema_batch.py
  - src/merlin/targetgen/torch_schema_batch_observer.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/m2m_runtime.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/evidence_status.py
  - src/merlin/targetgen/spec_fact_drift.py
  - src/merlin/targetgen/_m2m_capture_worker.py
  - src/merlin/targetgen/original_pointwise_sources.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_qualification.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_final_qualification.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/precision_staging.py
---

# Define the software contract before generating tests

Phase 0 combines selected hardware evidence, a software specification, and workload/test
policy. See the matching [Atlas](../../examples/atlas/phase0/README.md) and
[Gemmini](../../examples/gemmini/phase0/README.md) examples. The installed generator belongs
to `merlin-experiments`; target-specific inputs belong to examples or a selected
target-owned declaration package.

## Start with five decisions

Write what software may rely on, not everything you know about the accelerator:

1. Which operations run on the accelerator, which are fused, and which need the host?
2. What inputs are legal: precision, rank, layout, tails, broadcasting and aliasing?
3. What arithmetic must results obey: accumulation, rounding, overflow and exceptional values?
4. Which quantization formats apply to which operations, and with what scale/zero-point rules?
5. What crosses between host and accelerator, and is it copied or converted?

Start with one operation instead of promising an entire framework. For example, this is
one Gemmini contraction declaration inside `operations`:

```yaml
operations:
  contraction:
    placement: accelerator
    operand_dtypes: [int8]
    accumulator_dtype: i32
    ranks: [2, 3, 4]
    layouts: [row_major_contiguous]
    tails: [zero_pad_valid_window]
    broadcasting: [none, independent_batches]
    aliasing: [disjoint_inputs_outputs]
```

This is a partial starter, not a standalone SW spec or a certificate. Add the schema,
target and explicit numerical semantics; define quantization and transfers
before claiming those paths. Missing declarations are not inherited from this snippet or
guessed from an instruction name. For Gemmini, i32 readout does not make the internal i20
partial sum an i32 computation. Copy the matching full [Gemmini spec](../../examples/gemmini/target/software-spec.yaml)
or [Atlas spec](../../examples/atlas/target/software-spec.yaml) as a worked example, not as
defaults for another target. Omitted review status means **unreviewed**, never admitted.
Add reviewed declarations only when independent evidence supports exactly what you wrote.

Use the operation's name once. A shared family name such as `contraction` or `movement`
selects that family; a custom name must explicitly name its `ops` or `families`. Named
groups are normalized to the same internal operation rows as the legacy list syntax.
When a row names nonempty `ops`, only those exact frontend operations can match it;
`families` do not add other operations. Put an observed-family constraint in
`signature.family` when needed. Rows without an explicit `ops` roster can match
by semantic family. A direct operation match that fails its signature constraints
does not fall back to a broader family row.
A standalone declaration admits only that standalone use: it does not admit the same
family composed into another operation (`composed_with`) or, for an elementwise map,
applied as a fused epilogue. Declare and review such a composition explicitly.
Transfer names work the same way:

```yaml
transfer_contracts:
  operand_load:
    from: host
    to: accelerator
    copy: {dtype: int8, layout: row_major_contiguous}
```

`copy` explicitly means a bit-preserving transfer with the same dtype and layout
at both endpoints; it does not quantize, cast, transpose or dispatch. Add
`valid_window: required` when that is part of the software contract. Type-changing
transfers retain expanded `signature`/`semantics` declarations with explicit
source/result types and conversion rules. Do not mix compact and expanded fields
in one declaration. Both forms produce the same canonical consumer representation
in generated `software/software-spec.json`; the exact authored bytes are also saved.

No placeholder review/evidence fields are needed to author this unreviewed transfer.
The backend owns its physical storage and transport. Software-visible conversion,
layout and valid-window requirements remain explicit. Reviewed transfer claims still
require explicit evidence; missing or explicitly unqualified evidence cannot grant
admission. Quantization format IDs may also be omitted: a stable dtype-pair identity
is generated. Author eligible operations and residual scale/zero-point rules that
the selected readout cannot establish. Do not copy derived tensor mode,
granularity or symmetric zero-point values into this file. Inspect their basis
and remaining unknowns in the generated quantization contract: omission never
proves that an unknown format became supported.

## What is the SW spec?

`target/software-spec.yaml` is a versioned `merlin.software_spec.v1` declaration of
software-visible behavior. It is not a second handwritten hardware geometry table.

| Input | Define explicitly | Do not treat it as |
| --- | --- | --- |
| SW spec | Operation/signature constraints; layouts, tails, broadcasting, aliasing, placement, selected numeric behavior, quantization eligibility and transfer constraints | Automatically proven by an instruction name or storage width |
| Selected capability contract | Target ISA/runner intent and extraction anchors that RTL facts cannot establish; explicit same-target Phase 0 input | Executable OOT support, extracted geometry or certification |
| Selected OOT provider/backend config | Runtime implementation, ISA vocabulary/protocol ownership, extraction anchors and callable references | A second mandatory software spec to hand-maintain |
| Hardware selection | Which evidence is required and which source/configuration is selected | A capability declaration or certificate |
| Extracted RTL facts | Array and memory geometry, interfaces, observed decoder fields, datatype evidence (the array cell's operand and accumulator, and the element format of a lane engine beside it) and structural timing where established | Complete operation latency, numerical behavior, endpoint kind or software legality |
| Recipe/workload policy | Application roster, semantic seeds, tolerances, oracle tiers, holdouts and performance objectives | Hardware facts or generated capsules |

The authored spec holds `operations`, `numerical_semantics`, `quantization`,
`transfer_contracts` and optional `restrictions`. Review status defaults safely to unreviewed.
An operation may also carry its own `status: unreviewed`; an operation is admitted only when
both it and the spec are reviewed. Operation rows require placement and executable typed
constraints, unless they name a fact-derived `hardware` form (below). Use semantic `families` for a shared class, or exact
`ops` selectors when behavior must be operation-specific; do not duplicate both lists.
Numerical semantics select an independent model and its rounding/reduction policy,
never a target-name default. Generated source audits, test counts, qualification hashes
and long diagnostic reports belong in artifacts, not this YAML.

When the SW spec is minimal, `experiment.yaml` must select its same-target
`capability_contract` as a separate file. Installed Phase 0 freezes that file's
exact bytes and refuses a changed contract on resume. For a new run, use
`--phase0-capability-contract PATH` to select a reviewed replacement without
editing the experiment definition. An authored prototype remains diagnostic
until the required RTL and executable support evidence are separately qualified.

## Keep declarations small; derive the rest

Keep accelerator semantics, ISA effects, and deployment selections separate:

| Authored input | What it must establish |
|---|---|
| Accelerator software contract | Mathematical operations, ordered types, numeric behavior, and logical transfer requirements. |
| Public ISA declarations | Instruction effects and encoding fields, with their source identities. |
| Deployment and test policy | Hardware/frontend selections, host capability manifest, execution routes, and grading requirements. |

Record an input once. Host operation support belongs in the selected host capability
manifest; accelerator declarations need not repeat that roster. Before removing an
existing duplicate, compare complete operation admissions and coverage requirements
under the selected inputs. A missing declaration must retain its ordinary UNKNOWN
or refusal behavior.

Use the existing fact-resolution machinery for supported `hardware` declarations
and `eligible_operations: from_facts`. Derive typed operation inventories, tensor
geometry, graph interactions, coverage requirements and capsule membership through
the normal capture and requirements tools. Retain unsupported cases in the denominator.

Numeric semantics remain explicit: operand width, internal arithmetic width and
stored accumulator width can differ. Width extraction alone does not establish
overflow, rounding, narrowing, or complete instruction meaning. Keep provenance and
qualification records as separate machine-readable evidence. A selected execution
route does not establish that its deployment can run a program.

These inputs contain no kernels, instruction sequences, schedules or model-specific
mapping rules. Compiler authors receive only the inputs admitted at their selected
evidence level; this organization does not broaden their grants.

## Select corpus instruction roles for a neutral command runtime

A command contract can explicitly select `corpus_instruction_roles` instead of
the legacy `encoding.corpus_issue_order` baseline. Version 1 contains `version`,
`instruction_roster_sha256` and `by_role`, with optional `epilogue_roles`.
The digest binds the complete `rocc_operand_roles.instructions` list serialized
with sorted keys, compact JSON separators and no nonfinite numbers. Every selected
class must occur in that exact roster; aliases from a different decoder are refused.

`by_role` maps the existing corpus role vocabulary (`memory`, `weight_load`,
`matmul`, `acc_readout`, `acc_readout_scaled`, `tensor_compute_unary`,
`tensor_compute_binary`) to nonempty, distinct class lists. The existing role-slot
selector chooses the first class of the first available role for each slot.
An operation with a missing slot or no resulting classes refuses. Review these
as explicit corpus requirements: they do not establish hardware capabilities,
operand meanings, a schedule, or numerical execution. Unselected legacy and
taxonomy routes retain their existing meaning and failure behavior.

For a fused contraction epilogue, `epilogue_roles` maps an explicitly named stage
to its role/class selections. Those selections apply only when that stage occurs
in the requested contraction epilogue. Conflicting role selections refuse. A
fused readout or accumulator-seeding obligation therefore cannot supply a general
unary/binary role for reductions or elementwise operations. Keep source, placement,
complete numerical, linked-program policy and private full-model gates separate;
a passing class selection closes none of them.

## Bind the selected configuration to its elaborated source

An RTL source selection can verify that supplied FIRRTL produced selected HW MLIR,
while still knowing a configuration name only as an operator-supplied label. For
a new configuration campaign, run `python -m merlin.targetgen.rtl.elaboration`
with an exact Git revision, a committed file containing the configuration
symbol, explicitly selected submodule Git links, and a JSON argument vector.
Pass every relevant nested submodule as `--submodule PATH=COMMIT`, including
its pinned parent. Pass compiled generator JARs or other external executable
inputs as `--tool-input FILE`; the receipt hashes them and rejects a later
change. A successful source check still needs a separately recorded build
dependency and environment closure.
The vector must pass that symbol and contain one `{firrtl}` output placeholder.
This command runs twice in separate fresh directories and writes a receipt only
when both nonempty FIRRTL outputs have identical bytes. Pass that receipt as
`--elaboration-receipt` to `python -m merlin.targetgen.rtl.source_selection`.
The source-consistency view then verifies that the supplied FIRRTL, config,
selected tracked source and receipt still match.

This is a reproducibility and source-binding check, not complete hermetic build
provenance: undeclared environment variables, untracked build inputs, downloaded
dependencies, and elaborator semantics remain outside its scope. Record the
selected build environment and independently qualify the resulting hardware.
Historical source bundles without this receipt keep their prior FIRRTL-to-HW
assessment; they do not acquire source-to-FIRRTL evidence retroactively.

For a machine-dialect campaign, freeze the selected decoder population before
compiler authoring. `merlin-targetgen audit-dialect-modes --require phase1-inputs`
checks exact source revisions, every selected decoder row, parameter-domain
declarations with reviewed finite values and units, explicit exclusions, and reviewed source/model discrepancy
decisions. It deliberately does not require an operation grouping, typed
dialect plan, software admission, or executable instruction tests: those are
Phase 1 outcomes. The default `--require complete-dialect` criterion retains
those later typed-mode and declared-qualification obligations. Neither result
by itself proves arithmetic or hardware execution.

Parameter domains are machine-value sets, not prose labels. An integer domain
declares `kind: integer`, a physical `unit`, `reviewed: true`, and nonoverlapping
`intervals` of `{min, max, step}` with reachable endpoints. An enumeration
declares `kind: enum`, one physical `unit`, `reviewed: true`, and distinct
`values` of one type. Both forms cite one or more `evidence_sources` as
checkout-relative RTL file paths; the source-scope preflight checks those
files against the selected elaboration's pinned Git objects. This records the declared legal value set for later
verifier/allocation generation; source and execution tests still have to
qualify the declaration. Legacy prose domains remain visible as
`parameter_domain_unstructured` and cannot satisfy `--require phase1-inputs`.

After producing the selected source bundle, ISA census, and reviewed OOT mode
ledger, run `merlin-targetgen audit-dialect-source-scope --source-selection
<selection.json> --census <census.json> --inventory <ledger.json>
--expected-config <config-symbol> --out
<report.json>`. This rechecks the configuration-to-FIRRTL receipt, replays the
census from committed sources, and requires the pattern and decoder files to
belong to exactly one pinned RTL checkout in the selected elaboration. Every
required mode also needs nonempty RTL source references whose bytes match
committed files in that checkout; the report records their relative names and
hashes. Citing a file is a source basis, not a validated behavioral claim. A mode
scope audit alone cannot establish that its decoder belongs to the elaborated
machine. The report is a pre-authoring source and requirement population check;
it does not establish the instruction semantics, typed operations, emission,
or execution that Phase 1 must provide. Missing EE290 elaboration remains a
blocker for an EE290 machine-dialect launch even if another configuration's
source bundle is valid.

The later `audit-dialect-modes --require complete-dialect` check binds each
required mode's reviewed `parameter_domains` to fields in the generated typed
plan. Each mode ledger row supplies `parameter_bindings`, mapping every domain
name to one or more references such as
`[{"kind": "attribute", "name": "register_index"}]` or
`[{"kind": "type_parameter", "value": "source", "name": "first"}]`.
The referenced attribute or custom-type parameter declares the same `unit`
and exact finite domain. Integer domains use `intervals` of
`{min, max, step}`; enums use `choices`. Contiguous `min`/`max` is equivalent
to one step-one interval. This check catches an unconstrained odd register-pair
type or a byte offset accidentally interpreted in words. The generated MLIR
verifier enforces those declared values. A matching declaration remains an
authored legality assertion; independent RTL and execution tests must qualify it.
Phase 0 does not require these bindings before the compiler is authored.

## Select a frontend capture runtime for a frozen diagnostic run

Frozen Phase 0 never inherits `MERLIN_M2M_DIR`, `MERLIN_MODEL2MLIR`, or
`MERLIN_M2M_PYTHON` from the invoking shell. If the selected synthesis profile
requires live PyTorch capsules, provide both `--phase0-m2m-root` and
`--phase0-m2m-python` to `merlin experiment run ... --phase 0`, with
`--phase0-evidence-mode diagnostic`. The same two fields may be declared as
`m2m_root` and `m2m_python` in a Phase 0 experiment definition. Neither field
belongs in a target's SW spec.

The run copies the selected `m2m` package (without `__pycache__`) and exact workload
directories named by live model entries, then records their membership and bytes in
`phase0/private/m2m-runtime.json`. It checks the selected virtual environment
and base Python byte inventories before freezing and on resume, and routes all
M2M aliases to the copied source. The copy must stay a read-only, run-owned tree
with exactly its recorded members and no symlinks; reusing a completed run verifies
that archived copy without reopening the historical host venv. A requested frontend capsule cannot silently
disappear when the selected interpreter becomes unavailable. A workload that
names a different interpreter or external source requires a separate materialized
capture; it is not silently run in the wrong environment.

This is **diagnostic host execution**, not verified capture admission: the venv
and base Python remain at selected host paths, native libraries and arbitrary
loader file reads are not isolated, and old capture receipts gain no historical
authentication. For a reviewed corpus, select newly materialized, independently
verified capture inputs rather than promoting this runtime receipt.
When a completed frozen Phase 0 run is reused for a later release, its recorded
output, frozen-source seals and generation/capture attestations are checked as
historical evidence. That completed-artifact check does not reopen the old host
runtime or claim the current producer reran it. Starting or resuming execution
still requires the selected live runtime to match; the new Phase 1 source and
tool bundle is frozen separately.

Without `diagnostic`, the same selection instead configures the sealed Model2MLIR
runner for generation-time captures (operation probes, derived micro models): each
is preselected, run in the sandbox, replayed and attested against the selected
runtime, under `phase0/private/sealed-captures/`, and the capsule records the
attestation it was built from. A verified run with no selected sealed runtime is
refused before any capsule is written, and a capture request the sealed policy
cannot express (a declared loader environment, a pinned interpreter, a quantization
scheme instead of a recipe, an already-materialized model) fails closed rather than
running outside the seal.

## Let the selected facts fill hardware-shaped fields

An operation row can name the hardware form it relies on instead of authoring
hardware-shaped values: `hardware: standalone`, `fused` or `fused_operand_sum`,
with exactly one semantic family (from `families`, `ops` or the row name). Phase 0
selection fills its placement, `composed_with`, dtypes, epilogues and scale
granularity from the selected facts, under the experiment's
`policy.prohibited_instruction_roles` (`merlin.targetgen.spec_fact_drift.resolve_spec`).
Authoring any of those fields beside `hardware` is refused: one value, one authority.
A form the facts do not establish resolves to placement `unknown` with a
`software-spec-derivation` diagnostic; it is never filled with a plausible value.
A quantization format may likewise set `eligible_operations: from_facts`, which
resolves to every accelerator declaration covering a family the derived hardware
recipes quantize.

To narrow a derived value, add a top-level `restrictions` entry with `family`, `field`,
the declined `values` and a stated `reason`, optionally scoped to one fact-derived
`declaration`. A restriction that declines every value, or a declaration's own form,
is an authoring error. `software/software-spec.json` holds the resolved spec, and
`software/selection.json` records what was filled from which evidence; the authored
bytes remain the selected source identity.

Phase 0 also compares the resolved spec with the fact-derived capability field by field,
in both directions, and writes `coverage/spec-fact-drift.json`. An authored field narrower
than the facts without a recorded restriction (`forbids_established`), or one claiming
values the facts decide against (`exceeds_facts`), blocks the coverage commitment.
`restricted`, `unconfirmed` and `undetermined` findings are reported for review.

## Optional reference APIs

The [component diagnostic reference](../reference/component_phase0_diagnostics.md)
describes the separate component-only route and its qualification requirements.
It is not a prerequisite for catalog preparation.
The [instruction semantics reference](../reference/software_instruction_semantics.md)
describes reviewed effects and operand declarations used by the independent
checks. These declarations do not provide a compiler lowering or schedule.

## Derive rather than hand-list the default corpus

Atlas and Gemmini use `capsule_policy: derived_only` recipes with no authored
capsules. Historical seeds remain separately selectable compatibility inputs.
Capture the four independent iteration workloads with `--materialize-bundle`,
then use installed `merlin experiment corpus derive` with all four explicit
`--application-capture LABEL=PATH` selections and exact `--rtl-facts` bytes.
See each target's Phase 0 walkthrough for the complete command.
An external quantization adapter also requires one independently selected
`--application-quant-policy LABEL=PATH@SHA256` per quantized capture. Derivation
checks the exact policy bytes against the capture's manifest and checks that
the manifest names the selected software spec. This binds the operator's
choice. The derived artifact root retains those policy bytes and rechecks
their digest when selecting a synthesis profile. This does not review numerical
accuracy.

Derivation writes a complete demand census, requirements, a candidate synthesis
plan/profile and an exact evidence export. It neither constructs an agent nor
silently selects historical corpora or headline workloads. The generated plan
remains diagnostic until actual writers, oracles, placement and independent
numerical checks satisfy the obligations. Repeating the same selection into its
immutable output root must preserve bytes; changed inputs need a new root.

## Inspect exactly what the generator used

The experiment selects `software_spec`, `hardware_spec` and `evidence_mode`; the recipe
also references the same software spec. Select exact extracted bytes with
`--phase0-rtl-facts /absolute/generated/facts.json`. Extraction and Phase 0 selection are
separate: selection does not rebuild hardware or silently regenerate missing facts.

Each new run writes the following beneath `<run>/phase0/`:

| Artifact | What to inspect |
| --- | --- |
| `hardware/circt/facts.json` | Byte-for-byte copy of the selected extraction; absence remains a diagnostic gap |
| `hardware/effective-views/` | Actual resolved facts, target/performance profiles, execution capabilities, readout inputs, quantization and toolchain observations |
| `software/software-spec.json`, `software/contract.json`, `software/datapath.json` | Selected software declaration (with fact-derived declarations resolved) and the views passed to generation |
| `software/source-snapshots/` | Observed source bytes, indexed by role and hash; not candidate grants |
| `evidence-manifest.json` | Artifact hashes, source identities, consumer-to-artifact mapping and qualification blockers |
| `software/framework/pytorch-opset.json` | Versioned operator catalog observed in the selected capture interpreter: registered ATen overloads, Core ATen and decomposition sets, each with its own scope |
| `coverage/application-inventory.json` | Exact selected detailed inventory bytes, when an adjacent conformance sidecar is selected |
| `software/frontend/index.json` | Application-to-trace, typed graph and workload-specific framework catalog paths |
| `software/frontend/`, `coverage/application-graphs/` | Original/quantized/prepared PyTorch graphs, exact lowering lineage and typed MLIR producer–consumer edges when available |
| `software/host-capabilities.json` | Selected host compiler identities and explicit operation/precision declarations; a dtype profile alone is not operation support |
| `coverage/operation-accounting.json` | Per-application and combined operation partitions, provenance groups, signature/ordinal traceability and declared-versus-observed support |
| `coverage/phase1-capsule-coverage.json` → `phase1_witness_basis` | Finite source-operation and typed-edge witness universe, a compact inventoried selection from the selected cohort, uncovered obligations and the selection's minimum-proof status |
| `coverage/README.md` | Automatically rendered summary of those same operation and quantization views |
| `coverage/performance-basis.json` | Static, capture-bound work per selected operation: MACs for an observed single-MAC contraction, ABI tensor bytes when every extent and element width is known, the admitted placement and the reason for each value it cannot state; bound to the accounting, performance-facts, spec and RTL-facts digests it was built from. Selected operations only: not traffic, timing or a speedup claim. Generation derives capture-shape performance cohorts from it |
| `coverage/spec-fact-drift.json` | Field-by-field comparison of the resolved spec with the fact-derived capability; blocking findings stay open in the coverage commitment |
| `software/quantization-contract.json` | All authored formats, matching hardware recipes, parameter unknowns and operation-scoped quantization decisions |
| `coverage/generation.json` | Written/omitted capsules, failures, synthesis input identity and diagnostic status |
| `capsules/MANIFEST.yaml` | Actual members and distinct functional/performance/diagnostic selections |

Raw extraction and resolved consumer views have separate hashes. Current tool observations
do not retroactively identify the tools that produced unstamped old facts.
With an explicit capability contract, the selected executable provider's own contract is compared
with it: each unit whose scaling the provider states differently, or that the provider lacks, is
recorded as a `support-contract` contradiction bound to both contracts' digests, and neither
declaration replaces the other. A target-resolution home that has not been materialized is recorded
as an unknown support diagnostic, never inventoried as a provider. An evidence member may not be
absolute, contain `..`, or land under `private/`, where a frozen run keeps its source snapshot. Matching target
names, configuration names or dates does not prove that separate HW-MLIR, FIRRTL and hierarchy
files came from one elaboration.

Diagnostic selections remain diagnostic until their required semantics, source
consistency and coverage are qualified. They cannot become a verified release by
relabeling them or passing `--phase0-evidence-mode verified`: unresolved evidence
is refused. The selection's status is a function of its diagnostics alone: it is
`verified` only when none is on record, and verified generation refuses anything
else. Besides a reviewed spec and resolved fact-derived declarations, that requires
an `rtl-source-audit` report (`validation.json`) beside the selected facts that
verifies every audited fact and is bound to the exact facts and hardware-spec bytes,
an admitted sealed-runner attestation for every selected application capture,
re-checked against the capture bytes on disk, and admitted capsule outputs. Editing
a spec's review-status field alone does not establish it.

## Inspect the operator split and quantization contract

Accounting starts at the frontend, not by guessing PyTorch calls from MLIR tags.
The trace-capable model2MLIR conversion records the original captured graph, the
quantized graph actually lowered, the prepared/decomposed graph and the final MLIR.
These stages have separate counts and identities: one source operation may expand
into several lowered operations, or several source operations may fuse together.
Counts describe static captured call sites, not execution frequency inside loops.

The framework catalog inventories the selected PyTorch build's registered **ATen
overloads**, not every Python API or custom operator. Each workload retains its own
capture-process catalog when available; catalogs from another version are not silently
substituted. Missing frontend receipts remain unknown, including for older captures
and loaders that supply an already-quantized model without its original graph. Frozen
replay uses saved bytes and never queries another installed torch.

`coverage/operation-accounting.json` provides both per-application splits and their
conglomeration under `overall`. Its partitions distinguish accelerator candidates,
host work, unsupported/unresolved compute, support-lowering obligations, nested bodies
and structural operations. Candidate admission is screened against the selected hardware
contract and SW signature constraints; it is not proof that a compiler lowered the operation.
The declared support universe also retains operations absent from the selected workloads.
Do not use the observed subset to claim support for an entire untested ATen overload.

The exact normalized graph keeps separate obligations for independent computation and
support lowering. A `tensor.empty`, reshape, slice, constant or copy-like node remains
in the node/SSA denominator, but is not an independent accelerator arithmetic demand:
it needs a compiler-owned typed lowering and shape/value-preservation receipt. Until
that receipt is verified, `support_lowering` and support-mediated SSA dependencies stay
open. The operation-accounting report marks their independent
`accelerator_admission` as `not_applicable` while retaining every
`support_lowering_required` obligation; this is not a successful lowering.
They are not assigned a host/device lane or counted as direct transfers merely
because their source signature appears in a capsule. This avoids adding every lowered
helper operation to the target's SW spec or claiming a nonexistent host fallback.
Historical v1 coverage reports remain inspectable, but verified admission requires a
newly frozen v2 report with these role-specific checks.

Host and accelerator admission are independent: both may accept an operation, or neither
may have sufficient evidence. Requested placement and actual execution are separate
records. `host_required` means a host obligation, not demonstrated host compilation.
Inspect operand/weight storage and compute formats, accumulator/output precision and
shape/layout restrictions. Unknown host precision must not select a different default
profile. Host capability declarations belong beside the selected external compiler;
accelerator RTL does not establish host compiler support.

Each descriptor selects a host manifest through `host_lane.capability_spec`.
Gemmini keeps it in [`platform/`](../../examples/gemmini/platform/README.md);
other examples may select a manifest beside their descriptor. It binds the selected
compiler package digest and precision strategy without changing its payload.
Declarations carry their own review status and exact typed constraints. Neither
review nor an observed schedule establishes emitted compiler execution or support
for an entire operation family. Never populate support from a package name or a
blanket dtype claim. When selecting another host package, update this separate
input with its exact identity and independently reviewed operation constraints.

Typed producer–consumer edges identify casts, quantization conversions and possible
host/accelerator transfers. Their semantics and capsule witnesses must be explicit;
different dtype labels alone do not prove a conversion or an executed transfer.

Select a generated detailed sidecar through the existing conformance requirement.
Its byte copy, parsed accounting, content digest and declared-roster comparison travel
together. A missing sidecar is reported as `not_available`, never as zero uncovered work.
Held-out claim models are not added to this derivation inventory.
When a detailed inventory is selected, synthesis refuses a capture whose name
matches a held-out claim model after case and punctuation normalization; use
separate iteration workloads rather than renaming a validation capture.
Public synthesized capsule descriptions report counts and signatures, not source
model names. An exact integer-operation capsule uses a deterministic source ordinal
within the digest-bound detailed inventory together with the capture hash, so the
group matcher can still account for every occurrence. Keep the full name-to-ordinal
mapping in that inventory; publishing the inventory itself still exposes its
source names. An older frozen corpus that exposed names must be regenerated
rather than relabeled after the fact.

You can inspect a diagnostic subset before a complete requirement is ready, using
the existing inventory producer. It automatically exports the evidence bundle and
its generated coverage README beside the inventory:

```sh
python build_tools/scripts/check_conformance_coverage.py --target gemmini \
  --application-capture example_mlp=/absolute/capture/model.mlir \
  --software-spec examples/gemmini/target/software-spec.yaml \
  --rtl-facts /configured/out/artifacts/audits/gemmini/facts.json \
  --inventory-out /configured/out/artifacts/verification/gemmini/diagnostic.application-demands.json
```

Open `diagnostic.application-demands.evidence/coverage/README.md` beside that output.
The same producer accepts the Atlas example's target/spec/facts. A subset is explicitly
not the full declared roster, and an incomplete inventory still exits nonzero after
writing its diagnostic reports. Changed outputs require a new versioned destination.

Each `quantization.formats` entry references SW operation IDs through
`eligible_operations`, or sets it to `from_facts` (see above). Use a distinct `id` for each format or variant; same-width alternatives
must not borrow each other's scale/readout evidence. Values inferable from the selected
readout come from existing `quant_recipe` derivation. Residual authored parameters must be
explicit, with unknowns retained rather than filled by framework defaults.

The generated quantization contract keeps authored formats, matching hardware candidates,
parameter conflicts and observed operation decisions together. A host-only operation or
an unsupported signature cannot become eligible simply because another operation uses
that format. Unknown legality, numerics or format-specific evidence blocks eligibility.
Framework and compiler realization are separate, unevaluated routes until executed.

This is input for the existing `quant_layer_plan` and `_recipe_quantizer` adapters, which
use TorchAO's public `AOBaseConfig`/PT2E extension points. It does not patch TorchAO, select
a whole-model format automatically, or certify that a custom format is implemented there.

## Regenerate and verify coverage

Keep generated facts, application inventories, conformance requirements, synthesis plans,
capsules, goldens and weights under the configured output root, never committed in examples.
The example's `artifacts/` folder is a navigation guide, not an output destination.
Capsule MLIR references its copied `capsule.weights.safetensors` relatively. Its
frontend relocation receipt records a hash of the capture's original weight-reference
spelling, not the ephemeral absolute capture-cache path.

Changing the software spec or recipe changes synthesis identity. Generate a new requirement
and synthesis pair with the newly selected inputs. Legacy synthesis references without
these commitments remain diagnostic; a digest-bound mismatch refuses generation even in
diagnostic mode. Never edit old synthesis YAML, archived capsules or run receipts to repair
a mismatch. Select new artifacts explicitly through `--phase0-conformance-spec` and
`--phase0-synth-profile` on a fresh run.

Account for every captured application operation as accelerator work, explicitly allowed
host work, or an unresolved/unsupported obligation. A capsule count is not an operation
coverage proof. Inspect omissions, independent goldens and placement checks, then follow
the [generation and reviewed-release guide](generating_capsules.md).

Verified whole-workload admission checks exact source lineage, reviewed independent
compute placement/numerics, support-lowering and shape receipts, typed transfer
obligations, and coverage in the **actual admitted cohort**. A larger generated source
pool is not proof that a selected cohort covers them.
The Phase 0 `phase1_witness_basis` selects a small set of capsule witnesses for the
finite source-operation signatures and conditional typed edges recorded in that
report. It lists the complete scoped universe and obligations without a witness.
When a distinct obligation uniquely requires each selected capsule and those
capsules cover the witnessable universe, the report proves an exact minimum;
otherwise its selected size is only an upper bound. The generated corpus is not
pruned. The witness calculation does not cover numerical, ISA, precision, tail,
shape or other conformance axes, or establish that a compiler can execute the
selected cases.

This witness set is a plan for Phase 1 verification, not a correctness guarantee.
Phase 0 freezes the finite support domain, selected input identities and open proof
obligations. Phase 1 must discharge those obligations against emitted compiler
artifacts and bound target execution, including operation placement, typed support
routes, transfers and numerical behavior. Whole-module formal-proof eligibility is
reported separately; witness coverage does not establish it. A universal claim
also requires sound compositional proofs for the supported domain and hardware
conformance, beyond any finite capsule selection.
The completeness record establishes test obligations, not a working target compiler:
Phase 1 still has to lower, execute and numerically qualify its generated implementation.
Existing hardware-source and independent-reference qualification requirements remain
separate, and a complete coverage report cannot upgrade diagnostic evidence.

Phase 1 consumes the functional selection to establish correctness and freeze a compiler.
Phase 2 consumes its own performance selection plus that exact functional compiler/evidence.
Holdout claim models stay owner-only until the prescribed evaluation point. Full-model
compilation, executable numerical agreement, dispatch evidence and hardware qualification
remain separate requirements; none follows merely from producing these files.
