---
title: Calibrating and validating annotated timing
kind: guide
status: current
owner: core
last_verified: 2026-10-18
related: [architecture, performance_budget_unit]
code_refs: [packages/merlin-timing/src/merlin_timing/contracts.py, packages/merlin-timing/src/merlin_timing/capacities.py, packages/merlin-timing/src/merlin_timing/fitting.py, packages/merlin-timing/src/merlin_timing/validation.py, packages/merlin-timing/src/merlin_timing/design.py, packages/merlin-timing/src/merlin_timing/feedback.py, packages/merlin-timing/src/merlin_timing/cli.py, packages/merlin-timing/src/merlin_timing/archive.py, packages/merlin-timing/src/merlin_timing/replay.py, packages/merlin-timing/src/merlin_timing/provider_bundle.py, examples/gemmini/verification/extract_rtl_timing_facts.py, src/merlin/targetgen/rtl/timing_structure.py, src/merlin/targetgen/rtl/unit_correspondence.py]
---

# Calibrating and validating annotated timing

`merlin-timing` provides standalone Python tools for empirical screening and
validation of original annotated event predictions. The Python runtime uses only
the standard library. It does not import Merlin or xDSL, launch simulations, or
supply target prices.

## Provider ownership and standalone builds

The generic timing engine, source-bundle verifier, provider build orchestration
and calibration tools live in `merlin-timing`. Instruction semantics, device
configuration facts and optional observers live in the simulator repository's
`examples/` directory, independently of the compiler under evaluation. Observers
load separately from the functional extension. Phase 1 produces the target OOT
compiler from scratch; these trusted tools require no handwritten compiler or
generated compiler code. Candidate grants expose runtime endpoints through the
broker, without simulator source, vendor headers or selfcheck fixtures.

A target exporter calls `merlin_timing.provider_bundle.export_bundle` with its
source bytes and selected C++ entrypoint. It receives a closed, hashed bundle
with the canonical shared builder. The exported `build_provider.py` runs without
Merlin installed, using Python's standard library and one compiler process:

```sh
python3 /path/to/provider-source/build_provider.py --verify-only
python3 /path/to/provider-source/build_provider.py \
  --spike-root /path/to/annotated-spike \
  --output-library /path/to/libtiming_provider.so
```

The source roster is manifest data. No target-specific builder implementation is
needed. The builder rechecks bundle and runtime-header hashes before publishing
the library. Byte identity establishes custody, not timing accuracy. These
operator-side sources and headers are not candidate compiler inputs.

## Separate the two claims

A fitted **empirical screen** is a nonnegative linear relation between observed
features and measured cycles. Its coefficients and fixed cost are interpretable
screening terms. They can be exported as diagnostic parameter seeds, but fitting
that relation does not calibrate or qualify the nonlinear event scheduler.

**Native event timing** is validated from original complete event-engine reports
for the exact selected model, engine, device, executable, inputs and timing scope.
This route does not use the linear screen's predictions or coefficients. The
source importer and runtime must independently establish original artifact and
measurement provenance. Hashes and passing statistical checks alone cannot prove
that a caller supplied genuine measurements.

Neither route produces a physical timing bound or a guarantee of final runtime.
Missing facts, incomplete execution and observations outside the fitted domain
remain UNKNOWN.

## Commands

```bash
spike-timing fit --dataset public.json --output screen.json
spike-timing validate --dataset public.json --model screen.json --output screen-validation.json
spike-timing validate --dataset public-events.json --prediction-kind event_timing --output event-validation.json
spike-timing compare --model screen.json --cases candidates.json --output comparison.json
spike-timing compare-native --cases native-candidates.json --qualification event-receipt.json --output native-comparison.json
spike-timing probe-plan --dataset public.json --model screen.json --output probes.json
```

`fit` includes a nonnegative fixed invocation cost unless `--no-fixed` is selected.
The solver scales features and labels, refuses underidentified or ill-conditioned
terms, and checks projected NNLS convergence. It never fits on calibration or test
labels. `--maximum-condition` records the selected conditioning limit.

`compare` reports diagnostic estimates without a qualification receipt. Its
qualified intervals and ordering require `--qualification receipt.json`, where
the receipt contains the exact original `dataset` and its `report`. The tool
recomputes validation rather than trusting a supplied success flag. Only separated
empirical intervals decide an ordering; overlapping or unqualified intervals do
not. The receipt qualifies a screen, never an event engine.

`compare-native` uses original native report predictions, with no linear model.
Its optional qualification receipt must reproduce `validate` with
`prediction_kind=event_timing`. Exact profile, domain and declared features must
match. New candidates also need current independent numerical receipts. Only
in-domain candidates with separated qualified residual intervals decide an
ordering within the same workload. Missing qualification, overlapping intervals,
failed numerics and observations outside the training range remain diagnostic.

`probe-plan` exports missing original feature reports, passing measurements,
native reports and domain coverage. It contains no launcher or generated labels.

## Dataset contract

The closed top-level object has schema `merlin.spike_timing_dataset.v1` and fields
`profile`, `domain`, `feature_names`, `cases` and `records`. Contracts are validated
without an optional schema library; unknown top-level or record fields refuse.

| Object | Required information |
| --- | --- |
| `domain` | Digests for device configuration, timing scope, feature engine, feature schema, target model and memory regime; measured label engine and its digest |
| `case` | Unique case ID; whole-source leakage group; derived workload digest; mechanism family; explicit train/calibration/test split; source visibility, phase, manifest and content digests; ELF, input, scope and device digests |
| `record.features` | Exactly the declared feature names, each a finite nonnegative number or `null` |
| `record.feature_provenance` | Exact engine/model, original trace and execution-identity digests; explicit completeness |
| `record.label` | Original measured-cycle receipt, exact execution identity, engine identity, status, numerical status, freshness and positive cycles; `null` means unavailable |
| `record.event_report` | `null`, or an envelope containing the original native report and its canonical content digest |

`record_id` and `case_id` join each record to its declared case. An execution
identity hashes ELF, inputs, source, scope and device. A workload identity hashes
source, inputs, scope and device, so different compiled variants can be ranked
within the same workload. The native model digest is the actual model-file byte
hash supplied by the runtime. The fitted screen's artifact digest is separate.

Canonical content digests use SHA256 of JSON with sorted keys, compact separators
and no nonfinite numbers. A native report envelope is
`{"sha256": "<canonical-content-digest>", "report": {...}}`. Descriptive native
report extensions remain digest-bound but cannot override selected identity,
feature counts, completeness or timing fields.

## New native candidate contract

`compare-native --cases` accepts the closed schema
`merlin.spike_timing_native_cases.v1`, with `profile`, `domain`, `feature_names`
and `cases`. Each row contains:

- `case`: the dataset's exact case identity fields, omitting `split`;
- `features`: the declared feature projection from the original native report;
- `event_report`: its original report envelope, or `null` when unavailable;
- `numeric`: `null`, or an independent receipt with `status`, `receipt_sha256`,
  `execution_sha256` and boolean `measured_now`.

Numerical status must be `pass`, `fail` or `unknown`. Only a current `pass` for
the exact execution can authorize ranking; successful ELF exit is insufficient.
Public inference may use public Phase 1 or Phase 2 sources. This does not relax
the calibration dataset's restriction to public Phase 1 sources. Hidden,
validation, private or unknown sources cannot enter public inference. Historical
private comparisons remain private regardless of their statistical result.

## Public and private profiles

- `phase1-public` admits only public Phase 1 source records.
- `historical-private` admits only historical private source records. Its results
  are never eligible for public feedback.

Both use the same fitting and validation code. Profiles cannot mix source
visibility or phase. Exact domain digests prevent transferring prices across a
changed device, model, feature engine, scope or memory regime.

All variants and repeats of one whole source group stay in one split. Source and
workload identities cannot be assigned different groups. Repeats remain records
of one case and are aggregated without increasing pair counts. A failed, stale,
unverified or inconsistent repeat cannot be discarded to select a faster label.
The complete declared held case roster supplies the coverage denominator.

## Qualification gates

Held-out predictions must satisfy every gate:

- At least 90% agreement on within-workload ordering, considering measured
  separations strictly above 5% of the slower measurement. Predicted ties earn no
  correct-order credit.
- Median absolute relative error at most 15%; nearest-rank p95 at most 30%.
- At least 20 eligible held pairs across at least three mechanism families.
- Predictions for at least 80% of declared held cases.
- Residual interval evidence from at least two independent calibration groups.

Calibration residual quantiles provide empirical intervals. They do not establish
hardware bounds. Native timing qualification requires original native predictions;
a screen passing the same numerical gates cannot qualify that engine.
Unqualified outputs remain diagnostic. Synthetic tests exercise these policies
without establishing an actual engine, device or dataset qualification.

## Resource limits

Each JSON input is limited to 16 MiB; datasets admit at most 10,000 cases,
50,000 records and 64 declared features. Pair evaluation refuses above 20,000
comparisons rather than silently sampling or truncating the roster. Artifacts
contain compact records and hashes, without copying compiler checkouts, tensor
payloads or simulation environments.

## Original Phase 1 capsule archives

The archive intake accepts `merlin.phase1.capsule_timing_archive.v2` and `.v3`
from `merlin_experiments.phase1.capsule_timing_archive`. A v2 archive prepares
observations only when it is `timing_ready` (every passing gSIM record belongs
to a passing capsule). A v3 archive, normally the run's
`capsule_timing_training_<digest>.tar`, is judged per record: intake recomputes
each row's `training_eligible` verdict (a fresh, console-verified, kernel-window
gSIM measurement whose gSIM tier passed numerics with every identity present,
whatever the capsule's overall verdict) and refuses an archive whose declared
verdicts, visibility or training counts disagree. Only public eligible rows can
be bound; a hidden row refuses. A binding must cover every eligible public row
and every executed public repeat of a bound ELF; a result whose gSIM tier never
ran is not a repeat. Carried (copied) verdicts never count as measurements.
Supply the archive SHA256 from the immutable producer handoff, rather than
substituting a locally invented source or simulator identity:

```sh
spike-timing archive-inspect --archive capsule_timing.tar \
  --archive-sha256 ORIGINAL_ARCHIVE_SHA256 --output inventory.json
spike-timing archive-prepare --archive capsule_timing.tar \
  --archive-sha256 ORIGINAL_ARCHIVE_SHA256 --binding reviewed-binding.json \
  --output observation-plan.json
spike-timing archive-attach --plan observation-plan.json \
  --reports original-observations.json --output dataset.json
spike-timing validate --dataset dataset.json --prediction-kind event_timing \
  --output holdout-report.json
spike-timing archive-handoff --plan observation-plan.json --dataset dataset.json \
  --spike-executable /selected/spike --spike-configuration selected-spike.json \
  --model-artifact frozen-model.json --training-code training-code.tar \
  --output handoff.json
```

These archive commands perform host file processing only. They launch no
simulations and refuse to overwrite their outputs. Intake streams the
uncompressed tar, verifies every indexed artifact's bytes, and rejects links,
duplicate/traversing paths, missing files and contradictory summary counts.
Failed and unavailable rows, including null artifact references, remain in the
inventory. Caller, RTL console and original result are checked independently:
the caller must carry exactly one matching encoded timing selection, the console
exactly one positive matching `METRIC cycles` line, and the result the same
verdict, compiler identity, command-buffer identity and tensor shapes.

Run `spike-timing archive-design-check --plan observation-plan.json --output
design-check.json` immediately after preparation. It counts measured,
within-workload variant pairs separated by more than 5%, the mechanism families
they cover, calibration source groups and missing labels. A passing design check
only says the captured experiment can *attempt* native timing qualification; it
does not establish Spike accuracy. If the design check fails, add original
capsules or compiler variants and measure them with gSIM before freezing a model.
Repeating one ELF cannot create a variant pair. A static instruction trace is not
a dynamic schedule.

### Replaying a new native model against existing public labels

When resource or cache structure changes, the old count reports still identify
only their original model. Prepare a separate replay request without copying
those reports into the new domain:

```sh
spike-timing archive-replay-plan --plan observation-plan.json \
  --model-artifact structural-model.json --spike-executable /selected/spike \
  --provider-library /selected/functional.so \
  --provider-library /selected/timing-provider.so --output replay.json
```

This command performs no execution. It verifies the native model's exact ordered
library hashes and freezes the selected observer binary and model identities.
The hardware, observer source engine, label engine, memory regime and counter
scope must agree with the original archive plan. The output's `observation_plan`
contains the new model-file digest and otherwise preserves all original labels,
source splits, ELF/input bindings and per-repeat boundary receipts. Missing
reviews remain unavailable. Save that nested object as a separate plan; collect
fresh native reports under the ordinary resource admission controls, and use
`archive-attach` with that plan. Old-model reports refuse attachment even if an
attachment envelope names the new plan digest.

Before execution, the operator must verify the binary's source-engine identity,
functional/native provider compatibility, immutable ELF extraction and all input
hashes. The planner does not certify these runtime properties or complete missing
instruction/resource semantics and prices. Held-out event validation and capacity
extrapolation remain necessary before predictions may rank compiler candidates.
In particular, copying cache geometry from a named hardware configuration does
not establish its external-memory timing domain: use the actual label simulator's
selected memory implementation. The current retained gSIM binary uses
`mm_magic_t`, with response queuing and AXI backpressure, rather than a FireSim
DRAM bridge or DRAMSim.

### Structural facts for the selected 16 GiB gSIM target

The operator can extract one pinned, machine-readable `rtl_facts.json` from the
matching Chipyard source and generated gSIM FIRRTL/DTS/L2 metadata:

```sh
python3 examples/gemmini/verification/extract_rtl_timing_facts.py \
  --chipyard-source CHIPYARD_DIR --generated GENERATED_DIR \
  --firrtl-sha256 SELECTED_FIRRTL_SHA256 --dts-sha256 SELECTED_DTS_SHA256 \
  --spike-profile SELECTED_SPIKE_PROFILE.json \
  --output rtl_facts.json
```

The extractor requires the FireSim tweak → GemminiRocketConfig →
DefaultGemminiConfig source chain, checks the default INT8/INT32 geometry against
generated reservation station and controller queue structures, and reconciles
16 GiB memory and L2 size against the DTS and L2 metadata. It records exact
source/artifact hashes. The current selected elaboration yields 16×16 PEs,
256 KiB scratchpad in four banks, 64 KiB accumulator in two banks, 8/16/4
load/execute/store reservation entries, 8/8/2 controller queue entries and
16 GiB RAM at 0x80000000. The generated counter interface exposes 45 event
signals and eight external counter values; this does not imply those counters
were configured or sampled in a Phase 1 capsule. These are *structural*
constraints. The target cycle
frequency and primitive prices remain unknown. Do not copy reservation entries
into the generic event engine's `ResourceSpec.capacity` or `queue_capacity`:
those fields model concurrent occupancy and queued reservations, respectively,
and require a reviewed semantic mapping.
With `--spike-profile`, the same command also checks the selected Spike
profile's mesh, element widths, scratchpad/accumulator rows, memory base/size,
target name and original DTS hash. A structural match does not qualify timing.

### Loading structural facts into the device profile

Structural facts reach the observer's device profile through one closed,
target-neutral document, `merlin.timing_capacity_facts.v1`, and one command:

```sh
spike-timing apply-capacities --profile SELECTED_SPIKE_PROFILE.json \
  --facts capacity_facts.json --output profile-with-facts.json
```

The document names the exact profile it was derived against
(`device_profile_sha256`, the SHA256 of the profile file), the elaborated
design (`elaboration`: artifact name to SHA256, e.g. FIRRTL and DTS) and a list
of `facts`. Each fact has a unique `id`, a semantic `role`, the profile name it
`applies_to`, an integer `value` in the role's `unit`, 1..16 pinned `sources`
(`path`, `sha256`, `selector` naming the RTL element or field read) and a
`derivation`. The **role**, never a field name, decides where a value may go:

| Role | Unit | Placed at | Meaning |
| --- | --- | --- | --- |
| `occupancy_lanes` | commands | `resources.<name>.capacity` | Simultaneous occupancy lanes of an engine resource |
| `start_queue_entries` | commands | `resources.<name>.queue_capacity` | Reservations that may wait to start on that resource |
| `live_command_credits` | commands | `target.structure.live_command_credits.<name>` | Entries held from allocation through completion (a reservation station) |
| `command_queue_entries` | commands | `target.structure.command_queue_entries.<name>` | Buffer after issue, distinct from live-command credits |
| `outstanding_requests` | requests | `target.structure.outstanding_requests.<name>` | In-flight memory request limit |
| `max_transfer_bytes` | bytes | `target.structure.max_transfer_bytes.<name>` | Largest single transfer request |
| `datapath_bytes_per_beat` | bytes | `target.structure.datapath_bytes_per_beat.<name>` | Transfer datapath width |
| `minimum_latency_cycles` | cycles | `target.constraints.<class>.minimum_latency_cycles` | Fixed pipeline delay; a lower bound when fitting that price class |
| `declared_hardware_fact` | its own | `target.hardware.<name>` | Fills a hardware fact the profile already declares (e.g. a reset state left null) |

A fact can fill only a slot the profile declares (a resource, hardware fact or
price class), an already known value must agree, a role is stated once per
target, and nothing supplies or fits a price. The output carries a
`capacity_facts` receipt (document digest, profile digest, elaboration and every
placement). For the selected target, the extractor's reservation-station entries
are `live_command_credits`, its controller queue lengths `command_queue_entries`,
its in-flight memory request limit `outstanding_requests`, its DMA maximum
`max_transfer_bytes`, and its DMA bus width (in bits, divided by eight)
`datapath_bytes_per_beat`. `occupancy_lanes` and `start_queue_entries` assert a
reviewed mapping onto the generic engine's lane and wait-queue semantics; do not
derive them from reservation entries. The observer evidence pass reads the loaded
profile as its device profile: resources whose lanes and start queue are both
known become engine resources, the rest stay unbound and are reported
unresolved, and `structure`/`constraints` ride in the model for providers and
fitting.

### Timing structure derived from the elaborated RTL

The facts extractor's `timing_structure` section reads the timing-relevant
structure from the selected elaborated HW instead of Chisel sources
(`merlin.targetgen.rtl.timing_structure`). The target contract names anchors
only (`rtl_extraction.timing_structure`): the dispatch module and per-class
issue-port prefixes, the bus field vocabulary, memory-client port families,
the dataflow cell select, and port pairs for stage counts. The section then
records:

- dispatch entries per class (slots readable at the issue port) and the
  command buffers the intact issued command crosses;
- per memory client: beat width, request ID space, the request-size values
  the size field can carry, and in-flight transactions (state slots selected
  by the response ID);
- register stages between declared ports, classified `fixed`,
  `fixed_when_enabled` (enable-gated stages), `variable` (storage occupancy
  or feedback) or `unknown`;
- the dataflow modes the cell select chooses, classified structurally as
  weight- or output-stationary, and whether the select is runtime-driven.

Values the readers cannot reach are UNKNOWN with a reason. Memory round
trips, sustained DMA bandwidth, issue/completion overheads, translation
misses, contention and per-command occupancy are not RTL facts. They are
listed under `fit_from_measurement` and remain gSIM fits.

`simulation_correspondence` records whether the accelerator unit is the same
hardware under the simulation configuration and the chosen target
configuration. It compares FIRRTL module bodies, canonical HW per module and
SoC-level anchors (`merlin-target-tools unit-correspondence`). Measurements
on the simulation configuration stand for the target's unit-internal timing
only when the result is `identical` or `structurally_equivalent`.

The engine-resource mapping is a reviewed, target-owned declaration
(`profile_resources`): each `capacity`/`queue_capacity` names a structural
quantity, or `null` to leave it to fitting. Apply it to a profile copy:

```sh
merlin-target-tools timing-profile-structure --target TARGET \
  --facts facts.json --profile profile.json --out profile-structure.json
```

The copy records the mapping, its stated basis and every field that is still
to be fitted under `structural_mapping`. No price is derived, and the profile
stays unqualified until native validation passes.

### Fitting for Phase 2 ranking

Use Phase 1 gSIM cycle labels for the exact original ELF, embedded input,
kernel-call scope and selected simulator. Keep all variants of a source in one
train/calibration/test split. Capture at least 20 held variant pairs with measured
separation above 5% across at least three mechanism families, two independent
calibration source groups, and a train roster covering the held families. Sweep
tile shape, K blocking, reuse and DMA scheduling where numerics and memory bounds
hold. Include matched repeats to estimate noise; repeats are one case, not new
ranking pairs. The present Phase 1 capsule contract has no loop/FSM command
coverage, so neither fit nor claim those regimes.

Run the *functional* Spike extension with the optional timing observer on the
same original ELFs and inputs, with one frozen model file and reviewed PC scope
bindings. Tune Gemmini event latencies, occupancies and transfer rates only from
training labels; use calibration groups for residual intervals and held groups
once for validation. Native event reports must retain full feature counts,
unresolved events, cycle breakdown, exact model/engine hashes and numerical
receipts. Accept the model for Phase 2 LLM ranking only if `validate
--prediction-kind event_timing` qualifies it. The empirical screen is a
diagnostic seed, not a substitute for that native validation. When qualified
intervals overlap or a candidate leaves the training domain, report a tie or
UNKNOWN and spend gSIM measurement only where the choice matters.

The measured label is **one host-core cycle delta for the original capsule kernel
call through its selected completion wait**. Input staging, output serialization,
printing and parent readback are excluded. It is neither a command latency nor a
whole-model measurement. Counter/clock semantics come from the reviewed selected
record and pinned original accessor, ISA and clock sources. No mnemonic supplies
an implicit clock. Static instruction traces remain static; intake does not use
them as dynamic events. Cache warmth stays unknown without a separate protocol.

### Reviewed bindings and execution requests

The closed `merlin.phase1.spike_archive_binding.v1` object contains:

- `archive_sha256`, `domain` and `feature_names` under the existing contracts;
- `review`, containing the exact `timing_selection`, `cycle_scope`,
  `sim_provenance_sha256`, normalized `counter` and optional `cache_protocol`;
- `cases`, each with an existing dataset `case`, all original `results` paths,
  `candidate_sha256`, `shape_sha256` and a reviewed `counter_scope`.
  Different compiler versions can emit byte-identical programs; such an execution
  stays one case, and `candidate_sha256` is then the sorted list of every
  candidate that emitted those bytes. Each observation keeps its own result's
  candidate.

Binding version 2, `merlin.phase1.spike_archive_binding.v2`, replaces each
case's `counter_scope` with `counter_scopes`: an exact map from every original
result path to its own reviewed scope. Repeats can retain different original
review receipt digests while remaining records of one case. Their reviewed
PCs and occurrence must agree. A missing review is explicitly `null`; its
label and case remain declared, but no native report can attach to it.
No boundary is inferred from another execution. Version 1 keeps its original
nonmissing, single-review contract.

Version 2 emits `merlin.phase1.spike_observation_plan.v2`. The existing attach,
design-check and handoff commands accept both plan versions. Missing reviews
prevent design readiness. Missing reports exclude the entire affected case's
records from fitting while retaining its declared validation denominator, as
before. Incomplete native timing stays incomplete; per-record receipts do not
qualify feature counts, timing semantics or fitted coefficients.

`review.counter` requires `unit: cycles`, integer `width_bits: 64`,
`wrap: modulo`, the declared `clock`, integer `max_delta_cycles` strictly between
zero and 2^63, and `sources` with accessor/ISA/clock SHA256 values. These must agree
with the original selection and its `counter_semantics.source_pins`. The binary
identity in the original `sim_provenance.sha256` must equal the domain label
engine identity. The selection's own `selection_sha256` is preserved; it is **not**
replaced with a digest of the self-referencing full record.

The scope digest is the canonical digest of `merlin.kernel_call_scope.v1` with
`cycle_scope`, `timing_selection_sha256`, and `selection_record_sha256` (the full
record's canonical digest). The memory-regime digest binds
`{"cache_warmth":"unknown","protocol":null}` unless the review supplies a separate
protocol with `state` and `receipt_sha256`.

Each counter scope supplies exact `begin_pc`, `end_pc`, zero-based `occurrence`
and `evidence_sha256` for the independent original-ELF boundary review. Preparation
exports `merlin.spike_scope_binding.v1` objects for native Spike's
`--timing-scope-binding=FILE` with `scope.kind=counter_pc_pair`. Internal counter
reads cannot replace those boundaries. Addresses are execution bindings, separate
from the common frozen pricing file, so different linked ELFs retain one model
identity. Preparation does not recompile, replace or stage a different ELF.

The embedded input context is the canonical digest of
`{"schema":"merlin.embedded_input_context.v1","harness_sha256":H}` for the exact
original caller. This conservatively binds the complete generated caller, not
only its tensor bytes. It may decline comparisons between otherwise equivalent
callers; it cannot join changed inputs by shape alone.

Source visibility, sealed source manifests, mechanism families and splits need
independent operator admission; archive names do not establish them. Freeze
source/family/shape assignments before fitting. Every known repeat stays with its
case and source group. Omitting one repeat's observation makes the entire case
unavailable, rather than selecting the faster measured label. Unavailable cases
remain in the held-out coverage denominator.

`original-observations.json` has schema `merlin.phase1.spike_observations.v1`,
`plan_sha256`, and a `reports` object mapping plan `record_id` values to original
native report envelopes. Known registered feature counters include explicit
zeros. An arbitrary absent feature is unknown and is never filled with zero.

The handoff binds the immutable archive, dataset, split manifest, selected Spike
executable/configuration, model artifact and training code. Its held-out report
includes per-case absolute/relative errors and shape/family breakdowns. Selection
hashes do not independently prove which executable ran or which code fitted a
model; original process and fitting custody remain required. Missing observations
or failed gates produce a diagnostic handoff with ranking disabled. Qualification
from these labels applies to the declared gSIM capsule-call domain. It establishes
neither FireSim cycle accuracy nor full-model performance.
