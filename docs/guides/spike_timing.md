---
title: Calibrating and validating annotated timing
kind: guide
status: current
owner: core
last_verified: 2026-10-15
related: [architecture, performance_budget_unit]
code_refs: [packages/merlin-timing/src/merlin_timing/contracts.py, packages/merlin-timing/src/merlin_timing/fitting.py, packages/merlin-timing/src/merlin_timing/validation.py, packages/merlin-timing/src/merlin_timing/design.py, packages/merlin-timing/src/merlin_timing/feedback.py, packages/merlin-timing/src/merlin_timing/cli.py, packages/merlin-timing/src/merlin_timing/archive.py, packages/merlin-timing/src/merlin_timing/provider_bundle.py, packages/merlin-experiments/src/merlin_experiments/phase1/rtl_timing_facts.py]
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

The archive intake accepts `merlin.phase1.capsule_timing_archive.v2` from
`merlin_experiments.phase1.capsule_timing_archive`. Supply the archive SHA256
from the immutable producer handoff, rather than substituting a locally invented
source or simulator identity:

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

### Structural facts for the selected 16 GiB gSIM target

The operator can extract one pinned, machine-readable `rtl_facts.json` from the
matching Chipyard source and generated gSIM FIRRTL/DTS/L2 metadata:

```sh
python3 packages/merlin-experiments/src/merlin_experiments/phase1/rtl_timing_facts.py \
  --chipyard-source CHIPYARD_DIR --generated GENERATED_DIR \
  --firrtl-sha256 SELECTED_FIRRTL_SHA256 --dts-sha256 SELECTED_DTS_SHA256 \
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
