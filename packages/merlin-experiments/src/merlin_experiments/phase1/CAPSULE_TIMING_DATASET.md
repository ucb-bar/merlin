# Phase 1 capsule timing dataset for the Phase 2 Spike model

The training label is **one gSIM host-core cycle delta for one capsule's original
linked ELF and input**, measured immediately before `kernel_entry` through the
selected `kernel_complete` call. It includes the kernel and its completion wait.
Input staging, output serialization, metric printing and parent-side readback are
outside the bracket. The label is a whole capsule kernel-call cost, not an
individual Gemmini command, instruction, model layer or whole-model latency.
The counter/clock semantics must come from the selected contract; `rdcycle`
spelling alone does not establish the clock or its relationship to an internal
Gemmini counter. Cache warmth is recorded as unknown unless a separate protocol
establishes it.

## Collection

The release-selected capability contract must opt in with
`logical_harness.kernel_call_timing` (`merlin.kernel_call_timing.v1`), naming an
existing `counter_semantics` record and an explicit positive `max_delta_cycles`
below 2^63. The record declares `unit: cycles`, `width_bits: 64`, `wrap: modulo`,
the clock name, and SHA256-pinned original accessor/ISA/clock source files.
`logical_harness.cycle_counter_asm` supplies the reviewed instruction. There is
no implicit target default. The runtime renderer then emits one `METRIC cycles
<positive integer>` between completion and output publication. The normal
capsule result stores it at `tiers.L3.cycles`; missing/nonpositive markers become
null. `tiers.L3.timing` separately records host wall durations in seconds.
The kernel-call selection and `complete_call_timing` are mutually exclusive for
this caller because both publish the same `METRIC cycles` name.

Phase 1 retains ordinary agent selfcheck and simulator-job raw runs under
`RUN_DIR/_qa_work/selfchecks/selfcheck_*/runs/`, alongside periodic grading runs.
The host broker selects this location before launching workers; requests cannot
choose it. It must be outside the visible workspace, with no symlink components.
Public `selfcheck_out` files remain redacted copies. Both successful and failed
raw capsule executions are discoverable by the same archive audit; retention
does not make a failed or unmeasured execution eligible for calibration.

The selected contract block has this shape; the clock and source pins must be
filled from reviewed original sources before a release is frozen:

```yaml
logical_harness:
  cycle_counter_asm: "<reviewed counter instruction with %0 output>"
  kernel_call_timing:
    schema: merlin.kernel_call_timing.v1
    counter_semantics: selected_host_cycle_counter
    max_delta_cycles: <positive integer below 2^63>
counter_semantics:
  selected_host_cycle_counter:
    unit: cycles
    width_bits: 64
    wrap: modulo
    clock: <reviewed original clock identity>
    source_pins:
      - {path: /absolute/original/source, sha256: <64 lowercase hex digits>}
```

At completion, Phase 1 writes `capsule_timing_audit.json` and
`capsule_timing_archive_status.json` in the run directory. The audit indexes
authoring-time results under `_qa_work/**/runs/` and the formal
`grading_public/runs/` and `grading_hidden/runs/` suites. Each row carries its
`area` and `visibility` (`public`, or `hidden` for `grading_hidden`).

Training eligibility is decided **per record**, independently of the capsule's
overall verdict: a capsule may fail a host-side or structural rule while its
gSIM tier measured and numerically verified the same bytes. A record is
`training_eligible` when its gSIM tier passed and was measured now (not
carried), its cycles are positive, RTL-derived and cycle-accurate, its caller
carries exactly one kernel-window selection, its RTL console has exactly one
matching marker, its ELF, caller, compiler, command-buffer and simulator
identities and its input/numeric/trace/console artifacts are present, and the
current ELF is the one the gSIM process actually executed. Otherwise
`training_exclusions` names every reason. Hidden records are counted but never
enter a public calibration subset.

When at least one public record is eligible, finalization writes
`capsule_timing_training_<digest>.tar` (`export_scope: training_eligible_public`).
It contains only the indexed evidence (result, artifacts) of the eligible public
records plus every executed public repeat of the same ELF, whatever its verdict,
so a consumer cannot drop a slower or failed repeat. `source_audit` records the
full audit's counts and digest. If every passing gSIM record is also attributed
to a passing capsule, finalization additionally writes the full-directory
`capsule_timing.tar` (`timing_ready`), as before. The operator can rerun the
host-only audit or exports on a retained run, and `--finalize` performs the
complete finalization for a run whose supervisor stopped before doing so:

```sh
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR --training-out TRAINING.tar
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR --out RUN_DIR/capsule_timing.tar
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR --finalize
```

Ordinary capsule producers also retain the exact compiler tree under
`generated/compiler-source/<sha256>/`. Their `compiler_source` receipt checks the
source before lowering and afterward and binds the emitted ELF, harness, input
interface and command buffer. The archive verifies this retained source and its
artifact hashes; it never infers source identity from a mutable submission path.

Spike reads a capsule's outputs from its console and gSIM from a memory dump, so
the two tiers link different callers around the same kernel object into one
`generated/` directory, and the later link overwrites the earlier ELF. Every
completed simulator run therefore copies the exact ELF and caller it executed to
`generated/executed/<engine>/<elf-sha256>/` with an `executed_program.json`
record (engine binary digest, ELF, caller and kernel-object digests). The
receipt's `executed_programs` lists this invocation's records in `capsule_result.json`
(an earlier invocation's copies are removed when the receipt is captured), and the
build cache never stores this subtree. The audit joins each tier to the record
whose engine digest is that tier's simulator binary and indexes the copies as
`L2_executed_elf`, `L2_executed_harness`, `L3_executed_elf` and
`L3_executed_harness`, so the training subset carries the Spike-executed bytes
beside the gSIM-labelled ones. Older runs fall back to the retained process
records, which identify both programs by digest but kept only the last one's bytes.

A carried simulator certificate is labelled `measured_now: false`, retains its
original evidence reference and contributes to `gsim_reused_records`. It is not
a new timing observation. If strict training export is incomplete, finalization
still writes a separate content-named `capsule_evidence_*.tar` with all diagnostic
artifacts and the actual readiness status. This preserves paid execution data
without turning failed, incomplete or unattributed records into training labels.

The export refuses an existing destination or incomplete timing. The status
file names the reason and records the tar's SHA256 when ready; a missing tar is
not a measured calibration dataset. A resumed run with new capsule records gets
a new content-named tar, preserving the earlier archive.

## Archive format (`merlin.phase1.capsule_timing_archive.v3`)

The full and diagnostic tars have `index.json` and every file from every indexed
capsule directory under `run/<relative-path>`. This includes failed and
unavailable capsule executions for audit, even though only eligible gSIM records
can train the timing model. The training tar holds only indexed members.
`index.json.records` has one row per `capsule_result.json`, including repeated
paid executions:

| Field | Meaning |
| --- | --- |
| `result`, `capsule`, `status`, `result_status`, `result_failure`, `engine` | Exact run-relative result path, L3/overall verdicts and the overall failure category |
| `area`, `visibility` | Which result area held it (`authoring`, `grading_public`, `grading_hidden`) and whether it may enter public calibration |
| `cycles`, `cycles_measured`, `console_cycle_verified`, `cycle_scope`, `timing_selection_sha256`, `timing_selection` | L3 counter delta, exactly one matching RTL console marker, measured call boundary, reviewed selection identity and full pinned counter/clock record |
| `shape`, `command_buffer_sha256`, `candidate_sha256` | Logical tensor shapes/dtypes, command buffer and submitted compiler identities |
| `elf`, `elf_sha256`, `harness_sha256` | Exact linked program and generated caller identities |
| `artifacts` | Run-relative path, SHA256 and byte count for ELF, caller, kernel object, input interface, command buffer, numeric report, static instruction trace, and both console logs when present |
| `sim_provenance`, `derived_from_rtl`, `cycle_accurate`, `timing` | Simulator binary/provenance, L3 evidence class and host wall durations |
| `executions` | Per tier (`L2`, `L3`): the ELF, caller and kernel-object digests that tier's simulator binary executed, from the result's `executed_programs` record (`source: capsule_result`, with the retained copies) or else recovered from the capsule's retained process records (`source: invocation_records`) |
| `join` | Pairing keys: gSIM and Spike ELF digests, their kernel objects, whether they are the same bytes, the timing selection, a proposed source group (capsule plus its frozen input interface) and visibility |
| `training_eligible`, `training_exclusions` | The per-record verdict above and every reason it failed |

`index.json` also contains `capsule_records`, `records_by_area`,
`gsim_pass_records`, `gsim_pass_with_cycles`, `gsim_pass_complete`,
`timing_ready`, `training_eligible_records`, `training_eligible_public_records`,
`training_eligible_hidden_records`, `training_exclusions`, `training_ready`, and
`timing_target_scope: selected_gsim_elaboration_only`. A consumer
must verify tar members against indexed SHA256 values before use. The original
ELF embeds the harness's exact input bytes; the archived `harness.c`,
`input.interface.mlir`, `command_buffer.json`, result and logs expose the input
and compilation context. The archived instruction trace is a **static** trace;
it must not be treated as a dynamic event timeline.
`timing_ready` covers the retained passing gSIM L3 results. It does not assert
that every capsule in the admitted corpus ran at L3. Compare the archive's
capsule roster with the frozen corpus and grading manifest before describing a
dataset as an all-capsule measurement sweep. Failed or screened-only capsules
have no usable timing label.

## Annotated-Spike event counts (evidence only)

After a run finishes (never during authoring), the operator can observe every
training-eligible public gSIM program on the observer-enabled Spike:

```sh
python -m merlin_experiments.phase1.annotated_spike_evidence RUN_DIR --config CONFIG.json --jobs 4
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR --finalize
```

`CONFIG.json` (`merlin.annotated_spike_evidence_config.v1`) names the annotated
Spike and its timing-source manifest, the built timing provider and its source
bundle manifest, the exported device profile (optionally with RTL structural
facts placed by `spike-timing apply-capacities`; its fully known resources become
engine resources), a disassembler and the target
contract. The functional configuration is not configured: it is the exact argv
of the run's own functional-tier launches, recovered from their process records
(all launches must agree). The pass derives each program's counter window from
its linked ELF -- the one function calling the contract's entry and completion
symbols must contain exactly two reads matching the caller's reviewed
`counter_asm`, one before the entry call and one after the completion call --
and writes a `merlin.counter_window_review.v1` record and the
`merlin.spike_scope_binding.v1` that cites its digest. It then runs the program
twice, on the functional configuration and on the annotated Spike with the
provider loaded after the functional extension and a count-only model (every
price unknown, accelerator resources unbound), and requires byte-identical
consoles and exit codes. Where the run kept the functional tier's own program,
both reruns must also reproduce its retained functional console, which carries
the full output values. Evidence lands in
`<capsule>/annotated_spike/gsim_program/` (and `spike_program/`) beside
`annotated_spike_evidence.json`; the audit indexes the report and counter review
as artifacts only while they name the ELF still in place, so finalizing again
carries them into the training subset.

The report's `feature_counts` are the selected window's per-class event counts
(`target.commands`, requested DMA bytes, padded and useful MACs,
`target.class.<primitive>`, `cpu.<class>`), and its window instruction count
equals the functional `METRIC cycles` (Spike's counter is instret). With no
prices and unbound resources the report is honestly `timing_complete: false`
and lists every missing price, resource and semantic; the counts are features,
not a timing estimate.

## Spike calibration and Phase 2 handoff

The intended Phase 2 model is annotated Spike **with its Gemmini accelerator
observer**. Calibrate in two stages. First, derive the architectural structure
from the *elaborated, selected* RTL and its pinned source/configuration: mesh
geometry, scratchpad and accumulator banks/ports, reservation-station entries,
command-queue depths and issue rules, DMA maximum transfer and outstanding
request limits, cache geometry, and the CPU/Gemmini clock relationship. The
selected INT8 [GemminiConfigs.defaultConfig](https://github.com/copparihollmann/gemmini/blob/63f0b68a68f185a016a2993f2fb32e1661ea9e6c/src/main/scala/gemmini/Configs.scala)
source declares a 16x16 mesh,
4 scratchpad banks, 2 accumulator banks, 8/16/4 load/execute/store reservation
entries, 8/8/2 corresponding command-queue depths, 16 outstanding memory
requests and a 64-byte DMA maximum. Verify those effective values against the
FireSim-aligned gSIM elaboration rather than assuming no config override.
Record each fact's generated artifact or source path, SHA256, derivation and
mapping into the timing model. A hardware queue entry is not automatically a
generic observer resource lane; match the model field's admission/issue
semantics before assigning a number.

The existing FireSim-aligned gSIM receipt already contains a generated FIRRTL
(`6748b9756799c086a6fb06e6d80b4b052f2771f9eb1858c172e14dcf441ba149`)
and DTS (`c4a2009a405945ea98f731713e0f4e7dc483031ab8cd5156d871e4eb7594f1df`).
In the FIRRTL `ReservationStation` module, instantiated `entries_ld[0..7]`,
`entries_ex[0..15]` and `entries_st[0..3]` corroborate those source entry
counts; `LoadController` and `StoreController` instantiate the 8- and 2-entry
command queue modules. The DTS declares 16 GiB RAM at `0x80000000`, 32 KiB
I/D caches and a 512 KiB L2. A reproducible fact extractor should read the
pinned config chain and generated artifacts, emit a machine-readable
`rtl_facts.json` with each value, unit, source digest and evidence selector,
then check that source and elaboration agree. It should refuse a differing
generated design instead of silently filling the Spike profile from source
defaults. The current `fire_aligned_30mhz.json` is not yet populated this way.
The DTS's CPU `clock-frequency = <0>` is not a usable guest-cycle frequency;
clock/counter semantics need their own reviewed sources and timing contract.

Map each RTL fact to the behavior it limits, rather than matching field names:

| RTL fact | Timing-model role |
| --- | --- |
| 16x16 mesh, dataflow and type widths | Decode command work and legal packet shapes; calculate tile count, bytes and MACs. |
| 8/16/4 load/execute/store reservation entries | Per-class **live-command credits**, held from allocation until completion. |
| 8/8/2 controller command queues | Buffers after reservation-station issue, distinct from live-command credits. |
| Reservation-station issue logic | Per-class issue arbitration and dependency readiness. |
| 4 scratchpad and 2 accumulator banks/ports | Address-to-bank mapping and port conflicts for local-memory phases. |
| 16 outstanding memory requests and 64-byte DMA maximum | Split transfers into bounded requests; return credits on responses and share memory service. |
| 32 KiB I/D caches, 512 KiB L2, 16 GiB map | Cache geometry, residency and legal guest addresses; fit miss service separately. |

The current generic engine's `ResourceSpec.capacity` means simultaneous
occupancy lanes and `queue_capacity` means reservations waiting to *start*;
neither is an RTL reservation-station entry held from allocation through
completion. Set an issue-port resource's occupancy to its initiation interval,
not its full command latency, so independent commands can overlap. A small
Gemmini-specific scheduler should separately track reservation credits,
controller queues, banks and DMA credits, and publish command completion to
the generic dependency engine. Until those roles exist, use a plainly labeled
coarse approximation and validate its **ranking** on held-out matched variants;
do not put `8/16/4` directly into the generic lane or wait-queue fields and
present that as a faithful RTL mapping.

Second, fit only the remaining primitive latency/occupancy, transfer,
host/cache and memory-service prices against the gSIM whole-call labels and
targeted diagnostics. RTL pipeline structure can constrain minimum latencies,
but queueing, cache warmth, contention and effective DRAM service depend on the
executed workload. The labels constrain combined behavior; they do not identify
an individual command latency by themselves. Keep unsupported or
unidentifiable parameters unknown, and use held-out capsule families to test the
result. `merlin_timing.fitting.fit` currently produces a diagnostic linear
feature screen, not a calibrated Gemmini event-model configuration. The event
model still needs an explicit parameter-selection/fitting step before Phase 2
can use it as a tuned Spike timing model.

Use one passing, complete gSIM L3 row as one supervised example. Run the
archived ELF with the same input on the selected Spike configuration to produce
the model's features or prediction. Pair by ELF SHA256, capsule/shape,
candidate SHA256 and input identity; reject pairs with a changed ELF, input,
simulator selection or correctness verdict. Preserve repeated executions as
replicates, not independent train/test shapes. Split by capsule/kernel family
and shape before fitting, with all repeats of one identity in one split. Report
error on unseen held-out capsules, including absolute and relative cycle error,
and stratify by shape/operation family. A per-command coefficient fitted from
a whole-call label is a model parameter, not a measured per-command latency.

The Phase 2 handoff consists of this immutable tar and its SHA256, the selected
Spike executable/configuration identities, the fitted model artifact and
training code identities, the split manifest, holdout report, and the declared
prediction scope. Phase 2 may use the tuned model only for that scope. The
holdout labels come from gSIM, so its predictions estimate the selected gSIM
elaboration's capsule-call cycles. U250 measurements are not a prerequisite for
this handoff; they are required only to claim or calibrate board cycles. Full-model
executions are outside this capsule dataset.

## Dynamic command graph and efficient replay

Nicolas Rakanic's [MX-Gemmini performance extension](https://github.com/ucb-bar/libgemmini/tree/c17f961816b5ab48a447f7cd7abda0e5d2445e5d)
is a useful design reference, not a parameter set for this INT8 target. Its
[reservation station](https://github.com/ucb-bar/libgemmini/blob/c17f961816b5ab48a447f7cd7abda0e5d2445e5d/perf/control/reservation_station.cc)
derives dependencies from local row read/write spans and live queue entries;
load, execute, store and vector queues have distinct issue rules. Its
[event queue](https://github.com/ucb-bar/libgemmini/blob/c17f961816b5ab48a447f7cd7abda0e5d2445e5d/perf/sim/event_queue.h)
advances only to scheduled activity and separates same-cycle evaluation from
state commit. Its [model trace](https://github.com/ucb-bar/libgemmini/blob/c17f961816b5ab48a447f7cd7abda0e5d2445e5d/perf/model.cc)
can emit command allocation, issue, completion and row spans plus selected
unit events. These are model events to compare with RTL diagnostics, not measured
gSIM command timestamps.

For the Phase 1 corpus, preserve the existing exact ELF/input and single
kernel-call gSIM cycle label as the primary training record. On replay in Spike,
derive a **compact dynamic command graph** for each passing capsule: RoCC
instruction order and PC, funct and operand words, expanded loop commands where
used, decoded shape/bytes/MACs, local row read/write ranges, dependency edges,
fence/wait position, unknown decoding reasons and the selected counter-window
membership. Bind the graph to ELF, input, target-config and observer hashes.
This graph is a reconstruction from the functional execution, not a hardware
timestamp trace; keep its schema/version separate from the measured label.
The selected Phase 1 command contract excludes loop/FSM functs. Verify that
with a pre-launch replay census of the compiled capsules; if the census holds,
the current observer's rejection of those functs is irrelevant to this corpus.
The MX model's loop expansion remains a reference only for a future command
contract that admits them.

For a bounded diagnostic subset, add Gemmini performance-counter snapshots or
RTL event captures to separate load, compute, store, overlap, queue waits and
DMA traffic. Record the counter selector, start/end scope and perturbation check
for each diagnostic run. Hardware exposes fewer simultaneous counter slots than
event kinds, so selected counter groups may require repeated runs; do not merge
different runs into a single supposedly exact command timeline. Keep verbose
per-command or per-cycle RTL traces only for outliers and validation
microbenchmarks. The primary gSIM label must keep its reviewed uninstrumented
measurement window.

The MX extension's [three modes](https://github.com/ucb-bar/libgemmini/blob/c17f961816b5ab48a447f7cd7abda0e5d2445e5d/gemmini_perf.cc)
illustrate a performance tradeoff: `func` runs the functional extension,
`both` also runs the timing model, and `perf` skips data-moving and computing
functional commands. `perf` can accelerate timing-only sweeps when those
commands cannot affect later control flow or address generation, but it cannot
serve as the numerical oracle. Measure `func`, `both` and any timing-only mode on
the *same* Phase 1 capsules, recording wall time, command count, timing
completeness and cycle prediction; require functional equivalence separately.
The reported 100x speedup has no published comparable benchmark in this source
revision and must be treated as a claim until the selected build and workload
reproduce it.

## Qualification for Phase 2 optimization ranking

Phase 2 primarily needs to choose among candidate implementations of the
**same** workload, input and measurement scope. One measured implementation
per capsule is insufficient to test that decision, even if many different
capsules have labels. Before the Phase 1 measurement window closes, retain
several numerically passing ELFs per representative capsule: a baseline and
controlled variants of tiling, data reuse, transfer schedule, fence placement
and CPU/accelerator partition where the command contract permits them. Keep
these calibration/validation variants separate from the scored agent submission.
Run each variant on the same pinned gSIM, hash its ELF and input, and archive
its compiler provenance and kernel-call cycle label. If the Phase 1 agent
naturally produced multiple retained candidate ELFs, use them too; do not
assume that the final capsule archive contains enough matched alternatives.

Choose variants and shapes that separately stress command count, DMA bytes,
mesh work, local-memory reuse, queue pressure and host work, with both small
and large cases. This gives the fit a chance to distinguish fixed command
cost from per-byte and per-MAC cost. Read hardware geometry, capacities and
issue rules from the pinned RTL; fit only a small set of remaining timing
prices. Store every trial's parameter file and training/validation membership
so the selected model can be reproduced. Repeated gSIM runs on a subset should
measure label stability; near-tied candidates need more repeats or should be
reported as ties.

Hold out entire workload/shape groups before fitting, with each mechanism
family represented in both train and test; reserve some transformation types
for a separate extrapolation stress test. Evaluate within-workload pair ordering,
sign of predicted speedup,
top-choice regret and coverage, broken down by mechanism and by measured
speedup size. The existing `merlin_timing.validation` gate requires 90% order
agreement for held pairs separated by more than 5%, at least 20 eligible pairs
across three mechanism families, plus absolute-error limits. Report those
limits without silently weakening its existing qualification status. A separate
Phase 2 ranking decision may weigh held-out ordering and measured regret more
heavily than absolute cycle error. Preserve an uncertainty/abstention rule:
when predicted alternatives
are closer than observed label noise or the held-out error band, ask the LLM
to treat them as tied or request another measurement, not claim a winner.
Do not claim rank validation if the corpus supplies fewer than the required
matched pairs; cross-workload comparisons do not replace them.

The annotated Spike model at
[`806ec0adda22`](https://github.com/copparihollmann/riscv-isa-sim-gemmini/tree/806ec0adda2242597b253ebf17adfaa5cdb395e3)
accepts a `counter_pc_pair` scope binding with exact `elf_sha256`,
`input_sha256`, `scope_sha256`, opening/closing counter PCs, occurrence and an
independent boundary-review digest. The current timing archive retains the ELF,
generated caller, input interface and selected counter semantics, but does not
issue that binding or a separately defined digest of the embedded input tensor
bytes. The Phase 2 importer must verify the archived artifacts, define and
rederive that input digest, inspect the linked ELF to bind the two counter PCs
around `kernel_entry` through the selected completion call, and retain the
independent review evidence. A whole-program Spike estimate or a consecutive
counter-pair guess is not the capsule label's scope.

The annotated Spike branch now has a separate `fire_aligned_30mhz.json` profile
associated with archive SHA256 `0af6f26e…`, with the FireSim-selected `5b1254a7…`
functional receiver. The older `stock_u250.json` belongs to `a9a190b9…`.
The new profile's primitive prices and queue capacities are currently null.
Populate RTL-derived facts before fitting workload-dependent prices, and keep
any field without a defensible mapping unknown. Fitting to gSIM labels must
bind the gSIM executable, FIRRTL
and counter scope as the timing target; the archive association is only source
context. Matching ABI header `3758ae96…` does not make the two bitstreams
one timing target. Launch Spike with the selected 16 GiB map beginning at
`0x80000000` for these ELFs. Gemmini's local `address_bits: 32` parameter is
not the Rocket physical-memory size. The model's README example `-m64` is a
small test allocation, not this board's memory selection.
The next Phase 1 numerical oracle must select the exact `fire_aligned_30mhz`
functional Spike package and receiver. The annotated observer is a separate
Phase 2 timing consumer of the archived ELF/input records, enabled only after
calibration and held-out validation for the selected gSIM elaboration.

The compiler agent receives the selected Phase 1 contract and RTL facts through
the frozen input bundle. The Gemmini task grant also contains
`task/PHASE2_TARGET.md`, which names the planned full U250
`FireSimGemminiRocketConfig` bitstream, board address map and ABI identity.
Those facts are useful target context. Historical Phase 1 facts describe the
older `GemminiRocketConfig` elaboration. The next release must instead select
facts and capsule labels from the pinned FireSim-source
`GemminiGsimFireSimAlignedConfig` gSIM build. A model fitted to those labels
must declare its exact gSIM design identity; transferring it to FireSim still
requires a reviewed chip comparison and paired cycle measurements on the pinned
bitstream. The FireSim board name alone does not qualify Phase 2 predictions.

Derive Phase 0 facts and the Phase 1 gSIM build from the **same pinned Gemmini
generator revision** as the intended FireSim bitstream, with the FireSim chip
tweaks inside the gSIM harness. Pin the resulting FIRRTL, gSIM receipt,
contract, Spike extension and FireSim bitstream separately. Verify the chip
RTL/parameter and ABI match; then measure matched capsule ELFs on gSIM and the
board to calibrate any wrapper/clock/transport difference. Reusing a
configuration name or fitting old gSIM labels does not perform that verification.

The six whole-model Spike programs in the authored Phase 1 descriptor now select
`firesim_gemmini_rocket_u250_30mhz` directly. A fresh private model specification
must select that board, its FireSim DTS SHA256 and the updated board-catalog
SHA256. The operator `phase1.spike_alignment` check compares those inputs with
the full-U250 machine and exact Spike executable/extension, source commits and
ABI header before launch. Old R53b specifications still select the Verilator-DTS
functional board and cannot qualify this new selection.
