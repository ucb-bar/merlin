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
`capsule_timing_archive_status.json` in the run directory. If every passing gSIM
L3 result has a positive label, matching kernel-scope caller marker, original
ELF, input, numeric and simulator artifacts, and an RTL-derived/cycle-accurate/pass verdict,
it also writes `capsule_timing.tar`. The operator can rerun the host-only audit
or export on a retained run:

```sh
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR
python -m merlin_experiments.phase1.capsule_timing_archive RUN_DIR --out RUN_DIR/capsule_timing.tar
```

The export refuses an existing destination or incomplete timing. The status
file names the reason and records the tar's SHA256 when ready; a missing tar is
not a measured calibration dataset. A resumed run with new capsule records gets
a new content-named tar, preserving the earlier archive.

## Archive format (`merlin.phase1.capsule_timing_archive.v2`)

The uncompressed tar has `index.json` and every file from every retained
`_qa_work/**/runs/<suite>/<capsule>/` directory under `run/<relative-path>`.
This includes failed and unavailable capsule executions for audit, even though
only passing gSIM records can train the timing model. `index.json.records` has
one row per `capsule_result.json`, including repeated paid executions:

| Field | Meaning |
| --- | --- |
| `result`, `capsule`, `status`, `result_status`, `engine` | Exact run-relative result path and L3/overall verdicts |
| `cycles`, `cycles_measured`, `console_cycle_verified`, `cycle_scope`, `timing_selection_sha256`, `timing_selection` | L3 counter delta, exactly one matching RTL console marker, measured call boundary, reviewed selection identity and full pinned counter/clock record |
| `shape`, `command_buffer_sha256`, `candidate_sha256` | Logical tensor shapes/dtypes, command buffer and submitted compiler identities |
| `elf`, `elf_sha256`, `harness_sha256` | Exact linked program and generated caller identities |
| `artifacts` | Run-relative path, SHA256 and byte count for ELF, caller, input interface, command buffer, numeric report, static instruction trace, and both console logs when present |
| `sim_provenance`, `derived_from_rtl`, `cycle_accurate`, `timing` | Simulator binary/provenance, L3 evidence class and host wall durations |

`index.json` also contains `capsule_records`, `gsim_pass_records`,
`gsim_pass_with_cycles`, `gsim_pass_complete`, `timing_ready`, and
`timing_target_scope: selected_gsim_elaboration_only`. A consumer
must verify tar members against indexed SHA256 values before use. The original
ELF embeds the harness's exact input bytes; the archived `harness.c`,
`input.interface.mlir`, `command_buffer.json`, result and logs expose the input
and compilation context. The archived instruction trace is a **static** trace;
it must not be treated as a dynamic event timeline.

## Spike calibration and Phase 2 handoff

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
prediction scope. Phase 2 may use the tuned model only for that scope and must
keep gSIM as the source of actual RTL timing evidence. Full-model executions are
outside this capsule dataset.

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

The branch's shipped `examples/gemmini/timing/stock_u250.json` targets archive
SHA256 `a9a190b9…`, whereas this experiment selects the 30 MHz archive
`0af6f26e…`. Its primitive prices and queue capacities are unknown. Re-export
and pin a profile for the selected `0af6f26e…` bitstream before fitting or
deploying it; matching ABI header `3758ae96…` does not make the two bitstreams
one timing target. Launch Spike with the selected 16 GiB map beginning at
`0x80000000` for these ELFs. Gemmini's local `address_bits: 32` parameter is
not the Rocket physical-memory size. The model's README example `-m64` is a
small test allocation, not this board's memory selection.
The Phase 1 numerical oracle remains the selected stock functional Spike and
Gemmini extension; the annotated observer is a separate Phase 2 timing consumer
of the archived ELF/input records.

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
