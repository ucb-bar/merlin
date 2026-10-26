# Run the committed FireSim ELFs

For new ResNet/TinyLlama DSE start with the
[published delivery README](../../../../out/artifacts/handoff/gemmini-firesim-dse-publication/v1/handoff_gemmini-firesim-dse-publication_v1_20261026T173908Z_238ada8/README.md).
The same `comparisons/tvm-workload-tuning` branch contains the new baseline ELF,
all source changes and the complete input bundle in checksum-verified parts.
Its restore command needs Python 3.12+, without a compiler or TVM installation.
Do not apply the old archived source patch over the current branch.

[FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md) records current scope and
pending TinyLlama/observer gates. The six ELFs described below retain their
historical numerical identities.

These committed executables reproduce the historical numerical programs. All
new candidates/releases must follow [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md).
For the next profile-directed ResNet study read [TUNING_PLAN.md](TUNING_PLAN.md).
Before more paid DSE, qualify [live active-cycle enforcement](ADAPTER_CONTRACT.md);
the October 26 adapter's completed-result filter did not implement that policy.

The prepared ELFs are now **committed to the Merlin handoff branch**. Clone
only Merlin to run them; no separate ELF download or TVM checkout is required.

```bash
git clone --branch comparisons/tvm-workload-tuning https://github.com/ucb-bar/merlin.git merlin
export MERLIN="$PWD/merlin"
export DELIVERY="$MERLIN/out/artifacts/handoff/gemmini-firesim-elfs/v1/latest"
cd "$DELIVERY"
sha256sum -c SHA256SUMS
```

Use the existing ELF launcher and supplied **DDR3** FireSim bitstream. Its memory
model is already set. No compiler, Python environment, bitstream rebuild or
memory configuration is needed to execute these binaries.

## Included binaries

| File relative to DELIVERY | Role |
| --- | --- |
| ELFS/resnet-baseline.elf | Full ResNet with default schedules |
| ELFS/resnet-tuned.elf | Full ResNet with our current validated tuned values |
| ELFS/resnet-tuned-profiled.elf | Optional stage/operator attribution |
| ELFS/tinyllama-decode-kv.elf | Small decode K/V pair, full K=2048 |
| ELFS/tinyllama-prefill-kv.elf | Small 32-row prefill K/V pair, full K=2048 |
| ELFS/tinyllama-decode-down-panel.elf | Small output panel, full K=5632 |

See the [committed payload manifest](../../../../out/artifacts/handoff/gemmini-firesim-elfs/v1/handoff_gemmini-firesim-elfs_v1_20261026T045007Z_04f3e10/elf-manifest.json) for
sizes, hashes and scope. ELF bytes are unchanged from the positive/wrong-oracle
Spike checks. No new FireSim performance result is claimed.

The TinyLlama files are trained-weight W8A8 chunks, not full-model ELFs or the
calibrated EL4 numerical release. Full TinyLlama baseline/tuned E2E ELFs must
still be supplied or built from the actual remote exports.

## Run order and timing policy

1. TinyLlama baseline plus **ResNet baseline and our current tuned ResNet**.
2. TinyLlama chunk ML DSE.
3. TinyLlama tuned E2E with the same inputs/protocol as its baseline.

Run independent ready jobs on available slots. For DSE, stop a candidate when
active measured cycles reach the current best, excluding startup/loading/
initialization/warmup and tightening live limits. Keep the shared four-hour
maximum DSE deadline. No F2 specification or execution duration is assumed.
See [FIRESIM_CHUNK_TUNING.md](FIRESIM_CHUNK_TUNING.md) for the full protocol.

## Compiler sources are optional for these runs

The TVM branch is only needed to generate/re-tune candidates, not to run the
committed ELFs. It is a **custom Gemmini fork**, not unmodified upstream TVM.

- Upstream machinery: task extraction, EvolutionarySearch, gradient task
  scheduling, XGBoost, schedule traces and JSONDatabase.
- Custom extensions: Gemmini intrinsics/schedule rules, mathematical feature
  adapter, static packaging, profiling and measurement adapters. These use
  TVM extension APIs; they are not upstream Gemmini support. The fork also
  carries earlier framework correctness fixes described in GENERATION.md.
- Recent fork changes: optional profiling, prepared-graph equality audits,
  compact search/host admission, and a scalar input-binding fix.
- A historical Verilator-only watchdog rewrites a generated C guest template
  using string replacement. That brittle instrumentation is not enabled in the
  FireSim builder and is not part of executing this ELF handoff. Do not use it
  as a FireSim mechanism or runtime-patch TVM optimizer methods.

The published companion source is grammy117/tvm, branch gemmini/firesim-chunks,
commit 3423bd6818ab752ddd9cdb5d5773743e48c65f1d. Its supported extension APIs are
different from claiming that the whole integration is stock/upstream.
For future DSE, prepare actual model slices and the matching compiler separately.
The earlier tuning-input archive remains optional; it is not required for these
ELF runs. Do not apply old source patches on top of the published branches.
