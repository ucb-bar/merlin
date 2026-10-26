# FireSim handoff: baseline runs, TinyLlama DSE, then E2E

**Current preparation:** [FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md)
supersedes the run roster below for new DSE. Its portable bundle contains 18
ResNet regions from the new INT32-bias source, structural ownership, shared
budgets and a TinyLlama missing-coverage roster. The older committed ELFs below
remain historical controls with their original numerical identities.

For the next preparation, [OPTIMIZATION_KNOBS.md](OPTIMIZATION_KNOBS.md)
distinguishes available graph/operator choices from proposed host/packing work.
Actual trained-weight ResNet source regions and independent fixtures are in the
restorable bundle at `out/artifacts/handoff/gemmini-firesim-dse-publication/v1/latest/`.
Follow its README and pending admission gates. The new baseline passed Spike
controls; FireSim execution and fixed-set accuracy remain NOT_RUN.

**2026-10-26 follow-up:** all new runs use
[CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md). The archive from
the completed F2 run now informs [TUNING_PLAN.md](TUNING_PLAN.md), including
profile-directed ResNet regions. The old checkpoint probes are dynamic-row
diagnostics; explicitly request `--legacy-diagnostic` to reproduce them, and
use frozen calibrated scales for new qualification.

**Runner limitation found in the raw evidence:** the supplied remote adapter
used a total-target-cycle cap and a completed-result filter. It did not perform
the live active-cycle cutoff described below. Before another paid DSE campaign,
qualify [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md). The current pool's v3 request
requires explicit active-measured/in-flight scopes and rejects the old adapter's
boolean-only response. Baseline execution of committed ELFs remains available.

For a clean remote checkout use [BRANCH_HANDOFF.md](BRANCH_HANDOFF.md). The
prepared ELFs are committed on the Merlin branch. Running them requires no
TVM checkout, compiler setup or separate ELF archive. Compiler sources are only
needed when generating new candidates; do not apply old patches again.

The supplied FireSim bitstream already contains the memory model. **DDR3 is the
expected memory model.** Deliver and run ELFs with the existing launcher; do not
configure memory latencies, rebuild the bitstream, or infer F2 specifications.
No runtime estimate, F2 clock, memory capacity, bandwidth or slot count is assumed.
No new FPGA execution has happened locally.

The committed delivery is
`out/artifacts/handoff/gemmini-firesim-elfs/v1/latest/`. Its `ELFS/` folder
contains the prepared executables; `elf-manifest.json` and `SHA256SUMS` record
their actual sizes/hashes. Model fixtures and compiler inputs are separate
optional inputs for generating new candidates, not runtime dependencies.

## Requested run order

1. **Baselines and current ResNet result:** run the full TinyLlama baseline,
   the full ResNet baseline, and **ResNet with our current tuned values**.
   Use the same inputs and measurement policy for each baseline/tuned pair.
   These independent jobs may run in parallel on available independent slots.
2. **TinyLlama chunk tuning:** use small meaningful fragments of the actual
   model and stock TVM's ML-guided DSE. Keep the full reduction dimensions and
   surrounding quantization/scaling. Reuse a matching existing trace only after
   freshly measuring it on the supplied bitstream.
3. **TinyLlama tuned E2E:** install structurally matching, validated winners and
   run the complete model with the same input/protocol as its baseline. Check
   all outputs and compare full active inference cycles.

TinyLlama tuning depends on its baseline; it need not wait for an independent
ResNet job to finish. E2E depends on chunk/full-shape confirmation. Use as much
available parallel capacity as possible without putting two jobs on one slot.
The profiled ResNet ELF is optional diagnostics, not a required fourth baseline.

## ELFs already prepared

| File in `ELFS/` | Role |
| --- | --- |
| `resnet-baseline.elf` | Full unprofiled ResNet, default schedules |
| `resnet-tuned.elf` | Full unprofiled ResNet with our validated classifier winner |
| `resnet-tuned-profiled.elf` | Optional stage/operator attribution |
| `tinyllama-decode-kv.elf` | Two M=1, N=256, K=2048 projections |
| `tinyllama-prefill-kv.elf` | Two M=32, N=256, K=2048 projections |
| `tinyllama-decode-down-panel.elf` | M=1, N=128 output panel, full K=5632 |

The three historical chunk guests include row quantization, INT8 GEMM, FP32 rescaling and
all-output checks. Their ELF sizes are 0.73–1.34 MiB, measured from the files;
this is not a prediction of execution time. Each warms once and measures once.
All six prepared guests passed positive and wrong-oracle Spike controls.

**Full TinyLlama baseline/tuned ELFs are not present locally.** On the new
machine use an existing matching full baseline ELF or build it once from the
actual calibrated prefill/decode exports. Build the tuned full ELF after DSE.
Both must use the selected **32-token prefill + 32 teacher-forced decode calls,
cache capacity 64, all 99 output arrays**, real carried state, one complete warm
and one measured session. An older 16+1 diagnostic is not this baseline.
Use the existing `build_package_session.py` / `tinyllama_session.py` path.

The prepared chunks use trained layer-0 weights from revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`, v00-derived FP32 boundaries and a
fresh dynamic W8A8 recipe. Their independent NumPy references check every output.
They reflect the computation but do not replace the calibrated EL4 numerical
release. Prefer exact remote export slices before paid tuning. Avoid paying
for duplicate campaigns on equivalent probe and actual-export IR.

## DSE is ML-guided, not a static sweep

TVM enumerates legal structural sketch types, then **EvolutionarySearch**
proposes and mutates tile/reuse/dataflow/pipeline/unroll decisions. Its
**XGBoost cost model ranks candidates**, using measured results to update its
predictions. Early trials bootstrap the model; later batches use learned
ranking. The gradient task scheduler allocates trials using extracted task
weights. The final choice is the fastest valid measured result.

This is stock `extract_tasks` / `tune_tasks`, EvolutionarySearch, XGB and
JSONDatabase. It is not an exhaustive parameter grid followed by picking the
best. Defaults and replayed historical traces are comparison points, not a
replacement optimizer. Very short searches give the model limited training
information; retain `cost-model.json` and `trials.jsonl` when assessing progress.

## Active-cycle cutoff and concurrency

- Establish a valid active-cycle incumbent for each distinct workload first.
  **Keep a candidate running only while its active measured cycles are less
  than the current best. Abort at active_cycles >= current_best.** Tighten
  in-flight limits when another valid candidate improves the incumbent.
- Exclude boot, ELF loading, initialization and warmup from that comparison.
  Use the guest's measured `model_run` window, including CPU glue and the final
  completion fence. A long startup is not a reason to reject a faster candidate.
  Failed, equal, slower and aborted candidates are not improvement timings.
- The installed runner must observe/enforce this during execution; rejecting
  a completed result alone does not save compute. The ELF reports the completed
  interval; live observation comes from the existing runner. Do not substitute
  total-job cycles or elapsed wall time for active cycles.
- Start with batches of four and 12 proposals per chunk. Explore more as
  throughput and improvements permit, within **one shared absolute deadline,
  at most the user's four-hour DSE budget**. Do not renew it per process.
  There is no guessed per-job seconds/cycle/ELF-size ceiling.
- Keep managers and images running. Batch builds/submissions and reuse exact
  immutable payloads where supported. Parallelize across all actually available
  independent slots; share a physical-slot lock across every launcher. Size CPU
  build concurrency from the new machine's actual free resources. No F2 or host
  capacity is inferred here. Native batching does not combine variants into one
  ELF. Full-model cache dependencies remain serial within each session.

## Optional: generate and tune candidates remotely

This section concerns building new candidates, not executing the committed ELFs.
The Gemmini backend/feature adapter/packaging are custom extensions; the search
and XGBoost algorithms are upstream. No runtime replacement of optimizer methods
is required. The historical Verilator C-template rewrite is not enabled here.

For compiler reconstruction use the original full handoff source pins in
`SOURCE_PINS.json`: TVM `tvm-firesim-chunks`, base
`c50c4e26992eb79ea4fc1c693d3aa16f62aaea43` plus its delivered patch/overlay,
and matching native library SHA256
`6fc46068a7dc3b231e7fd3fec5be7041c703149b26f292cdf0d330e2dc3995fd`.
[GENERATION.md](GENERATION.md) remains the pass/ownership/knob reference.
No Python/compiler setup is needed merely to load an already prepared ELF.

For automated tuning, `tune_firesim_chunks.py` reuses the existing builder and
stock TVM search. `firesim_job_pool.py` connects those measurements to the
installed ELF runner. The optional pool integration uses `pool.template.json`
and [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md): ELF in, correctness/active cycles out, a live incumbent
limit, and the shared study deadline. Record an existing deployment identifier
if available; do not start a hardware/memory characterization task.

Set `PY`, `TVM`, `MERLIN`, `DELIVERY`, `POOL` and `TOOLCHAIN` for the remote
installation; `RUN` is a fresh artifact product. Put `$TVM/python` and
`$TVM/apps/gemmini` first in `PYTHONPATH` and select the pinned native library.
Set one deadline before launching any study, and reuse it for all processes:

```bash
DEADLINE_EPOCH=$(date -d '+4 hours' +%s)
CHUNK="$FROZEN_RELEASE/exact-chunks/decode-kv"
"$PY" "$MERLIN/examples/gemmini/comparisons/tvm/tune_firesim_chunks.py" \
  --target-clock-hz "$TARGET_CYCLE_COUNTER_HZ" \
  --program "$CHUNK/program.json" --fixture "$CHUNK/fixture.npz" \
  --pool "$POOL" --toolchain "$TOOLCHAIN" \
  --output-dir "$RUN/decode-kv" --deadline-epoch "$DEADLINE_EPOCH" \
  --trials 12 --batch 4 --builders "$BUILDERS" --graph-fusion-level 2
```

Run prefill K/V and the down panel concurrently when resources permit, sharing
that pool/deadline and using distinct outputs. Increase the proposal budget if
useful work fits the same deadline. Add
`--seed-database "$DELIVERY/tinyllama-verilator-library"` to remeasure matching
prior traces; never import their historical Verilator costs as FireSim timings.
`validated-database/` exports admitted labels; `database/` retains search history
including failure penalties. Retain raw runner output and partial progress.

`FROZEN_RELEASE` must identify a prepared release conforming to the current
numerical policy, with actual calibration/scales, independent references and a
layer placement/output ledger. It is not supplied by renaming the old dynamic
probe directory. No replacement static-scale TinyLlama release is created by
the audit; quality gates remain explicit until evaluated.

For actual exports, use `prepare_firesim_chunks.py inventory --program ACTUAL.pt2
--output-dir NEW`, collect independent live-in/live-out references, and freeze
an exact small interval:

```bash
"$PY" "$MERLIN/examples/gemmini/comparisons/tvm/prepare_firesim_chunks.py" slice \
  --program "$ACTUAL_STAGE" --start "$START" --stop "$STOP" \
  --fixture "$BOUNDARY_NPZ" --fixture-provenance "$BOUNDARY_PROOF" \
  --output-dir "$RUN/exact-chunk"
```

The slicer preserves full dimensions/constants and residual/cache live-outs.
`prepare_exported_chunks.py` additionally freezes explicit functional source-FX
regions with source-derived live boundaries and checked INT64 contraction
references. It is a preparation tool, not a compiler optimization. Its proof
records numerical admission as pending rather than inferring it from timing.

Compile and independently validate each winning whole chunk. The N=128 down
panel cannot supply an N=2048 record: confirm promising choices with a short
stock search on the full shape before E2E. CSE is already a no-op on the shared
quantization in the prepared K/V probes; do not spend trials on identical graphs.

ResNet's current library matches **1/68 tasks, 1/67 eligible tasks, 1/21
contractions**. Its 24 distinct proposed traces are 3.57% of 672 declared compact
combinations before legality/code-equivalence pruning. The classifier's 2.06x
Verilator improvement is not a measured full-model FireSim gain. Compare the
baseline and current tuned ELFs on the supplied bitstream. Profiling is optional;
report complete model intervals rather than summed operator/chunk minima.
