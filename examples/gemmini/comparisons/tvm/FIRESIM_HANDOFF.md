# FireSim handoff: ResNet50 and 16 complete kernel cases

Active scope is **ResNet50 plus G01–G06, C01–C03 and T01–T07** from
[KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md](KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md).
**Do not run full TinyLlama or SmolVLA.** Their isolated kernels remain active.
This file supersedes older three-model launch instructions and local `out/`
paths in historical releases. Read [ARCHITECTURE.md](ARCHITECTURE.md) and
[TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md) for the compiler and operation limits.

## Source, evidence and readiness

Use TVM **bb4ad9f8dbd21dd96a4dc63121dfebc24742f260** on
[`grammy117/tvm:gemmini/direct-intrinsics`](https://github.com/grammy117/tvm/tree/gemmini/direct-intrinsics)
and Merlin builder commit `b632e4c97bcc5e1d05881db41aa5e22360740601` plus
the documentation/evidence commit containing this handoff. The executable
TVM implementation is unchanged from `64bed0cb9`; later changes are documentation.
Do not use the historical `firesim-chunks` handoff pin or relabel old ELFs.

The current local Spike evidence and reproducible build commands are linked in
[the verification delivery](SPIKE_HANDOFF_EVIDENCE.md). These are diagnostic
functional checks, not RTL or FireSim performance results. The delivery publishes
small receipts/checksums; build ELFs on the remote from the tracked sources and
inputs below. Prebuilt ELF paths in receipts identify the original test machine.
They are not download locations or a prerequisite to reproducing the builds.

The remote worker owns deployment and physical slots. Before timing, qualify its
actual hardware facts, guest output checks, startup exclusion, repeated timings,
wall cancellation and safe slot reuse under [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md).
No live active-cycle observer or incumbent-based cancellation is required.
Do not add a sequential Spike preflight to each remote candidate: that work has
been done locally for these baselines; remote candidates use guest checks.

## Budgets, defaults and search

| Campaign | Search | Aggregate occupied-slot budget | Final run timeout |
| --- | --- | --- | --- |
| ResNet50 | Up to 25 ReplayTrace proposals per owner chunk, shared across its tasks and graph plans | 4 hours | 30 minutes per baseline/tuned E2E run |
| 16 kernels | Up to 25 ReplayTrace proposals per complete case, shared across its tasks and graph plans | 24 hours total for TVM | Calibrated per-case wall watchdog; no invented cycle limit |

The 24-hour kernel ceiling is a planning allowance, not a measured prediction.
Close admissions at the first trial/time limit, finish active jobs under their
frozen watchdogs and report overrun. Restarts do not renew budgets. CPU preparation
and final ResNet E2E are outside DSE occupancy. Baseline calibration, failures,
repeats and connected confirmations consume the DSE allowance. The launcher
absolute deadline is an additional bound; the remote aggregate occupancy ledger
must account for concurrent physical slots.

Use **stock ReplayTrace, no ML, no 50-sample warmup and no guided phase**.
A trial is a schedule proposal, not a new fixture. Duplicate/failed proposals
count; report attempted, unique, built and successful candidates separately.
The prepared study initially allocates 20 of 25 trials, reserving five for stock
graph alternatives. This is one shared 25-trial allowance, not an additional
five-trial phase. `--initial-trials 25` uses the whole allowance on the initial
plan. A continuation must preserve the ledger and per-chunk total, re-extract
changed graphs, and reuse only structurally identical tasks. Do not claim that
stochastic generation failures prove search-space exhaustion.

Start with fusion=2, CSE off, automatic prepacking and bias folding off. The
backend's shape-derived default chooses the largest exact legal intrinsic tile
(up to 16), ranks bounded M/N reuse groups by logical operand traffic and K
batching, and validates actual aligned storage. It tries at most eight choices
plus a one-group fallback. It is a heuristic, not a cost prediction or optimum.
Keep the correct measured default in the database. Search legal tiles, reuse,
OS/WS, traversal, hoisting, pipelining, staging and tails through existing traces;
then stock fusion/CSE within the same trial and time budgets. Physical addresses
and required fences are fixed lowering. Secondary ops are not separately tuned.

The runner currently measures **extracted contractions**. These labels select
schedules, but final results must time the rebuilt complete kernel/chunk. T01/T02
include all 32 heads; T04/T06 include all outputs. Never sum task minima to
manufacture complete-case or E2E cycles. Share records only after exact structural
matching; remeasure old traces instead of importing old costs.

## Wall watchdog and cycle measurement

The remote pilot sets positive bootstrap and ceiling seconds. Run three correct
uncached baseline jobs per selected task. Use three times their median as the
floor. After nine successful uncached candidates, use the larger of that floor
and three times the latest-nine median, capped by the explicit ceiling. A ceiling
below the baseline floor refuses search. Freeze the limit at slot admission.
Timeouts/failures do not update the median and never become cycle labels.

Record completed warm target cycles, including host work, packing, DMA and
completion; exclude startup and the first warmup call. Retain slow correct
measurements. Convert cycles to seconds at the stock TVM Runner boundary using
the actual documented target counter frequency. Do not mix legacy cycle-valued
JSON databases with these seconds-valued records. This watchdog still censors
very slow candidates; report timeout counts and the policy in the paper.

## Restore and build on the remote CPU worker

Set `MERLIN`, `TVM_SOURCE`, `PY`, `RISCV_GCC`, `ADAPTER_RECEIPT` and a fresh
`WORK` directory to the worker's installed paths. No commands below deploy or run
FireSim. Use Python 3.12+ for checked archive extraction and bounded build workers.

```sh
export COMPARISON="$MERLIN/examples/gemmini/comparisons/tvm"
export PYTHONPATH="$MERLIN/src:$TVM_SOURCE/python:$TVM_SOURCE/apps/gemmini:$COMPARISON:$PYTHONPATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TVM_NUM_THREADS=1

# Uses the committed split archive and frozen catalog; no full large-model restore.
"$PY" "$COMPARISON/restore_resnet_inputs.py" --merlin-root "$MERLIN" \
  --output-dir "$WORK/resnet-inputs"
"$PY" "$COMPARISON/prepare_active_study.py" --kind resnet \
  --input-root "$WORK/resnet-inputs" --output-dir "$WORK/resnet-study"

# Generate exact signed-nibble inputs and independent references for all 16 cases.
for CASE in $("$PY" "$COMPARISON/prepare_kernel_benchmarks.py" list); do
  "$PY" "$COMPARISON/prepare_kernel_benchmarks.py" prepare --case "$CASE" \
    --output-dir "$WORK/kernel-inputs/$CASE"
  "$PY" "$COMPARISON/prepare_kernel_benchmarks.py" build \
    --prepared "$WORK/kernel-inputs/$CASE" --output-dir "$WORK/baseline/$CASE" \
    --riscv-gcc "$RISCV_GCC" --adapter-receipt "$ADAPTER_RECEIPT"
done
"$PY" "$COMPARISON/prepare_active_study.py" --kind kernel \
  --input-root "$WORK/kernel-inputs" --output-dir "$WORK/kernel-study"

# Repeat for every ID in resnet-study/study.json coverage (18 chunks).
"$PY" "$COMPARISON/build_shared_chunk_elf.py" --tvm-source "$TVM_SOURCE" \
  --restored-root "$WORK/resnet-inputs" --chunk resnet-stem \
  --output-dir "$WORK/baseline/resnet-stem" --riscv-gcc "$RISCV_GCC" \
  --adapter-receipt "$ADAPTER_RECEIPT" --host-math-policy native \
  --tensor-memory-bytes 268435456
```

ResNet has the stem, 16 complete bottleneck blocks and classifier. Its initial
inventory has 21 distinct contractions across ten owners; all 18 connected chunks
still need baseline/tuned confirmation even where records are shared. The frozen
catalog SHA256 is `839fad75688a9c2a87b975842f072f217a1f0fbc0ff3ba26e9b1707600e37c2a`.
The restorer preserves historical source/policy files as provenance. **Do not
apply archived source patches or follow archived campaign instructions.**

Each builder emits `package/`, `guest/valid.elf`, `guest/bad-oracle.elf`, reference
bindings, memory/placement and instruction audits. Kernel builds reject CPU
fallback. Convolution kernels explicitly declare biased reductions in ordinary
TensorIR with the benchmark's flattened output ABI; no automatic bias-folding
pass was restored. Host gather/staging work is included. Use the whole output
and guards; do not narrow the checker to one head or projection.

For tuned builds use fresh output directories, `--tuning-database "$DATABASE"`
and the matching `--graph-fusion-level`/`--graph-cse`. The compiler rejects an
explicit database with zero structural hits. Record misses and keep defaults;
do not force traces onto different programs. The current harness measures
`warm_repeat`; it does **not** implement the historical 2 MiB eviction profile.
Other compilers must use the same declared timing profile for comparison.

## Run tuning, then connected confirmations

After the remote qualification in ADAPTER_CONTRACT.md, invoke
`run_firesim_study.py --study <study.json> --toolchain <toolchain.json>
--pool <qualified-pool.json> --output-dir <new-dir> --campaign-kind resnet|kernel
--target-clock-hz <verified-counter-Hz> --candidate-bootstrap-seconds <pilot>
--candidate-timeout-ceiling-seconds <pilot-ceiling> --ledger <persistent-ledger>`.
It first writes a dry-run command plan; `--execute` starts work. Use a separate
persistent ledger for each active campaign, preserve it across graph phases,
and enforce the aggregate occupancy caps remotely. For execution, supply `--deadline-epoch` with the campaign's fixed absolute
admission deadline, and use `--phase continuation` for reserved graph trials.
The remote agent supplies actual paths and concurrency; this handoff supplies no fake
qualification receipt or guessed measured cycles.

Replay selected traces into complete packages and confirm every promoted chunk
and all 16 complete cases, including shared-owner cases. Preserve untuned correct
defaults for unconfirmed or uncovered work. Save input/output/ELF hashes, logs,
cycles, clock, source/hardware identities, traces, failures, watchdogs and occupancy.

## ResNet baseline and tuned E2E

The connected parent passes local diagnostic Spike, including wrong-reference
rejection; see the evidence link above. The same ResNet delivery includes
`resnet-release/full-resnet/program.json`,
`fixture.npz` and the relocated independent reference. Build the connected parent
using `build_shared_chunk_elf.py` above with `--chunk full-resnet`; use separate
baseline/tuned directories and add the matching database only to the tuned build.
This executes the actual parent graph; it does not substitute boundary fixtures
or sum chunk timings. Retain all 53 convolutions, one linear, residual/pool/host
work, 32 requant/ReLU and 22 raw exits. Check the complete 1000-logit output.

The sole model-running authority is [protocol at 8ec99308](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol),
also committed in the frozen `gemmini-firesim-elf-plan` release's `run-contract/`.
Bind the parent checkpoint, recipe, selected input and numerical ledger to that
protocol before claiming selected-model qualification; frozen source proofs
still say `REVIEW_REQUIRED`. Use functional_v1: exact arity/shape/dtype and integer
values, finite floats with rtol=1e-5 and atol=2e-6. FP quality is report-only.
Do not substitute the archive's old CURRENT_RUN_POLICY as the authority.

The selected model target is full Gemmini/Rocket U250 at 30 MHz. The remote must
check the actual image, capability/header/runtime facts and memory map. A kernel
result on an F2 image is a separately attributed experiment. Baseline and tuned
E2E each have a **1,800-second wall timeout**, startup through collection after
slot admission, excluding queue wait. These final runs are outside DSE time.

## Limits to report

- Diagnostic repaired Spike is not the missing spec-pinned libgemmini oracle.
  Exact per-program spec proofs and deployed hardware bindings remain required.
- Local functional success does not establish FireSim speedup or selected-model
  quality; report actual complete-call and E2E results separately.
- This is bounded random trace sampling with an engineered default, not an
  ML-tuned optimum. Publish trials, failures, timeouts and actual tuning effort.
- Full TinyLlama/SmolVLA are deferred as too expensive. Isolated kernels do not
  establish their full-model performance or quality.
- Nine inherited module-size failures and eight no-regex call sites in unchanged
  files remain repository lint debt; they are not failed Spike tests.
