# Reproducing the F2 FireSim results (2026-10-28)

This guide re-creates the TVM–Gemmini numbers from the 2026-10-28 FireSim
campaign on AWS F2: 16 complete kernel cases, 18 ResNet50 chunks and the
connected ResNet50 parent, each built untuned and tuned and timed in target
cycles. It lists the experimental setup, the exact sources, every step that was
run, and the results with the caveats needed to read them.

| Where | What |
| --- | --- |
| [Measurement product][m] | Imported result, setup and qualification files plus the scripts that ran (about 0.5 MB, SHA256SUMS). `measured/summary.json` is the source of every table below. |
| `grammy117/grammy-trash-can` commit `5c1fab823`, `tvmfs-f2-20261028/` | The complete remote tree (about 207 MB): per-trial candidates, logs, run transcripts. Private repository. |
| [USAGE.md](USAGE.md) | How to install and drive the tools used here. |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | What the compiler and runner do. |

[m]: ../../../../out/artifacts/measurements/firesim_f2_gemmini_rocket_ddr3llc_60mhz/resnet50_kernels16/tvm_replaytrace_v1_20261105T022306Z_5998ce4/

## Headline results

All cycles are completed warm-call target cycles on the F2 image below
(60 MHz nominal). Lower is better.

- **Kernels:** the tuned default plan (fusion 2) beats the untuned default on
  15 of 16 cases. The geometric-mean speedup is 1.48×; the largest gains are the
  convolutions (C01 −88.1%, C02 −87.3%, C03 −54.6%). T02 got slower when tuned
  (+2.68%), so its untuned default is kept.
- **ResNet50 chunks:** the geometric-mean speedup with the tuned default plan is
  1.02×. It is 1.08× if each chunk takes its best confirmed stock graph plan,
  usually fusion 0.
- **ResNet50 end to end:** tuned (fusion 2) is **−0.32%** against the untuned
  default (2,229,304,805 vs 2,236,439,673 cycles). The per-chunk winner, fusion 0,
  is **+2.96% slower** end to end, so chunk results do not compose and must not
  be summed.
- Every one of the 34 units repeated to the identical cycle count on all three
  baseline runs and all three tuned runs. All 170 alternative-plan builds and
  runs succeeded. No candidate hit the wall watchdog.

## Experimental setup

### Hardware

| Item | Value |
| --- | --- |
| Platform | AWS `f2.48xlarge`, us-east-1, 8 FPGA slots (Xilinx VU47P); all 8 slots in the pool |
| FireSim design | `GemminiRocketConfig` with `FRFCFS16GBQuadRankLLC4MB_BaseF2Config` (DDR3 FR-FCFS quad-rank memory model + 4 MB LLC), no NIC |
| Image | AGFI `agfi-02410ef1a8461b435` (AFI `afi-0b759e97614a74b1b`, HWDB `f2_gemmini_rocket_ddr3llc_60mhz`) |
| Target clock | 60 MHz build; post-route WNS +0.092 ns; driver reported FMR 1.07 (56.04 MHz effective host rate). Seconds below divide cycles by the nominal 60 MHz. |
| Hardware sources | Chipyard `157dd69e`, FireSim `e1bdea8bb042`, Gemmini `8c3f9923a44a`; `gemmini_params.h` sha256 `3758ae96…` (16×16 mesh, 1×1 tiles, `SInt(20)` output) |
| Identity | This is **not** the selected U250 30 MHz model target. It is a separately attributed F2 60 MHz experiment. |

Receipt: `setup/hardware-receipt.json` in the measurement product.

### Software sources

| Component | Pin | How it was verified |
| --- | --- | --- |
| TVM compiler | [`grammy117/tvm` `f3cfd1a7d`](https://github.com/grammy117/tvm/commit/f3cfd1a7dc95d00191225fa8ac5fc8aee3a6940d) on `gemmini/direct-intrinsics` (tree-identical to pre-identity-fix `bb4ad9f8d`; base Apache TVM v0.19.0 `c4dc0c29`); pinned in this branch as the submodule `third_party/tvm` | All 62 TVM `compiler_sources` sha256 values in `setup/resnet-study.json` match this commit and the submodule checkout. The later `fee495aca` is a Python-only refactor (58 of the 62 files still match; no C++ changes); the branch tip after it adds documentation only. |
| Merlin runner and builders | `fe0306676` on `mlsys27/feat-SoTA-tvm`, formerly `comparisons/tvm-workload-tuning` (tree-identical to pre-identity-fix `9530ee83d`) | The 5 Merlin `compiler_sources` match. Merlin code (`src/`, `packages/`, the comparison `*.py`) is unchanged from that commit to the branch tip. |
| Gemmini software for the adapter | gemmini `6ad65b90`, gemmini-rocc-tests `7c540b3a`, rocc-software `fddb795a`, libgemmini `ea8f7ed7` | `setup/adapter-receipt.json` |
| Cross compiler | `riscv64-unknown-elf-gcc` 16.1.0 (`g6afcc4f6d-dirty`) on the F2 build host | `setup/adapter-receipt.json` |
| Python | 3.12 virtual environment with the TVM build; single-threaded BLAS (`OMP/OPENBLAS/MKL/TVM_NUM_THREADS=1`) | `tools/env.sh` |
| Model protocol | Merlin [`8ec99308` protocol](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol) for ResNet inputs/references; kernels from [KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md](KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md) | ResNet restore verified 182 payload files, archive sha256 `0f4abd85…` |

### Workloads

| Set | Units | Tuning owners | Distinct contractions |
| --- | --- | --- | --- |
| Kernels | G01–G06 (GEMM), C01–C03 (convolution), T01–T07 (transformer pieces; T01/T02 all 32 heads, T04/T06 all outputs) | 16 | 17 |
| ResNet50 chunks | stem, 16 bottleneck blocks, classifier | 10 (chunks with identical contractions share an owner's records) | 21 |
| ResNet50 parent | connected full graph: 53 convolutions, 1 linear, residual/pool/host work, 1000-logit output | uses the merged ResNet database | — |

Kernel inputs are deterministic signed-nibble generators; ResNet inputs come from
the committed split archive (frozen catalog sha256 `839fad75…`). Every ELF has a
positive guest (`valid.elf`) and a deliberately wrong reference (`bad-oracle.elf`).

### Search and measurement

| Setting | Value |
| --- | --- |
| Search | Stock MetaSchedule `ReplayTrace`, round-robin over extracted contraction tasks; no learned cost model, no warmup or guided phase |
| Trials | 20 per owner (initial phase) of a 25-trial cap; the 5 reserved trials were not used (below) |
| Starting graph | Relax fusion level 2, CSE off, automatic weight prepacking off, bias preload off |
| Starting schedule | Backend default: largest exact intrinsic tile ≤16, traffic-ranked M/N reuse groups, checked against real SRAM allocation (at most 8 choices + 1 fallback) |
| Tuned scope | Extracted contractions only; other operators keep fixed schedules or host code, and their cost is in every complete-call time |
| Cycle label | Completed second (warm) call of the whole unit, including host work, packing, DMA and completion; boot and first call excluded. Seconds for the TVM Runner = cycles / 60 MHz. |
| Candidate watchdog | `median_wall_v1`: 3 baseline wall samples, floor 3× their median, then 3× the latest-nine median, bootstrap 120 s, hard ceiling 600 s; frozen at slot admission |
| Budgets | ResNet 4 occupied-slot-hours; kernels 24 occupied-slot-hours; final E2E wall timeout 1,800 s per run |
| Confirmation | Per unit: untuned baseline ×3, tuned fusion-2 build ×3, then the first 5 distinct stock graph alternatives (fusion 3, 1, 0; fusion 2 + CSE; fusion 3 + CSE) built with the same database, one run each |

### Runner qualification (before tuning)

All five ADAPTER_CONTRACT controls passed through the real pool on slot `h0s7`
(`qualification/*.json`): repeated G01 agreed exactly (5 runs, 1,972,011 cycles,
spread 0); the warm label excluded boot and first call; the wrong oracle was
rejected; a 45 s wall limit stopped full ResNet remotely; and the slot was reused
safely after both a remote timeout and a forced local kill. Three earlier
qualification attempts are kept in the full tree and were not used: one on an
`f2.6xlarge` superseded by the 8-slot host, one with a local-kill bug and one
with a boot-parse bug.

## Recipe

The commands below are the ones the F2 operator ran, taken from
`tools/` and `setup/*-commands.json` in the measurement product. Replace
`/work/exp/tvmfs` with your own root. Steps 1–4 need only a CPU host; steps 5–9
need a FireSim pool.

### 1. Check out the pinned sources

```sh
git clone -b mlsys27/feat-SoTA-tvm https://github.com/ucb-bar/merlin.git "$EXP/merlin"
git -C "$EXP/merlin" submodule update --init third_party/tvm        # TVM f3cfd1a7d
export MERLIN="$EXP/merlin" TVM_SOURCE="$EXP/merlin/third_party/tvm"
git -C "$TVM_SOURCE" submodule update --init 3rdparty/dlpack 3rdparty/dmlc-core 3rdparty/rang
```

The branch pins the F2 TVM as the `third_party/tvm` submodule. From the F2
Merlin commit `fe0306676` to the branch tip only documentation, imported
artifacts, `.gitignore` and that submodule changed, so the tip plus the
submodule is the F2 source set. Build `libtvm.so` from `$TVM_SOURCE` with LLVM 18
(see [USAGE.md](USAGE.md#2-build-tvm)).
Set the environment as in `tools/env.sh`: `MERLIN`, `TVM_SOURCE`, `TVM_BUILD`
(= `TVM_LIBRARY_PATH`), `PY`, `RISCV_GCC`, `ADAPTER_RECEIPT`, `COMPARISON`,
`PYTHONPATH="$MERLIN/src:$TVM_SOURCE/python:$TVM_SOURCE/apps/gemmini:$COMPARISON"`
and the four single-thread variables.

### 2. Build the Gemmini adapter

```sh
"$PY" "$TVM_SOURCE/apps/gemmini/verify_matmul.py" \
  --gemmini-repo "$BASE/gemmini-source" --rocc-tests-repo "$BASE/rocc-tests-source" \
  --rocc-software-repo "$BASE/rocc-software-source" \
  --build-root "$BASE" --output-dir "$BASE/adapter-<name>" --cc "$RISCV_GCC"
```

`$BASE` must end in `baselines/tvm-gemmini`. Check that the new receipt's
`adapter_sources`, `headers` and `public_symbols` equal `setup/adapter-receipt.json`.

### 3. Prepare inputs and baseline ELFs

```sh
"$PY" "$COMPARISON/restore_resnet_inputs.py" --merlin-root "$MERLIN" --output-dir "$WORK/resnet-inputs"
"$PY" "$COMPARISON/prepare_active_study.py" --kind resnet --input-root "$WORK/resnet-inputs" --output-dir "$WORK/resnet-study"
for CASE in $("$PY" "$COMPARISON/prepare_kernel_benchmarks.py" list); do
  "$PY" "$COMPARISON/prepare_kernel_benchmarks.py" prepare --case "$CASE" --output-dir "$WORK/kernel-inputs/$CASE"
  "$PY" "$COMPARISON/prepare_kernel_benchmarks.py" build --prepared "$WORK/kernel-inputs/$CASE" \
    --output-dir "$WORK/baseline/$CASE" --riscv-gcc "$RISCV_GCC" --adapter-receipt "$ADAPTER_RECEIPT"
done
"$PY" "$COMPARISON/prepare_active_study.py" --kind kernel --input-root "$WORK/kernel-inputs" --output-dir "$WORK/kernel-study2"
for CHUNK in <18 ResNet chunk ids> full-resnet; do
  "$PY" "$COMPARISON/build_shared_chunk_elf.py" --tvm-source "$TVM_SOURCE" --restored-root "$WORK/resnet-inputs" \
    --chunk "$CHUNK" --output-dir "$WORK/baseline/$CHUNK" --riscv-gcc "$RISCV_GCC" \
    --adapter-receipt "$ADAPTER_RECEIPT" --host-math-policy native --tensor-memory-bytes 268435456
done
```

The chunk ids are the `coverage[].id` values of `resnet-study/study.json`.
Check the regenerated studies against `setup/resnet-study.json` and
`setup/kernel-study.json`: the per-unit `program_sha256`/`fixture_sha256` and
`workload_keys` should match. Fresh extraction finds 10 ResNet owners with 21
distinct contractions and 16 kernel owners with 17. (`kernel-study2` is the
study that was tuned; an earlier `kernel-study` in the full tree was not used.)

### 4. (Optional) functional check on Spike

`check_kernel_spike.py --guest "$WORK/baseline/$CASE/guest" --simulator-receipt <spike receipt>`
runs the positive and wrong-oracle guests. This is a functional check only. The
published [SPIKE_HANDOFF_EVIDENCE.md](SPIKE_HANDOFF_EVIDENCE.md) used a
repaired-RNE libgemmini plugin, recorded only by its patch sha256 `f723f188…`.
That plugin build no longer exists on a26, and the patch was not found in any
retained archive. The plain libgemmini `ea8f7ed7` plugin (`simulator-a26`) is
what remains, so that evidence cannot currently be re-run byte for byte.

### 5. Set up the F2 host and pool

Flash every slot with the AGFI and install the forced-command runner
(`tools/f2-host/`: `f2-flash` loads all slots and reloads one slot before every
run; `f2-exec` streams one ELF from stdin to a free slot and returns the target's
exit code; `f2-idle-stop` powers the host off after 20 idle minutes; settings in
`f2run.conf`). The workstation side is `tools/tvmfs_adapter.py`, which implements
the `gemmini_firesim_probe_request_v4` contract: it enforces the absolute
deadline on both ends, probes and resets the physical slot, and accepts a result
only if the booted ELF hash and AGFI match.

The pool file (`f2/pool.json`, not copied because it holds a key path) has:
`deployment_id`, `hardware_receipt`, `adapter_sources` (path + sha256 of the
adapter and the four host files), `persistent_manager: true`,
`adapter_cancels_remote_on_timeout: true`, `runner_qualification` (path + sha256),
and 8 `slots` `h0s0`…`h0s7`, each with a lock file and the command
`python3 tvmfs_adapter.py --hosts hosts.json --key <ssh key> --request {request} --result {result} --slot {slot}`.
`hosts.json` maps `h0` to the F2 host's address.

### 6. Qualify the runner, then pilot

```sh
python tools/qualify.py h0s7      # five ADAPTER_CONTRACT controls; writes f2/qualification/
python tools/pilot.py             # every baseline valid.elf once; sizes bootstrap/ceiling seconds
```

The pilot's warm cycles (`measured/pilot-summary.json`) equal the later baseline
cycles exactly. The longest ResNet chunk took 25 s of wall time and full ResNet
100 s, consistent with the 120 s bootstrap and 600 s ceiling used for tuning.

### 7. Tune

```sh
sh tools/launch-tuning.sh resnet <jobs> <deadline-hours>
sh tools/launch-tuning.sh kernel <jobs> <deadline-hours>
```

Each runs `run_firesim_study.py --study <study.json> --toolchain f2/toolchain.json
--pool f2/pool.json --output-dir work/tune-<kind> --campaign-kind <kind>
--target-clock-hz 60000000 --candidate-bootstrap-seconds 120
--candidate-timeout-ceiling-seconds 600 --deadline-epoch <now + hours> --jobs <jobs>
--ledger f2/ledger-<kind>.json --execute`. The job count was not recorded; the
recorded absolute deadlines were 2026-10-28 10:24:53Z (ResNet) and 14:23:47Z
(kernels). The per-owner `tune_firesim_chunks.py` command lines are in
`setup/tune-*-commands.json`. All 26 owner campaigns finished with return code 0
before their deadlines.

`make_continuation.py` then re-extracted the seven other stock graph plans for
every ResNet owner. **No plan introduced a new contraction** (`continuation_rows:
0`), so no continuation trials ran and the 5 reserved trials per owner stayed unused.

### 8. Merge, rebuild and confirm

The ResNet database `work/db-resnet` is the union of the ten owners'
`validated-database` directories: 21 workloads, 221 records, identical by
(workload structural hash, trace, measured seconds). Each kernel case uses its
own `tune-kernel/<case>/validated-database`. Then:

```sh
POOL=f2/pool.json python tools/stage2.py              # 34 units: baseline x3, tuned x3, 5 alternatives x1
python tools/stage2.py e2e work/db-resnet             # full ResNet baseline + tuned (fusion 2), wrong-oracle control
python tools/stage2.py e2eplan work/db-resnet 0 nocse # full ResNet with fusion 0
python tools/summarize.py                             # summary.json
python tools/accounting.py                            # tuning-accounting.json
```

Tuned builds pass `--tuning-database` with the matching `--graph-fusion-level`
and `--graph-cse`; the compiler refuses a database with zero structural hits.
`best_confirmed` in `summary.json` is the minimum over the untuned default, the
tuned fusion-2 build and the measured alternatives.

## Results

Generated from `measured/summary.json`, `measured/e2e*.json` and
`measured/tuning-accounting.json`. "Best confirmed" takes the lowest measured
build per unit; alternative plans have one run each.

### Kernels

| Unit | Baseline cycles | Tuned cycles (fusion 2) | Tuned vs baseline | Best confirmed plan | Best cycles | Best vs baseline |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| G01 | 1,972,011 | 1,959,012 | -0.66% | tuned, fusion 2 | 1,959,012 | -0.66% |
| G02 | 3,424,353 | 3,192,686 | -6.77% | tuned, fusion 2 | 3,192,686 | -6.77% |
| G03 | 4,283,444 | 3,696,135 | -13.71% | tuned, fusion 2 | 3,696,135 | -13.71% |
| G04 | 3,082,783 | 2,814,698 | -8.70% | tuned, fusion 2 | 2,814,698 | -8.70% |
| G05 | 3,606,097 | 3,037,933 | -15.76% | tuned, fusion 2 | 3,037,933 | -15.76% |
| G06 | 3,297,937 | 2,932,182 | -11.09% | tuned, fusion 2 | 2,932,182 | -11.09% |
| C01 | 327,109,776 | 38,872,761 | -88.12% | tuned, fusion 2 | 38,872,761 | -88.12% |
| C02 | 348,516,304 | 44,215,831 | -87.31% | tuned, fusion 2 | 44,215,831 | -87.31% |
| C03 | 228,703,886 | 103,739,715 | -54.64% | tuned, fusion 2 | 103,739,715 | -54.64% |
| T01 | 48,718 | 47,623 | -2.25% | tuned, fusion 2 | 47,623 | -2.25% |
| T02 | 48,581 | 49,885 | +2.68% | untuned default | 48,581 | 0.00% |
| T03 | 5,475,334 | 4,507,132 | -17.68% | tuned, fusion 2 | 4,507,132 | -17.68% |
| T04 | 11,164,602 | 9,324,924 | -16.48% | tuned, fusion 2 | 9,324,924 | -16.48% |
| T05 | 1,709,140 | 1,590,971 | -6.91% | tuned, fusion 2 | 1,590,971 | -6.91% |
| T06 | 2,229,270 | 2,077,751 | -6.80% | tuned, fusion 2 | 2,077,751 | -6.80% |
| T07 | 32,172,886 | 25,781,071 | -19.87% | tuned, fusion 2 | 25,781,071 | -19.87% |

Geometric-mean speedup over the 16 cases: 1.4847× tuned, 1.4872× best confirmed.

### ResNet50 chunks

| Unit | Baseline cycles | Tuned cycles (fusion 2) | Tuned vs baseline | Best confirmed plan | Best cycles | Best vs baseline |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| resnet-stem | 386,810,844 | 386,810,844 | 0.00% | tuned, fusion 0 | 326,187,976 | -15.67% |
| resnet-stage1-block1 | 281,526,463 | 280,863,141 | -0.24% | tuned, fusion 2 | 280,863,141 | -0.24% |
| resnet-stage1-block2 | 260,109,631 | 259,471,872 | -0.25% | tuned, fusion 2 | 259,471,872 | -0.25% |
| resnet-stage1-block3 | 259,681,832 | 259,074,145 | -0.23% | tuned, fusion 2 | 259,074,145 | -0.23% |
| resnet-stage2-block1 | 177,860,135 | 177,440,013 | -0.24% | tuned, fusion 2 | 177,440,013 | -0.24% |
| resnet-stage2-block2 | 125,241,868 | 125,203,576 | -0.03% | tuned, fusion 0 | 116,287,609 | -7.15% |
| resnet-stage2-block3 | 123,089,858 | 123,051,402 | -0.03% | tuned, fusion 0 | 114,261,680 | -7.17% |
| resnet-stage2-block4 | 125,241,868 | 125,203,576 | -0.03% | tuned, fusion 0 | 116,287,609 | -7.15% |
| resnet-stage3-block1 | 90,207,878 | 90,207,878 | 0.00% | untuned default | 90,207,878 | 0.00% |
| resnet-stage3-block2 | 60,346,878 | 60,353,072 | +0.01% | tuned, fusion 0 | 55,674,442 | -7.74% |
| resnet-stage3-block3 | 60,346,878 | 60,353,072 | +0.01% | tuned, fusion 0 | 55,674,442 | -7.74% |
| resnet-stage3-block4 | 60,346,878 | 60,353,072 | +0.01% | tuned, fusion 0 | 55,674,442 | -7.74% |
| resnet-stage3-block5 | 60,346,878 | 60,353,072 | +0.01% | tuned, fusion 0 | 55,674,442 | -7.74% |
| resnet-stage3-block6 | 60,346,878 | 60,353,072 | +0.01% | tuned, fusion 0 | 55,674,442 | -7.74% |
| resnet-stage4-block1 | 54,718,932 | 53,061,993 | -3.03% | tuned, fusion 2 | 53,061,993 | -3.03% |
| resnet-stage4-block2 | 35,061,802 | 33,864,200 | -3.42% | tuned, fusion 0 | 30,629,309 | -12.64% |
| resnet-stage4-block3 | 35,489,090 | 34,138,172 | -3.81% | tuned, fusion 0 | 30,605,957 | -13.76% |
| resnet-classifier | 2,028,945 | 1,534,393 | -24.37% | tuned, fusion 2 | 1,534,393 | -24.37% |

Geometric-mean speedup over the 18 chunks: 1.0221× tuned, 1.0808× best confirmed.

### ResNet50 end to end

| Build | Cycles | vs baseline | Seconds at 60 MHz |
| --- | ---: | ---: | ---: |
| Baseline (untuned default) | 2,236,439,673 | 0.00% | 37.27 |
| Tuned, fusion 2 (default plan) | 2,229,304,805 | -0.32% | 37.16 |
| Tuned, fusion 0 | 2,302,656,248 | +2.96% | 38.38 |
| Wrong-oracle control | rejected (guest failure, no cycles) | | |

Each run took about 100–115 s of slot wall time, well inside the 1,800 s limit.
One fusion-0 attempt aborted when the host went down and was rerun; the full
tree keeps the aborted folder.

### Tuning accounting

| Campaign | Kind | Attempted | Unique | Built | Successful | Timeouts | Occupied slot-seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet-classifier | resnet | 20 | 20 | 20 | 20 | 0 | 233.5 |
| resnet-stage1-block1 | resnet | 20 | 20 | 20 | 20 | 0 | 300.6 |
| resnet-stage1-block2 | resnet | 20 | 19 | 20 | 20 | 0 | 229.7 |
| resnet-stage2-block1 | resnet | 20 | 20 | 20 | 20 | 0 | 312.8 |
| resnet-stage2-block2 | resnet | 20 | 20 | 20 | 20 | 0 | 219.7 |
| resnet-stage3-block1 | resnet | 20 | 20 | 20 | 20 | 0 | 314.4 |
| resnet-stage3-block2 | resnet | 20 | 20 | 20 | 20 | 0 | 222.5 |
| resnet-stage4-block1 | resnet | 20 | 20 | 20 | 20 | 0 | 312.7 |
| resnet-stage4-block2 | resnet | 20 | 20 | 20 | 20 | 0 | 226.7 |
| resnet-stem | resnet | 20 | 20 | 20 | 20 | 0 | 269.5 |
| C01 | kernel | 20 | 20 | 20 | 18 | 0 | 586.4 |
| C02 | kernel | 20 | 20 | 20 | 19 | 0 | 566.6 |
| C03 | kernel | 20 | 20 | 20 | 18 | 0 | 509.3 |
| G01 | kernel | 20 | 20 | 20 | 20 | 0 | 221.1 |
| G02 | kernel | 20 | 19 | 20 | 20 | 0 | 240.4 |
| G03 | kernel | 20 | 20 | 20 | 20 | 0 | 259.3 |
| G04 | kernel | 20 | 20 | 20 | 20 | 0 | 229.0 |
| G05 | kernel | 20 | 18 | 20 | 20 | 0 | 231.1 |
| G06 | kernel | 20 | 20 | 20 | 20 | 0 | 233.9 |
| T01 | kernel | 20 | 15 | 20 | 20 | 0 | 223.6 |
| T02 | kernel | 20 | 15 | 20 | 20 | 0 | 226.1 |
| T03 | kernel | 20 | 20 | 20 | 20 | 0 | 250.8 |
| T04 | kernel | 20 | 20 | 20 | 20 | 0 | 256.2 |
| T05 | kernel | 20 | 20 | 20 | 20 | 0 | 231.8 |
| T06 | kernel | 20 | 20 | 20 | 20 | 0 | 252.2 |
| T07 | kernel | 20 | 18 | 20 | 20 | 0 | 400.7 |

Totals: ResNet 0.734 slot-hours of a 4 h cap; kernels 1.366 slot-hours of a 24 h cap.

"Unique" counts distinct (task, sketch, trace) proposals; ReplayTrace may resample
a trace, and duplicates still count against the 20 trials. The five unsuccessful
convolution candidates (C01 ×2, C02 ×1, C03 ×2) built but failed the guest
check. They had no cycle label and did not update the watchdog.

## Reading the results

- **Hardware attribution.** These numbers belong to the F2 60 MHz DDR3+LLC
  image only. They are not results for the selected U250 30 MHz target, and
  they do not transfer to other memory models.
- **E2E vs chunks.** The connected parent is the only end-to-end number. The sum
  of the 18 chunk baselines (2,258,803,536) is not the parent's baseline
  (2,236,439,673). Fusion 0 wins 11 chunks yet slows the parent by 2.96%.
- **Tuned equals baseline.** For `resnet-stem` and `resnet-stage3-block1` the
  tuned ELF differs in bytes but measures the identical cycle count. No sampled
  trace beat the measured default on any of their tasks, so the database kept the
  default schedule. For T02 the extracted task improved by one cycle
  (1,116 → 1,115), but the complete 32-head call got 2.68% slower. Task labels do
  not predict complete-call time.
- **Alternative plans** reuse the fusion-2 database. They were not tuned
  separately, because their contractions were already owned. Each was measured
  once; repeat agreement on baseline and tuned runs (always identical) is the
  only basis for a single run being representative.
- **Search strength.** 20 random ReplayTrace proposals per owner with an
  engineered default is a small, seeded search, not an optimum. The large
  convolution gains show that the default schedule is weak for C01–C03, not that
  search converged.
- **Accounting scope.** Slot-hours count the tuning campaigns only (`tune-*`).
  Qualification, pilot, the stage-2 confirmations and the E2E runs are outside
  the ledger, although the handoff counted confirmations against DSE time.
  Even with them, total use stays far below both caps.
- **Not covered.** Full TinyLlama and SmolVLA were not run. The diagnostic
  Spike oracle is not the spec-pinned libgemmini oracle. Per-program
  software-spec admission is still open; see [ARCHITECTURE.md](ARCHITECTURE.md).

## Re-checking without F2 (a26, 2026-11-04)

Without F2 access the cycle counts cannot be re-measured. These checks were rerun
on the a26 build host at the F2 pins (TVM `f3cfd1a7d`, Merlin `5998ce4ef`); the
evidence is in the [handoff regression product][v]:

- The regenerated G01 and T01 `program.json`/`fixture.npz` are byte-identical
  to the F2 study's recorded sha256.
- A fresh adapter from the pin has the same `adapter_sources`, `headers` and
  `public_symbols` as the F2 receipt. The objects differ because a26 uses GCC
  13.2.0 and F2 used 16.1.0, so ELF bytes also differ: G01 baseline `2d8a564f…`
  here vs `a676c0aa…` on F2.
- G01 and T01, both baseline and tuned from the published F2 databases, build
  with Gemmini placement. They pass on Spike and reject their wrong oracle using
  the plain libgemmini `ea8f7ed7` plugin (`simulator-a26`), not the repaired-RNE
  plugin behind SPIKE_HANDOFF_EVIDENCE (step 4).
- `db-resnet` equals the union of the ten owner databases (above).

[v]: ../../../../out/artifacts/verification/gemmini-tvm-handoff-regression/v1/
