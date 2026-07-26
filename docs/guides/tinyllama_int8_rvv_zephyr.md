---
title: TinyLlama int8 on multicore RVV under Zephyr — end to end
kind: guide
status: current
owner: runtime
last_verified: 2026-07-25
related: [getting_started, rvv_e2e, zephyr, model2mlir, reproducibility, compilation_strategies]
code_refs:
  - merlin/python/merlin/compile_cli.py
  - merlin/python/merlin/llvmlower/pipeline.py
  - merlin/python/merlin/runtime/backends/zephyr_model.py
  - merlin/runtime/c/libomp_zephyr.c
  - merlin/runtime/c/omp_static_schedule.h
  - merlin/python/merlin/rvvgen/k1.py
  - merlin/python/merlin/targetgen/publish.py
  - build_tools/scripts/k1_multicore_scaling.py
  - build_tools/scripts/check_repro_env.py
---

# TinyLlama int8 on multicore RVV under Zephyr

Compile **TinyLlama in int8 (W8A8, real quantized compute)** with Merlin's tuned RVV schedule, and
run it end to end on **multiple RISC-V cores under Zephyr** — as a one-shot inference, or as a
sustained loop for application development.

This guide is the whole path on a fresh machine. It assumes nothing except the base install in
[Getting started](getting_started.md); every stage states its oracle and what a `not_run` means.

> **The one thing to know up front.** There are four places a number can come from here and they
> are not interchangeable. **spike** proves correctness and cannot tell you anything about speed —
> it simulates every hart at full speed, so an idle or spinning core costs exactly what a working
> one does. **Verilator** is cycle-accurate but runs ~10⁴ cycles/second, several orders of
> magnitude short of a whole inference; it certifies the *mechanism*, not the model.
> **FireSim** runs the *same* RTL on an FPGA at ~25 MHz, which is the only way to get
> cycle-accurate whole-model numbers out of **our own** SoC. **Real silicon** (the SpacemiT K1)
> gives wall-clock on a shipping chip, but that chip is a fixed vendor design, not ours.
> The guide says which one is in play at every step, and never quotes a speedup from spike.

## Status — read before you rely on this

**Full TinyLlama int8 currently FAILS its accuracy gate on the K1 board.** This is an open
defect, not a gate artefact, and it is under investigation. What is measured today:

| path | result |
|---|---|
| `tiny_llama_int8_full`, **fp32** weight-only, host | **cos = 1.0** (rel 1.5e-5) — bundle, golden and dequant scales are all correct |
| `tiny_llama_int8_full`, **int8**, host | cos = 0.976 — ordinary W8A8 degradation |
| `tiny_llama_int8_full`, **int8**, K1 board | **cos = 0.484** — fails |
| every isolated int8 contraction (weight×act, act×act, batched attention, 50× scale spread, 100× activation outliers) | spike is **bit-identical to host** — the compiled RVV int8 codegen is faithful |

So the fault needs the whole model rather than any single operation, and it is **not** the
recipe (two different recipes give bit-identical wrong output), **not** the bundle or golden,
and **not** the isolated int8 kernels.

Everything else in this guide is verified and unaffected: the spike correctness path, the
multicore images (bit-exact 1 vs 2 vs 4 harts), sustained inference (drift 0.0), and the K1
scaling curve. `small_llama` int8 passes end to end. Until the above is closed, treat
whole-model TinyLlama **int8 on the board** as known-bad and use the fp32 path or spike.

## 0. Prerequisites

Do the base install and `.env` setup in [Getting started](getting_started.md) first, then confirm
what will actually run here:

```bash
.venv/bin/python build_tools/scripts/check_repro_env.py
```

For this guide you want these capabilities `[OK]`:

| capability | what it gives you | if missing |
|---|---|---|
| `llvm_m2m_toolchain` | model2MLIR + clang-23 (the lowering) | **required** — nothing below runs |
| `spike_rv64gcv` | bit-exact RVV correctness | **required** for §3–§5 |
| `zephyr_spike` | the Zephyr whole-model build (`ZEPHYR_BASE`, SDK 0.17.0, `MERLIN_CHIPYARD`) | **required** for §4 onward |
| `zephyr_multicore` | the multicore image (adds the in-repo OpenMP shim) | required for §5 |
| `saturn_multicore_verilator` | cycle-accurate multicore Saturn RTL | optional — §6 skips |
| `k1_board` | real silicon cycles | optional — but §7 is the ONLY honest speedup source |

The Zephyr workspace is [`ucb-bar/zephyr-chipyard-sw`](https://github.com/ucb-bar/zephyr-chipyard-sw)
(branch `dev`); point `MERLIN_ZEPHYR_SW` at the checkout and `ZEPHYR_BASE` at its
`zephyr_ws/zephyr`. It needs `git submodule update --init` plus its own conda env and the Zephyr
SDK — see that repo's README.

## 1. Get the model bundle

Quantization and export happen in **model2MLIR, not Merlin**. The int8 (W8A8) bundle carries the
model, the weights, and the goldens the gates use:

```bash
ls out/artifacts/recaptures/tiny_llama_int8_full/
# model.mlir  weights.safetensors(+manifest)  inputs.npz  extra.npz
# golden.npy  golden_w8a8.npy
```

`merlin-compile` resolves (and, with `MERLIN_M2M_DIR` set, auto-captures) this for you, so you only
need this step to check what you have. `_full` is the real 22-layer TinyLlama-1.1B-Chat with its
pretrained weights; `_consistent` is an older **2-layer, randomly initialised** capture kept for
regression — useful as a fast smoke test, useless as evidence about the real model.

**The two goldens are not interchangeable, and confusing them looks exactly like a bug.**
`golden.npy` is a **weight-only-int8** reference: model2MLIR quantizes with torchAO
`int8_weight_only`, so the weights are int8 but the activations stay fp32. Merlin's int8 path
computes **W8A8** — activations quantized too. Grading a W8A8 run against `golden.npy` therefore
measures activation-quantization error, which on a real pretrained LLM is large and on a small
random-init model is nearly zero. That is what `golden_w8a8.npy` is for, and it is why the gate
below has two tiers. If a bundle lacks it, regenerate with
`run_model(bundle, workdir, int8_compute=True)` and `np.save` the result.

int8 is the only measured-working quantized format today — `fp8`/`int4` are a documented plan, not
a claim. See [model2MLIR frontend](model2mlir.md).

## 2. Get the tuned RVV schedule

The codegen package (transform schedule + knobs) is published as its own repo:

```bash
git clone -b <branch> git@github.com:ucb-bar/rvv-mlir.git rvv-mlir
```

Clone the default branch first — it is a landing page listing every published branch, its package,
its datatype and its certification status. Pick the `int8_w8a8` champion branch from that table.

A package is a **vector schedule, not a dialect**:

- `payload/schedule.mlir` — the transform-dialect schedule (tiling + vectorization of the contractions)
- `payload/knobs.yaml` — cflags, `dtype_strategy`, `op_match` tile/vector sizes, `lmul_policy`, and
  the `expected_instructions` the emitted code must actually contain
- `.merlin/` — provenance and the recorded certification

In-repo you do not need the clone at all: `merlin-compile` resolves the certified champion for the
requested datatype automatically. Pass `--package <dir>` to pin a specific one.

## 3. Compile and verify on spike (correctness)

```bash
merlin-compile --workload tiny_llama --dtype int8 --target rvv --run spike --verify --json
```

This lowers the bundle through the package's schedule with the **integer** datapath
(`llvmlower/quant_passes.py`: i8×i8→i32 `vwmacc` plus integer softmax/norm), builds a Zephyr image,
runs it on spike, and gates the output.

The gate is multi-tier, because a W8A8 model legitimately fails an fp32-tight threshold:

| tier | reference | bar |
|---|---|---|
| T1 `w8a8` | `golden_w8a8.npy` | cos > 0.999 **and** rel < 1e-2 **and** per-element max-rel < 5% |
| T2 `fp32` | `golden.npy` | cos > 0.99 **and** top-1 argmax matches **and** per-element max-rel < 5% |

`ok = T1 or T2`. The per-element term exists because aggregate cos and rel can both look perfect
while a single element is catastrophically wrong — a measured fp16-accumulate GEMM once passed at
cos 0.9999986 while being 1209% wrong on one element.

## 4. Sustained inference

A single inference tells you almost nothing about a service: it cannot show a per-iteration cost
that creeps (arena growth, allocator churn), and its cold caches make the one number you quote
either optimistic or pessimistic depending on which you pick.

```bash
merlin-compile --workload tiny_llama --dtype int8 --target rvv \
               --run spike --iters 20 --warmup 3 --verify --json
```

The image runs `warmup + iters` invocations **against the same arena** and emits one
`METRIC iter_cycles <i> <cycles>` line per timed iteration. The host reports `min` / `median` /
`p95` and — the point of the mode — `drift`, the late-third median against the early-third median.
A rising drift means the arena is not really being reused.

Report the **median**, never a single run: the K1 noise floor alone is ≥1.9%.

Measured, `small_llama` int8 on spike (`--iters 5 --warmup 1`), abridged:

```json
"status": "verified",
"sustained": { "n": 5, "min": 369131311, "median": 369131311, "p95": 369131311,
               "max": 369131311, "drift": 0.0 },
"verify":    { "gate_ok": true, "w8a8_cos": 0.9999999, "w8a8_rel": 0.0, "w8a8_max_rel": 0.0 }
```

`drift: 0.0` with every iteration identical is what a correctly reused arena looks like — spike is
deterministic, so any nonzero drift there would be real allocator churn rather than measurement
noise. `w8a8_rel: 0.0` means the integer datapath reproduces the W8A8 golden exactly, not merely
within tolerance.

## 5. Multicore

```bash
merlin-compile --workload tiny_llama --dtype int8 --target rvv \
               --run spike --harts 4 --iters 20 --warmup 3 --verify --json
```

`--harts N` changes both halves of the stack:

- **Compiler.** An outer `scf.forall` is layered *under* the package's schedule and lowered to
  `omp.parallel` + `omp.wsloop`, so the object carries real RVV vectors **and** `__kmpc_*` calls.
  Only parallel dims are split — matmul over N, batch_matmul over B. K is the reduction dim and is
  never tiled, because splitting it would race the accumulator.
- **Runtime.** `merlin/runtime/c/libomp_zephyr.c` provides the OpenMP surface Zephyr does not have,
  over one COOP worker **pinned 1:1 per hart**.

### The pinning is a correctness requirement, not tuning

Two independent reasons, both learned the hard way:

1. The Saturn fork's `arch/riscv/core/v.c` corrupts vector state across a context switch — the bug
   `samples/merlin_model_runner/prj.conf` documents, where yolov8n produced two *different* wrong
   hashes. Pinned COOP workers that **never block** never enter that path. Note that removing
   preemption is not enough: a blocking `k_sem_take` is a *voluntary* switch through the same code,
   which is why the pool spins on an atomic generation counter instead.
2. The emitted IR passes `__kmpc_global_thread_num()` — not the outlined region's `tid` argument —
   to the worksharing loop. The shim derives that id from the running hart, which is exact *only*
   because workers are pinned 1:1. Get it wrong and two harts silently claim the same output slice.

Neither failure crashes; both produce a plausible wrong answer. So the gate is **bit-exactness**:

```bash
.venv/bin/python -m pytest merlin/tests/runtime/test_zephyr_multicore.py -q
```

1 hart and N harts must agree **bit for bit**. That is a legitimate bar here, not an overreach: the
split touches only parallel dims and reductions stay serial, so there is no reordering that could
explain a difference. Any difference is corruption.

The gate runs a **whole model on spike** (~12 s per thread count) and, when a board is present, the
same comparison on the **K1**. Measured, `small_llama` int8:

| threads | spike cycles | vs 1 hart | differing elements |
|---|---|---|---|
| 1 | 369,140,000 | — | — |
| 2 | 192,395,001 | 1.92× | **0** / 2048 |
| 4 | 97,374,999 | 3.79× | **0** / 2048 |

(Those spike cycle counts show the *work split*, not speed — spike is IPC≈1. For real cycles see
§6b; for wall-clock see §7.)

### Two bugs this gate exists to catch — and once failed to

Worth reading before trusting a multicore image, because both produced silence rather than an error
and neither is visible in a single-kernel test:

1. **Nested parallel regions.** merlin's lowering does emit them — 4 of the 358 fork sites in a
   `small_llama` int8 module sit *inside* an outlined region. An inner fork that touches the shared
   generation counter releases the workers into a region their master is not joining, and the
   master then waits forever on a generation they already raced past. Nested regions are now run
   inline as a team of one, which is what libomp does with `OMP_NESTED=false`.
2. **Worker stack size.** An OpenMP worker runs the *same* outlined model code as the master,
   spilling scalable vectors with dynamic stack adjustment — it is not "just a loop body". At the
   original 256 KB a worker overflowed into the master's live frames (the pool array is laid out
   directly above the master's stack, and there is no MPU on these SoC configs to catch it), and
   the *master* then faulted on a corrupted register 143 regions later. Worker stacks now match
   the master's 8 MB.

Both bugs needed a **whole model** to appear at all, which is why the shim gate is no longer a
single synthesized kernel. If a multicore run ever goes quiet again, rebuild with
`MERLIN_OMP_DEBUG_SPLIT=1`: the shim dumps every worksharing partition, reports which worker it is
waiting on and what that worker last published, and gives up on a wall-clock deadline instead of
spinning forever.

## 6. Cycle-accurate multicore RTL (optional)

Every stock Saturn config in chipyard is single-core, and the only multi-tile SoC gives just one
tile a vector unit. `generators/chipyard/src/main/scala/config/MerlinSaturnConfigs.scala` adds
4-tile configs where **every** tile has its own Saturn unit (vLen=256, dLen=128 — matching the K1,
so a schedule tuned on the board transfers without a re-tune):

```bash
cd $MERLIN_CHIPYARD/sims/verilator
make CONFIG=MultiSaturnV256D128ShuttleConfig -j16     # ~minutes; produces simulator-chipyard.harness-<CONFIG>
```

Confirm the elaborated SoC is what you asked for — 4 harts, all with `v` and `zvl256b`:

```bash
grep -E "cpu@|riscv,isa" generated-src/chipyard.harness.TestHarness.MultiSaturnV256D128ShuttleConfig/*.dts
```

Measured on this sim — Zephyr SMP booting on 4 harts and running the level-synchronous multicore
RVV dispatch executor (`samples/merlin_mt_rvv_dispatch`):

```
Merlin multicore RVV dispatch executor: 3 dispatches over 4 harts (chipyard_riscv64)
level 0: 2 dispatch(es) done      level 1: 1 dispatch(es) done
RESULT: PASS (1024/1024 elems correct)
```

That is the multicore-RVV **mechanism** certified against cycle-accurate RTL: several V-using
threads on distinct Saturn tiles, with a dependency barrier between levels, agreeing with a
scalar reference.

Be precise about what it does and does not establish. The sample is **fp32** (`vfmacc` at
`e32/m8`) and its check is a 1e-2 RELATIVE tolerance, not bit-exactness — so it says nothing
about int8 numerics, and "1024/1024 correct" means within 1%, not identical. It is evidence
that threads on separate Saturn tiles execute V correctly and synchronize correctly; it is not
evidence about any compiled model's accuracy.

**Scope — read this before pointing it at a model.** Booting Zephyr and running those three
dispatches took ~40 minutes of wall clock. A whole 22-layer TinyLlama inference is ~10¹⁰ cycles;
at that rate it is not hours but weeks. Verilator here certifies the multicore RVV **mechanism**
and small kernels, nothing more. Whole-model functional truth comes from spike (§3–§5) and
whole-model timing from the K1 (§7).

Two practical notes. Redirect the sim's stdout through `stdbuf -o0` to watch it live — it
block-buffers to a file otherwise, and a run killed by a timeout loses everything it had
buffered. And do **not** redirect its stdin from `/dev/null`: the harness reads stdin for UART
and exits immediately on EOF.

## 6b. FireSim — the same SoC on an FPGA (whole-model RTL)

Verilator certifies the mechanism but cannot reach a model (~10⁴ cycles/s). The **same** Saturn
SoC on an FPGA runs at tens of MHz, which is what makes whole-model multicore RVV measurable
against real RTL rather than a simulator.

Build the bitstream (hours of Vivado, unattended). `build_tools/chipyard/setup_multicore_saturn.py`
installs the target configs and Alveo U250 recipes; then add the recipe to `builds_to_run` and:

```bash
cd $MERLIN_CHIPYARD/sims/firesim && source sourceme-manager.sh --skip-ssh-setup
cd deploy && ./firesim buildbitstream
```

Measured for `alveo_u250_firesim_dual_saturn_v256d128` (2 tiles, a Saturn unit on **each**):
**28.65% LUTs** of the U250 (495k/1.73M), FFs 6.7%, BRAM 8.9%, DSP 1.7%, and timing **closed at
25 MHz (WNS +0.034 ns, TNS 0.000)**. The FireSim shell is 6.7% of that, so the 4-tile variant
should also fit (~47% projected). When the build finishes, register the entry it prints into
`sims/firesim/deploy/config_hwdb.yaml` and point `config_runtime.yaml`'s `default_hw_config` at
it — note that file is SHARED, so restore it if others use the machine.

### Running — always through the queue

There is one physical FPGA, so **every run must go through the job queue**, never a direct
`firesim` invocation:

```bash
/scratch2/agustin/firesim_queue/bin/firesim-queue status      # daemon must be ALIVE
/scratch2/agustin/firesim_queue/bin/firesim-queue daemon      # start it if not (leave running)
```

`zephyr_model.run_on_firesim()` already defaults to `queue=True`. It resolves ModelBlaster's
runner via `MERLIN_MODELBLASTER` and FireSim's paths from `MERLIN_CHIPYARD`, all through `.env`.

Three failure modes worth recognising, because none of the errors names its real cause:

- **`ModuleNotFoundError: No module named 'modelblaster'`** — `MERLIN_MODELBLASTER` is unset or
  wrong. It names neither the setting nor the path it wanted.
- **`insmod: ERROR: could not load module poll_mode=1`** at `INFRASETUP` — the XDMA kernel module
  is not loaded. FireSim's helper searches for `xdma.ko`, but a modern kernel ships
  `xdma.ko.zst` (compressed), so the search finds nothing and `poll_mode=1` is mistaken for the
  module path. Check with `lsmod | grep xdma`.
- **`cmake: error while loading shared libraries: libidn.so.11`** in some *later, unrelated*
  step — `sourceme-manager.sh` puts Xilinx's bundled cmake 3.3.2 first on `PATH` and it is
  linked against a library no current distro ships. Merlin's own build steps now probe cmake by
  running it, but anything else you run in that shell will hit this.

### Measured — multicore RVV on the FPGA

`small_llama` int8, RVV backend, dual-Saturn v256/d128 bitstream at 25 MHz:

| harts | cycles | speedup | vs 1 hart |
|---|---|---|---|
| 1 | 362,862,321 | 1.00× | — |
| 2 | 230,548,059 | **1.574×** | **bit-identical** |

Note the honesty gap against spike, which reports 1.92× for the same pair: spike charges every
hart the same price per instruction and models no memory system, so it sees the *work* split but
not what the split costs. 1.574× is the number that includes real DRAM and cache behaviour, and it
is the one to quote for the SoC.

## 7. Wall-clock speedup on shipping silicon

§6b already answers "is it faster **on our SoC**", cycle-accurately. This step answers the
different question of wall-clock on a chip you can buy — and it is the only one that can, because
the K1's memory system, clocks and core count are a real product rather than a bitstream.

```bash
.venv/bin/python build_tools/scripts/k1_multicore_scaling.py \
    --model tiny_llama --dtype int8 --threads 1,2,4,8 -n 3
```

One binary, `OMP_NUM_THREADS` varied at run time — rebuilding per point would put a different
object under each measurement. Every point is accuracy-gated; a thread count that fails carries no
timing. It reports speedup, parallel efficiency, the Amdahl serial fraction the measurement
implies, and the parallel-region count, so a poor curve can be *attributed* rather than guessed at.

Measured, `small_llama` int8, K1 (8 cores, VLEN=256), median of 3:

| threads | median | spread | speedup | efficiency |
|---|---|---|---|---|
| 1 | 222.0 ms | 0.3% | 1.00× | — |
| 2 | 118.4 ms | 2.1% | 1.88× | 94% |
| 4 | 66.5 ms | 0.5% | 3.34× | 83% |
| 8 | 53.3 ms | 25.1% | 4.17× | 52% |

How to read it. The implied serial fraction is **6.6%, the same at T=2 and T=4**, and it predicts
the 4-core result exactly (1/(0.066 + 0.934/4) = 3.34). That fraction is the reductions the
lowering deliberately leaves serial, so 4 cores is at the ceiling the design implies rather than
leaving something on the floor. The ~358 fork/joins per inference are **not** the bottleneck — if
they were, 2 cores could not reach 94% efficiency.

**Do not quote the 8-core number.** Its 25% run-to-run spread (against 0.3–2% elsewhere) and its
shortfall against its own Amdahl prediction (5.47× predicted, 4.17× measured) point at memory
bandwidth across the K1's two 4-core clusters. It needs more reps and a bandwidth probe before it
means anything.

## What each stage proves

| stage | oracle | establishes | cannot tell you |
|---|---|---|---|
| host `run_model` | torch golden | the lowering is numerically right | anything about RVV |
| spike rv64gcv | Merlin reference | real RVV instructions, bit-exact | **speed** (idle harts cost full price) |
| Zephyr N-hart on spike | the 1-hart image | the parallel split is not corrupting | **speed** |
| Verilator multicore Saturn | RTL | the mechanism, cycle-accurate | a whole model (~10⁴ cycles/s) |
| **FireSim multicore Saturn** | the same RTL on an FPGA | **whole-model cycles on our own SoC** | wall-clock on shipping silicon |
| K1 board | on-device | real wall-clock and real speedup | anything about *our* SoC |

The pattern to reproduce is stage-over-stage agreement, not a single absolute number.

## Troubleshooting

**`k_thread_cpu_pin` fails / pool clamps to fewer threads.** The image is asking for more harts
than the SoC has. The DT overlay is generated from the hart count; check `CONFIG_MP_MAX_NUM_CPUS`
in the generated `prj.conf` against your board or `spike -pN`.

**A multicore image boots but never prints.** Zephyr's SMP boot waits for every CPU in
`CONFIG_MP_MAX_NUM_CPUS`. Running a 2-hart image under `spike -p1` hangs before `main()` — match
the hart counts.

**Multicore output differs from single-hart.** Do not tune around it. That is the bit-exactness
gate firing, and it means either the hart→thread mapping or the vector-state handling is wrong;
both silently corrupt results. Fall back to `--harts 1` (unaffected) and report it.

**Multicore is slower on spike.** Expected, and not a regression: spike executes the spinning
worker's instructions at full speed. Never take a speedup number from spike; use §7.
