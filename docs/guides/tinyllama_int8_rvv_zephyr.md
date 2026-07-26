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

> **The one thing to know up front.** There are three places a number can come from here and they
> are not interchangeable. **spike** proves correctness and cannot tell you anything about speed —
> it simulates every hart at full speed, so an idle or spinning core costs exactly what a working
> one does. **RTL simulation** (Verilator) is cycle-accurate but runs ~10⁴ cycles/second, which is
> several orders of magnitude short of a whole inference; it certifies the *mechanism*, not the
> model. **Real silicon** (the SpacemiT K1) is the only oracle that times a whole model on many
> cores. The guide says which one is in play at every step, and never quotes a speedup from spike.

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
# model.mlir  weights.safetensors(+manifest)  inputs.npz  golden.npy  extra.npz
```

`merlin-compile` resolves (and, with `MERLIN_M2M_DIR` set, auto-captures) this for you, so you only
need this step to check what you have. `_full` is the real 22-layer TinyLlama; `_consistent` is an
older truncated capture kept for regression.

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

The gate runs on the **K1 board**, not spike — same comparison, but seconds instead of tens of
minutes, because spike has to simulate the spinning worker at full speed. (A spike leg exists
behind `MERLIN_RUN_SLOW=1`; a 2-hart run of a model that takes <200 s at 1 hart ran past 25
minutes there, which is not a gate anyone will actually run.)

Measured, `small_llama` int8, one binary with `OMP_NUM_THREADS` varied:

| threads vs 1 | differing elements |
|---|---|
| 2 | **0** / 2048 |
| 4 | **0** / 2048 |
| 8 | **0** / 2048 |

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

**Scope — read this before pointing it at a model.** A whole 22-layer TinyLlama inference is ~10¹⁰
cycles; at RTL-simulation speed that is not hours but weeks. Verilator here certifies the multicore
RVV **mechanism** and small kernels, nothing more. Whole-model functional truth comes from spike
(§3–§5) and whole-model timing from the K1 (§7). Redirect the sim's stdout through `stdbuf -o0` if
you want to watch it live; it block-buffers otherwise.

## 7. Speedup — real silicon only

This is the only step that can honestly answer "is it faster".

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
| K1 board | on-device | real cycles and real speedup | bit-exactness vs a simulator |

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
