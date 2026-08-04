---
title: Vision, audio and control workloads on Kodiak — multicore RVV under Zephyr
kind: guide
status: current
owner: runtime
last_verified: 2026-08-03
related: [tinyllama_int8_rvv_zephyr, model2mlir, rvv_e2e, zephyr, compilation_strategies]
code_refs:
  - merlin/python/merlin/compile_cli.py
  - merlin/python/merlin/rvvgen/apply.py
  - merlin/python/merlin/llvmlower/frozen_blocks.py
  - merlin/python/merlin/llvmlower/impr_features.py
  - merlin/python/merlin/llvmlower/c_runtime.py
  - merlin/python/merlin/common/mlir_query.py
  - merlin/python/merlin/runtime/backends/zephyr_model.py
  - merlin/tests/rvv/test_vision_workloads_rvv.py
  - merlin/tests/rvv/test_microkernel_shape_policy.py
  - merlin/tests/runtime/test_zephyr_ram_sizing.py
---

# Vision, audio and control workloads on Kodiak

The destination for this guide is the **Kodiak test chip** — a `chipyard` SoC with **3 working
Saturn-vector cores** and a bounded DRAM budget. [TinyLlama int8 on multicore
RVV](tinyllama_int8_rvv_zephyr.md) walks the same pipeline for a decoder-only LLM on spike, FireSim
and the K1; this guide covers the four **non-transformer-only** workloads and what it takes to fit
and run them on a real tapeout.

| workload | upstream | capture unit |
|---|---|---|
| `spectformer` | [`badripatro/SpectFormers`](https://github.com/badripatro/SpectFormers) | one SpectFormer-Ti forward: 224², patch 16, embed 256, depth 12 (4 spectral + 8 attention blocks) |
| `whisper_tiny` | `openai/whisper-tiny` (HF weights) | audio encoder + **one** cross-attending decoder step |
| `lstmnetvit` | [`anish-bhattacharya/vitfly`](https://github.com/anish-bhattacharya/vitfly) | one `LSTMNetVIT` forward, command output only, zero initial LSTM state |
| `deepjscc` | [`mingyuyng/DiffJSCC`](https://github.com/mingyuyng/DiffJSCC) | JSCC encoder + AWGN channel + decoder (**not** the diffusion refinement stage) |

Nothing here is workload-specific in the compiler. Everything below is either a new model2MLIR
decomposition (shared capability) or a policy that derives a decision from the model instead of
pinning it.

## The one fact that drives the design

The certified RVV package declares `op_match: [linalg.matmul, linalg.batch_matmul]`, and everything
the schedule does not match is compiled through `convert-linalg-to-loops` with `-fno-vectorize
-fno-slp-vectorize`. `parallel_transform_schedule` likewise splits only matmul-over-N and
batch_matmul-over-B.

> **An op that does not land on a named contraction gets neither RVV vectors nor `--harts`
> parallelism. It runs scalar, on one core.**

So every op these four models need was decomposed **onto contractions**:

- **conv → im2col gather + `linalg.matmul` + reshape** (padded, strided, dilated, grouped and
  transposed conv all normalize into this path)
- **`rfft2`/`irfft2` → real DFT matmuls** with constant twiddle operands and an explicit
  real/imaginary split (xDSL 0.65 has no complex type; a complex tensor of shape `S` is a real
  tensor of shape `S + [2]` — torch's own `view_as_complex` layout, so `view_as_complex` /
  `view_as_real` are identities)
- **LSTM → per-timestep gate matmuls** (`torch.export` already unrolls the recurrence)

This is also the target-agnostic choice: any target with a matmul provider inherits conv, FFT and
recurrence with no new backend code.

## Building for Kodiak

Kodiak is a `chipyard*` board, so **no compiler or runtime change is needed to target it** — `board`
is already a parameter, the generated CPU overlay derives from the hart count, and a region larger
than the 256 MB default is emitted as a `&ram0` overlay. Board support (`chipyard_kodiak`) lives on
the `kodiak` branch of [`ucb-bar/zephyr-chipyard-sw`](https://github.com/ucb-bar/zephyr-chipyard-sw),
whose bumped `zephyr_ws/zephyr` adds `boards/chipyard/kodiak/`:

```bash
git -C "$MERLIN_ZEPHYR_SW" checkout kodiak
git -C "$MERLIN_ZEPHYR_SW" submodule update
```

Build with `board="chipyard_kodiak"` and `n_harts=3`. Three facts about that board config matter,
and two of them are traps:

| board setting | value | consequence |
|---|---|---|
| `&ram0` | `0x80000000 + 0x10000000` (**256 MB**) | a DTS default, not necessarily the physical DRAM. Merlin's overlay grows the region to what the model needs — confirm the real DRAM before trusting a >256 MB region |
| `CONFIG_MP_MAX_NUM_CPUS` | `2` | merlin's generated `prj.conf` overrides it, so 3 harts works; anything built from board defaults caps at 2 |
| `CONFIG_FPU_SHARING` | `y` | **the setting that silently hangs a Saturn tile** — V-illegal-instruction traps get mis-routed to the FP path and retry forever, with no fault printed. Merlin sets `n` and app config wins; images built from board defaults are exposed |

**3 harts is safe on a non-divisible extent.** `parallel_transform_schedule` splits with
`tile_using_forall ... num_threads`, so MLIR emits a `ceil(dim/3)` tile with an `affine.min` bound on
the last thread — not a fixed tile size that would drop work.

Running on the chip uses the UART loader from that branch (`pyuartsi --elf <elf>` against the board's
USB serial device), with its reset script and `scripts/record_power.py` sampling in parallel — which
is what makes **energy per inference** measurable on Kodiak rather than only cycles. The USB devices
are attached to the Kodiak host, so the runner executes there (same shape as the K1 route) and parses
the usual `OUT`/`METRIC`/`DONE` protocol.

One `march` note: the certified package compiles with plain `-march=rv64gcv`, i.e. **VLEN=128**,
which matches spike's default. Saturn on Kodiak is VLEN=256, where the same fixed-width vectors halve
their LMUL. That should be a deliberate `_zvl256b` choice rather than an accident — see the K1 VLEN
note in the TinyLlama guide.

## What fits in Kodiak's DRAM

Footprint = the linked image (`.text` + `.data`, where the weights blob lands, + `.bss` + `noinit`)
plus the activation arena Zephyr's malloc claims from the leftover. Weights are exact (safetensors
payload); activation peaks are measured by liveness over the captured IR
(`mlir_query.activation_peak_bytes`); image overhead is `size` on the linked ELFs (8.6 MB, mostly the
8 MB worker stack).

| workload | dtype | weights | peak activations | total | 256 MB | 512 MB |
|---|---|---:|---:|---:|---|---|
| `deepjscc` | int8 | 0.8 MB | 12.8 MB | **22 MB** | yes | yes |
| `deepjscc` | fp32 | 1.0 MB | 12.8 MB | **22 MB** | yes | yes |
| `spectformer` | int8 | 10.5 MB | 3.1 MB | **22 MB** | yes | yes |
| `spectformer` | fp32 | 35.5 MB | 3.0 MB | **47 MB** | yes | yes |
| `lstmnetvit` | int8 | 6.3 MB | 11.3 MB | **26 MB** | yes | yes |
| `lstmnetvit` | fp32 | 13.6 MB | 18.0 MB | **40 MB** | yes | yes |
| `whisper_tiny` | int8 | 116.8 MB | 210.4 MB | **336 MB** | no | yes |
| `whisper_tiny` | fp32 | 220.9 MB | 210.4 MB | **440 MB** | no | yes, 86 % full |
| `smolvla` | int8 | 482.6 MB | 199.0 MB | 690 MB | no | no |
| `tiny_llama` (22 layer) | int8 | 1241.7 MB | 312.7 MB | 1563 MB | no | no |

So **three of the four fit either way**; `whisper_tiny` needs the DRAM to really be 512 MB. Its peak
is two buffers: a 76 MB `tensor<384x51865xf32>` vocab projection materialized as fp32 inside the
*int8* bundle, and 51.5 MB `tensor<6x1500x1500xf32>` encoder attention matrices — keeping the
projection in int8 and chunking encoder attention is what would take it from "tight" to
"comfortable".

TinyLlama and SmolVLA do not fit, and it is weights, not activations. Note that the GGUF bundles
(`tiny_llama_gguf_q4k` / `q6k` / `q8`) carry **4196 MB** of weights — ingest dequantizes to fp32 — so
they are *not* a footprint win over int8's 1242 MB. Getting an LLM onto this chip needs a real
sub-8-bit weight path or weight streaming.

## The correctness ladder

Run it in this order; each stage has a different oracle.

```bash
merlin-compile --workload spectformer --dtype int8 --target rvv --run host  --verify --json
merlin-compile --workload spectformer --dtype int8 --target rvv --run spike --verify --json
merlin-compile --workload spectformer --dtype int8 --target rvv --run spike --harts 3 \
               --iters 20 --warmup 3 --verify --json
```

A missing bundle is captured automatically, in the workload's own venv (resolved from its
`capture.toml`). Generate the W8A8 reference **before** grading an int8 run:

```bash
.venv/bin/python build_tools/scripts/make_w8a8_golden.py spectformer_int8_full
```

and check `tier_ok == "w8a8"`. Grading a W8A8 run against the weight-only `golden.npy` measures
activation-quantization error, not correctness — `merlin-compile` warns when the W8A8 reference is
missing, and the gate reports which tier carried the verdict.

## Status, measured

`spike`, 1 hart, int8 (W8A8), graded against the W8A8 reference:

| workload | cycles | `w8a8_cos` | `w8a8_rel` | verdict |
|---|---:|---:|---:|---|
| `spectformer` | 4,198,885,000 | 1.0 | 0.0 | **verified** — bit-exact |
| `deepjscc` | 485,985,000 | 0.9176 | 0.889 | run_mismatch — open RISC-V-side defect |
| `lstmnetvit` | 857,315,000 | 0.9943 | 0.259 | run_mismatch — same |
| `whisper_tiny` | — | — | — | builds and lowers; spike run exceeds the default timeout |

Multicore (`lstmnetvit`, int8, spike): **857,315,000 cycles at 1 hart → 381,565,000 at 4 harts
(2.25×)**, with every gate metric identical to 16 digits across the two — i.e. the parallel split is
bit-exact, which is the only thing a multicore run is allowed to prove about correctness. spike
cannot tell you anything about speed; the 3-hart cycle numbers have to come from the chip.

Host (`--run host`, all four, int8 and fp32): **verified**, `tier_ok = "w8a8"`, `w8a8_rel = 0.0`.
All four lower with **zero opaque ops** (`spectformer` 1240 linalg ops, `whisper_tiny` 1296,
`lstmnetvit` 653, `deepjscc` 380).

### The open defect

`deepjscc` and `lstmnetvit` diverge **only on RISC-V**. Ruled out by measurement, not argument:

| hypothesis | test | verdict |
|---|---|---|
| quantization math / wrong golden | `--run host` (same int8 passes) | ruled out — `w8a8_rel = 0.0` |
| the vectorized IR itself | same prepared IR + schedule + features, **x86 backend** | ruled out — `cos = 1.000000000`, 0 bad elements |
| the shape-adaptation policy | `deepjscc` keeps the frozen block; `spectformer` *adapts* and is bit-exact | ruled out |
| the externalized-buffer path | `lstmnetvit` has **0** externalized buffers and still diverges; `deepjscc`'s 27 arg-table rows match the manifest offsets exactly | ruled out |
| stack overflow from `hoist-static-allocs` | all 40 allocas in the image are memref *descriptors*; intermediates are heap | ruled out |
| output reordering / layout | spike's values do not occur in the reference at all, and one exceeds its max | ruled out — real arithmetic difference |

What remains is RISC-V backend codegen. The live discriminator is an N-block sweep on one bundle
(`deepjscc`'s matmul extents are all multiples of 16, so NR ∈ {4, 8, 16} are all legal — a clean
single variable): the **passing** `spectformer` runs NR=8, `deepjscc` NR=16 (which emits
`vsetvli e32,m4` — LMUL 4), `lstmnetvit` NR=2. Do not quote either model's numbers as an accuracy
result until this closes.

## What the compiler learned (target-agnostic)

### A register block is a claim about extents

`from_strategy._rvv_blocking_lowers` encodes, as a measured predicate, that a block masking a
parallel dim of a contraction does not lower at all on the integer path (LLVM-23 rejects the multi-op
`vector.mask` a masked `transfer_write` needs) and degrades ~34× on fp32. The certified champion pins
its block as constants **inside** a hand-frozen feature, so the one package anyone compiles with was
the one package that could not adapt — and a model whose extents the frozen tails do not fit simply
failed to build.

`llvmlower/frozen_blocks.py` now describes that point's caps and its per-op-class realization as data
(sourced from the feature's own module constants, so it cannot drift from the registration), and
`rvvgen.apply` re-resolves it against the workload's real contraction extents. The adaptation is
deliberately minimal — each op class is checked independently and the frozen block is kept wherever
it holds:

| workload | resolved feature |
|---|---|
| `tiny_llama`, `small_llama`, `deepjscc` | `accumulator_resident_wholemodel_vf` (**unchanged**, byte-identical codegen) |
| `spectformer` | `accum_resident_v3p_1_8_1_16_16` (matmul N=8 — the DFT's frequency bins) |
| `lstmnetvit` | `accum_resident_v3p_1_2_1_2_16` |
| `whisper_tiny` | `accum_resident_v3p_1_16_x_x_16` (see below) |

That `tiny_llama` is unchanged is not luck: the policy *derives* exactly the frozen block from
`tiny_llama`'s own extents. Substitutions are printed and recorded in the result under
`features_shape_adapted` — they change the emitted kernel, so they are never silent.

### Sometimes the honest answer is "not vectorized"

`whisper_tiny`'s `batch_matmul` class holds both a 1500-wide encoder attention and a single-token
decode step whose N=1. The only block legal for *every* extent in the class is one lane wide — and a
1-lane block is not a vectorization: it emits

```mlir
vector.contract {iterator_types = ["reduction"]} vector<1xi8>, vector<1xi8> into i32
```

a parallel-dim-free dot product that no `lower_contraction` strategy matches, so the build dies at
LLVM translation after a full compile. The policy therefore **declines the class** (`x_x` in the
feature name): those contractions go to `convert-linalg-to-loops` while the other class keeps its
vectors. `whisper_tiny`'s 67 matmuls — the projections, where the compute is — vectorize at
`[1, 16]`; its 24 attention `batch_matmul`s run scalar. `merlin-compile` prints the unclaimed classes
and records them in `features_shape_adapted.unclaimed_op_classes`.

This is a **per-op-class** decision, so a class mixing a tiny and a large extent is clamped by its
smallest member. Per-**op** blocking (tag each contraction with its own legal block, match by
attribute in the schedule) would recover whisper's encoder attention and is the identified follow-up.

### RAM is provisioned from the measured working set

Sizing the region's headroom from weight bytes alone assumes activations are small relative to
parameters, which is false for an encoder whose attention matrices dwarf its weights.
`mlir_query.activation_peak_bytes` measures peak simultaneously-live intermediate bytes by liveness
over the captured IR (every tensor is statically shaped, so footprints are exact; arguments are
excluded because weights bind from the blob), and `_ram_for_weights` uses
`max(weight-scaled, peak + 128 MB)` — strictly ≥ the old value, so no image that boots today gets a
smaller region. Measured effect: `deepjscc` / `lstmnetvit` / `spectformer` stay at the 256 MB default,
`tiny_llama` int8 stays at 1744 MB, and `whisper_tiny` grows 288 → 464 MB.

## Reproducing

```bash
# model2MLIR: every new op has an oracle-checked regression; every workload captures clean
cd "$MERLIN_M2M_DIR" && .venv/bin/python -m pytest m2m/tests -q
for w in spectformer whisper_tiny lstmnetvit deepjscc; do
  .venv/bin/python workloads/capture.py "$w" --formats fp32,int8    # must print opaque=0
done

# merlin: gates and sizing
.venv/bin/python -m pytest merlin/tests/rvv merlin/tests/runtime -q
.venv/bin/python build_tools/scripts/check_repro_env.py
```
