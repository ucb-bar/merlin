---
title: Vision, audio and control workloads on multicore RVV under Zephyr
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

# Vision, audio and control workloads on multicore RVV under Zephyr

[TinyLlama int8 on multicore RVV](tinyllama_int8_rvv_zephyr.md) walks the same pipeline for a
decoder-only LLM. This guide covers what changed to bring **four non-transformer-only workloads**
through it — a spectral vision transformer, a speech encoder-decoder, a recurrent control policy,
and a convolutional codec — and reports what each one measures today, including where it does not
yet pass.

| workload | upstream | capture unit |
|---|---|---|
| `spectformer` | [`badripatro/SpectFormers`](https://github.com/badripatro/SpectFormers) | one SpectFormer-Ti forward: 224², patch 16, embed 256, depth 12 (4 spectral + 8 attention blocks) |
| `whisper_tiny` | `openai/whisper-tiny` (HF weights) | audio encoder + **one** cross-attending decoder step |
| `lstmnetvit` | [`anish-bhattacharya/vitfly`](https://github.com/anish-bhattacharya/vitfly) | one `LSTMNetVIT` forward, command output only, zero initial LSTM state |
| `deepjscc` | [`mingyuyng/DiffJSCC`](https://github.com/mingyuyng/DiffJSCC) | JSCC encoder + AWGN channel + decoder (**not** the diffusion refinement stage) |

Nothing here is workload-specific in the compiler. Everything below is either a new
model2MLIR decomposition (shared capability) or a policy that derives a decision from the model
instead of pinning it.

## The one fact that drives the design

The certified RVV package declares `op_match: [linalg.matmul, linalg.batch_matmul]`, and everything
the schedule does not match is compiled through `convert-linalg-to-loops` with `-fno-vectorize
-fno-slp-vectorize`. `parallel_transform_schedule` likewise splits only matmul-over-N and
batch_matmul-over-B.

> **An op that does not land on a named contraction gets neither RVV vectors nor `--harts`
> parallelism. It runs scalar, on one core.**

So every op these four models need was decomposed **onto contractions**, not onto a fused
`linalg.generic`:

- **conv → im2col gather + `linalg.matmul` + reshape** (padded, strided, dilated, grouped and
  transposed conv all normalize into this path)
- **`rfft2`/`irfft2` → real DFT matmuls** with constant twiddle operands and an explicit
  real/imaginary split (xDSL 0.65 has no complex type; a complex tensor of shape `S` is a real
  tensor of shape `S + [2]`, which is torch's own `view_as_complex` layout, so `view_as_complex` /
  `view_as_real` are identities)
- **LSTM → per-timestep gate matmuls** (`torch.export` already unrolls the recurrence)

This is also the target-agnostic choice: any target with a matmul provider inherits conv, FFT and
recurrence with no new backend code.

## Running one

```bash
merlin-compile --workload spectformer --dtype int8 --target rvv --run host  --verify --json
merlin-compile --workload spectformer --dtype int8 --target rvv --run spike --verify --json
merlin-compile --workload spectformer --dtype int8 --target rvv --run spike --harts 4 \
               --iters 20 --warmup 3 --verify --json
```

A missing bundle is captured automatically, in the workload's own venv (resolved from its
`capture.toml`). Generate the W8A8 reference **before** grading an int8 run:

```bash
python build_tools/scripts/make_w8a8_golden.py spectformer_int8_full
```

and check `tier_ok == "w8a8"` in the result. Grading a W8A8 run against the weight-only
`golden.npy` measures activation-quantization error, not correctness — `merlin-compile` warns when
the W8A8 reference is missing, and the gate reports which tier carried the verdict.

## Status, measured

`spike`, 1 hart, int8 (W8A8), graded against the W8A8 reference:

| workload | cycles | `w8a8_cos` | `w8a8_rel` | verdict |
|---|---:|---:|---:|---|
| `spectformer` | 4,198,885,000 | 1.0 | 0.0 | **verified** — bit-exact |
| `deepjscc` | 485,985,000 | 0.9176 | 0.889 | run_mismatch — RISC-V-side defect, see below |
| `lstmnetvit` | 857,315,000 | 0.9943 | 0.259 | run_mismatch — same |
| `whisper_tiny` | — | — | — | builds and lowers; spike run exceeds the default timeout |

Multicore (`lstmnetvit`, int8): **857,315,000 cycles at 1 hart → 381,565,000 at 4 harts (2.25×)**,
with every gate metric identical to 16 digits across the two — i.e. the parallel split is
bit-exact, which is the only thing a multicore run is allowed to prove about correctness.

Host (`--run host`, all four, int8 and fp32): **verified**, `tier_ok = "w8a8"`, `w8a8_rel = 0.0`.
All four lower with **zero opaque ops** (`spectformer` 1240 linalg ops, `whisper_tiny` 1296,
`lstmnetvit` 653, `deepjscc` 380).

### The open defect

`deepjscc` and `lstmnetvit` diverge **only on RISC-V**. Localized by three runs of the same model:

| run | what it shares with the spike run | result |
|---|---|---|
| `--run host` (dispatch runtime) | the int8 quantization math | `w8a8_rel = 0.0` |
| host, **vectorized** lowering (same prepared IR, same schedule, same features, x86 backend) | everything except the target | `cos = 1.000000000`, zero bad elements |
| `spike` | the target | `cos = 0.9176`, 96.6 % of elements off by >1 % |

So the vectorized IR is correct and the fault is on the RISC-V side. The error is structural, not
numerical noise: the output carries period-2 repeats (`0.8667, 0.8698, 0.8667, 0.8698`) where the
reference varies, which is a stride/indexing signature. It is **not** caused by the shape-adaptation
policy below (`deepjscc` keeps the frozen champion block) and **not** a masked-tail case (every
`deepjscc` matmul N is a multiple of 16). Both models are conv-dominated (154 and 142 conv ops)
while the passing `spectformer` has 11, which is where to look first. Do not quote either model's
numbers as an accuracy result until this closes.

## What the compiler learned (target-agnostic)

### A register block is a claim about extents

`from_strategy._rvv_blocking_lowers` encodes, as a measured predicate, that a block masking a
parallel dim of a contraction does not lower at all on the integer path (LLVM-23 rejects the
multi-op `vector.mask` a masked `transfer_write` needs) and degrades ~34× on fp32. The certified
champion pins its block as constants **inside** a hand-frozen feature, so the one package anyone
compiles with was the one package that could not adapt — and a model whose extents the frozen tails
do not fit simply failed to build.

`llvmlower/frozen_blocks.py` now describes that point's caps and its per-op-class realization as
data (sourced from the feature's own module constants, so it cannot drift from the registration),
and `rvvgen.apply` re-resolves it against the workload's real contraction extents. The adaptation is
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
`[1, 16]`; its 24 attention `batch_matmul`s run scalar. `merlin-compile` prints the unclaimed
classes and records them in `features_shape_adapted.unclaimed_op_classes`.

This is a **per-op-class** decision, so a class mixing a tiny and a large extent is clamped by its
smallest member. Per-**op** blocking (tag each contraction with its own legal block, match by
attribute in the schedule) would recover whisper's encoder attention and is the identified
follow-up.

### RAM is provisioned from the measured working set

The Zephyr RAM region has to hold the weights blob plus the activation arena. Sizing its headroom
from weight bytes alone assumes activations are small relative to parameters, which is false for an
encoder whose attention matrices dwarf its weights. `mlir_query.activation_peak_bytes` measures peak
simultaneously-live intermediate bytes by liveness over the captured IR (every tensor is statically
shaped, so footprints are exact; arguments are excluded because weights bind from the blob), and
`_ram_for_weights` uses `max(weight-scaled, peak + 128 MB)` — strictly ≥ the old value, so no image
that boots today gets a smaller region.

## Memory footprint — what fits a 512 MB SoC

Weights are exact (safetensors payload); activation peaks are the liveness measurement above; image
overhead is `size` on the linked ELFs (8.6 MB, mostly the 8 MB worker stack).

| workload | dtype | weights | peak activations | total | fits 512 MB |
|---|---|---:|---:|---:|---|
| `deepjscc` | int8 / fp32 | 0.8 / 1.0 MB | 12.8 MB | ~22 MB | yes |
| `lstmnetvit` | int8 / fp32 | 6.3 / 13.6 MB | 11.3 / 18.0 MB | 26 / 40 MB | yes |
| `spectformer` | int8 / fp32 | 10.5 / 35.5 MB | 3.1 MB | 22 / 47 MB | yes |
| `whisper_tiny` | int8 | 116.8 MB | 210.4 MB | 336 MB | yes |
| `whisper_tiny` | fp32 | 220.9 MB | 210.4 MB | 440 MB | yes, 86 % full |
| `smolvla` | int8 | 482.6 MB | 199.0 MB | 690 MB | no |
| `tiny_llama` (22 layer) | int8 | 1241.7 MB | 312.7 MB | 1563 MB | no |

`whisper_tiny`'s peak is two buffers: a 76 MB `tensor<384x51865xf32>` vocab projection materialized
as fp32 inside the *int8* bundle, and 51.5 MB `tensor<6x1500x1500xf32>` encoder attention matrices.
Keeping the projection in int8 and chunking encoder attention is what would take it from "tight" to
"comfortable". Note also that the GGUF bundles (`tiny_llama_gguf_q4k` / `q6k` / `q8`) carry 4196 MB
of weights — ingest dequantizes to fp32 — so they are **not** a footprint win over int8's 1242 MB.

## Reproducing

```bash
cd /scratch/agustin/projects/model2MLIR && .venv/bin/python -m pytest m2m/tests -q
for w in spectformer whisper_tiny lstmnetvit deepjscc; do
  .venv/bin/python workloads/capture.py $w --formats fp32,int8      # must print opaque=0
done

cd /scratch/agustin/projects/oscar-merlin
.venv/bin/python -m pytest merlin/tests/rvv merlin/tests/runtime -q
.venv/bin/python build_tools/scripts/check_repro_env.py
```
