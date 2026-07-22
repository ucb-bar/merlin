---
title: model2MLIR frontend
kind: guide
status: current
owner: frontends
last_verified: 2026-07-22
related: [getting_started, lowering_pipeline, reproducibility, rvv_e2e]
code_refs: [merlin/python/merlin/frontends]
---

# model2MLIR frontend (smolVLA)

[model2MLIR](https://github.com/ucb-bar/model2MLIR) (`m2m`) converts PyTorch models to
**standard linalg-on-tensors MLIR** (`tensor`/`linalg`/`arith`/`scf`/`func`, weights
externalized to safetensors, `prov.*` provenance on every op). Merlin treats it as a
frontend: `merlin/python/merlin/frontends/` parses its artifacts and lifts matmul
facts into the contract → schedule → interface → target → runtime pipeline.

## Prerequisites

**Shared base:** complete the base install + `.env` setup in [Getting started](getting_started.md)
first (`uv sync --all-extras`, `cp .env.example .env`).

**Workflow-specific prerequisites:**

- **Required — a model2MLIR checkout**: clone
  [`ucb-bar/model2MLIR`](https://github.com/ucb-bar/model2MLIR) and point `MERLIN_M2M_DIR` (and
  `MODEL2MLIR_DIR`) at it; the compile path runs inside `MERLIN_M2M_VENV` (defaults to
  `$MERLIN_M2M_DIR/.venv`). The setup script below creates both m2m's torch/torch-mlir venv and a
  dedicated capture venv.
- **Required for capturing (not for ingesting committed bundles)** — the model's own weights/repo,
  reachable from the capture venv. A full smolVLA capture is RAM-heavy (~80 GB peak); the committed
  bundle artifacts ingest without re-capture.
- Confirm the `llvm_m2m_toolchain` capability with `check_repro_env.py`.

## Setup

```bash
build_tools/scripts/setup_model2mlir.sh
# env overrides: MODEL2MLIR_DIR=/path/to/model2MLIR  SMOLVLA_CAPTURE_DIR=/path/to/capture
```

## Capturing full smolVLA

The capture unit is one flow-matching denoise step (SmolVLM2-500M prefix + action
expert). From the capture venv:

```bash
cd $SMOLVLA_CAPTURE_DIR
.venv/bin/python $MODEL2MLIR_DIR/workloads/capture.py smolvla --formats fp32 int8 fp8
```

Produces `workloads/smolvla/smolvla{,_int8,_fp8}.mlir` (25–29k lines, 0 opaque ops) +
weights `*.safetensors` (fp32 1.2 GB / int8 506 MB) + manifest JSONs.

## Adding a new model — the capture bundle (the shared baseline input)

A model enters Merlin as a **capture bundle**: the single, framework-neutral input that every
baseline (ours, Buddy, TVM, ExecuTorch, ggml, …) ingests, so each arm starts from identical bytes
and the comparison is apples-to-apples. Bundles are resolved by
`merlin/python/merlin/baselines/bundle.py` (`resolve(model, variant)` →
`CaptureBundle`) and live under `merlin.common.artifacts.recaptures_dir()`, i.e.

```
out/artifacts/recaptures/<model>_<variant>_consistent/
  model.mlir                       # linalg-on-tensors export (Buddy / TVM-via-relax ingest directly)
  weights.safetensors              # HF weights …
  weights.safetensors.manifest.json #   … + arg-index map
  inputs.npz / input_order.json    # the seeded inputs
  golden.npy                       # torch reference output — the correctness gate
  extra.npz                        # registered buffers + lifted constants
```

`bundle.resolve()` prefers a full-fidelity `<model>_<variant>_full` recapture (real/native
architecture) over the older truncated `_consistent` bundle when present. `variant ∈ {fp32, int8,
fp8}`. The essential inputs are `model.mlir` + `golden.npy`; `.require()` fails closed if either is
missing so a runner can report a clean `gap_reason` instead of a crash. Per-model correctness
tolerances (`min_cos`, `max_rel`) live in `bundle.TOLERANCES`, mirroring
`merlin/tests/rvv/test_vla_models_rvv.py` so a baseline is gated exactly as our own runtime is.

The PyTorch loader for frameworks that ingest torch directly (ExecuTorch export, TVM `from_pytorch`)
lives **outside** this repo at `$MERLIN_MODEL2MLIR/workloads/<model>/loader.py` (default
`/path/to/model2MLIR`) — it is not vendored.

So "add a model" = **capture it in model2MLIR to produce this bundle, then point Merlin at it.**
The recaptures tree is regenerable (PURGEABLE) and gitignored; do not hand-build a bundle path — go
through `recaptures_dir()`.

## Quantization (torchAO) — what works, what's planned

**Quantization is applied in the external model2MLIR repo, not in Merlin.** The m2m capture pipeline
calls `m2m.capture.torchao_pipeline.apply_quantization` on the torch model before export; Merlin only
*consumes* the already-quantized bundle. (Merlin invokes that same external function in exactly one
place — its TVM baseline re-quantizes a live-loaded model to match the capture:
`merlin/python/merlin/baselines/tvm.py:282`.) You add a quantized model by capturing it there and
ingesting the resulting bundle:

```bash
# in the model2MLIR capture venv (see build_tools/scripts/setup_model2mlir.sh for m2m setup)
.venv/bin/python $MODEL2MLIR_DIR/workloads/capture.py <model> --formats fp32 int8 ...
```

**Working and tested today — int8 only.** The int8 path is **weight-only / W8A8**
(`int8_dynamic_activation_int8_weight`) and is the one format with a *measured* accuracy gate:
int8 passes **5/5** (see the accuracy-gate report referenced from
`merlin/python/merlin/baselines/`). This is the only quantized path you should treat as real.

**Aspirational — not a working path.** `fp8` (`float8_dynamic_activation_float8_weight`),
`int4_weight_only`, and `int4-weight + fp8-activation` are a documented **plan**, not a sweep. Their
accuracy status is `unavailable` (no quantization run is executed and no accuracy number is asserted
for them). The plan is emitted verbatim by
`merlin/python/merlin/dse_guidance/numerical_contract.py::torchao_integration_plan_md()`, which
states which format informs which DSE candidate and what must be measured (accuracy gate, packed
layout + scale metadata preserved through capture, low-bit kernel cost) before each becomes
DSE-legal. Treat fp8/int4 as **planned/unmeasured**, never as an available format.

**Honest gap even for int8.** Per `merlin/python/merlin/dse_guidance/quant_metadata.py`, the int8
qdq capture is *torchao int8 weight-only*, which is **not necessarily a model's native scheme** — for
example bitvla's native format is W1.58 ternary (packed int2 + absmean scale) BitLinear, so a
torchao-int8 bundle is a stand-in, not the native datapath. This gap is recorded per workload (never
hidden); a separate native capture (`recaptures_native/bitvla`) is what actually exposes the packed
ternary storage + scale when present.

## Merlin ingestion

```python
from merlin.frontends import linalg_mlir as fl, facts as ff

mod = fl.parse_mlir_file("workloads/smolvla/smolvla.mlir")        # ~5 s
inv = fl.matmul_inventory(mod, fl.load_manifest(".../smolvla.safetensors.manifest.json"))
# 302 linalg.matmul ops; weights resolved via the manifest.
facts = ff.lift_weight_reuse(inv, invocations=10)  # reuse across denoise steps
rec = ff.select_gemm(inv, max_macs=2_000_000)      # action_out_proj: 50x720x32
res = ff.drive_pipeline(rec, reuse=2, target="saturn")
# -> run on spike via merlin.runtime.backends.spike; dse records via ff.record_dse
```

The integer pipeline executes a layer's i8 deployment GEMM with the model's real
(M, K, N); capture dtype is preserved as provenance.

## Whole-model lowering (`merlin/python/merlin/llvmlower/`)

The `llvmlower` package compiles an entire model2MLIR module to native code via the
**MLIR → llvm-dialect → LLVM IR → clang** path (LLVM 23, from the IREE/torch-mlir
install), targeting x86 (correctness oracle) or rv64gcv (deployment). Merlin-authored
passes (`passes_xdsl.py`) handle `quant_ext.dequantize_per_channel` → `linalg.generic`,
`emit_c_interface`, and textual normalization of printer quirks; the upstream pipeline
(`pipeline.py`) does bufferize → loops → llvm dialect; `translate` emits LLVM IR;
`codegen.py` runs clang. Many-arg models use a generated C trampoline (`abi.py`).

Validated end-to-end on host (== PyTorch, consistent-capture golden):
- **tiny_llama** (full transformer): cos 1.0000, argmax exact.
- **smolVLA int8** (full VLM + action expert, 1 denoise step): cos 0.943 (the residual is
  bf16 matmuls accumulating in bf16 vs torch's f32 accumulation — a bounded precision gap).
The whole tiny_llama also compiles to a real RVV `rv64gcv` object (auto-vectorized:
`vsetvli`/`vle32`/`vfmul`). `truncate.py` provides subgraph truncation for per-op
bisection (it found the bug below).

## model2MLIR correctness fixes (made at source)

- **Uninitialized matmul accumulators** (`m2m/ir/import_fx.py::_zero_fill_contraction_accumulators`):
  `linalg.matmul`/`quantized_matmul` compute `out += A·B`, but m2m fed an unfilled
  `tensor.empty` as `outs` — undefined memory, read as garbage/NaN whenever the allocator
  returned dirty pages. Now every contraction accumulator gets an explicit `linalg.fill 0`.
  This was the root cause of NaN/uncorrelated whole-model output.
- **slice_scatter step arg** (`m2m/ir/decompositions.py::decompose_slice_scatter`): read
  `step` from the `end` arg slot (index 4 vs 5), corrupting RoPE strides. Fixed.

## Known upstream issues

- m2m's section splitter emitted use-before-def SSA references (values captured
  inside `linalg.generic` bodies — e.g. the embedding table — were never added as
  section inputs; `%2034` in `sections/smolvla.model.mlir` was undefined). **Fixed in
  the local checkout** (`m2m/transforms/sections.py::_free_values`, uncommitted);
  both sections re-parse cleanly after re-splitting. Re-run the capture to refresh
  the committed `sections/*.mlir` artifacts, which still predate the fix.
- xDSL 0.65 rejects MLIR's parenthesized multi-result `linalg.generic` tail
  (`} -> (T1, T2)`); the frontend normalizes the text before parsing.
- m2m's `backend="torch_mlir"` path emits MLIR custom assembly xDSL cannot fully
  parse (`tensor.extract_slice` has no custom-format parser in xDSL). For Merlin
  ingestion, convert with `backend="fx_importer"` — verified end-to-end (tiny_llama:
  155 matmuls inventoried from a fresh capture).
- A full smolVLA capture needs more free RAM than this shared box typically has
  (~80 GB peak); two attempts were OOM-killed at the worst case. The committed
  artifacts are valid and ingest cleanly.
