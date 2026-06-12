# model2MLIR frontend (smolVLA)

[model2MLIR](https://github.com/ucb-bar/model2MLIR) (`m2m`) converts PyTorch models to
**standard linalg-on-tensors MLIR** (`tensor`/`linalg`/`arith`/`scf`/`func`, weights
externalized to safetensors, `prov.*` provenance on every op). Merlin treats it as a
frontend: `merlin/python/merlin/frontends/` parses its artifacts and lifts matmul
facts into the contract → schedule → interface → target → runtime pipeline.

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
