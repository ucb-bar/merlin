# Local Qwen integration

The compiler wheel and `atlasc` driver live in atlas-mlir. Qwen capture, experiments, model inventories, benchmarks, and historical reports live in this Merlin subtree. The model code is a path-rewritten replay of the original work; the commit messages retain its original source revisions and co-author credit.

The shared atlas-kernels APIs provide D128 attention with 12 query heads and 2 KV heads, width-1536 RMSNorm with explicit `eps=1e-6`, and full-K projection panels. The mapping carries epsilon into launch identity, generator arguments, reference computation, and reports. The attention schedule derives bounded key counts from its static geometry and guards its final delay before halt. No Qwen kernel implementation is copied into Merlin.

The offline random loader constructs all 28 layers with hidden width 1536, intermediate width 8960, vocabulary 151936, and 1,777,088,000 parameters. It uses seed 20261105, random FP32 weights, a 32-token prefill, and 2 cached decode steps. This is an architectural bringup fixture, with no pretrained-quality claim. The separate pretrained loader still refuses a missing checkpoint.

From the Merlin root, use a fresh output directory and explicitly provision the compiler, kernel library, simulator, capture interpreter, model interpreter, and job-local `QWEN_TMP` directory:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$CAPTURE_PYTHON" -B examples/atlas/model_bringup/qwen/full_model/launch_bench.py \
  --library "$ATLAS_KERNELS_ROOT" --model-root "$NPU_MODEL_ROOT" \
  --atlas-bin "$ATLAS_MLIR_BIN" --model-python "$MODEL_PYTHON" \
  --out "$OUT" --tmp "$QWEN_TMP" --workers 2
```

The validated capture environment uses Python 3.12, PyTorch 2.11.0, and Transformers 5.5.4. The pinned simulator uses Python 3.14. Set `MODEL_PYTHON` to the separate simulator interpreter. The script invokes the library's exporter, coverage builder, generators, compiler verification, independent projection golden, and functional/timed simulator APIs. It requires the `verified` compiler tier and assembler/emitter word equality. Interrupted campaigns retain their evidence; use a new output directory to retry.

The export census is 591 projections, 84 attention invocations, and 171 RMSNorm invocations. Coverage expands these into 18,681 projection panels, 84 attention launches, and 171 norm launches across 9 distinct variants. The launch bench stages seeded synthetic operands at those shapes. The additional gate/down checks use actual boundary tensors from the random full-model graph, preserve K=1536 and K=8960, and check the first 128 output columns.

Independent target-arithmetic byte equality applies to projection panels. Attention and RMSNorm have FP32 fidelity metrics and exact timed/functional agreement, but no independent target-arithmetic byte golden in this path. Five extracted PyTorch regions are checked against the exported session on their actual boundary tensors. Those source checks do not establish connected device execution, connected SwiGLU accuracy, pretrained accuracy, hardware performance, or end-to-end token latency. These remain PENDING.

Fresh torch-mlir capture requires its optional capture dependency. `--skip-linalg` records an explicit skipped Linalg stage while retaining exact PyTorch extraction checks. The fresh random campaign and its result receipts do not replace the historical pretrained evidence retained in the other documents.
