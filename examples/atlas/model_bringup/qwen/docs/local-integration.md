# Local Qwen integration

The setup requires two atlas-mlir sources. Jeremy Mills' separate `feat/pip-package` work at [56b7673f151e56587b6793d44802df5881e0b4ba](https://github.com/ucb-bar/atlas-mlir/tree/56b7673f151e56587b6793d44802df5881e0b4ba), including the initial `25bbaf5` and `a92d173` commits, supplies the compiler wheel, `atlasc` driver, Python capture APIs, and toolchain support and discovery. Compiler-only [atlas-mlir PR #26](https://github.com/ucb-bar/atlas-mlir/pull/26) at `0dac108`, built in Release mode, supplies `atlas-opt` and `atlas-emit`; it does not provide the Python package. Qwen capture, experiments, model inventories, benchmarks, and historical reports live in this Merlin subtree. The model code is a path-rewritten replay of the original work; the commit messages retain its original source revisions and co-author credit.

The shared atlas-kernels APIs provide D128 attention with 12 query heads and 2 KV heads, width-1536 RMSNorm with explicit `eps=1e-6`, and full-K projection panels. The mapping carries epsilon into launch identity, generator arguments, reference computation, and reports. The attention schedule derives bounded key counts from its static geometry and guards its final delay before halt. No Qwen kernel implementation is copied into Merlin.

The offline random loader constructs all 28 layers with hidden width 1536, intermediate width 8960, vocabulary 151936, and 1,777,088,000 parameters. It uses seed 20261105, random FP32 weights, a 32-token prefill, and 2 cached decode steps. This is an architectural bringup fixture, with no pretrained-quality claim. The separate pretrained loader still refuses a missing checkpoint.

From the Merlin root, use a fresh output directory and explicitly provision the compiler, kernel library, simulator, capture interpreter, model interpreter, and job-local `QWEN_TMP` directory:

```sh
export ATLAS_MLIR_ROOT=/path/to/atlas-mlir-package # feat/pip-package at 56b7673
export ATLAS_OOT_BIN_DIR=/path/to/atlas-mlir-compiler/build/bin # PR #26 at 0dac108, Release
export PYTHONPATH="$ATLAS_MLIR_ROOT/python:$PWD/src:$NPU_MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$CAPTURE_PYTHON" -B examples/atlas/model_bringup/qwen/full_model/launch_bench.py \
  --library "$ATLAS_KERNELS_ROOT" --model-root "$NPU_MODEL_ROOT" \
  --atlas-bin "$("$CAPTURE_PYTHON" -B -c 'from atlas_mlir.toolchain import bin_dir; print(bin_dir())')" \
  --model-python "$MODEL_PYTHON" \
  --out "$OUT" --tmp "$QWEN_TMP" --workers 2
```

`ATLAS_MLIR_ROOT` selects the package source through Merlin's source-root bootstrap. `PYTHONPATH` exposes that package's `python` directory, Merlin's `src` directory, and the simulator source while preserving any existing entries. `ATLAS_OOT_BIN_DIR` is the supported toolchain override; `atlas_mlir.toolchain.bin_dir()` resolves it, and the command passes that same directory to `--atlas-bin`. `ATLAS_MLIR_BIN` is not a supported toolchain environment override.

The validated capture environment uses Python 3.12, PyTorch 2.11.0, and Transformers 5.5.4. The pinned simulator uses Python 3.14. Set `MODEL_PYTHON` to the separate simulator interpreter. The script invokes the library's exporter, coverage builder, generators, compiler verification, independent projection golden, and functional/timed simulator APIs. It requires the `verified` compiler tier and assembler/emitter word equality. Interrupted campaigns retain their evidence; use a new output directory to retry.

The export census is 591 projections, 84 attention invocations, and 171 RMSNorm invocations. Coverage expands these into 18,681 projection panels, 84 attention launches, and 171 norm launches across 9 distinct variants. The launch bench stages seeded synthetic operands at those shapes. The additional gate/down checks use actual boundary tensors from the random full-model graph, preserve K=1536 and K=8960, and check the first 128 output columns.

Independent target-arithmetic byte equality applies to projection panels. Attention and RMSNorm qualification requires FP32 fidelity metrics and exact timed/functional agreement, but has no independent target-arithmetic byte golden in this path. Five extracted PyTorch regions are checked against the exported session on their actual boundary tensors. Those source checks do not establish connected device execution, connected SwiGLU accuracy, pretrained accuracy, hardware performance, or end-to-end token latency. These remain PENDING.

Eager PyTorch, export, and shared-kernel launch qualification are separate from fresh native attention and SwiGLU compilation. At `56b7673f151e56587b6793d44802df5881e0b4ba`, capture writes raw Linalg without the historical normalization step or raw/normalized sidecars. Fresh native compilation remains PENDING for compiler support, including sigmoid lowering and `add_bf16` admission, and torch-mlir is unavailable in the offline environment.

Fresh torch-mlir capture requires its optional capture dependency. `--skip-linalg` records an explicit skipped Linalg stage while retaining exact PyTorch extraction checks. The fresh random campaign and its result receipts do not replace the historical pretrained evidence retained in the other documents.

Using only the unmodified `56b7673f151e56587b6793d44802df5881e0b4ba` package tree and its compiler build refuses all 3 Qwen attention launch variants at the required `verified` tier with a `vstore`/DMA channel 1 WAW conflict. Those package-only failures are expected because that compiler lacks PR #26's CFG DMA timing verifier, whose path-sensitive address tracking resolves the false WAW conflict across a loop merge. PR #26 also provides attention metadata contracts; these statically bounded Qwen programs do not require those contracts for their `verified` classification. Shared-kernel qualification requires the split setup above, with the Python package from `56b7673` and Release compiler executables from PR #26 at `0dac108`. All 9 launch variants pass with that split setup at the required `verified` tier, with exact timed/functional agreement, both compiler checks, and assembler/emitter word equality.
