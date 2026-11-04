# Source captures and fixed-input host builds

Capture native operands, export ResNet plans and references, and build
fixed-input TinyLlama and SmolVLA host ELFs with an explicit Atlas provider.

## Inputs and environment

| Input | Contract |
| --- | --- |
| Native Python and packages | Recorded versions and original source files. |
| Checkpoint, corpus, observations | External, hash-pinned capture inputs. |
| Atlas tools and runtime | Explicit provider paths and recorded hashes. |
| Output directories | Fresh, beneath the configured `artifacts_dir()`. |

Run from the Merlin root with the recorded native interpreter and packages. Export the settings so capture subprocesses inherit the same import roots and tensor thread limits:

```sh
export PYTHONPATH="$PWD/src:$NATIVE_PACKAGES"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MERLIN_OUT_ROOT=/absolute/path/to/writable-output
```

`artifacts_dir()` is `$MERLIN_OUT_ROOT/artifacts`; TinyLlama's native capture requires its output beneath that directory. Replace input variables with physical paths and authenticated hashes; see `--help` for complete arguments. Inputs cannot contain symlinks or parent traversal. ResNet and native replay accept `--relocate /old=/new` rules; authentication precedes mapping, preserving payload hashes and versions.

## Capturing a new model

Capture records real native tensor values, shapes, dtypes, parameters, and
reference outputs, together with source identities and input hashes. Pair this
with a `torch.export` op and shape inventory when bringing up another model.
The existing model-specific entry points are:

```sh
python examples/atlas/comparisons/capture_tinyllama_native.py \
  --checkpoint "$CHECKPOINT" --corpus "$CORPUS" --full-query \
  --out "$TINY_NATIVE"
python examples/atlas/comparisons/capture_smolvla_policy.py \
  --source-receipt "$DYNAMIC_RECEIPT" --observation-input "$OBSERVATIONS" \
  --out "$SMOL_NATIVE"
```

TinyLlama writes `operands.npz` and `manifest.json` for the first decoder layer's
query projection, while SmolVLA writes parameter and invocation tensors,
`trajectory.npz`, and `receipt.json` for fresh vision and ten denoising steps.

These are model-specific adapters with external prerequisites, not checkpoint-to-host bootstrap commands. TinyLlama requires the pinned checkpoint, development corpus, tokenizer files, and frozen Transformers source. The corpus is not supplied by this repository, and a different text with the same token count fails authentication. SmolVLA requires a pre-existing dynamic-block receipt whose status is `passed_dynamic_block_model_and_retained_policy_continuation`, its authenticated source bookends, retained policy archive, weights, backbone configuration, exact native environment, and the compiled observation archive. A checkpoint plus raw dataset frame does not satisfy that receipt contract.

For independent setup, use atlas-mlir's [input and environment contracts](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-performance/docs/bringing-up-a-model.md#inputs-and-environments) and [new model adapter contract](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-performance/docs/bringing-up-a-model.md#new-model-adapter-contract). Record a missing corpus, receipt, archive, or provider as an unmet prerequisite. For a new model, implement its input preparation, source authentication, op inventory, and real-value capture explicitly before mapping kernels; these scripts do not generate those adapters.

Host graphs are hand-written C, including [tiny_block.c](tiny_block.c), [smol_vision_graph.c](smol_vision_graph.c), and [smol_policy_graph.c](smol_policy_graph.c). Capture does not generate these graphs. TinyLlama and SmolVLA reach the device through model-specific capture adapters and these hand-written host graphs, bypassing Merlin's semantic compiler and provider. The Merlin-intended semantic compilation route is Agustin's `feat/atlas-model-compilation` branch, currently proven on small shapes.

Follow the
[model bringup guide](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-performance/docs/bringing-up-a-model.md)
for inventory, kernel mapping, compilation, exactness checks, and host execution.

## ResNet plan and reference

Start with `refresh/capture_resnet.py original`, supplying pinned `--receipt`,
`--preparation`, their `-sha256` arguments, and a fresh `--output-dir`. It uses
torchvision ResNet50 v1.5 `IMAGENET1K_V2`, resize 232, center crop 224, and
recorded normalization, retaining original FP32 pre-BatchNorm weights.

```sh
python examples/atlas/comparisons/refresh/partition_resnet.py \
  --source-manifest "$SOURCE" --source-manifest-sha256 "$SOURCE_SHA" \
  --tile-shape 32 32 32 --out "$INVENTORY"
python examples/atlas/comparisons/refresh/export_full_host_plan.py \
  --source-manifest "$SOURCE" --source-manifest-sha256 "$SOURCE_SHA" \
  --inventory "$INVENTORY/partition.json" --inventory-sha256 "$INVENTORY_SHA" \
  --tile-shape 32 32 32 --atlas-tools "$ATLAS_TOOLS" \
  --packing-provider-sha256 "$PACKING_SHA" \
  --vector "$VECTOR" --vector-sha256 "$VECTOR_SHA" \
  --scalar "$SCALAR" --scalar-sha256 "$SCALAR_SHA" --out "$PLAN"
python examples/atlas/comparisons/refresh/export_full_reference.py \
  --source-manifest "$SOURCE" --source-manifest-sha256 "$SOURCE_SHA" \
  --host-plan "$PLAN/host-plan.json" --host-plan-sha256 "$PLAN_SHA" \
  --atlas-tools "$ATLAS_TOOLS" --packing-provider-sha256 "$PACKING_SHA" \
  --vector "$VECTOR" --vector-sha256 "$VECTOR_SHA" \
  --scalar "$SCALAR" --scalar-sha256 "$SCALAR_SHA" --out "$REFERENCE"
```

The plan exports 53 convolution weights, classifier weights, normalization, and
the image. References use pinned Atlas arithmetic/packing and Torch pooling.

## TinyLlama full capture

Start with `capture_tinyllama_native.py --checkpoint ... --corpus ...
--full-query --out ...` for native Q operands. Input hashes are fixed in source.
The historical session receipt also has a fixed digest and binds the native full-Q manifest, checkpoint, corpus, interpreter, Torch library, and frozen Transformers sources. `capture_tinyllama_full.py` does not consume the newly written `$TINY_NATIVE` directory directly and does not create that historical receipt. Without the authenticated receipt and its referenced providers, this full-capture command is blocked; generating a selective Q capture alone does not satisfy it.

```sh
python examples/atlas/comparisons/capture_tinyllama_full.py \
  --historical "$SESSION_RECEIPT" --atlas-arithmetic "$VECTOR" \
  --atlas-arithmetic-sha256 "$VECTOR_SHA" --decode-count 2 --out "$TINY_CAPTURE"
```

All 22 decoder layers and the LM head use greedy feedback. Operands are FP32
views of BF16 words; decode rows use positive-zero M32 padding and full K order.

## Full-model and fused captures

`capture_tinyllama_fused.py` captures fused projection boundaries, and `capture_atlas_attention.py` records exact staged attention operands and model replies. `refresh/capture_resnet_device.py` captures device patch gathers, max pooling, and fused convolution outputs from a graph plan.

The full C graph capture entry is `d1-capture-c-v2.py`. Its projection worker and concurrent stream backfill drivers are in this directory. Set `ATLAS_ROOT`, `MERLIN_ROOT`, `ATLAS_NPU_MODEL_ROOT`, and `TINY_WORK_ROOT` to physical absolute paths, use one tensor thread per process, and use at most 16 workers, the validated concurrency limit. The full capture, export, qualification, and portable receipt workflow is documented in atlas-mlir `tools/tiny_full/REPRODUCING.md`. The driver requires external checksum-bound parameter and scale inputs, as listed by `--help`.

## SmolVLA policy capture and replay

The dynamic-block receipt must bind the native full-query manifest and retained
policy inputs. Supply observations for fresh vision and ten denoising steps:

```sh
python examples/atlas/comparisons/capture_smolvla_policy.py \
  --source-receipt "$DYNAMIC_RECEIPT" --observation-input "$OBSERVATIONS" \
  --out "$SMOL_NATIVE"
python examples/atlas/comparisons/capture_smolvla_policy.py \
  --source-receipt "$DYNAMIC_RECEIPT" --observation-input "$OBSERVATIONS" \
  --native-capture "$SMOL_NATIVE/receipt.json" --atlas-tools "$ATLAS_TOOLS" \
  --out "$SMOL_ATLAS"
```

For the maintained instruction-backed C-graph route, follow atlas-mlir's [Smol attention capture runbook](https://github.com/ucb-bar/atlas-mlir/blob/sota/bringup-performance/docs/smol_attention_capture.md#inputs). It uses an immutable compiled-observation FullBundle and baseline providers instead of loading a raw checkpoint. `tools/capture_smolvla_attention.py --prepare --execute` produces `run/manifest.json`, `run/replay.json`, `run/device-events.jsonl`, `run/phase-reference.json`, and provider identities. A complete capture covers 1,421 logical projection calls, 2,715 projection panels, and 200 attention launches. A limited phase run is incomplete, and reply reuse retains its narrower identity-bound scope. Capture outputs are references; the separate original-model admission and host execution gates remain required.

## Fixed-input block ELF

`build_block_elf.py` takes a pinned JSON recipe with `workload` (`tiny` or
`smol`), ordered `projections` (24 Tiny panels or seven Smol projections),
`data`, and `parameters`. File records bind `path` and `sha256`; relative paths
are beneath the recipe directory. Array records select an NPZ `array`, or NPY
`array: null`; records without `array` hold raw bytes. Supported arrays are
uint8, BF16 uint16, and float32, serialized little-endian. See
[the recipe reader](build_block_elf.py) and
[kernel input checks](block_kernel_inputs.py) for required fields and the five
regenerated payload comparisons. No command exports a complete recipe from a
checkpoint. C computes successor activations; saved outputs are comparisons.

```sh
python examples/atlas/comparisons/build_block_elf.py \
  --recipe "$RECIPE" --recipe-sha256 "$RECIPE_SHA" \
  --provider-runtime "$ATLAS_RUNTIME" --atlas-tools "$ATLAS_TOOLS" \
  --model-root "$MODEL_ROOT" --startup "$STARTUP" --linker-script "$SCRIPT" \
  --clang "$CLANG" --linker "$LLD" --native-cc "$CC" \
  --libm "$LIBM" --libm-sha256 "$LIBM_SHA" \
  --expf-table-sha256 "$EXPF_TABLE_SHA" \
  --expected-elf-sha256 "$ELF_SHA" --out "$BLOCK"
```

Use `tiny_start.S`/`tiny_link.ld` or `smol_start.S`/`smol_link.ld`; match the
qualified toolchain, libm, recipe, and parameters. Native C expf generates the
lookup table. Builds emit kernels, C/assembly, logs, and `build.json`.

## Tests and limits

Run `python -m pytest merlin/tests/infra merlin/tests/runtime` for source and
runtime controls. Embedded `test_smol_policy_*.py` and `test_smol_vision_*.py`
tests need their explicit external fixtures.

Native substitution needs an archived cold-contraction validator; workload
receipts have a separate admission contract. Numerical tolerances stay fixed.
These tools alone establish neither complete instruction-model execution nor
FPGA, FireSim, or application-quality results.
