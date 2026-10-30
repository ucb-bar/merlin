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

Run from the Merlin root with `PYTHONPATH="$PWD/src:$NATIVE_PACKAGES"` and
`PYTHONDONTWRITEBYTECODE=1`. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`MKL_NUM_THREADS` to 1. Use the recorded native interpreter for `python` below.
Replace variables with physical paths and authenticated hashes; see `--help`
for complete arguments. Inputs cannot contain symlinks or parent traversal.
ResNet and native replay accept `--relocate /old=/new` rules; authentication
precedes mapping, preserving payload hashes and versions.

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
The historical session receipt also has a fixed digest; regeneration requires
its original source revision and archived provider.

```sh
python examples/atlas/comparisons/capture_tinyllama_full.py \
  --historical "$SESSION_RECEIPT" --atlas-arithmetic "$VECTOR" \
  --atlas-arithmetic-sha256 "$VECTOR_SHA" --decode-count 2 --out "$TINY_CAPTURE"
```

All 22 decoder layers and the LM head use greedy feedback. Operands are FP32
views of BF16 words; decode rows use positive-zero M32 padding and full K order.

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
