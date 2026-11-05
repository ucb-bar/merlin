# Source captures and fixed-input host builds

Capture native operands, export ResNet plans and references, and build fixed-input TinyLlama and SmolVLA host ELFs with an explicit Atlas provider.

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

Use the [native TinyLlama recipe](#capturing-a-new-model) for native Q operands; input hashes are fixed in source.
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

### Fixed-session scale producer dependency: PENDING

The scale inputs are historical full export contracts, each supplying 66 selected consumer exponents for 22 layers and three steps. They are authenticated inputs, not calibration outputs generated by these capture entrypoints. `capture_tinyllama_fused.py:down_scales` and `d1-capture-c-v2.py:attention_scales` authenticate the entire corresponding file before selecting its entries.

| Consumer selection | Required contract SHA-256 | Original producer capture SHA-256 |
| --- | --- | --- |
| `(step, site)` for `mlp.down_proj` | `a34fa90b3dac3a6e131c8f9d0bdf8da57978e2a3d76af1325d02af197e46b938` | `6ea15b123a9c647333e328021bc84ca3b54a0e85fd36e39ba90c118c182825f0` |
| `(step, layer)` for `self_attn.o_proj` | `c6f5d6407ea784f6cc3c871d6cb93eb8cbf680cf276f8f37005f584f0858b8eb` | `9dfe0737bb6ad68a086b7448727c84232bd3f8f549afa692e13668d0a3e4e803` |

The retained original down-scale procedure was located in the V2 job evidence: `A/capture_m3_command.json`, `A/capture_m3_freeze.json`, and `A/capture_m3_run.json` record `capture_tinyllama_full.py` at Merlin `88e959a8dd15f1bc7330f362916d292b7eb5fbcf`, with all 22 layers, 128 prefill tokens, two actual greedy decode steps, and original npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. `A/export_m3_command.json` and `A/export_m3_summary.json` record the successful export at atlas-mlir `f70bad229846f74eefe495855457287405b4e23b`, producing `A/m3-assets/contract.json` with the first digest above. These records are external historical evidence, not repository fixtures or a successful new reproduction run.

The historical export command below requires that original source and environment closure. It is not a command to run against today's checkout. `V2_ROOT` denotes the original physical job root; changing recorded source paths changes the contract bytes.

```sh
"$V2_ROOT/A/.venv/bin/python" -B \
  "$V2_ROOT/A/atlas-mlir/tools/tiny_full/export_capture.py" \
  --capture "$V2_ROOT/A/m3-capture/receipt.json" \
  --capture-sha256 6ea15b123a9c647333e328021bc84ca3b54a0e85fd36e39ba90c118c182825f0 \
  --model-root "$V2_ROOT/A/npu-model" \
  --out "$FRESH_SCALE_EXPORT"
```

For each down-projection consumer, the original producer selects an E8M0 exponent from the maximum absolute finite BF16 word in that call's `A_bf16` tensor using `tiny_fp8_pack.tensor_scale_bits`. The zero tensor selects exponent 0; a normal maximum with unbiased exponent `e` and integer significand `s` selects `e - 8 + int(s > 224)`. The retained codec also handles BF16 subnormals. Its source SHA-256 is `fff76584a936c357668a4aea5f4d24b7dc756a4067f74131f732f4fce72f949f`. FP8 packing uses round-to-nearest, ties-to-even for the significand, flushes underflow to positive zero, and saturates magnitude to the finite E4M3 value 448. Decode operands have positive-zero padding to M32. The exporter checks scalar-policy agreement and packed values against the selected RTL numerical implementation. The producer receipt binds each tensor's shape, dtype, byte extent, and SHA-256; the contract joins its exponent by `(step, site)`. In the fused capture, that fixed down exponent configures the up-projection FP8 output and is carried into the down-projection consumer. It is not recomputed from the later fused producer tensor.

The retained down-scale inputs were checked again during the conformance investigation: all 66 producer tensors matched their descriptor hashes and lengths, and all 66 exponents recomputed with that authenticated codec matched the capture panels and selected contract entries. This is a retained-tensor scale audit, not byte-for-byte regeneration of the full contract, a fresh model capture, or qualification of new inputs. The attention contract's retained producer chain is separately recorded by `D/d1-full-native-command.json`, `D/d1-full-native-run.json`, and `D/d1-stage2-assets/contract.json`; it depends on the preceding down-scale contract.

Full contract regeneration remains **PENDING**. The original down-contract closure currently has eight missing historical model/environment inputs: the checkpoint, corpus, frozen `modeling_llama.py` and `cache_utils.py`, model configuration, and three tokenizer files. Four original source files have changed: `capture_tinyllama_full.py`, `capture_atlas_attention.py`, `export_capture.py`, and `attention_export.py`. The historical exporter sources are recoverable from Git, but executing relocated copies records different physical source paths. The exporter's explicit source-archive option also records substitutions, changing the full contract bytes. The attention-contract closure additionally has historical system loader and `libm` identity drift. Provision the authenticated original inputs and source/environment closure before attempting exact replay; do not replace these dependencies with newly chosen scales, rewritten provenance, or a copy of the expected contract. Byte-identical regeneration and changed-exponent/producer-tensor refusal checks for a reproduced contract remain **PENDING**. A general calibration policy and arbitrary-input accuracy are unproved.

## SmolVLA policy capture and replay

The dynamic-block receipt must bind the native full-query manifest and retained policy inputs. Run the [native SmolVLA recipe](#capturing-a-new-model) with observations for fresh vision and ten denoising steps, then replay it on Atlas:

```sh
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
