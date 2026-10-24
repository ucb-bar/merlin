# Fixed-input block ELF reproduction

`build_block_elf.py` regenerates the captured Tiny first-block prefill and first decode inputs, or the selected Smol vision-language layer inputs, then links an RV64GC host ELF from the current Merlin C sources and an explicitly selected Atlas provider. It runs packing, assembly and compilers. It does not execute a device model, RTL, QEMU, full inference or a policy trajectory. The two workload ELFs remain unexecuted.

The reproduction boundary is a supplied, hash-pinned data recipe. No public command currently exports the complete recipe directly from a fresh checkpoint. Native capture commands in [refresh/README.md](refresh/README.md) export operands and provenance for narrower edges; they do not supply all block inputs automatically. Distribution of the reviewed fixed-input fixtures, model source and recorded native environments is a separate decision for the artifact owner. No weights, recipes, generated lookup tables, ELF files or result receipts are included in this source change.

## External recipe

The recipe is JSON with `workload` (`tiny` or `smol`), ordered `projections` (24 Tiny panels or seven Smol projections), `data` and `parameters`. Each file record has `path` and `sha256`. Relative paths are interpreted beneath the recipe's directory; absolute physical paths are also accepted. Parent traversal and symlink components are refused. Moving fixtures requires updating path spelling and authenticating the resulting recipe's new SHA-256; the referenced payload digests remain unchanged.

Every projection supplies `source_npz` and `source_manifest` records plus `source_dtype`. The NPZ contains the provider-admitted `A`, `B` and `native_C` arrays, and exact BF16 word arrays when required by that dtype. The manifest binds the dtype and original NPZ. `program_source`, `program`, `initial`, `activation` and `weights` records are comparisons against fresh generation, rather than sources of launch authority. `expected_output` is a read-only comparison payload. The current Atlas `compile_workload_inputs.prepare` function regenerates conversion, resource selection, tiled operands, assembly, machine words, guards and initial DRAM from the pinned source arrays. All five generated payloads must match the recipe exactly before compilation continues.

Tiny projections specify `rows`, `physical_rows`, `k`, `n`, `column`, `width`, `logical`, `a_scale`, `rescale_exponent`, `memory` and `requested_dma`. Smol projections specify `role`, `physical_rows`, `native_rows`, `input_width`, `output_width`, `program_words`, `rescale_word`, `rescale_exponent`, `activation_bytes`, `weight_bytes`, `output_bytes`, `output_offset`, `dram_bytes`, `activation_scale`, `weight_scale` and `requested_dma`. These fields must match the regenerated provider descriptors. Physical memory includes `output_offset`, `output_bytes` and `total_dram_bytes`; DMA counts include load and store bytes.

Tiny `data` contains `step0-entry`, `step1-entry`, `step0-expected`, `step1-expected`, `input-norm`, `post-norm`, `cosine`, `sine`, and `step0-keys`, `step0-values`, `step1-keys`, `step1-values`; `parameters` contains `epsilon`. Smol `data` contains `input`, `pad`, `attention_boundaries`, `input_norm_weight`, `post_norm_weight`, `expected_output`, `expected_key_cache`, `expected_value_cache`, `cosine` and `sine`; `parameters` contains `input_epsilon`, `post_epsilon` and `first_position`. Data records contain raw bytes, or an NPZ plus `array` key, or an NPY with `array: null`. Supported array types are uint8, uint16 BF16 words and float32, serialized little-endian. Expected outputs and caches are comparisons only; successor activations are computed by C.

## Build

Run from a parent directory containing `merlin`, `atlas`, `npu-model` and `fixtures`. Install Merlin's Python dependencies and use the selected model's assembler environment. The Atlas checkout must provide `tools/compile_workload_inputs.py`, its producer helpers, raw runtime/adapters and the target startup/linker templates. The builder snapshots `runtime/atlas_host_layout.h` when present and recursively includes quoted Atlas runtime headers, including any headers the layout header includes; an older provider without that header remains supported. Supply physical compiler, linker and native libm paths with their recorded identities. The native C expf helper generates the BF16 lookup at build time without a Python float round trip; the expected table digest refuses a different libm result. No generated table is checked in.

```sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/merlin/src"
export MERLIN_OUT_ROOT="$PWD/out"
python merlin/examples/atlas/comparisons/build_block_elf.py \
  --recipe fixtures/tiny/recipe.json --recipe-sha256 "$RECIPE_SHA256" \
  --provider-runtime atlas/runtime --atlas-tools atlas/tools --model-root npu-model \
  --startup atlas/runtime/host_templates/tiny_start.S \
  --linker-script atlas/runtime/host_templates/tiny_link.ld \
  --clang "$CLANG" --linker "$LLD" --native-cc "$CC" \
  --libm "$LIBM" --libm-sha256 "$LIBM_SHA256" \
  --expf-table-sha256 "$EXPF_TABLE_SHA256" \
  --expected-elf-sha256 "$ELF_SHA256" \
  --out out/artifacts/tiny-block
```

For Smol select its recipe and `smol_start.S`/`smol_link.ld`, and use a fresh output directory. The same tool emits kernel inputs, generated C/assembly assets, compile logs and `build.json` with source, toolchain, library and output bookends. To reproduce an existing binary, all toolchain and parameter identities must match that fixture. Supplying another supported toolchain creates a separately identified build rather than qualifying it by resemblance.

The default shared host library includes mixed BF16-plus-FP32 residual addition. Tiny does not call this API; its build defines `SMOL_HOST_EXCLUDE_MIXED_RESIDUAL` only for that translation unit to retain the qualified linked symbol set. Smol and normal library builds retain the full API. This option changes the available symbols, not the arithmetic of retained functions.

## Result scope

The integrated ResNet reference separately has whole-host-ELF RV64 QEMU execution with an npu_model MMIO bridge: 54 launches and 1,000 logits bit-exact to its packaged reference. The separate C54 Verilator AtlasCore run took 63,480,310 cycles. These are distinct gates; neither is an FPGA or full-SoC result. Integrated task-quality accuracy is not yet measured. Recorded native-fidelity failures remain failures under their original unchanged diagnostics.

Tiny and Smol expose partial native and selected-block arithmetic/RTL gates. Their generated ELFs are build evidence only. No full Tiny inference, complete Smol policy, platform timing or application-quality acceptance follows from this builder.

## Supporting entry points

These tools are explicit bounded diagnostics or helpers used by the builder and exporters. Use each command's `--help` for its required paths and receipt hashes; none launches automatically from the ELF build.

| Source | Inputs and role |
| --- | --- |
| `block_kernel_inputs.py` | Internal builder helper: the authenticated block recipe and Atlas tools regenerate all 24 Tiny or seven Smol kernel payloads. |
| `smol_vlm_block_injected_gate.py` | Pinned capture and rotary-table receipts plus a native C compiler; checks selected-layer C composition with archived outputs injected by callbacks. |
| `tiny_host_ops_parity.py` | Pinned native/block/decode receipts and shared C sources; checks host primitive parity and provides admission helpers to the layout/rotary tools. |
| `tiny_host_layout_parity.py` | Primitive receipt and supplied tables; checks head layout, cache and grouped-query attention transforms. |
| `tiny_rotary_parameters.py` | Layout receipt and native rotary contexts; emits admitted rotary parameters and checks C lookup results. |
| `tiny_workload_quality.py` | Informational scoring helper imported by the full arithmetic tool; it does not relax the original numerical acceptance gate. |
| `evaluate_smolvla_dynamic_block.py` | Selected-layer source, producer/model paths and retained-policy inputs; checks native continuation with admitted replacement outputs. |
| `refresh/resnet_provider.py` | Shared provider binding and conversion used by both full exporters and the partition inventory. |
| `refresh/resnet_graph_inputs.py` | Shared source-manifest and graph-input admission for the ResNet exporters and partition controls. |

ResNet capture uses `refresh/capture_resnet.py` with `original`, `original-3x3`, `after-conv1` or `after-conv2`. Replay uses `refresh/evaluate_resnet.py` with `single`, `coupled` or `three`; the final mode accepts an optional complete shortcut binding. These modes share one report lifecycle and preserve their distinct source/provider admissions. See [the source capture contracts](refresh/README.md).

Atlas source filenames are declared once in `atlas_provider_files.py`. Packing binds the producer helpers, path/tooling modules and the runtime layout header; replay receipts bind the producer tuple declared by the selected Atlas producer. Update that shared declaration when moving to another provider epoch, then rerun the identity gates.

Native-substitution replay requires an archived Atlas cold-contraction producer root providing `run_model_contraction.py` and `validate_model_contraction.py` with `bind`, `recheck` and `admit_contraction`, which current atlas-mlir no longer ships. Select that archived root explicitly for the native replay entry points in `refresh/README.md`. Current workload replay instead admits `run_workload_contraction.py` receipts with `producer="workload_contraction_v1"`; fixed-input ELF reproduction uses the current packing/runtime providers declared in `atlas_provider_files.py`.
