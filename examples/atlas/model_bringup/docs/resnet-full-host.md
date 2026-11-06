# Complete ResNet50 host graph

The generators support a batch-one ResNet50 plan with 177 ordered nodes and 54
Atlas contractions. Convolution and linear geometry, padding, groups, parameter
extents, finite values, topology and asset hashes are checked before C
generation. Host operations use separate FP32 multiply/add rounding and
sequential average pooling. The generated graph packs each activation from the
current C tensor and consumes actual device replies. Captured successor
activations are never embedded as graph inputs.

The device callback stages live packed A with immutable B, patches one admitted
rescale word, and launches the sequenced raw runtime. It checks input/weight
preservation and output guards before returning the physical BF16 result.
Descriptors, staging buffers and the device context must remain alive after
failure: timeout can retain device ownership until external quiescence and
reinitialization. The selected DMA address is `0x100000000`; emulation may map a
different CPU staging pointer through the memory callback.

## Regenerate the programs and host ELF

Use [model bringup setup](../README.md#build) and Merlin's development Python environment. Keep `ATLAS_ROOT` and `ATLAS_MLIR_BIN` on the canonical compiler; export `ATLAS="$VIEW"`, `ATLAS_SOURCE_ROOT="$VIEW"`, and `ATLAS_MLIR_ROOT="$VIEW"`, and use `cd "$VIEW"`. A source manifest identifies the frozen
checkpoint, image stream and selected image; those immutable data inputs remain
external. This reproduction route requires externally supplied fixed-input
fixtures and their hash-bound assets. The checkout does not distribute those
fixtures or provide an automated fresh-checkpoint capture runner. The public
Merlin entrypoints export the complete operator inventory, host plan and
arithmetic reference:

```sh
MERLIN="$MERLIN_ROOT"
REFRESH="$MERLIN/examples/atlas/comparisons/refresh"
PACKING_SHA=$(sha256sum tools/compile_workload_inputs.py | cut -d ' ' -f1)
VECTOR_SHA=$(sha256sum third_party/atlas-kernels/lib/golden/atlas_vector_arithmetic.py | cut -d ' ' -f1)
SCALAR_SHA=$(sha256sum third_party/atlas-kernels/lib/golden/atlas_tile_arithmetic.py | cut -d ' ' -f1)
SOURCE_SHA=$(sha256sum "$SOURCE_MANIFEST" | cut -d ' ' -f1)
python "$REFRESH/partition_resnet.py" --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_SHA" --out "$MODEL_OUTPUT_ROOT/resnet-inventory"
INVENTORY_SHA=$(sha256sum "$MODEL_OUTPUT_ROOT/resnet-inventory/partition.json" | cut -d ' ' -f1)
python "$REFRESH/export_full_host_plan.py" --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_SHA" --inventory "$MODEL_OUTPUT_ROOT/resnet-inventory/partition.json" \
  --inventory-sha256 "$INVENTORY_SHA" --tile-shape 32 32 32 --atlas-tools tools \
  --packing-provider-sha256 "$PACKING_SHA" --vector third_party/atlas-kernels/lib/golden/atlas_vector_arithmetic.py \
  --vector-sha256 "$VECTOR_SHA" --scalar third_party/atlas-kernels/lib/golden/atlas_tile_arithmetic.py \
  --scalar-sha256 "$SCALAR_SHA" --out "$MODEL_OUTPUT_ROOT/resnet-provider"
PLAN_SHA=$(sha256sum "$MODEL_OUTPUT_ROOT/resnet-provider/host-plan.json" | cut -d ' ' -f1)
python "$REFRESH/export_full_reference.py" --source-manifest "$SOURCE_MANIFEST" \
  --source-manifest-sha256 "$SOURCE_SHA" --host-plan "$MODEL_OUTPUT_ROOT/resnet-provider/host-plan.json" \
  --host-plan-sha256 "$PLAN_SHA" --atlas-tools tools \
  --packing-provider-sha256 "$PACKING_SHA" --vector third_party/atlas-kernels/lib/golden/atlas_vector_arithmetic.py \
  --vector-sha256 "$VECTOR_SHA" --scalar third_party/atlas-kernels/lib/golden/atlas_tile_arithmetic.py \
  --scalar-sha256 "$SCALAR_SHA" --out "$MODEL_OUTPUT_ROOT/resnet-provider/reference"
PROVIDER="$MODEL_OUTPUT_ROOT/resnet-provider"
```

If a captured manifest was relocated, the exporters accept explicit repeatable
`--relocate OLD_PREFIX=NEW_PREFIX` mappings and continue checking asset hashes.
The reference exporter performs offline target arithmetic, not Core, RTL,
platform or task-quality execution. The selected model checkout must match the
revision checked by `run_model_tile.py`. Compiling the following templates uses
its assembler without constructing a Core or executing a program.

```sh
python tools/compile_resnet_templates.py --model-root "$MODEL" \
  --host-plan "$PROVIDER/host-plan.json" --out "$MODEL_OUTPUT_ROOT/resnet-templates"
python tools/generate_resnet_graph.py --plan "$PROVIDER/host-plan.json" \
  --templates "$MODEL_OUTPUT_ROOT/resnet-templates/receipt.json" --out "$MODEL_OUTPUT_ROOT/resnet-graph"
python tools/check_resnet_graph_replay.py --generated "$MODEL_OUTPUT_ROOT/resnet-graph" \
  --math runtime/atlas_resnet_graph_math.c --reference "$PROVIDER/reference" \
  --cc "$NATIVE_CC" --out "$MODEL_OUTPUT_ROOT/resnet-replay"
python tools/build_resnet_graph_host.py --generated "$MODEL_OUTPUT_ROOT/resnet-graph" \
  --replay-gate "$MODEL_OUTPUT_ROOT/resnet-replay/graph-replay.json" \
  --reference "$PROVIDER/reference" --templates "$MODEL_OUTPUT_ROOT/resnet-templates/receipt.json" \
  --rescale-table "$MODEL_OUTPUT_ROOT/resnet-templates/rescale-words.bin" --atlas-root . \
  --clang "$RISCV_CLANG" --lld "$LLD" --out "$MODEL_OUTPUT_ROOT/resnet-host"
```

The native replay checks all 54 C-produced activation streams and scales and the
1,000 final target logit words against hash-bound saved device replies. It
executes host C, not device instructions. The ELF builder snapshots the
graph/runtime and audits every embedded asset and ELF load segment. Its
RV64GC/LP64D startup enables FP, sets RNE, and supplies a 64 KiB stack, aligned
mailbox and high-address DMA arena. Linking does not execute the ELF.

## EE290 SoC kernels

The frozen per-kernel network-model fixture supplies `kernel-*/model/{program.bin,initial-dram.bin,expected-bf16-tiles.bin}`. Build one independent ELF per kernel with a RISC-V GCC toolchain that supplies `htif_nano.specs` and `htif.ld`; the builder links the shared host runtime in place. Provide physical paths without symbolic links for all inputs and toolchain executables. LOADMEM places the initial device image at `0x100000000`; HTIF reports completion and the ELF checks every output word, both guards and the runtime sequence markers.

```sh
python tools/chipyard/resnet_kernels.py build --network-model "$NETWORK_MODEL" \
  --cc "$RISCV_CC" --out "$MODEL_OUTPUT_ROOT/resnet-soc-kernels"
# Use an already built EE290GenericIOSimConfig simulator; set its library/license environment separately.
export ATLAS_SOC_SIMULATOR="$EE290_SIMULATOR"
export ATLAS_DRAMSIM_INI="$DRAMSIM_INI"  # Enable +dramsim to reproduce the recorded cycle totals.
python tools/chipyard/resnet_kernels.py run --build-root "$MODEL_OUTPUT_ROOT/resnet-soc-kernels" \
  --out "$MODEL_OUTPUT_ROOT/resnet-soc-runs" --jobs 1
python tools/chipyard/resnet_kernels.py results --build-root "$MODEL_OUTPUT_ROOT/resnet-soc-kernels" \
  --runs-root "$MODEL_OUTPUT_ROOT/resnet-soc-runs" --out "$MODEL_OUTPUT_ROOT/resnet-soc-results.json"
```

The runner records logs, exit status and ELF identity with finite cycle and wall limits; select them with `--max-cycles` and `--wall-seconds`. Increase `--jobs` only after allocating simulator licenses and CPU resources. Results require a matching ELF identity and runner status in `run.json`, zero exit, no timeout and exactly one matching PASS for every built kernel; the complete fixture must report 54/54 exact. No simulator is built by these tools.

For a full host loaded by a loader that zero-fills every PT_LOAD memory tail, `build_resnet_graph_host.py --skip-bss-clear` omits the redundant startup clear. The default still clears BSS.

## Opt-in full-graph traffic reuse

`compile_resnet_templates.py --traffic-reuse` selects the qualified public generator dispatch for every declared contraction while preserving the plan, physical operand layout and arithmetic references. This remains compile-only preparation. The graph runtime admits at most 32,763 raw words, reserving five additional words for the sequenced completion wrapper inside the 32,768-word instruction memory; relocated programs are checked again after expansion.

Generate the graph and run the native captured-reply gate with the selected templates as above, then build the host with an explicit physical DRAM contract. For a generic QEMU UART profile, pass `--uart-kind ns16550 --uart-base 0x10000000`; a hardware UART profile produces a separately bound ELF. The physical DRAM options validate all load segments, allocated sections and stack space, and `--arena-base` relocates the program, runtime layout and linker reservation together.

```sh
python tools/compile_resnet_templates.py --traffic-reuse --model-root "$MODEL" \
  --host-plan "$PROVIDER/host-plan.json" --out "$MODEL_OUTPUT_ROOT/resnet-reuse-templates"
# Generate, replay-check and link using this template receipt and rescale table.
python tools/prepare_resnet_graph_bundle.py --host "$MODEL_OUTPUT_ROOT/resnet-reuse-host" \
  --templates "$MODEL_OUTPUT_ROOT/resnet-reuse-templates/receipt.json" --reference "$PROVIDER/reference" \
  --model-root "$MODEL" --out "$MODEL_OUTPUT_ROOT/resnet-reuse-bundle"
python tools/host_iss/run.py --workload resnet --elf "$MODEL_OUTPUT_ROOT/resnet-reuse-bundle/host/host.elf" \
  --bundle-root "$MODEL_OUTPUT_ROOT/resnet-reuse-bundle" --model-root "$MODEL" --qemu "$QEMU" \
  --wall-seconds 21600 --out "$MODEL_OUTPUT_ROOT/resnet-reuse-live"
```

The bundle preparer regenerates each scaled program from the current public dispatch, checks it against the host image, preserves all 1,000 expected logit words, and binds captured inputs, immutable weights, guards and expected replies with checksums. Its new kernel receipts use `atlas.resnet_kernel_preparation.v1` and `PREPARED_NOT_EXECUTED`; these are inputs for future execution, never successful numerical receipts. The live bridge accepts this preparation schema as well as legacy `passed_model_only` payloads, then executes every launch and checks its actual DMA counts, preserved bytes and output. Only a fresh successful whole-host run qualifies the reuse graph.
