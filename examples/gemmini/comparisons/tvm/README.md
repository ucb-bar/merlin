# TVM comparison: actual EL4 workloads through FireSim

This directory is the model/session side of the TVM Gemmini comparison. The
companion TVM fork owns mathematical lowering, device schedules and static
packages. Merlin owns workload contracts, references, session routing and the
existing queued execution infrastructure. The next execution milestone is full
ResNet50, TinyLlama and SmolVLA sessions on the pinned EL4 FireSim device.

The delivered operator studies are complete at their stated scope. They do not
qualify those full-model FireSim sessions. Start with [AGENTS.md](AGENTS.md) and
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). [EL4_ASSETS.md](EL4_ASSETS.md) identifies the
recorded private capture/bundle paths, required files and known hashes. The
historical host, cutout and legacy BYOC
reports remain in [README_HISTORICAL.md](README_HISTORICAL.md); their original
inputs, tolerances, failures and attribution remain unchanged.

## Source checkout and documentation map

The clean delivery branch `comparisons/tvm-el4-handoff` starts at EL4
`0cc289de52a10bed256f32c90cfbb54a21b3c81f`. Original comparison work was on
`comparisons/tvm-gemmini-byoc` at `1bd601422fc1defc4343bacbf4227ca17e8aefef`
plus the saved working changes. The original comparison branch and dirty tree
are preserved. This delivery adds comparison files/tests/docs to EL4 and does
not rewrite its compiler/core/campaign code.

TVM's local delivery branch is `gemmini/curated-handoff`, based on Apache v0.19.0
`c4dc0c29ff81ddae688da24625a603da3b4a2c0e`. Its historical measured source is
`87f07883b65ee55dbd1b93f665aaab87cd8a1e50` on `gemmini/metaschedule-dse`.
See the final handoff artifact for delivery commit IDs, bundles, hashes and tests.
TVM is a separate repository, not a Merlin submodule. Inside its source tree:

| Document | Authority |
| --- | --- |
| `apps/gemmini/README.md` | Supported entrypoints and qualification boundary |
| `apps/gemmini/OWNERSHIP.md` | Stock/supplied/external responsibilities, all custom passes, optimization ownership chart, upstream changes |
| `apps/gemmini/TUNING.md` | Stock MetaSchedule flow, measurement policy and extension priorities |
| `apps/gemmini/FIRESIM.md` | Static ABI, independent fixtures, final linking/deployment gates |
| `apps/gemmini/AGENTS.md` | Compiler continuation instructions |

The retained `target.yaml`, `HOST_SETUP.md` and `byoc/` describe the older
`6b376cc8` bringup profile. They are reproducibility references, not the current
TensorIR configuration. The new compiler uses its source-bound
`GemminiCapabilities`, primitive adapter receipt and `firesim_target.json`.
Do not install the legacy snapshot over a new development checkout.

## First session: establish an isolated environment

Read root `AGENT.md`/`CLAUDE.md`. Work in separate checkouts, set the actual paths
below and use a fresh artifact root. Do not copy another server's absolute paths
or source-epoch receipts as if they qualified this server.

```bash
export MERLIN=/absolute/merlin-el4-checkout
export TVM=/absolute/tvm-curated-checkout
export TVM_BUILD=/absolute/matching-tvm-build
export PY=/absolute/python-with-torch-onnx-tvm-dependencies
export M2M=/absolute/model2mlir-checkout
export GCC=/absolute/riscv64-unknown-elf-gcc
export MERLIN_REPO_ROOT="$MERLIN"
export PYTHONPATH="$MERLIN/src:$TVM/python:$M2M"
export TVM_LIBRARY_PATH="$TVM_BUILD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$MERLIN"
"$PY" - <<'PY'
import tvm
from tvm._ffi.base import _LIB
from merlin.common.artifacts import new_product
print('TVM Python:', tvm.__file__)
print('TVM native:', _LIB._name)
p = new_product('compare', version=1, target='gemmini-tvm-el4',
                notes='Actual workload intake; no hardware qualification inferred')
print('Set RUN to:', p.path)
PY
export RUN=/absolute/path/printed/above
```

Add only the explicit dependency roots required by that environment. Record
Python, Torch, ONNX, NumPy, TVM, LLVM, GCC and source hashes. The completed local
study used LLVM18/native compiler-v5 and a separate Python3.12 environment; a
clean history does not require recompiling byte-identical native sources.
Any subsequent C++ change does. Confirm the native source identity before reuse.
The TinyLlama cache/session integration requires the selected Transformers 5.4.0
overlay; the SmolVLA environment's 4.57.1 cache API is incompatible. Source-exact
SiLU profiles additionally require AVX512; the current AVX2 host cannot qualify them.
The handoff artifact includes a local-path environment example; it is not portable.

## Select the actual workload, not a development stand-in

The current EL4 model roster is
[examples/gemmini/numerics/models.yaml](../../numerics/models.yaml), with source
and numerical policy in [the protocol](../../protocol/README.md). Its selected
run configuration and input manifests are authoritative if they override defaults.
[Phase 0 workload patterns](../../target/phase0-workloads.yaml) are independently
authored development inputs, not actual pretrained model checkpoints. The old
`el4_20261018/common.py` embeds server-specific paths and a stopped campaign;
do not rerun it to discover a new model workflow.

| Workload | Actual complete boundary | Required checks and unresolved work |
| --- | --- | --- |
| ResNet50 v1.5 | Real checkpoint; frozen preprocessing/calibration; every selected image; all logits and selected labels | Full-device graph and accuracy gates; separate quantized-vs-source quality from compiled-vs-quantized correctness; actual FireSim timing pending |
| TinyLlama | EL4 default prefill32 + 32 teacher-forced decode steps; every stage output, full KV cache and position carried exactly; reset and repeat | Comparison factory defaults are a short frontend diagnostic and must be overridden; bind tokenizer/corpus/checkpoint and chosen precision; full memory/loader/E2E qualification pending |
| SmolVLA | Prefix encode, ten flow/denoise updates, final action decode; real images/masks/language/state/noise; full cache, time and latent recurrence; all padded `[1,50,32]` action outputs | Synthetic fixture and one transformer-block cutout are insufficient. Six semantic channels are a separate quality report. The old complete 16-layer diagnostic failed its floating gate; numerical closure and full FireSim qualification remain required |

EL4's default numerical policy names all declared outputs at every step,
finite values and exact integer/boolean comparison. The roster's default
functional thresholds are `atol=2e-6`, `rtol=1e-5`; its quality thresholds are a
separate contract (e.g. TinyLlama KL/top-one agreement). Read the **selected frozen
run policy** before adopting these defaults. TVM comparison CLI defaults differ
and do not override EL4. Preserve historical recipes as historical; do not silently
replace a full-model contract with the old 128+32 host study or an INT8 cutout.

Checkpoint/data hashes, factory arguments and session manifests must identify:
model revision, original dtype, all quantization scales/zero points/rounding,
input order/layout, source preprocessing, calibration/development/held-out roles,
sequence/observation counts, cache capacity, recurrence/reset behavior, output
meaning and accuracy policy. Protect held-out inputs/answers from tuning and
public compiler-development artifacts. No checkpoints or tensors enter Git.

## Import and save real semantic programs

The reusable [session verifier](verify_session.py) follows model2MLIR's
`ExternalRuntimeSession`; factories supply model-specific contracts. It supports
source preservation, explicit routes, signature limits and repeated resets.
`--save-programs` persists `stage_N/semantic.json` with a source hash for the same
program input used by TVM tuning and compilation. This is opt-in because constants
can be large. Import-only mode executes native source stages to build real stage
inputs but does not build or execute the TVM candidate.

For TinyLlama, prepare local factory arguments (JSON) with at least
`checkpoint_dir`, `model2mlir_root`, `corpus_path`, `prefill_tokens: 32`,
`decode_tokens: 32`, and the explicitly selected `precision`/`quantized` policy.
The defaults in [tinyllama_session.py](tinyllama_session.py) are diagnostic.
For SmolVLA use [smolvla_session.py](smolvla_session.py): `checkpoint_dir`,
`backbone_dir`, `model2mlir_root`, `input_npz`, `input_manifest`, `input_source`,
selected cameras and the declared original precision/quantization policy. Select
`fixture_kind="dataset"`; its default `fixture_kind="synthetic"` is a diagnostic. Verify the processor and physical action dimension.

```bash
"$PY" examples/gemmini/comparisons/tvm/import_models.py \
  --model tinyllama --tvm-source "$TVM" --tvm-build "$TVM_BUILD" \
  --model2mlir-root "$M2M" --factory-arguments "$RUN/tinyllama-arguments.json" \
  --save-programs --output-dir "$RUN/tinyllama-import"

# Repeat for smolvla with its own real-input arguments and directory.
# ResNet intake accepts --model resnet50 --checkpoint /absolute/checkpoint.
```

For complete host candidate/reference checks use the verifier, not just import:

```bash
"$PY" examples/gemmini/comparisons/tvm/verify_session.py \
  --model2mlir-root "$M2M" --tvm-source "$TVM" --tvm-build "$TVM_BUILD" \
  --factory examples/gemmini/comparisons/tvm/tinyllama_session.py \
  --factory-arguments "$RUN/tinyllama-arguments.json" \
  --onnx-exporter dynamo --opset 22 --graph-mode optimized --repeats 2 \
  --rtol 0.00001 --atol 0.000002 --save-programs \
  --output-dir "$RUN/tinyllama-host"
```

This example uses the roster defaults; a selected frozen contract may require
other flags/source-preserving export options. Native BF16 source reproduction is
not established by generic FP32 fused export. An import/refusal or host mismatch
must be resolved and recorded before any device performance claim.

For ResNet, [prepare_resnet_inputs.py](prepare_resnet_inputs.py) can create distinct
streams from the selected local image archive; it never downloads the dataset.
The comparison verifier's `--quantized --calibration-inputs … --inputs … --labels …`
path requires explicit sources/preprocessing and accepts a frozen quantization
plan for held-out evaluation. `--export-baremetal` also saves
`integer_semantic.json`, the full integer recipe including its declared outputs.
Its Imagenette development stream is not a substitute for an EL4 selected dataset
or canonical ImageNet accuracy. See `verify_resnet.py --help` and retain the
source-versus-quantized quality gate before handing the program to the compiler.

## Extract, tune and compile exactly those programs

```bash
export PROGRAM="$RUN/tinyllama-host/stage_0/semantic.json"
"$PY" "$TVM/apps/gemmini/tune_models.py" --program "$PROGRAM" \
  --work-dir "$RUN/stage0-extraction" --extract-only
```

Read `extraction.json`: all extracted tasks, call weights, eligible tuning indices
and CPU exclusions are explicit. Compare it with the entire stage/session roster.
A stage imported once may execute 32 or ten times; IR-local extraction does not
know the outer session cadence. `--model tinyllama-decode` and the other built-in
TVM model-shaped programs use synthetic constants and are only pilot workloads.

```bash
"$PY" "$TVM/apps/gemmini/tune_models.py" --program "$PROGRAM" \
  --work-dir "$RUN/stage0-tuning" --trials-per-task 16 \
  --trials-per-iteration 4 --parallel 4 --builders 2 \
  --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --spike-receipt "$SPIKE_RECEIPT" --hardware-receipt "$RTL_RECEIPT"

"$PY" "$TVM/apps/gemmini/compile_package.py" --program "$PROGRAM" \
  --output-dir "$RUN/stage0-package" --riscv-gcc "$GCC" \
  --adapter-receipt "$ADAPTER_RECEIPT" \
  --tensor-memory-bytes "$STAGE_TENSOR_BUDGET" \
  --tuning-database "$RUN/stage0-tuning/database"
"$PY" "$TVM/apps/gemmini/compile_package.py" --verify "$RUN/stage0-package"
```

Set the receipt and budget variables to verified actual artifacts, not example
filenames. The tuning command measures on local Verilator, not FireSim. The
budget is a starting experiment setting, not evidence of convergence. Repeat for
all signatures; preserve CPU/default baselines and all failed/capped trials.
Inspect each package's placement, constant/workspace plan and archive audit.
No function matching a database is not the same as complete database coverage.
`--tuning-result` and the older custom controller are retired.

## Build the full session and functional controls

A TVM package exports `model_run(inputs, outputs, workspace, workspace_bytes)`.
The caller owns aligned, correctly sized, nonaliasing buffers. It must route every
stage and state update, preserve immutable constants and execute the selected
reset/repeat contract. Multiple archives export the same generic symbols: link
through an explicit namespaced-stage plan or the existing `baremetal.export_graphs`
API, with hashes binding each entrypoint. Do not blindly link several `model_run`
definitions, nor treat a TVM pointer-array ABI as a gemmini-mlir command-buffer ABI.

[`build_package_session.py`](build_package_session.py) now links static packages
through Merlin's existing startup, HTIF, linker, warm-profile generator and
OUT_BIN readback. It namespaces each stage's defined symbols and shares one
runtime/adapter. [`package_session.py`](package_session.py) validates explicit
routes, carries candidate state, resets input/output/workspace storage, guards
buffers and checks every output plus constant/input immutability. The result
includes positive and wrong-oracle ELFs, memory/instruction/symbol audits, source
bindings and a Merlin UART validation policy. It does not deploy a job.

Use the session verifier's existing route/fixture format, with real compiled
packages and independent references from the **selected** quantized captures:

```sh
"$PYTHON" examples/gemmini/comparisons/tvm/build_package_session.py \
  --tvm-source "$TVM" --stages "$SESSION/stages.json" \
  --route "$SESSION/route.json" --fixture "$SESSION/fixture.npz" \
  --fixture-provenance "$SESSION/reference.json" \
  --riscv-gcc "$RISCV_GCC" --output-dir "$OUT/linked-session" \
  --rtol "$SELECTED_RTOL" --atol "$SELECTED_ATOL" --readback
```

`stages.json` is `{"version":1,"stages":{"prefill":"prefill-package","decode":"decode-package"}}`;
paths are relative to that file. Each route call selects a stage and lists input
edges as `{"fixture":N}` or `{"output":[earlier_call,output_index]}`.
`fixture.npz` contains dense `input_N` and `output_N` arrays, outputs flattened
in invocation/output order. `reference.json` binds `fixture_sha256`, the route
path/SHA, `reference_contract` and `reference_sources` (path/SHA records).
Use independently admitted references; a provenance string alone cannot prove
their independence. Read tolerances from the selected numerical policy.

The bridge has local host tests and a real two-package Spike control: four
calls/eight outputs, independent NumPy reference, warm/reset/measured execution,
full readback, and wrong-oracle rejection after successful warmup. The same
control passes with constants/references at 9 GiB and session buffers at 13 GiB,
using an explicit 16 GiB guest span starting at `0x80000000`. This is a
small recurrent integration proof, not model or FireSim qualification. The v2
handoff artifact retains the exact source, ELF, simulator and reference hashes.

For large models, select `--weights-base ADDRESS --arena-base ADDRESS` together
with `--memory-bytes BYTES`. Both bases must be page aligned, inside the declared
map and leave room for code/stack. `session_memory.py` places stage constants and
input/reference blobs in Merlin's existing `weights_blob.o` region, and reserves
guarded buffers/workspace in an explicit NOLOAD arena. The final link checks
code/stack bounds and exact weight extent; ELF audits include the arena. The plan
rejects overlap, overflow and unsupported older packages before linking.

Absolute mode requires packages whose graph manifests declare
`constants_binding.absolute_address_macro=TVM_MODEL_CONSTANTS_ADDRESS`. It
recompiles only their generated `model.c` wrappers with literal constant bases;
the original scheduled `operators.o` and constant bytes are reused. Near mode
continues to link the packaged objects directly. Package hashing is streamed and
final ELF audits memory-map the image instead of copying multi-GB blobs into
Python. Record the selected bases and qualification; the small high-address
control does not establish full-model capacity or FireSim loader support.

Current limits: all invocation outputs and their references remain in memory
for validation outside timing; the report explicitly accounts for this cost.
The default near-code layout admits at most 2 GiB; larger maps require the
explicit absolute layout above. Stages must use the same capability,
runtime and adapter. Float16 is refused; FP32, FP64, BF16 and integer/boolean
storage are supported. Full-model capture-to-package intake, actual large-model linking,
selected-reference correctness and hardware execution remain unfinished.
`prepare_firesim.py` still stages **isolated block** ELFs only.

Use TVM `qualify_package.py` for an independently referenced stage and
`verify_tensorir.py run` for Spike/RTL positive and wrong-oracle controls. Then
verify the entire linked session on the selected functional engine. Check every
output and state at every invocation, not just a final cosine. Resolve the preserved
full SmolVLA numerical failure. Keep source/quantization quality and compiler
agreement as separate verdicts. Spike gives functional evidence, not hardware cost.

After execution, export complete UART outputs for the shared offline checker:

```sh
"$PYTHON" examples/gemmini/comparisons/tvm/read_package_session.py \
  --session "$OUT/linked-session/session.json" --uart "$RUN/uart.log" \
  --validation-policy "$OUT/linked-session/validation-policy.json" \
  --output-dir "$RUN/readback"
```

[`read_package_session.py`](read_package_session.py) reuses Merlin's OUT_BIN
decoder and warm/measured UART validator. It checks checksums, completeness,
storage ABI, successful exit and the policy's session-manifest hash before
writing one NPZ per invocation with `out0`, `out1`, etc. `readback.json` records
the stage name, zero-based stage invocation, array metadata and file hashes.
It preserves scalar shapes, integer widths and floating bit patterns; BF16 uses
explicit `raw_uint16_bfloat16` storage, which needs a BF16-aware consumer.
The selected EL4 interfaces use FP32/int64/bool and need no BF16 conversion.

Use these `steps` entries as the actual-array records in the
[baseline output manifest](../../protocol/baselines.md), pairing each with its
independently admitted quantized reference and binding the selected inputs,
protocol and numerical policy. Run `python -m merlin.model_check.baseline_numerics`
as described there. Export alone does not compare references, identify the
executed ELF/device, or qualify timing; retain the original execution receipts.
The exporter has local malformed-log tests and exact all-output comparisons
against the two-stage near/absolute Spike controls, not full EL4 model evidence.

## Bind the actual FireSim device and use the existing queue

The intended device is the full one-hart RV64GC U250 design, with no RVV:

| Artifact | Expected identity / remaining gate |
| --- | --- |
| Hardware pin | `firesim_gemmini_rocket_u250_30mhz` in `merlin/contract/hardware_pins.yaml` |
| Queue hw_config | `alveo_u250_firesim_gemmini_rocket_30mhz` |
| Bitstream archive SHA-256 | `0af6f26e1cbad1c03f9e5873217505a5100bf2d0d068205cfa486522b24fc603` |
| Elaborated design SHA-256 | `7b6280272c7d8db3e013c7e5fd07592d2fd317a42c669e8b3b08f29c32944d08` |
| Params header SHA-256 | `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924` |
| DTS SHA-256 | `c4a2009a405945ea98f731713e0f4e7dc483031ab8cd5156d871e4eb7594f1df` |
| Matching driver | Must be recovered/hashed with this image; TVM target record is currently null |
| DRAM | Declared 16 GiB starting at `0x80000000`; loader/high-address capacity needs actual qualification |

The similarly named stock and lean images are different devices. Header agreement
is necessary but not whole-SoC proof; adapter and FireSim source lineages differ.
The `30mhz` name is not a qualified clock/cycle conversion. Verify actual bytes.

Follow [FireSim guide](../../../../docs/guides/firesim.md),
[whole-model execution](../../../../docs/guides/whole_model_on_accelerator.md),
[whole-model machines](../../phase2/whole-model-machines.yaml) and the selected
[EL4 setup](../../../../docs/guides/phase2_el4_v26_setup.md). The objective JSON is
an `/ABSOLUTE/…` template, not a runnable configuration. The selected V26 generated
compiler admission is separate from the trusted TVM baseline; do not grant the
candidate TVM schedules, reference answers or compiler-bearing support.

Use the manifest schema documented in the FireSim guide. Set raw queue path,
queue state root, chipyard, exact hardware pin/hw_config/one-entry hwdb artifact,
workload/bootbinary, final ELF and an independent validation-policy file per entry.
The existing checkpoint tool validates the installed queue contract:

```bash
"$PY" -m merlin.perf.firesim_checkpoint "$RUN/checkpoint.yaml" --dry-run
# After the concrete image/driver/loader/control/session gates above pass:
"$PY" -m merlin.perf.firesim_checkpoint "$RUN/checkpoint.yaml"
```

The final caller must emit the existing warm/measured protocol and exact output
validation markers expected by the receipt. TVM's operator HTIF harness does not
emit that protocol automatically. Reuse Merlin's explicit warm-profile support;
do not rewrite UART logs to appear compliant. Keep warm and measured calls in
one boot, ensure one positive measured cycle count and preserve full output gates.
The queue owns `kill -> infrasetup -> runworkload -> kill` under one lock. No direct
FireSim invocations, manual staging races or parallel UART readers.

Before submission audit all final ELF instructions, undefined symbols, LOAD
segments, entrypoint, ISA/ABI, code, .bss/.noinit, stack, weights/packed duplicates,
inputs/outputs, KV caches, host scratch, session arena and verifier buffers.
Qualify needed high-DRAM addresses through the actual loader. A stage tensor budget
excludes these other reservations; a multi-GB blob cannot be treated as ordinary
medany-relative data. Use the existing absolute-pointer loader convention.

## Completion criteria for the next agent

For each actual workload retain checkpoint/input/export/program/package/ELF hashes,
source and quantized references, full output, state and reset checks, unchanged quality
metrics, a meaningful wrong-oracle control, peak memory accounting, hardware and
driver identities, job ID, submitted argv, client/daemon/UART logs and sealed
receipt. Declare exactly what the timing window includes: all stages, CPU work,
packing, state updates, completion fences and repetition policy. Do not report
summed operator minima as measured E2E latency. Preserve failed cases and CPU work
in the report. Compare only matching hardware and numerical/timing contracts.

Further tuning priorities are real-stage coverage, more post-XGB-training search
batches, CPU-only task tuning, a queue-backed FireSim Builder/Runner, full-session
memory/residency planning and independently tested new legal sketch families.
The detailed ownership/optimization chart is in TVM `apps/gemmini/OWNERSHIP.md`;
the extension plan and cost-model caveats are in its `TUNING.md`.
