---
title: model2MLIR frontend
kind: guide
status: current
owner: frontends
last_verified: 2026-10-20
related: [getting_started, extending_the_stack, phase0_specification, model_lowering, reproducibility]
code_refs: [src/merlin/frontends, src/merlin/capture/bundle.py, src/merlin/targetgen/_m2m_capture_worker.py, src/merlin/targetgen/frontend_trace.py, src/merlin/frontends/compile_inputs.py, src/merlin/semantic_compiler/linalg_bridge.py, src/merlin/targetgen/cli.py, packages/merlin-experiments/src/merlin_experiments/phase0/m2m_runtime.py]
---

# model2MLIR frontend

Merlin consumes model2MLIR's typed linalg-on-tensors MLIR and external tensor
payloads. Framework capture and importer/decomposition extensions belong in the
selected model2MLIR environment. Target encoding and drivers belong in OOT support
packages. Parsing a capture or inventorying its matmuls is not proof of complete
model lowering, numerical agreement, or accelerator execution.

The compiler-only staged signature marks each function argument as a runtime
input, parameter, constant, or buffer. The application compiler binds parameters
and declared constants as immutable values. Buffers require an explicit state
interface, including their initialization and any returned updates; storing a
buffer initializer does not make the buffer immutable. Runtime arguments remain inputs.

The bridge currently preserves these source operations as typed graph nodes:

- Fully static `tensor.insert_slice`, including non-unit strides and the
  initialized destination outside the inserted area. Dynamic and rank-reduced
  slices are unsupported.
- All-parallel `linalg.generic` gathers with an identity output map and an
  in-bounds integer-affine input map.
- Strict f32 add, subtract, and multiply with static singleton-projected
  inputs, including a rank-one bias over a rank-two result.
- One-axis f32 `linalg.reduce` when the source declares
  `merlin.reduction_order = "axis-ascending"` and the body adds each input to
  the initialized accumulator with unflagged `arith.addf`. The graph keeps
  one binary32 rounding per addition.

An unannotated reduction has no assumed order. The source annotation records
the chosen order but does not establish that an external model intended it.
Captured f32 arithmetic still needs a selected precision policy or compiled
host route before it can execute on a target with different arithmetic.

For the complete model → kernel → MLIR → machine-code distinction, start with
[Extending the compiler stack](extending_the_stack.md#understand-the-two-lowering-routes).
For the independent iteration loaders and held-out TinyLlama, SmolVLA and ResNet50
policy, see [Iteration workloads and headline validation](../../examples/workloads/README.md).

## Configure an isolated capture environment

Complete [Getting started](getting_started.md), then select the external
model2MLIR checkout with `MERLIN_M2M_DIR` and its interpreter with
`MERLIN_M2M_VENV` where the integration requires it. Keep machine-specific paths
in local configuration. PyTorch, TorchAO and model-specific dependencies live in
that capture environment; Merlin's artifact consumers do not need to import them.

`build_tools/scripts/setup_model2mlir.sh` is an optional setup helper for an
existing checkout. It accepts `MODEL2MLIR_DIR` and `SMOLVLA_CAPTURE_DIR` and can
install framework/nightly dependencies. Inspect its choices before running it;
record the resulting repository revisions and package versions for each capture.
An installed dependency or a filename alone does not establish a reproducible
toolchain or a supported model precision.

For a frozen Phase 0 diagnostic run, select both the checkout and interpreter
explicitly with `--phase0-m2m-root` and `--phase0-m2m-python`; shell aliases are
not inherited. The run's `phase0/private/m2m-runtime.json` records copied M2M
source/workload inventories and the checked host Python environment. It does
not certify a sealed capture or grant Phase 0 admission. See
[Phase 0 specification](phase0_specification.md#select-a-frontend-capture-runtime-for-a-frozen-diagnostic-run).

Weights and model loaders are separate inputs. Select the intended checkpoint,
revision, representative input source and preprocessing explicitly. Loading a
randomly initialized architecture, truncating a graph, or capturing one denoising
step must be recorded as that scope, not complete checkpoint/session validation.

## Start from PyTorch and inspect the frontend trace

The trace-capable model2MLIR public API provides a small inspection loop:

```python
import torch
from m2m import convert

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU()).eval()
inputs = (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,)
result = convert(model, inputs, backend="fx_importer", capture_trace=True)
assert result.ok, result.diagnostics
assert result.capture_trace is not None
print(result.path_taken)
print(result.mlir_text)
```

This returns conversion evidence, not a complete deployment bundle. To retain
external weights, inputs, goldens and the selected PyTorch catalog, use the
existing capture worker from Merlin's checkout:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out out/artifacts/workloads/iteration/mixed-mlp/fp32
```

Use a fresh output directory. The worker seeds construction before importing the
loader and requires deterministic framework algorithms. It retains the exact
conversion/model instance instead of recapturing a second model. Inspect:

| Artifact | Purpose |
| --- | --- |
| `frontend-trace.json` | Original → quantized → prepared graphs and exact MLIR correspondence |
| `pytorch-opset.json` | The selected interpreter's versioned ATen/Core ATen/decomposition catalog |
| `linalg.mlir` | The exact capture used for demand and lowering analysis |
| `model.mlir`, `weights.safetensors`, argument manifest | Executable model input and separate tensor payloads |
| Inputs, golden and `capture_receipt.json` | Invocation data, independent framework reference and byte-bound capture identity |

Original frontend call counts, prepared calls, MLIR operations and runtime
invocations are different denominators. Inspect the trace and typed SSA edges;
do not infer a source operation from a final MLIR name. Missing original snapshots
remain unknown. For an external quantizer, retain the original frontend snapshot
before mutation and pass `original_frontend_snapshot` through the conversion API.

## Quantization is operation-scoped

Standalone model2MLIR captures and Merlin Phase 0 recipe captures are separate
routes. A standalone bundle records the quantization actually performed by its
external capture pipeline. Its storage dtype does not prove that the selected
accelerator implements that scheme, scale layout, accumulator precision or readout.

For Phase 0, Merlin derives recipes from selected datapath/readout facts and the
reviewed software/quantization declarations. The capture worker uses public
TorchAO interfaces: static PT2E `Quantizer`/`QuantizationSpec` with calibration, or
dynamic `AOBaseConfig`-derived configurations with per-module `FqnToConfig`.
Neither route edits PyTorch or TorchAO source.

Inspect the derived recipe, per-layer eligibility/refusals, actual annotations,
scale axes/values, storage tensors and numerical outputs. A supported contraction
format is not permission to quantize LSTM, normalization or every `Linear`.
FP16/BF16 capture, integer quantization and multiple low-bit formats have different
obligations. An authored format or recipe candidate is not a working framework
adapter or target compiler. Unsupported formats must fail explicitly; no blanket
claim of available or unavailable formats replaces per-run evidence.

See [TorchAO extension interfaces](extending_the_stack.md#extend-quantization-through-public-torchao-interfaces)
for the extension seam and [Phase 0 specification](phase0_specification.md)
for operation partitions, precision contracts and exact generated evidence.

## Consume complete capture bundles

To give a compiler only declared source inputs, stage a fresh compiler directory
from the materialized capture:

```bash
merlin-targetgen stage-capture --capture /absolute/capture \
  --out /absolute/artifacts/compiler-inputs \
  --status-file /absolute/artifacts/staging-status.json
```

The staged directory contains parsed `program.mlir`, external weights and their
argument manifest, the frontend source trace, a static invocation signature,
and a digest manifest. When the MLIR function has externalized parameters or
buffers, the signature records each function argument's weight name or runtime
input position. Staging checks those roles, types, shapes, and weight bytes
against the receipt-bound manifest. It does not copy or read runtime samples or goldens.
The capture receipt binds the copied member bytes; source closure, target
numerical admission, and model execution remain separate checks. The output
directory must be new, and failure returns a nonzero status.
If metadata omits the entry name, staging accepts only a single parsed function.
Captures with several functions must name the entry explicitly.

When the capture stores declared lifted constants or buffer initializers in
`extra.npz`, staging selects only those tensors. It checks their parsed argument
types and the producer's declarations, then writes
`capture-constants.safetensors`. The original weights keep their bytes. The
v3 signature names one of these two archives for each stored argument, and the
digest manifest records the extraction. The raw `extra.npz` archive is excluded
from compiler inputs, along with runtime samples and reference outputs.

Staging performs no quantization or general dtype conversion. It accepts the
producer's BF16-as-f32 storage only when every original BF16 bit can be
recovered exactly; NaN payloads remain unqualified. Lossy legacy buffer storage,
missing initializers, and ambiguous buffer names produce an error. Existing
v1/v2 stages retain their format and checks.

Staging copies and rehashes the weight archive in chunks before publishing it.
`load_compile_inputs` hashes tensor archives in chunks and reads their bounded
headers without copying the complete payload into Python. Argument extraction
can also select roles and limit the total unique payload bytes:

```python
from merlin.frontends.compile_inputs import load_compile_inputs, load_lifted_argument_bytes

staged = load_compile_inputs(compiler_directory)
constants = load_lifted_argument_bytes(
    compiler_directory, staged, kinds=("constant",), max_bytes=1024 * 1024,
)
```

This still rechecks every archive's digest, including unselected parameters.
Selected bytes and the digest come from the same read stream. Repeated
arguments naming one stored tensor share one payload and count once against
the extraction budget. Parameter and buffer roles remain in the signature;
this selection does not turn them into runtime samples or make them optional.
Without `kinds`, all stored argument roles are returned. The embedded C
parameter exporter checks its byte budget before materializing payloads.

An explicitly selected precision policy can be staged with
`stage-capture --precision-policy /absolute/policy.json`. The JSON must carry a
`source_sha256` matching the original captured `model.mlir`; staging copies its
exact bytes and records their hash. It does not validate the policy's arithmetic
or accept it as a target lowering. Keep the policy outside the capture bundle.

The native commands accept `--compile-inputs /absolute/artifacts/compiler-inputs`
in place of `--linalg` or `--request`. They recheck the declared file set and
member hashes, and use the staged signature's entry function. A staged precision
policy still returns `unsupported_semantics` for whole-source compilation: no
native numerical transformation covers the full captured graph.

For an explicitly selected, source-bound matmul tile, `native-select` and
`native-compile` also accept `--diagnostic-region REGION_ID --mode diagnostic`.
The installed target support must provide that exact tile profile; pass
`--support SUPPORT_NAME` to both commands and to `native-verify`. The compiler
checks every policy matmul region against parsed source operations, builds one
typed target tile, and records input reindexing and precision conversions as
**unmaterialized boundary obligations**. `native-compile` keeps that record in
`precision-plan.json`, tracked by the compilation manifest. Its emitted binary
is a diagnostic target-program region, not a source-equivalent model artifact;
the boundary conversions, source-model quality, and host execution still need
separate qualification. A missing region or a target mode absent from the
selected support fails explicitly.

To inspect every declared matmul in one staged capture, run the diagnostic
region sweep with the same snapshot and support. An exact `--abi` file applies
when every tile has the same input roles. For regions with different operand
counts, pass an explicit slot pool instead:

```bash
merlin-targetgen native-probe-model-regions --engine merlin_native \
  --support SUPPORT_NAME --snapshot /absolute/native-snapshot \
  --compile-inputs /absolute/artifacts/compiler-inputs \
  --abi-pool /absolute/diagnostic-slots.json \
  --target-source /absolute/target-source \
  --out /absolute/artifacts/model-region-probe
```

The slot pool is JSON with `input_slots` and `output_slots` arrays of distinct,
nonnegative physical slot numbers, for example `{"input_slots":[3,4,5,6],
"output_slots":[35]}`. The probe derives an exact ABI for each tile from its
declared operand roles and writes it under `abis/`. The selected target compiler
still checks those addresses. A pool that cannot fit a tile returns a resource
limit; it does not justify moving the computation to the host.

`manifest.json` records each source region's SSA inputs and result, its compile
status and artifact, plus every top-level operation still lacking a route.
For explicit source-bound precision decisions, it also attempts named and
batched output tiles. It checks that emitted tile plans cover distinct source
windows without gaps before reporting all tiles emitted. A direct whole-region
failure remains visible beside its isolated tile results.
`source-graph.json` retains the exact parsed SSA inventory. The command
also writes `stitching.json` with ordered kernel, host and transfer steps;
those steps are obligations, not executable dispatches. The command
continues after one region fails and returns a nonzero `incomplete` status:
separate region programs are not yet a composed model, and a per-tile ABI does
not imply that their physical addresses can be reused across a model schedule.

For a single admitted static rank-two integer contraction, the installed native
selector can parse Linalg directly. Supply a native snapshot built for the selected
target profile, and select the entry function explicitly:

```bash
merlin-targetgen native-select --engine merlin_native \
  --snapshot /absolute/native-snapshot --linalg /absolute/region.mlir \
  --linalg-entry work --mode strict-native --out /absolute/selection.json
```

When the selected target requires a physical boundary representation, pass
`--physical-layouts /absolute/layouts.json` to `native-select`, `native-compile`,
and `native-verify`. Both `merlin.physical_boundary_layouts.v1` and `.v2`
contain `target_identity`, `source_identity` (SHA-256 of the exact Linalg file),
`physical_extents` and `boundary_layouts` keyed by every input/constant ID,
and `output_layouts` in result order. Extents use the named storage bank's
address unit. Version 2 also requires `request_digest` for the exact typed
request after any selected precision transformation and before layouts are
bound. A changed transform or numerical policy therefore requires a new layout
declaration. Version 1 applies only to direct parsed Linalg. Merlin checks the
declaration before selection, and verification reconstructs the same request
and compares the canonical `physical-boundary-layouts.json` stage file. This
option still refuses outer tiling and typed `--request` inputs; a typed request
already carries its own layouts.

The optional `native-select --abi` file fixes boundary addresses with
`fixed_inputs` and `fixed_outputs`. It may also list `reservations`, which are
`{storage, start, extent}` intervals that the caller already owns. The selector
places no value inside a reserved interval and records the reservations in its
report. A malformed reservation is refused, not ignored. The `native-compile`
ABI takes only the fixed input and output addresses.

`native-compile` accepts the same `--linalg` and `--linalg-entry` pair alongside
its required OOT support, ABI, target source and output arguments. Both commands
also accept `--request` for an already typed semantic graph. The Linalg bridge
currently admits signed i8/i32 or i32 contractions with ordered wrapping i32
accumulation, explicit initialization and static rank-two shapes. It also
admits type-preserving tensor movement, static reshapes, same-rank slices and
insertions, checked affine gathers, and a bounded i64-indexed embedding lookup.
An unflagged, all-parallel f32 multiplication is retained as a typed semantic
node, including a one-input square. These entries describe source semantics;
Target instruction selection and physical execution remain separate checks.
Bare f32 `linalg.reduce` still requires a qualified numerical-order policy or
a compiled host route. The bridge does not assign it a K-ascending fold merely
because a sequential implementation would be convenient.
Closed, explicitly annotated Linalg regions admit a finite-domain ordered
FP8/BF16 reset reduction, bounded BF16 add, nearest-even BF16 multiply,
sign-clamp ReLU, and a BF16-to-E4M3 `arith.truncf` with declared numerical
policy, input domain,
and scale code. These forms do not equate an f32 model with a quantized target
program. The multiply form requires the exact unflagged FP32 product and
nearest-even BF16 bit-rounding body, with finite normal operands and a normal
rounded result supplied as a caller premise. The cast bridge checks the scalar
body and a direct input's
domain declaration. It retains a computed producer as a separate semantic
node; the selected OOT target binding must establish that producer's result
satisfies the cast's declared domain. The caller must establish runtime
input-domain assumptions.
Other operations or numerical policies return a nonzero
structured status. This single-region route does not compile a complete
capture or model.

After compilation, reopen the published package against the same source and
native snapshot:

```bash
merlin-targetgen native-verify --engine merlin_native \
  --snapshot /absolute/native-snapshot --linalg /absolute/region.mlir \
  --linalg-entry work --mode diagnostic \
  --artifact /absolute/compiled-region --out /absolute/verification.json
```

`native-verify` checks source, target, compile mode, snapshot, binary, execution
plan, and every stage file declared in the package manifest. If the package
records a selected graph, it must also carry a checked selection record with
the same fingerprint. The command returns a nonzero status and replaces its
status JSON when a file or identity changes. This is an artifact-integrity
check; it does not execute the program or establish numerical or RTL behavior.

`merlin.capture.bundle.CaptureBundle` is the canonical capture-bundle interface.
`merlin.baselines.bundle` retains a compatibility import. Legacy roster resolution
uses `merlin.common.artifacts.recaptures_dir()` and recorded variant/scope metadata;
prefer explicit freshly produced bundles when qualifying a new workflow.

The bundle format keeps `model.mlir`, `weights.safetensors` and its argument-index
manifest, `inputs.npz`/`input_order.json`, `golden.npy`, and lifted buffers/constants
separate. Session bundles may contain multiple programs and a session contract.
`CaptureBundle.require()` checks essential MLIR/golden presence, including selected
session programs; it does not validate every external payload or certify outputs.
Consumers must also check their required argument order, shapes, payload hashes,
precision and complete entrypoint/state interface.

For matmul inventory inspection:

```python
from merlin.frontends import linalg_mlir

module = linalg_mlir.parse_mlir_file("/absolute/capture/model.mlir")
manifest = linalg_mlir.load_manifest(
    "/absolute/capture/weights.safetensors.manifest.json"
)
inventory = linalg_mlir.matmul_inventory(module, manifest)
for operation in inventory:
    print(operation.kind, operation.m, operation.k, operation.n, operation.dtype)
```

This is a contraction inventory, not whole-PyTorch operator coverage. Use Phase 0's
frontend/operation accounting for the complete captured graph and its host,
accelerator-candidate and unresolved partitions. Preserve unknown shapes and
missing lineage instead of treating them as supported operations.

## Lower and audit the model

[Inspecting whole-model MLIR lowering](model_lowering.md) documents the existing
`merlin lower --ir-audit both --audit-sidecar ...` workflow. It retains inspectable
named stages from preprocessing through MLIR LLVM dialect and LLVM IR. Host shared
libraries and RISC-V objects are separate code-generation outputs; an object is
not a linked platform executable and lowering alone does not execute a model.

MLIR parsing may normalize known printer differences or reprint custom assembly
to generic form through the configured toolchain. Parse failures must remain
actionable; an empty inventory is not an acceptable substitute. Check section
free-value/argument capture before compiling a split subgraph, and check explicit
initialization of contraction accumulators when auditing importer changes.
For an unsupported ATen operator, extend the actual importer/decomposition
boundary in model2MLIR and compare outputs with the independent framework reference.

Close a change with one joined capture → accounting → lowering → execution check
on an independent iteration workload, including a refusal case. Evaluate held-out
headline models separately with their exact checkpoints, invocation/session scope,
observed host/accelerator placement and numerical receipts. Historical local runs
do not certify a new capture, checkout or accelerator configuration.
