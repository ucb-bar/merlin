---
title: model2MLIR frontend
kind: guide
status: current
owner: frontends
last_verified: 2026-10-05
related: [getting_started, extending_the_stack, phase0_specification, model_lowering, reproducibility]
code_refs: [src/merlin/frontends, src/merlin/capture/bundle.py, src/merlin/targetgen/_m2m_capture_worker.py, src/merlin/targetgen/frontend_trace.py, src/merlin/frontends/compile_inputs.py, src/merlin/semantic_compiler/linalg_bridge.py, src/merlin/targetgen/cli.py, packages/merlin-experiments/src/merlin_experiments/phase0/m2m_runtime.py]
---

# model2MLIR frontend

Merlin consumes model2MLIR's typed linalg-on-tensors MLIR and external tensor
payloads. Framework capture and importer/decomposition extensions belong in the
selected model2MLIR environment. Target encoding and drivers belong in OOT support
packages. Parsing a capture or inventorying its matmuls is not proof of complete
model lowering, numerical agreement, or accelerator execution.

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
and a digest manifest. It does not copy or read runtime samples or goldens.
The capture receipt binds the copied member bytes; source closure, target
numerical admission, and model execution remain separate checks. The output
directory must be new, and failure returns a nonzero status.

For a single admitted static rank-two integer contraction, the installed native
selector can parse Linalg directly. Supply a native snapshot built for the selected
target profile, and select the entry function explicitly:

```bash
merlin-targetgen native-select --engine merlin_native \
  --snapshot /absolute/native-snapshot --linalg /absolute/region.mlir \
  --linalg-entry work --mode strict-native --out /absolute/selection.json
```

`native-compile` accepts the same `--linalg` and `--linalg-entry` pair alongside
its required OOT support, ABI, target source and output arguments. Both commands
also accept `--request` for an already typed semantic graph. The Linalg bridge
currently admits signed i8/i32 or i32 contractions with ordered wrapping i32
accumulation, explicit initialization and static rank-two shapes. Other
operations or numerical policies return a nonzero structured status; this
single-region route does not compile a complete capture or model.

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
