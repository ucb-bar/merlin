# Inspect and check the native host route

This example checks a saved standalone capture through Merlin's existing generic
MLIR→LLVM→host shared-library route. It is useful for the independent iteration
workloads used by [Gemmini](../../compiler-generation/gemmini/test-generation/README.md) and
[Atlas](../../compiler-generation/atlas/test-generation/README.md), without claiming accelerator offload or changing
their historical RVV deployment packages.

Use trusted, locally produced captures. The process has resource limits, but is
not a security sandbox for arbitrary malicious MLIR or native code. No framework
loader is imported or recaptured, and TorchAO/PyTorch sources are not modified.

## Select actual generated inputs

The test generation requirements producer retains complete iteration captures below
`materialized/<workload>/`: `coverage_mlp`, `residual_cnn`, `causal_decoder`, and
`multimodal_policy`. Each directory contains its exact `model.mlir`, weights and
argument manifest, inputs, eager reference, frontend trace and capture receipt.
Do not substitute held-out headline models or infer input identity from a name.

Install Merlin and `merlin-experiments` with their lowering dependencies, and
select the already provisioned compiler tools explicitly:

```sh
export MERLIN_COMPILER_PYTHON=/absolute/compiler-environment/bin/python
export MERLIN_CLANG=/absolute/llvm/bin/clang-23
export MERLIN_MLIR_TRANSLATE=/absolute/llvm/bin/mlir-translate
```

Choose a generated capture directory and a fresh output path:

```sh
python -m merlin.experiments.model_qualification \
  --native-host-only \
  --bundle /absolute/generated-phase0/materialized/coverage_mlp \
  --out out/artifacts/native-host/coverage-mlp-1 \
  --atol 0.00001 --rtol 0.0001
```

Repeat with the other independent iteration captures. Tolerances are explicit
inputs, not automatically relaxed after a mismatch. This mode requires one
standalone FP32 result with a complete saved reference; recurrent sessions and
other result precisions are separate interfaces. It does not take `--package`,
`--target`, `--execute`, or target evidence.
The FP32 result requirement does not imply float-only computation: saved programs
may contain integer contractions, quantized i8 weights and mixed host precisions.
The qualifier checks the recorded argument ABI and compares the complete result,
without recapturing or silently changing the quantization scheme.

## Read the automatically generated artifacts

- `request.json`: byte identities of every captured input/reference and selected
  resource bounds.
- `program-000/native/`: LLVM IR, object, host shared library, and exact/compact
  intermediate lowering views. Follow its `ir-audit-*/index.json` via the recorded
  path rather than guessing the newest directory.
- `program-000/output.npy`: actual complete native output, not copied eager data.
- `observations.json`: exact tensor ABI, immutable input-buffer checks, explicit
  tolerance, maximum error, and numerical result.
- `qualification.json`: compiler payload, upstream MLIR and executable tool byte
  identities, saved input identities, and output identities. Original source
  closure retains its recorded status; materialization is not source certification.
- `README.md`: invocation-specific navigation and scope.

A successful finite check records `native_host_numerical_verified: true` and
`target_executed: false`. It does not certify individual operations independently,
all shapes/precisions, an accelerator, RVV deployment, or application accuracy.
Outputs and receipts are read-only; regeneration needs a fresh output directory.

## Freeze a native check alongside test generation

Add a generated receipt to an existing `merlin experiment corpus derive` invocation
using the same label as its exact capture selection:

```sh
--application-capture coverage_mlp=/absolute/capture/model.mlir \
--native-qualification coverage_mlp=/absolute/native-check/qualification.json
```

These are additional arguments, not a standalone command: retain the experiment
definition, extracted RTL facts, full declared capture roster and fresh output root.
Receipt selection is optional and is not another authored SW-spec requirement.
The producer refuses changed model, input, weight, reference, output or ABI bytes.
A new capture needs a new matching native check, even if its workload name is unchanged.

Inspect `evidence/software/native-baseline-observations.json` for the exact input
and result precisions, numerical policy, compiler identity and measured error.
Its `frozen_artifact_root` points to readable copies of the original qualification,
request, observations, lowering audits and actual output. The per-application
operation-accounting report retains this as a separate `precision_execution`
observation; frozen inspection does not reopen the original capture or compiler.
It does not upgrade RVV board support, accelerator admission, or independent
operation/numerical qualification.

## Keep native verification and board deployment distinct

Select this explicit mode for local native verification. Existing Gemmini/Atlas
descriptor `host_lane` profiles select immutable RVV schedule packages for board
deployment, not this native verifier. The current `HostLane` protocol loads real
RVV manifests/schedules/knobs; no fake schedule package is minted to make a native
receipt look like RVV support. Preserve the historical packages unchanged.

Before admitting a broader host capability, independently qualify its exact typed
operation signatures and numerical semantics, then select its compatible runtime
and deployment ABI. Whole-model agreement alone is not that evidence.

See [model lowering](../../../docs/guides/06-lowering.md) for stage inspection and
[the compiler stack](../../../docs/guides/12-experiments.md) for ownership boundaries.
