---
title: Inspecting whole-model MLIR lowering
kind: guide
status: current
owner: ir
last_verified: 2026-09-23
related: [llvm_toolchain, llvm_integration, triton_kernels]
code_refs:
  - src/merlin/llvmlower/cli.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/common/ir_audit.py
  - src/merlin/xdsl_dialects/ir_inspection.py
---

# Inspecting whole-model MLIR lowering

`merlin lower` accepts an existing linalg-on-tensors MLIR file and calls Merlin's
shared whole-model lowering API. It does not capture a model, select an accelerator,
launch an agent, or execute the result. Unlike the broader `merlin compile` workflow,
it needs no workload catalog or deployment configuration.

```sh
merlin lower /path/to/capture/model.mlir \
  --out out/build/model-lowering/inspection-001 \
  --ir-audit both \
  --audit-sidecar /path/to/capture/weights.safetensors \
  --audit-sidecar /path/to/capture/manifest.json
```

The output directory must not exist, including as a symlink. Use a fresh directory
for every invocation; failures preserve any completed intermediate evidence.
Successful output is JSON with `ll_path`, optional native outputs, lowering statistics,
and `audit_index`: the exact index for this invocation, not a guessed latest directory.
Configure the upstream MLIR toolchain as described in [LLVM integration](llvm_integration.md).

By default, lowering stops at LLVM IR. `--target host` additionally compiles a host
shared library; `--target riscv` additionally compiles the existing RISC-V object route.
Repeat the option for both. These are code-generation routes, not accelerator support
or hardware qualification. Neither output is run. `--textual` selects the existing
text-only preprocessing route; `--feature NAME` selects registered lowering features.

## Choosing inspection detail

- `--ir-audit` or `--ir-audit exact` retains exact named-stage IR, which may be large.
- `--ir-audit compact` records stage identities and available compact inspection views.
- `--ir-audit both` retains exact snapshots and available compact views.

Compact views are labeled **inspection-only**, bind their exact parent hash, and must
not be compiled. Some text-only stages have no compact printer; their index says so.
For xDSL stages, dense attributes above 64 elements are represented by references
to separate `tensors/<sha256>.bin` files. The stage's `inspection.tensors` list records
the MLIR element type, shape, byte count and hash. These are exact xDSL dense storage
bytes, not safetensors or numeric conversions; identical bytes share one file across
stages in an audit. Completion rechecks tensor payloads and retained named-stage exact
and compact files against their recorded hashes and sizes; missing or changed files
refuse completion. Native pass-printer views
still elide constants without this payload export.
Compact means readable text, not a disk-size guarantee: raw storage for expanded
splat tensors can exceed their short MLIR spelling. `exact` does not export these blobs.
The audit records completed preprocessing stages and upstream pass evidence available
from the existing pipeline. It is not a promise of one complete module per upstream pass.
Without the flag, the lowering API's normal intermediate outputs still exist but no
audit index is created. The Python API returns the same `LowerResult.audit_index` field.

## External weights and biases

Existing safetensors and argument manifests stay separate from MLIR. Repeated
`--audit-sidecar` flags commit their observed hashes and recheck them on successful
completion; missing files refuse the invocation. This requires `--ir-audit`.
Declared external sidecars are not copied or converted. Embedded xDSL dense constants
are separately available through the inspection payloads described above, without
changing executable MLIR or its ABI. Restoring executable IR from compact views, generic
safetensors conversion, and native-stage tensor export remain separate work. An audit
is not a compiler certificate or atomic input snapshot.

## Reading recorded tensor payloads

`merlin.common.ir_audit.read_tensor_payload(audit_directory, descriptor)` reopens
one entry from a stage's `inspection.tensors` list and returns its exact bytes.
It refuses malformed descriptors, linked/non-file payloads and size/hash mismatches.
The descriptor must come from an index you trust; a hash is not authentication.

To reconstruct an xDSL dense attribute, call
`merlin.xdsl_dialects.ir_inspection.reconstruct_tensor(audit_directory, descriptor,
tensor_type=expected_type)`. Supply the expected static, unencoded `TensorType`;
its shape and element type must match the descriptor. Supported elements are
8/16/32/64-bit integers of any signedness and f16, bf16, f32 and f64. Reconstruction
preserves storage bytes, including NaN payloads and signed zero, without numeric
conversion. Other types and inconsistent storage lengths are refused.

This reconstructs a tensor attribute, not a whole executable module. Descriptors
do not retain the original tensor/vector container or encoding, and compact views
remain inspection-only. Raw xDSL storage is not a portable safetensors encoding.
The reader verifies the payload's recorded byte count and SHA-256 before
reconstruction, while the caller supplies the expected tensor type; neither
check authenticates an untrusted audit index.
