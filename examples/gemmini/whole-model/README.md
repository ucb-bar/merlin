# Whole-model lowering and inspection

This walkthrough inspects an existing model capture through shared Merlin
lowering. It does **not** yet provide an independently qualified end-to-end
Gemmini deployment. A Gemmini capsule compiler certificate does not establish
whole-model correctness or performance.

Use a linalg-on-tensors MLIR capture produced by the capture workflow, with its
existing weights and manifest kept alongside it. Framework capture and
quantization belong to [model2MLIR](../../../docs/guides/model2mlir.md), not a
second implementation in this example. Configure the required
[LLVM/MLIR tools](../../../docs/guides/llvm_integration.md) before lowering.

For a capture that has the two named sidecars, run:

```sh
merlin lower /absolute/capture/model.mlir \
  --out /configured/out/build/model-lowering/gemmini-inspection-001 \
  --ir-audit both \
  --audit-sidecar /absolute/capture/weights.safetensors \
  --audit-sidecar /absolute/capture/manifest.json
```

Replace paths with actual inputs and your configured output root. Omit a sidecar
flag if that file is not part of the capture; do not invent empty weights.
The destination must be fresh. This command invokes native lowering tools, but
does not execute the resulting program. The output JSON's `audit_index` points
to this invocation's recorded stages, including any available compact views.

The default result is LLVM IR, not an accelerator executable. `--target riscv`
is an additional host code-generation route, not a Gemmini selection. For the
separate accelerator offload flow and its qualification limits, consult the
[whole-model accelerator guide](../../../docs/guides/whole_model_on_accelerator.md)
and select the [OOT support](../target/README.md) explicitly. Historical results
in that guide do not qualify a newly generated compiler or changed support code.

## Inspect weights without bloated text

`both` preserves exact stages and available compact inspection views. Large xDSL
dense attributes in compact views refer to hash-addressed binary tensor payloads;
the audit records their types and shapes. Those payloads are raw xDSL storage,
not safetensors. Compact views are inspection-only and must not be compiled.

Existing external safetensors files are hashed in place, not converted or copied.
Native pass-printer views do not yet export tensor payloads. Generic safetensors
conversion and reconstruction of executable MLIR from compact views remain
unfinished. See the [lowering inspection guide](../../../docs/guides/model_lowering.md)
for exact guarantees and failure behavior.
