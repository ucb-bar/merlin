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

## Check model readiness

For an explicit model2MLIR capture directory, first compare a *requested*
deployment format with what the captured graph actually contains:

```sh
merlin-compile --target gemmini --model-preflight \
  --capture-bundle /absolute/capture-directory \
  --deployment-dtype int8 --json
```

The directory must contain `model.mlir`. The report also checks for
`weights.safetensors`, its manifest, `inputs.npz`, `input_order.json`, and
`golden.npy`. Read `contractions.captured_operand_dtypes`,
`contractions.captured_accelerator_groups`, `contractions.capsule_form_groups`,
and `blockers` together. The command exits nonzero when blocked. Even when
static checks pass, `target_binary_emitted` remains false: this is an inventory,
not a whole-model compiler or correctness test.

The available ResNet-50 and SmolVLA denoise-step captures expose the current
gap. Requested `int8` routing identifies Gemmini-capable contractions, but
the standard captures retain FP32/BF16 operands; the W8A8 ResNet capture
still has 53 FP32 contractions. No target-native quantization bridge or
general whole-model Gemmini binary is established by those routes. A separate
historical ResNet program from an alternate capture is useful evidence for a
tracer bullet, not certification of this generated Phase 1 compiler.

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
