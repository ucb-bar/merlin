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
  --out /configured/out/build/model-lowering/model-inspection-001 \
  --textual \
  --ir-audit both \
  --audit-sidecar /absolute/capture/weights.safetensors \
  --audit-sidecar /absolute/capture/weights.safetensors.manifest.json
```

Replace paths with actual inputs and your configured output root. Omit a sidecar
flag if that file is not part of the capture; do not invent empty weights.
The destination must be fresh. This command invokes native lowering tools, but
does not execute the resulting program. The output JSON's `audit_index` points
to this invocation's recorded stages, including any available compact views.
`--textual` selects the whole-model preprocessing route; it does not skip the
native MLIR-to-LLVM passes.

## Follow the lowering stages

Open the printed `audit_index` (a JSON file under a fresh `ir-audit-*` directory).
Its `outcome` says whether lowering completed or failed, and each `stages` row
binds a stage name to its exact SHA-256, byte count and, with `both`, a file to
inspect. A failed invocation retains the stages completed before the failure;
do not infer that later stages or a target binary exist. For example:

```sh
AUDIT=/configured/out/build/model-lowering/model-inspection-001/ir-audit-XXXX
python -m json.tool "$AUDIT/index.json" | less
ls "$AUDIT"/*-input.mlir "$AUDIT"/*-upstream.mlir "$AUDIT"/*-llvm-final.ll
find "$AUDIT/passes" -name '*.mlir' | sort -V | less
```

The exact snapshots follow captured linalg-on-tensors input → preprocessed
upstream MLIR → scheduled upstream MLIR → translated/normalized/final LLVM IR.
`passes/` holds native pass-manager inspection views when the selected tools
emit them; those are diagnostic prints, not executable replacements. In one
local DeepJSCC capture smoke, `outcome` was `completed`, six named exact stages
and 52 native pass views were recorded, and both weight sidecars were
SHA-bound. That validates this inspection route for that capture, **not**
Gemmini offload, numerical equivalence, ResNet-50 or SmolVLA compilation.

To navigate a new audit without guessing its numbered filenames, print the
ordered stage map and then inspect the corresponding files:

```sh
python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); x=json.loads(p.read_text()); print("outcome:", x["outcome"]); [print(s["name"], "->", p.parent / s["file"], s["sha256"][:12]) for s in x["stages"]]' "$AUDIT/index.json"
rg -n 'linalg\.|tensor\.|memref\.|llvm\.' "$AUDIT"/000-input.mlir "$AUDIT"/001-upstream.mlir | less
diff -u "$AUDIT"/001-upstream.mlir "$AUDIT"/002-upstream-scheduled.mlir | less
```

The first command is authoritative for filenames; the numbered paths in the
other commands illustrate the current six-stage route. If a run fails early,
use only stages present in its index. The `input` snapshot shows captured
operations and shapes, `upstream` shows preprocessing, and
`upstream-scheduled` shows the IR handed to native translation. LLVM stages
show the subsequent host lowering. Equal stage hashes mean that particular
boundary did not change the recorded bytes; an empty diff is valid. For
finer-grained inspection, open the
indexed files under `passes/` in pass order and compare adjacent views; a pass
view can be scoped to a nested operation rather than the whole module.

For an accelerator-specific kernel, the separate OOT compiler route can emit
`contract`, `schedule`, `interface`, `target`, `runtime` and command-buffer
artifacts; see the [published compiler smoke](../README.md#use-the-published-compiler-without-merlin).
Those stages must be inspected under the exact compiler/package identity used
for the run. The generic whole-model command above targets the shared host
lowering path and does not silently substitute a Gemmini dialect pass.

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
