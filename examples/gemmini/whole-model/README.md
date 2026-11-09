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
some standard captures retain FP32/BF16 operands. A separately integerized
W8A8 ResNet capture has exact `i8 × i8 → i32` candidates, but isolated kernel
emission does not establish reviewed SW admission, host/device stitching or
whole-model execution. No general whole-model Gemmini binary is established by
those routes. A separate historical ResNet program from an alternate capture
is useful evidence for a tracer bullet, not certification of this generated
Phase 1 compiler.

For an already-integerized capture, you can materialize the exact signed
`i8 × i8 → i32` contraction kernels as standalone `merlin_iface` inputs:

```sh
merlin-target-tools outline-int-mm --target gemmini \
  --mlir /absolute/capture/model.mlir \
  --software-spec /absolute/phase0-artifacts/software/software-spec.json \
  --capability-contract /absolute/phase0-artifacts/software/contract.json \
  --out /configured/out/artifacts/model-kernels/int8-iteration-001
```

The fresh directory contains `manifest.json` plus one interface MLIR file per
outlined contraction. Both operator-selected Phase 0 files are required: the SW spec must
declare accelerator contraction with a matching precision/rank, and the capability
contract must explicitly declare the resident-packed, accumulator-commit and
command-buffer class.
Missing, foreign or malformed selections refuse; an unsupported declaration emits
no interface. The manifest binds the exact model, SW spec and capability-contract
byte hashes, MLIR operation and operand-producing SSA values. Its `stitching`
inventory lists ordered source operations, function return bindings, and typed
SSA crossings into or out of each contraction. These are *transfer requirements*,
not generated DMA or a working dispatch. For the integerized `coverage_mlp`
capture, both matmuls consume host-produced quantization/transpose results and
feed further host math. Each crossing keeps the byte-bound source value and
consumer identity so later placement and pointer order need not be guessed from
tensor names. The inventory remains `diagnostic_unexecutable`. Each candidate's
`software_admission` can still be `unknown` (the current unreviewed example is),
and `compiler_support` remains `not_evaluated`; this is an inspectable diagnostic,
not a Phase 0 provenance certificate or a claim that the target compiler supports the operation. Submit an interface
to the selected OOT compiler to check kernel code generation separately. This does
not lower the intervening quantization, transpose, dequantization or host operations, connect the kernels back to the
model, or establish numerical execution. The published Gemmini compiler still
declines the *whole* upstream Linalg module; an isolated kernel command buffer
does not change that verdict.
To record what the selected OOT package emits for every candidate in the
captured model, without hand-writing scripts in `out/`, run:

```sh
merlin-target-tools probe-int-mm-route --target gemmini \
  --mlir /absolute/capture/model.mlir \
  --software-spec /absolute/phase0-artifacts/software/software-spec.json \
  --capability-contract /absolute/phase0-artifacts/software/contract.json \
  --package /absolute/selected-oot-package \
  --out /configured/out/artifacts/model-kernels/route-probe.json
```

The fresh receipt binds model/spec/contract/package bytes, source SSA operands,
the OOT command buffer for each distinct interface, explicit declines, and a
separate direct whole-model emission observation. Read `emission_counts`,
`complete_model_direct_emission`, `stitching.obligations` and
`whole_model_offload_verified` together. Successful isolated kernels do not
imply that their inputs and outputs are connected back into a running model.

The outline's focused test numerically checks its isolated signed `i8×i8→i32`
interface on a non-square K-tail against scalar arithmetic. It does not execute
the OOT compiler output, connect the host operations, or compare a model golden.

An opt-in whole-model handoff now exists for *qualified* outlines. Its identity
is the normalized MLIR file produced by `prepare_for_lowering`, not an earlier
raw capture that preparation may rewrite. `ExactOffloadSelection.from_outline`
re-derives the candidate IDs and interfaces from that file and the selected SW
spec/contract bytes; it refuses the current `unknown` SW admission. Before
certification, the experiments owner must call `bind_exact_offload` with the
reviewed Phase 0 release seal, selected descriptor, and application label. The
binding checks that the exact model and both contract byte strings are selected
sources of that release, and is reopened at certification, rewrite, and build.
If preparation changes the raw capture, the normalized model needs its own
selected capture receipt; a matching outline alone cannot confer admission.
The `certify` step runs the selected interface through the OOT numerical oracle
with an accelerator trace, and only then may `DeviceRouting(exact_selection=...)`
replace those exact operations. The rewrite and object build recheck model,
package, transport, pointer ABI, release lineage, and interface identities;
unselected operations stay on the host path. This is a trusted-host evidence
gate, not a sandbox against arbitrary Python in the host process, and is not a
full-model numerical certificate. The present example has neither reviewed
admission nor a demonstrated OOT whole-model execution, so its outline remains
diagnostic.

For a new numerical accelerator certificate, the compiler must emit
`compiler_pointer_abi: {version: 1, arguments: [...]}` in its command buffer.
Merlin compares that asserted pointer order with the selected target runner's
`kernel_abi_from_commands` order before execution. A missing or mismatched
declaration stops qualification; older compiler packages remain inspectable.
Matching declarations are only a structural prerequisite: the emitted kernel
must still execute and match the independent numerical reference.

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

## Evaluate a generated compiler without tuning on validation models

Keep full TinyLlama, SmolVLA and ResNet-50 validation captures separate from
Phase 0 iteration inputs and Phase 2 tuning. A truncated model, one denoising
step or synthetic input remains a diagnostic, not full application validation.
The experiments package provides a deterministic, offline observation command:

```sh
python -m merlin.experiments.model_qualification \
  --bundle /absolute/held-out-capture \
  --package /absolute/generated-compiler-package \
  --target gemmini \
  --evidence-bundle /absolute/saved-phase0-evidence \
  --lower-native \
  --out /configured/out/artifacts/model-qualification/inspection-001
```

The package must actually be the generated compiler to evaluate, not a reference
compiler renamed as an experiment result. The fresh output is generated by the
command: no hand-written scripts are needed in `out/`. `qualification.json`
binds the compiler tree, model, weights, inputs, goldens, session roster and
selected target evidence to exact bytes. `program-*/` contains compiler output
and diagnostics; `operation-accounting.json`, when evidence is supplied,
partitions observed operations without feeding them into capsule generation.

`--lower-native` additionally lowers every complete declared program through the
shared pipeline to LLVM IR. It requires verified materialized capture receipts,
binds weight sidecars and records exact/compact stages in `program-*/native/`.
Read `native_lowerings` and `full_native_lowering_verified` separately from the
OOT compiler observations: native lowering can pass while accelerator command
generation declines. The command generates a `README.md` linking its audit indexes.

Preserve the complete workload roster rather than substituting a diagnostic stage:

| Workload | Declared programs | What remains explicit |
| --- | --- | --- |
| ResNet50 | Classifier forward plus the image stream | Checkpoint identity, image count and preprocessing |
| TinyLlama | Prefill and recurrent decode | Full decoder, fixed cache capacity, token stream and decode count |
| SmolVLA | Prefix encode, recurrent flow denoise and action decode | Full policy configuration, conditioning/cache bindings and denoising count |

The capture worker's `--materialize-bundle` mode preserves explicit deferred
multi-program sessions from model2MLIR, including carried-state ABI bindings and
per-program original/transformed/prepared traces. It currently accepts these
sessions without requested precision conversion or quantization; it does not
silently cast shared parameters or turn native FP8 into FP32. Session formats,
quantization and accelerator execution are independent qualification questions.
Parameter-free stages receive a valid empty safetensors container from the bundle
writer, not fabricated tensors. BF16 lifted constants are stored losslessly in
parallel payloads rather than omitted from the runtime ABI.

Read `compiler_observations` beyond the exit code: `emitted_ir` records typed
operation/dialect counts and structural verification, `canonical_ir_changed`
detects a pass-through graph, and `command_buffer` records empty commands and
explicit declines. An `emit_command_buffer` process that exits successfully but
declines the model is recorded as `status: declined`; unchanged target lowering
is `unchanged`, and an LLVM artifact emitted without an accepted command buffer
is `unqualified`. `model_routes` joins those observations to each program's
exact capture hash, selected operation ledger (when supplied), and native-host
LLVM result. It names missing source closure, placements, typed transfers and
execution evidence without treating host LLVM lowering as accelerator offload.
A shared grammar cannot verify the semantics of unregistered OOT operations.
`status: completed` means observation completed, not that a compiler or model
passed. `whole_workload_validation_verified` remains false.

`--execute` additionally attempts the existing single-forward mesh dispatch
runtime, using the explicitly saved contract/facts. Inspect `target_executed`,
numerical results, dispatch decisions and host fallbacks together. This is not
native whole-model executable certification or application accuracy. A staged
multi-program session is inventoried and each program is checked, but that
runtime does not execute its stage bindings or recurrence; it reports the gap.
The evaluator checks that mesh certification starts from the requested compiler
package, then permits the certifier's byte-checked private build copy to run.
The request records the selected Chipyard path and explicit mesh-engine policy;
the selected OOT support provider is hashed before and after the run. Worker
temporary and shape-keyed certification runs stay inside the generated
qualification directory; each evaluation gets its own output root. A wall-budget
exhaustion after a kernel ELF is built is still incomplete execution evidence;
neither a host fallback nor an earlier successful compiler command upgrades it.
For an FP32 capture routed to the integer mesh, this diagnostic runtime applies
per-tensor symmetric boundary quantization and rescales each integer result. That
is not a TorchAO-derived model quantization recipe. A successful Spike kernel
trace can therefore coexist with a failed whole-model comparison; inspect the
actual error and do not widen tolerances merely to make it pass.
For an integer iteration check, first recapture the small
[`coverage_mlp`](../phase0/README.md#4-derive-requirements-and-candidate-capsules)
with the generated Phase 0 recipe and `--materialize-bundle`. Confirm
`meta.json` records integerized contractions, calibrated quantization parameters,
and which layers stayed on the host, then evaluate that *same* captured graph and
golden here. A matching host result does not establish accelerator execution:
read `mesh_ran`, `mesh_fell_back`, `mesh_unavailable`, and `target_executed` together.
A worker that denies bubblewrap's NETLINK setup cannot reach the compiler or
simulator; Merlin reports those layers as unmeasured, not backend fallbacks.
Full validation also needs original-source tracing, the full pretrained
checkpoint, attributed real inputs and the correct complete application session.
The evaluator does not download weights, derive capsules or change compiler code.
Local process-group/PID cleanup is bounded; arbitrary worker-loss cleanup still
requires the managed-worker deployment described by the execution workflow.

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
