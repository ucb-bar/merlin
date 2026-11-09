# Headline model capture and lowering

TinyLlama, SmolVLA and ResNet50 are held-out validation workloads, not inputs to
Phase 0 capsule selection. Their real model loaders live in model2MLIR. Merlin
does not copy those implementations or infer target support from host lowering.
Use a fresh artifact directory for every run.

| Model | Capture scope | What one capture does **not** prove |
| --- | --- | --- |
| TinyLlama | Full checkpoint, one prefill forward (`M2M_SEQ=8`) | KV-cache decode/session correctness, real-corpus quality, accelerator support |
| SmolVLA | Full checkpoint, explicit prefix → recurrent denoise → action-decode programs | Policy quality on a real trajectory, target execution |
| ResNet50 | Full pretrained `IMAGENET1K_V2` checkpoint, one synthetic image | ImageNet accuracy, real-input preprocessing, target execution |

The commands below are compiler checks with seeded synthetic inputs. For
application-quality validation, supply attributed real inputs through each
model2MLIR loader's documented environment and inspect its `paper_ready` and
source fields. Do not turn a truncated, random-initialized, or single-step
diagnostic into a whole-model claim.

## Capture from PyTorch

Choose the model2MLIR checkout and its model-specific interpreter, then run the
Merlin worker from this repository root. Keep checkpoints cached and use offline
mode when checking an existing local selection. The TinyLlama adapter
[tiny_llama_loader.py](tiny_llama_loader.py) delegates model construction to
model2MLIR; it only records the ordinary prefill path's otherwise missing
checkpoint and input scope. It verifies that cached config and weights resolve
to the same revision and that the loaded layer count matches that config.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 M2M_SEQ=8 \
  "$LLAMA_CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/headline_validation/tiny_llama_loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$HEADLINE_ROOT/tinyllama-prefill"

M2M_RESNET_RANDOM=1 M2M_SESSION_STEPS=1 \
  "$RESNET_CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader "$MODEL2MLIR_ROOT/workloads/resnet50_v1_5/loader.py" \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$HEADLINE_ROOT/resnet50-synthetic"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  M2M_SMOLVLA_PRETRAINED=1 M2M_SMOLVLA_SESSION=e2e \
  "$SMOLVLA_CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader "$MODEL2MLIR_ROOT/workloads/smolvla/loader.py" \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out "$HEADLINE_ROOT/smolvla-session"
```

Do not set `M2M_RESNET_PRETRAINED=0` for a full-checkpoint check. ResNet's
`M2M_RESNET_RANDOM=1` above selects synthetic *inputs*, not random weights.
SmolVLA's `--dtype fp32` selects an unquantized session, not an all-f32 tensor
ABI: the captured KV cache has bf16 leaves. Inspect `meta.json` per stage.

Before lowering, require `ok: true`, `opaque: 0`, the intended checkpoint and
scope, and `capture_receipt.json` in every selected bundle. The SmolVLA root
`session-receipt.json` and `session_contract.yaml` must name all three programs
and the prefix/cache/flow bindings. The receipt's
`source_closure_verified: false` is a blocking fact for verified release, not
a field to edit. A capture may still be useful for diagnostic compiler checks.
Inspect the whole declared session with the selected Merlin compiler interpreter
before treating independently lowered stages as one program:

```sh
PYTHONPATH=src:packages/merlin-experiments/src "$MERLIN_COMPILER_PYTHON" - "$CAPTURE" <<'PY'
import sys
from pathlib import Path
from merlin.experiments.model_qualification import inspect_workflow

report = inspect_workflow(Path(sys.argv[1]))
print([(row["name"], row["missing_runtime_inputs"]) for row in report["programs"]])
print(report["session_abi"], report["session_capture"])
PY
```

For TinyLlama and SmolVLA, `session_abi.status` of `verified_structure` checks the
ordered stage roster and typed cross-stage MLIR bindings; `session_capture.status`
of `producer_receipts_bound` checks the root receipt against the selected stage
receipts. ResNet50's version-1 single-program session has no cross-stage ABI.
These checks do not verify source closure or numerical execution.
Inspect each stage's
`frontend-trace.json` separately: `ok: true` and zero opaque calls do not imply
complete PyTorch-to-MLIR operation correspondence. A `diagnostic` trace leaves
that lineage obligation open even if later LLVM lowering succeeds.

## Inspect target demand without admitting the held-out models

For each captured program, inspect the groups the selected target would route to
an accelerator. This writes a small derivation report without creating goldens,
running a compiler, or adding validation shapes to the Phase 0 corpus:

```sh
merlin experiment corpus groups \
  --target "$TARGET" --definition "$DEFINITION" \
  --rtl-facts "$FACTS" \
  --capture "$CAPTURE/model.mlir" \
  --manifest "$CAPTURE/weights.safetensors.manifest.json" \
  --model "$MODEL" --out "$REPORT" --plan-only
```

Inspect `group_capsules.json`: `accelerator_groups` is the routed denominator;
`stated` and `unstated` say whether those groups can be expressed in the shared
capsule vocabulary; `entries` names each distinct program and its source groups.
Also inspect `routing_summary`: `host_groups`, `elements_on_host`,
`growth_stopped_by`, `host_by_gap_class`, `readout_underived`, and
`device_groups_requiring_input_format_change` show work that
the group count alone conceals. A stated group is a possible target program,
not evidence that its source operations were legally transformed, emitted, or
executed on the device; zero unstated groups does not establish complete offload.
`inputs` records SHA-256 of the capture, manifest, experiment definition,
selected software spec, capability contract, OOT provider contract and RTL facts actually
read. The command refuses absent or source-unverified facts before routing any groups;
`--rtl-facts` pins the same selected artifact used for Phase 0. This checks the
artifact's recorded source-consistency status, not independent source qualification.
`missing_input_receipts` exposes any
selection that could not be byte-bound. `materialization: not_requested` means
none of these entries has yet been built or graded. A complete plan is neither
capsule conformance nor whole-model numerical validation. Omit `--plan-only` to
materialize diagnostic capsules under a separate output directory; never use
`--promote` with the held-out validation models. For an integer contraction,
inspect `integer_partial_sum_bound` in the built capsule: an `unknown` or
`may_overflow` mathematical golden does not qualify the selected internal MAC
width. The bound itself is not full-kernel execution evidence.

## Lower and inspect every program

`merlin lower` consumes one complete `model.mlir` at a time. Run it with the
selected compiler interpreter and MLIR toolchain configured as in
[the lowering guide](../../../docs/guides/model_lowering.md). For TinyLlama and
ResNet50, `CAPTURE` is their bundle. For SmolVLA, repeat once each for
`stages/prefix_encode`, `stages/flow_denoise`, and `stages/action_decode`:

```sh
merlin lower "$CAPTURE/model.mlir" --out "$LOWER_ROOT/$PROGRAM" \
  --ir-audit exact \
  --audit-sidecar "$CAPTURE/weights.safetensors" \
  --audit-sidecar "$CAPTURE/weights.safetensors.manifest.json"
```

`model.mlir` retains typed operation and provenance attributes; its weights
and biases are in the parallel safetensors file. The lowering result's
`audit_index` points to named intermediate MLIR and LLVM IR stages under the
fresh output directory. Inspect the index and terminal `model.ll` for **each**
program. Successful host LLVM lowering proves neither OOT accelerator codegen
nor numerical execution. Phase 1 must still account for host/accelerator
placement, precision, complete sessions and independent numerical results.

## Current diagnostic evidence

The checks below used selected local model2MLIR and compiler builds, full model
checkpoints, and seeded synthetic inputs on 2026-09-28. Their capture receipts
still say `source_closure_verified: false`,
so none of these bundles is an admitted Phase 0 corpus or a target/compiler
certificate. Keep producer trace status, Merlin's exact-byte join, LLVM lowering,
and numerical execution as separate checks.

| Capture | Frontend correspondence | LLVM lowering observed |
| --- | --- | --- |
| ResNet50, full `IMAGENET1K_V2` checkpoint, one image | 0 opaque; producer trace and exact-byte Merlin join complete | Host LLVM IR from the selected diagnostic capture; no target execution claim |
| TinyLlama, full 22-layer prefill/decode session | Both programs: 0 opaque, complete producer traces, complete exact-byte Merlin joins and verified materialized receipts | Both programs reached LLVM IR through `merlin lower`; each compact audit completed 10 named stages and bound the weights and manifest sidecars |
| SmolVLA, full prefix/flow/action session | All three programs: 0 opaque, complete producer traces, complete exact-byte Merlin joins and verified materialized receipts | All three reached LLVM IR from the current captured bytes; each compact audit completed 10 stages and bound both sidecars. Prefix retains three runtime shape assertions. |

A separate ResNet50 W8A8 diagnostic capture selected the SW spec's independent
integer reference. All 54 PT2E-selected contractions were integerized (53
convolutions and one linear); its reference comparison was exact, while the
portable Q/DQ comparison differed. Its frontend trace was complete. Against
the selected Gemmini contract and RTL facts, the held-out capture stated all
54 accelerator groups and materialized their 21 distinct capsule programs
without generator refusals. This tests vocabulary and capsule generation, not
corpus admission: the capture's source closure is still unverified, and no
whole-model OOT command buffer or numerical execution is claimed. A bounded
synthetic i8 window derived from one source-identified ResNet contraction
matched scalar arithmetic on Gemmini Spike; that check does not use the
model's actual operand values or prove RTL-simulator execution.

A separate full-checkpoint TinyLlama W8A8 prefill diagnostic selected and
integerized all 155 linear contractions. Its fresh original-to-quantized and
quantized-to-prepared traces have no unresolved call sites, and its output
matched an independent PT2E integer reference exactly on the seeded eight-token
input. A Gemmini demand plan states all 155 accelerator groups as five distinct
candidate programs, with no unstated group. All five programs were materialized
without generator refusal; a repeat generation produced identical capsule files
(the summary manifest differs only in its output paths). The generated programs
have not been executed against the selected target. The capture still records
unverified source closure,
and the portable Q/DQ diagnostic differs from the selected integer reference.
Neither the trace nor the demand plan admits a Phase 0 corpus or certifies a
Phase 1 compiler.

For a bounded device check, prepared node `g:prepared:root:n283` in that
TinyLlama capture is an i8×i8→i32 contraction with source geometry 8×2048×256.
The Gemmini probe derived an 8×32×16 window from the selected mesh, compiled one
ELF, and matched independent scalar arithmetic on both Spike and Verilator.
Its operands are synthetic; this does not execute the model's operand values,
the full contraction, or the complete frontend-to-device route. The generated
probe and numerical receipt live under the local `out/artifacts/probes/` tree.

The SmolVLA `flow_denoise` gate-projection window also matched scalar arithmetic
on Spike and Verilator, but its captured body is BF16 and the Gemmini probe
projects it to synthetic i8. It is a source-identified geometry check, not a
numerical check of SmolVLA's captured dtype or operands. A full int8 session
needs one consistent shared-weight quantization policy across all three
programs, plus a fresh complete session receipt. The separate one-step int8
diagnostic cannot fill that gap; the existence of a deferred shared-weight
capture path in model2MLIR does not itself qualify the session.

The earlier TinyLlama lowering proof covers captured prefill and recurrent
decode programs, not compiled host or accelerator numerical execution. The
W8A8 reference check above is PyTorch-side only. The SmolVLA prefix guard branches
reached LLVM IR, but their runtime behavior has not been qualified. A complete
frontend trace proves operation correspondence, not host/device placement,
supported precision, target code generation or model accuracy.

In a derived `coverage/operation-accounting.json`, inspect each application's
`completeness.source_trace.transition_obligations`. It lists the exact uncovered
call-site IDs, operators and observed input/result dtypes, and checks the
producer's unresolved-ID roster against the relation edges. Any uncovered row is
a diagnostic obligation, not an elimination or equivalence proof.

Replay the graph-correspondence check on any selected bundle (or one SmolVLA
stage) without recapturing or changing its evidence:

```sh
PYTHONPATH="$MODEL2MLIR_ROOT" "$CAPTURE_PYTHON" - "$CAPTURE/frontend-trace.json" <<'PY'
import json
import sys
from m2m.capture.trace import graph_relation

with open(sys.argv[1], encoding="utf-8") as stream:
    trace = json.load(stream)
observed = graph_relation(trace["graphs"]["quantized"], trace["graphs"]["prepared"])
recorded = next(item for item in trace["transformations"] if item["from_stage"] == "quantized")
assert observed == recorded, "selected model2MLIR implementation differs from the captured relation"
print(observed["status"], len(observed["unresolved_source_ids"]), len(observed["unresolved_destination_ids"]))
PY
```
