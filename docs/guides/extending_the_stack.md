---
title: Extending the compiler stack
kind: guide
status: current
owner: compiler
last_verified: 2026-10-06
related: [phase0_specification, model_lowering, model2mlir, triton_kernels, target_resolution, llvm_integration, simulator_selection]
code_refs:
  - src/merlin/targetgen/software_spec.py
  - src/merlin/targetgen/instruction_semantics.py
  - src/merlin/targetgen/semantic_search/search.py
  - src/merlin/targetgen/quant_recipe.py
  - src/merlin/targetgen/quant_layer_plan.py
  - src/merlin/targetgen/_recipe_quantizer.py
  - src/merlin/targetgen/rtl/circt_introspect.py
  - src/merlin/targetgen/rtl/timing.py
  - src/merlin/xdsl_dialects/schedule.py
  - src/merlin/xdsl_dialects/lowering/pipeline.py
  - src/merlin/xdsl_dialects/lowering/target_lowering.py
  - src/merlin/xdsl_dialects/targets/factory.py
  - src/merlin/xdsl_dialects/lowering/dispatch_program.py
  - src/merlin/runtime/dispatch_runtime.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/codegen.py
---

# Extending the compiler stack

Start with one independently authored workload and one concrete signature: operation,
shape, layout, operand/storage precision, compute and accumulator precision, and result
precision. Carry that same case through capture, lowering and execution before expanding
the support surface. Target-specific semantics, dialects, code generation and drivers
belong in selected out-of-tree (OOT) packages, not branches on target names in Merlin core.

This guide maps the extension boundaries and their existing interfaces. It is not a claim
that every route accepts every model, or that a software declaration proves hardware support.

## Find the boundary

| Area | Start here | Boundary to preserve |
| --- | --- | --- |
| Software specification and RTL extraction | [Phase 0 specification](phase0_specification.md), `examples/<target>/target/` | Authored semantics versus extracted facts; unknown versus verified |
| Framework capture and quantization | [model2MLIR](model2mlir.md), `merlin.targetgen._recipe_quantizer` | PyTorch/TorchAO stay in the capture interpreter; Merlin consumes artifacts |
| Kernel scheduling and target dialects | [Core dialects](../reference/core_dialects.md), `merlin.xdsl_dialects.lowering` | Generic decisions versus target-owned encoding and execution |
| Native code generation | [Model lowering](model_lowering.md), `merlin.llvmlower` | MLIR, LLVM IR, machine code and runtime ABI are distinct stages |
| Dispatch and model composition | [Runtime reference](../reference/runtime.md), `merlin.xdsl_dialects.lowering.dispatch_program`, `merlin.runtime.dispatch_runtime` | Ordered buffer dependencies, placement and transfers must remain explicit |
| Simulator execution and qualification | [Selecting a simulator](simulator_selection.md), selected OOT execution adapter | Availability, source identity and numerical agreement are separate checks |

The [Atlas](../../examples/atlas/README.md) and
[Gemmini](../../examples/gemmini/README.md) examples organize inputs into `target/`,
`phase0/`, `phase1/`, `phase2/` and `whole-model/`. Their `artifacts/` directories explain
navigation; generated payloads belong under the configured output root, not in examples.

## Understand the two lowering routes

```text
PyTorch model + actual inputs + optional operation-scoped quantization
  → exported/decomposed graph → typed linalg/arith MLIR + external tensor payloads
      ├─ admitted kernel/region
      │    → contract → schedule → interface → OOT target dialect
      │    → runtime representation → command buffer / target-owned execution
      └─ generic native computation
           → bufferization → loops/control flow → MLIR LLVM dialect
           → LLVM IR (.ll) → machine object (.o) → runtime/platform link
```

A **model** is the complete computation and its invocation/state interface. A **kernel**
is an outlined callable region; it can contain one operation or several fused operations.
A model may become one kernel, multiple kernels, or a mixture of host and accelerator
regions. An MLIR file can represent either a whole model or a region: its entrypoint,
arguments, results and stage determine which, not its file extension.

The MLIR **LLVM dialect is not assembly**. Translation produces LLVM IR, which LLVM code
generation turns into machine code. Assembly text is another optional output, not a
necessary intermediate file. An object file is not an executable: linking supplies runtime
symbols, drivers, platform libraries and, where needed, startup code and a linker script.
`merlin.llvmlower.codegen.compile_ll` emits objects; `build_host_shared` links a host
shared library for a caller to load. `merlin lower --target riscv` does not automatically
produce or run a complete platform executable. See [LLVM integration](llvm_integration.md)
before considering any LLVM backend changes.

The staged kernel route is intentionally narrower than arbitrary whole-model MLIR.
`lower_module` checks its supported function/block and matmul-family interface shape;
do not force an unsupported graph through it or silently drop operations. A target may
consume command buffers/runtime calls without introducing an LLVM backend at all.

## Inspect semantic selection before writing a target lowering

The diagnostic search path consumes actual model2MLIR/capsule
`linalg-on-tensors` MLIR and the normalized instruction model frozen by Phase 0:

```text
PyTorch capture → typed linalg/arith MLIR → exact scalar/indexing inventory
                                 + Phase 0 OOT instruction semantics
                                 → bounded instruction candidates
                                 → modeled local-memory allocation
                                 → per-region selection/refusal receipt
```

An operator may run this diagnostic outside the Phase 1 agent sandbox on a fresh
output path, using the exact target and frozen model from one Phase 0 run:

```sh
python -m pip install '.[semantic-search]'  # from a Merlin source checkout
merlin-target-tools semantic-search --target "$TARGET" \
  --mlir "$CAPTURE_LINALG_MLIR" \
  --instruction-model "$PHASE0/software/instruction-semantics.json" \
  --out "$RUN/semantic-search.json"
```

The receipt binds the MLIR and model bytes and summarizes candidate/refusal counts
by parsed operation kind. Currently the search accepts static,
pure, exact typed `linalg.generic` bodies; its built-in equivalences are guarded
integer commutation and modular-add reassociation, not arbitrary algebra. Allocation
models declared memory capacity, alignment, value liveness and bank conflicts with
a bounded solver.
An unmatched or timed-out region remains unresolved; it is **not** automatically
host-admitted. A selected row is only a candidate instruction graph with a modeled
allocation. It does not emit target code, prove numerical equivalence, establish
whole-model coverage, or certify the Phase 1 compiler. Target-specific encodings,
transfers, runtime behavior and compiler passes remain OOT and need independent
execution evidence. This initial implementation is a narrow semantic seam for
growing verified rules, not unrestricted algebraic optimization.

When a reviewed experiment selects a Phase 0 evidence bundle, its
`software/instruction-semantics.json` is copied into the sealed corpus release
as an owner-only input. Phase 1 records
`semantic_search_diagnostic.json` beside the host run record from the frozen
public Linalg capsules; it is not served to the agent or used by the grader.
Phase 2 may record `_host_semantic_diagnostics/semantic_search.json` inside a
fresh optimization stage, using its frozen model capsule and matching Phase 0
evidence when that link exists. Missing or unknown instruction semantics remain
explicitly unavailable; neither receipt grants target support or changes timing or
qualification. To make search an agent-visible tool, define and evaluate a
separate experiment treatment rather than changing an existing run in place.

## Phase 0: deterministic derivation, not an agent

Phase 0 must be a deterministic transformation of selected input bytes, explicit policy
and pinned tools. It extracts facts, screens software-visible signatures, derives precision
and transfer obligations, and generates capsules and independent goldens. It does not ask
an agent to invent missing semantics, select favorable tests or repair a target compiler.
Phase 1 is the agentic functional compiler experiment. Phase 2 optimizes and measures an
already functionally qualified compiler using a separate performance cohort.

For a live model workload, capsule generation reads `workloads/<name>/capture.toml`
from the selected model2MLIR source, not another checkout named by the host environment.
A malformed declaration or missing pinned interpreter fails that capsule explicitly.
The generated capsule's `input_provenance.capture_declaration` records the selected
declaration's relative path, byte count and SHA-256 (or records its absence). This
is inspectable input lineage, not proof of a sealed PyTorch runtime or Phase 0
admission; those claims still require a preselected sealed capture and independent
replay through the separate capture-execution attestation gate. A raw materialized
Model2MLIR receipt or a later source-tree digest cannot upgrade an old capture.

Authored inputs still have a role. The SW spec supplies behavior not yet established by
extraction: operation legality, layouts and tails, numerical semantics, ABI ordering,
quantization eligibility and host/accelerator transfer rules. Workload policy supplies
test objectives and selection constraints. Do not duplicate extracted geometry or turn
an unknown into a default. Where the facts establish a hardware form, an operation can name
it with `hardware:` so Phase 0 selection fills its hardware-shaped fields from the selected
facts; narrow a derived value with a reasoned `restrictions` entry rather than re-authoring
it (see [Phase 0 specification](phase0_specification.md)). Validate selected declarations through
`merlin.targetgen.software_spec.load_software_spec`; validation checks structure, not truth.

To improve extraction, audit a small exact RTL cone yourself and compare it with the
existing extractor. Record source identity, the expected structural fact, the observed
result and the missing rule. Implement the rule generically and compare at least one
different design/configuration. The manual audit is development evidence for the tooling,
not an agentic step that production Phase 0 must repeat.

For a provisioned target support package, run the existing extraction entrypoint:

```sh
export MERLIN_TARGET_PATH=/absolute/oot-target-support
python -m merlin.targetgen.rtl.circt_introspect \
  --target "$TARGET" --source-bundle /absolute/rtl-source-bundle.json \
  --out out/artifacts/rtl-audits/example-1/facts.json --validate
```

Use a fresh output destination and the source-bundle format accepted by the existing
source-selection tooling; see the target's Phase 0 example. Inspect agreements,
divergences and unknowns. A successful extraction command is not certification of
every fact or proof that separate elaborations share a source owner. Phase 0 selection
archives the exact `facts.json` bytes separately from effective consumer views.

Reproducibility includes fixed model construction seeds, calibration samples, source and
tool versions, transformation policy and stable selection/order rules. Compare content
identities and membership across repeat runs; destination paths and observation timestamps
are not semantic differences. Keep failures and missing facts explicit. Regeneration
creates new artifacts and newly frozen runs, never edits old receipts to fit changed inputs.
Follow [Generating capsules](generating_capsules.md) for the actual generation/review flow.

## Extend quantization through public TorchAO interfaces

There are four independent questions: does the hardware format exist, is it legal for this
operation/signature, can this framework build express it, and can the selected compiler
lower and execute it? A supported int8 matrix multiply does not imply int8 LSTM or general
normalization. Multiple formats on one accelerator need separate operation-scoped evidence;
formats with the same storage width must not borrow each other's scale or rounding rules.

The existing path is:

```text
selected datapath/readout facts + explicit semantic declarations
  → quant_recipe: formats, scales, zero points, families, unknowns
  → quant_layer_plan: eligible module/weight signatures and refusal reasons
  → capture worker's TorchAO adapter
      ├─ static: public PT2E Quantizer + QuantizationSpec + calibration
      └─ dynamic: public AOBaseConfig-derived configs + FqnToConfig + quantize_
  → quantized frontend graph + scale/layout payloads + lowering evidence
```

`quant_recipe.derive_candidates` inventories format alternatives without claiming that
TorchAO or a compiler implements them. `quant_layer_plan.plan` checks module families and
weight shapes. `_recipe_quantizer.build_quantizer` additionally checks exported operations
and owners for static PT2E annotation. `build_fqn_config` maps rejected dynamic layers to
`None`, using TorchAO's built-in per-module configuration rather than modifying TorchAO.
Functional arithmetic a container owns between its children (a residual add, for example) is
planned per operation; an operation whose owner cannot be established is refused.
Module eligibility is a first filter, not complete proof of arbitrary functional-operator
or fused-region placement.

For a new format, first add its structural vocabulary and derivation rules, including
granularity, packing, ranges, zero points, bias domain and rounding. Then extend the
appropriate adapter using a public `Quantizer` or `AOBaseConfig`/registered-handler
extension point supported by the pinned TorchAO build. The current adapter already uses
those extension points; a generated per-target Python subclass is not required. Do not
patch PyTorch or TorchAO sources, quantize every `Linear` indiscriminately, or substitute
a nearby dtype or weight-only scheme when the intended route fails.

Prove the format on a mixed model containing both eligible and ineligible operations:
inspect the layer plan, actual PT2E annotations, scale axes/values, storage tensors,
remaining float contractions and numerical outputs. Keep target and framework refusals
separate. Preserve original→quantized→prepared source identities through conversion.
For custom formats not expressible by current adapters, report that implementation gap;
an authored format entry is not a working quantizer.

## Extend capture from PyTorch to typed MLIR

Start with [the four independent iteration workloads](../../examples/workloads/README.md):
mixed MLP, residual CNN, causal decoder and multimodal policy. They exercise important
operator patterns but are not extracted headline subgraphs or numerical equivalents of
the held-out TinyLlama, SmolVLA and ResNet50 checkpoints. Do not use held-out captures,
shape frequencies or validation results to derive or tune capsules.

The trace-capable model2MLIR public API provides a small inspection loop in the isolated
capture interpreter:

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

Use the existing capture worker in [the iteration examples](../../examples/workloads/README.md)
to generate a complete artifact bundle with external weights, input/golden data and the
operator catalog. `m2m.convert` alone returns conversion evidence; it does not create that
complete deployment bundle. When an external quantizer mutates the model, capture its
original snapshot first and pass `original_frontend_snapshot`; an already-quantized model
without that snapshot has unknown original coverage, not an inferred original graph.

Add an unsupported operator at the actual frontend/importer/decomposition boundary in
model2MLIR, not by teaching Merlin to guess an ATen name from final MLIR. Inspect typed
SSA edges and source IDs, preserve exact one-to-many/many-to-many correspondence, and
check executable numerical agreement. Names or module paths alone are not lineage proof.
Static captured call counts, MLIR operation counts and runtime invocation counts are
different denominators. Record them separately, including host, accelerator and unresolved
precision/transfer obligations.

The optional [Triton kernel frontend](triton_kernels.md) accepts a declared `@triton.jit`
kernel and lifts its supported TTIR subset into generic computation. It is not an
implemented arbitrary PyTorch→Triton→TTIR→linalg whole-model route. A new model-to-Triton
path must preserve operator semantics, pointer/shape/effect information and source lineage,
then compare with direct capture on the same independent workload. Unsupported TTIR,
cross-program accumulation and effects must fail explicitly.

## Extend a target dialect out of tree

A target support package supplies `contracts/target_contract.yaml` and
`contracts/dialect_plan.yaml`. The plan maps generic interface operations into the
target's operations. The existing `merlin.xdsl_dialects.targets.factory.build_dialect`
generates the common tensor-resident xDSL shape from that data: pack, contraction,
commit and evict over resident-tensor and accumulator types. Do not copy this machinery
for each accelerator name, or use it to claim unrelated target semantics.

When a target needs executable custom dialect behavior, its contract can declare
`plugin.dialect`. The selected OOT module contributes a `TargetSpec` through
`register_dialect_spec`, with its own operation/type classes and optional opcode map.
Inspect `merlin.xdsl_dialects.lowering.target_lowering` for the exact existing contract.
An xDSL prototype is not automatically a native C++ MLIR plugin: native registration,
verification, passes and build dependencies require a separately implemented route.

Add one operation by defining its types/effects/verifier, concrete capability constraints,
lowering semantics and independent execution checks together. Inspect the generic
interface and resulting target MLIR; make invalid shapes/precision/order fail at a named
boundary. Keep ABI bit fields, instruction encodings and device-specific state machines
OOT. Shared additions should describe accelerator classes or general obligations and
work for more than one design without importing its support library.

A manually implemented reference backend can help validate an interface; it must remain
separate from a Phase 1 candidate and private oracle inputs. Selecting a support plugin
does not qualify a generated compiler. Preserve compiler payloads and retain publication
and certification records separately, as described in [Target publishing](../design/target_publishing.md).

## Extend scheduling and delay handling

`merlin.xdsl_dialects.schedule` records selected layout, packing, memory placement,
accumulator lifetime, dispatch grouping, vector strategy and interface decisions.
It does not submit work or encode target commands. Keep accelerator-class abstractions
here; put a particular design's command fields and instruction semantics in OOT support.
Use [target selection](target_resolution.md) rather than introducing a target-name table.

Existing generic timing extraction in `merlin.targetgen.rtl.timing` counts feed-forward
register crossings in HW-dialect dataflow. Feedback/sequenced modules retain unknown
latency and separate partial-depth evidence. Pipeline depth is not automatically an
instruction latency: operand-dependent sequencing, stalls, queueing, handshakes, memory
and hazards require their own facts or qualified measurements. The resource timelines
in `merlin.perf.activity_schedule` consume supplied durations; they are not RTL proof.

A general instruction-delay/LUT derivation and downstream LLVM delay-annotation backend
remain extension work, not a universal existing API. Introduce them at a documented
boundary: exact RTL/configuration → qualified operation/resource timing → scheduling
constraints → target-owned instruction ordering. Define units, source identity, applicable
operands/resources and unknowns before choosing annotation syntax. Verify the consumer
honors annotations and that their identity survives lowering; arbitrary MLIR attributes
may otherwise disappear. A delay relevant to correctness must survive scheduling and
execution, not merely appear in an inspection file. Validate with an independent RTL
trace or hardware observation, including hazards and a changed pipeline configuration.

## Extend dispatch without hiding host work

`outline_dispatches` forms callable kernels; `build_dispatch_program` emits ordered
dispatch/view nodes and typed buffer identities; a call to an explicitly declared external
catalog symbol stays a dispatch node in its original position and is neither an outlined kernel
nor an implementation. `verify_program` checks dependencies.
The Python `dispatch_runtime` executes supported driver glue and compiled host kernels
as a reference route. That is not proof of accelerator offload. Region captures, argument
order and buffer lifetimes must remain explicit; missing captured reads cannot be assumed
away by a memory planner.

Choose one-kernel versus multi-kernel execution by semantics and legal fusion, not by the
model's name. Introduce runtime changes using a small host→accelerator→host composition:
inspect generated dispatches, transfer contracts, ownership/lifetimes, dtype/layout
conversions, ordering/completion and actual outputs. Track requested placement and observed
execution separately. Transfers require executed witnesses; different dtype labels or an
accelerator-looking symbol are not evidence that bytes crossed a boundary.

For a new device, implement its drivers/ABI and execution adapter OOT. Keep shared dispatch
and dependency logic generic. Do not silently execute refused device work on the host and
report it as accelerator coverage. Extend full-model support only after the minimal seam
works, then qualify the complete entrypoint/state interface rather than just one favorable
kernel.

## Inspect and exchange real stage artifacts

Use the existing lowering audit to retain inspectable stage artifacts:

```sh
merlin lower /absolute/capture/linalg.mlir \
  --out out/build/lowering/example-1 --textual --ir-audit both \
  --audit-sidecar /absolute/capture/weights.safetensors \
  --audit-sidecar /absolute/capture/weights.safetensors.manifest.json
```

The destination must be fresh. The returned `audit_index` identifies this invocation's
ordered stages and hashes. `lower_module(..., ir_audit="both", workdir=...)` provides the
staged kernel audit; [Triton](triton_kernels.md) exposes the same flag. Exact snapshots
and available native pass views are distinct: this does not promise a complete module
after every internal pass. See [Model lowering](model_lowering.md) for the full audit contract.

The whole-model reference route has a matching audit:
`merlin.runtime.dispatch_runtime.run_model(..., transform_audit="exact")` (or
`MERLIN_MODEL_TRANSFORM_AUDIT=exact`) retains the exact captured, normalized and outlined
modules with their normalization recipe. The result records the outlined dispatch inventory
and the audit's qualification, which replays normalization and outlining from the captured
bytes and requires identical output; that checks archival integrity and deterministic passes, not value
equivalence. A `compact` audit keeps only hashes and is refused because it cannot be replayed.

Pass the exact executable MLIR, external weights/biases and manifest, input/golden data,
entrypoint/ABI description, producer identity and stage digest together. The capture trace
and raw/effective Phase 0 views establish how the artifact was derived. Confirm required
members and schemas, hashes, tensor types/shapes, ordered arguments/results and declared
dependencies before consuming it. A digest detects changed bytes; it is not authentication
or numerical certification. A stage file without its external payloads is not a complete
handoff.

Keep safetensors sidecars separate from MLIR. Compact inspection views externalize large
xDSL dense storage into typed, hashed raw tensor payloads; those are not safetensors and
must not replace executable IR. Native printer elision does not export tensor bytes.
`--audit-sidecar` binds existing files in place; it does not copy them into a portable
bundle. Do not feed an inspection-only compact view back into a compiler.

## Close an extension with coarse evidence

Use one joined artifact/execution check over the independent workload before growing a
large granular test suite: capture it, inspect coverage and precision, lower it, execute
the relevant route and compare with the independent reference. Include one legal case,
one boundary case and one deliberate unsupported/refusal case. Re-run with pinned inputs
to check deterministic derivation and source/artifact identities. Exercise the installed
package outside its checkout when the extension changes packaging or dependencies.

Then evaluate the held-out headline models at their prescribed evaluation point. Report
complete model scope, checkpoint/input identity, accelerator versus host work, unresolved
operations, numerical agreement and actual hardware/performance evidence independently.
Iteration success, host execution or a sealed corpus does not establish full headline
accelerator compilation. Functional and performance capsule selections remain separate;
performance results must bind the exact functionally qualified compiler and new run inputs.
