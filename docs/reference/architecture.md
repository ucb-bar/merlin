---
title: Architecture
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [repo_structure, core_dialects, lowering_pipeline]
code_refs: [src/merlin, packages, experiments/catalog.yaml]
---

# Architecture

Merlin derives capsules and develops target compilers through shared authoring,
evaluation and optimization infrastructure. The joint experiment combines
correctness and performance development in one evolving compiler using the Phase 1
engine; separate Phase 1/Phase 2 workflows and module names remain available. Start at
[`experiments/catalog.yaml`](../../experiments/catalog.yaml) through `merlin experiment`.
Shared artifacts (schemas) connect those phases and the supporting research workstreams.
Two compiler planes remain: xDSL for prototyping and MLIR/C++ for stable code.

## Implementation ownership

### Compiler ownership migration

The heterogeneous refactor preserves the current branch's compilation, CCA,
scheduling, validation and development capabilities. Its complete destination
is specified in the architecture Word document. The
[preservation rules](../design/refactor_preservation.md) govern removal and
consumer migration; the destination tree is not a claim that all stages have
already moved.

Current implementation entrypoints are:

| Responsibility | Source to open | Current caller |
| --- | --- | --- |
| Scalar computation pass sequence | `src/merlin/compiler/lower/scalar/pipeline.py` | Existing whole-model LLVM pipeline |
| Shared numerical and source preprocessing | `compiler/frontend/preprocess.py`, `repairs.py`, `quantization/linalg.py`, `transforms/matmul.py`, `transforms/dead.py`, `lower/scalar/casts.py`, `ir/provenance.py` | Existing lowering, editable shared pass exports and fresh SDK27; historical preprocessing import remains an exact module alias |
| Shared terminal LLVM conversion and index-width binding | `src/merlin/compiler/lower/llvm/passes.py` | Scalar, vector and OpenMP pipelines |
| Selected OOT region emission and receipt checks | `src/merlin/compiler/lower/regions.py` | Existing device object builder |
| Provider identity and explicit component composition | `src/merlin/compiler/backend/load.py`, `composition.py` | Target provider selection; explicit composition API |
| Legacy interface-to-command-buffer process adapter | `compiler/backend/package.py` | Explicit ordinary `merlin.package` selection; original package process, schema and decline handling |
| Scalar/RVV package loading and ISA admission | `src/merlin/compiler/backend/host.py`, `isa.py` | Host selection, execution inputs, package application and scalar minting |
| Quantized accumulator/readout contracts | `src/merlin/compiler/quantization/contract.py` | Group numerics, readout capabilities and independent numerical planner |
| Captured quantization scope | `src/merlin/compiler/quantization/scope.py` | Shared feature discovery, integer lowering and model preparation; respects the captured arithmetic boundary |
| Explicit application preparation policy | `compiler/quantization/methodology.py`, `preparation.py` | Existing methodology derivation and capture recipe selection; source-bound profiles and exact numerical alternatives |
| Compact semantic composition and inspection | `target/spec/load.py`, `source.py`, `normalize.py` | Opt-in software-spec v2, legacy consumer projection and `merlin inspect --spec` |
| Closed numerical declarations and extensions | `target/spec/quantization.py` | Canonical validation reused by legacy and compact intake; unmodeled execution remains unknown |
| Numerical compatibility admission and recipe projection | `compiler/quantization/admission.py`, `torchao/recipes.py` | Existing evidence/requirements callers retain exact diagnostic schemas and recipe digests; framework-free admission is separate from TorchAO projection and realized quantization |
| Typed computation and source-bound instruction records | `target/spec/computation.py`, `signature.py`, `instructions.py` | Compact expansion, selected instruction search and semantic diagnostics; old targetgen module is an exact alias |
| Native staged pass invocation | `src/merlin/compiler/passes/native.py` | Existing lowering runners; source included in lowering receipts |
| Ordered xDSL rewrite invocation | `src/merlin/compiler/passes/xdsl.py` | Existing numerical preprocessing, with original callbacks and observations |
| Zephyr image memory layout | `src/merlin/compiler/build/zephyr/memory.py` | Existing Zephyr application builder |
| SoC-dependent Zephyr image configuration | `src/merlin/compiler/build/zephyr/config.py` | Existing Zephyr application builder |
| Selected Zephyr tree capabilities | `src/merlin/compiler/build/zephyr/tree.py` | Image configuration; per-build symbol/default discovery |
| Placement candidates, selection and measured cost | `src/merlin/compiler/planning/candidates.py`, `placement.py`, `cost.py` | Prebuild placement and device-routing selection |
| Source-bound group selection and reports | `src/merlin/compiler/planning/groups.py`, `reports.py` | Prebuild routing, offload materialization and compile reports |
| OOT-discovered overlapping fusion alternatives | `compiler/planning/fusion.py`, backend `propose` hook | First signed dense/readout family; selected-contract checking and existing global selection |
| Live SSA inputs, outputs and region captures | `src/merlin/compiler/ir/boundary.py` | Outlining, fusion discovery, dispatch liveness and source-region lowering |
| Explicit matrix implementation offers and lowering | `src/merlin/compiler/backend/proposal.py` | Existing prepared-IR matrix route |
| Shared prepared contraction/object service orchestration | `src/merlin/compiler/lower/prepared.py` | Installed OOT services supply explicit contract/geometry/rewrite/object callbacks; partial IR and objects retain incomplete-image scope |
| Public compile and inspect service | `src/merlin/api.py`, `compiler/request.py`, `compiler/artifact.py` | Python callers and `merlin compile --backend` / `merlin inspect` |
| Source-driven system compilation | `compiler/system.py`, `components.py`, `backend/scalaroffers.py` | The same public Python compile call; discover against post-capture IR, admit proposals, select once, then invoke `PlannedCompiler` |
| Existing prepared-program intake | `compiler/frontend/staged.py` | `api.StagedProgram` and `merlin compile --input-kind=compiler-stage`; closed compiler inputs and actual precision-policy identities reach discovery through original-caller admission |
| Explicit source semantic obligations | `compiler/ir/semantics.py` | Original-caller typed attribute coverage, source binding and revalidation; implementation admission remains separate |
| Shared source and selected callback observations | `compiler/sourceguard.py` and stage-local implementation rosters | Discovery, admission, partition and selected lowering recheck actual source bytes and callbacks; observations do not establish loaded-bytecode or semantic equivalence |
| Saved-export input and compiler-only staging | `compiler/frontend/torch/export.py`, `capture.py`, `frontend/inputs.py`, `signature.py`, `archives.py`, `constants.py` | Saved-export and native application inputs; exact ABI, archive and lifted-argument handling |
| Installed compiler and runtime selection | `src/merlin/compiler/backend/service.py` | Explicit `merlin.backends` and `merlin.runners` entry points |
| Metadata-only installed backend inventory | `api.inspect_backends`, `compiler/backend/service.py` | `merlin inspect --backends`; interpreter, distribution and declared origin, without importing providers |
| Experiment definition, selection and execution | `experiments/definition.py`, `spec.py`, `adapters.py`, `runner.py` | `merlin experiment`; research orchestration stays outside the compiler service |
| Selected topology and actual-source proposal admission | `src/merlin/system/topology.py`, `compiler/backend/sourceproposal.py` | Explicit source-bound heterogeneous planning |
| Bounded region/resource cover and shared-memory budgeting | `src/merlin/compiler/planning/execution.py` | `select_execution`; costs remain unknown without a selected cost model |
| Complete-SSA region partitioning | `src/merlin/compiler/partition/regions.py` | `PlannedCompiler`; original CPU operations remain inline |
| Independent partition fidelity | `src/merlin/verification/partition.py` | Beta-expands actual calls and compares the complete original program; old import is an exact alias |
| Independent translation and proof services | `verification/{refine,receipts,smt_export,lean_checker,lean_cli,lean_contract}.py`, `verification/lean/Merlin/` | Existing SMT translation and live Lean checks; 41 legacy verifier modules are exact compatibility identities; mathematical sources have one authority |
| Outline, scalar IR and split-reduction verification | `src/merlin/verification/outline_ir.py`, `scalar_ir.py`, `split_reduction.py` | Verification CLI and ordinary integer epilogue admission; unchanged independent algorithms, exact legacy aliases |
| Packing lifetime and payload verification | `verification/packing.py`, `packing_ir.py`, `packing_types.py` | Ordinary source packing admission; original per-island budgets and checked LLVM pointer lifetimes |
| Actual host-call alignment | `verification/host_call_alignment.py` | Host packing and borrowed-call admission; exact native import/source observations |
| Saved-model lifecycle | `model.py`, `compiler/build/modeloptions.py`, `compiler/build/saved.py`, `runtime/model.py`, `verification/model.py` | Existing model API and CLI; build, execution and independent checks have real separate owners |
| Saved-model input custody and readiness | `runtime/modelinputs.py`, `compiler/frontend/materialized.py`, `preflight.py` | Saved build/runtime callers and explicit capture readiness; original materializer/source identities remain separate |
| Prebuild source routing and coverage | `compiler/planning/model.py`, `complete.py`, `coverage.py` | Existing ordinary model build, complete-group selection and source-boundary census |
| Original host/device interfaces and source builds | `compiler/partition/model.py`, `compiler/build/owned.py`, `siblings.py`, `spatial.py` | Existing interface publication and original host/source builders around separately selected candidate device code |
| Independent saved-model replay and qualification | `verification/completed_model.py`, `scalar_host.py` | Completed-link source replay and explicitly selected single-capture scalar checks; original acceptance and source-access roles remain unchanged |
| Mesh selection and execution | `compiler/selection/mesh.py`, `runtime/mesh.py`, `mesh_backend.py`, `mesh_model.py` | Existing tile/package selection, selected simulator transports and recursive/model execution |
| Independent mesh certification and reference arithmetic | `verification/mesh.py`, `mesh_reference.py` | Existing tile/route certification and numerical references; runtime compatibility hooks pass actual selected callbacks |
| Selected region compilation and artifact admission | `src/merlin/compiler/pipeline.py`, `invocation.py` | Same public compile service; arbitrary region artifact kinds |
| Exact selected-offer lowering | `compiler/backend/selected.py`, `sourceproposal.py` | Recipe-bearing offers use selected hooks, live source/partition tickets and actual artifact acknowledgments |
| Artifact-bound C call contracts and shared control assembly | `src/merlin/compiler/backend/cabi.py`, `build/control.py` | Explicit `merlin.emit=c-dispatch` on a selected plan |
| Caller-owned stateful C calls | `compiler/backend/stateful.py` | C ABI v2, CPU context/workspace reservations and exact binary embedding; existing context tables retain error state, while initialization and loader execution remain explicit obligations |
| Scalar model workspace and typed image initialization | `compiler/backend/workspace.py`, `lower/llvm/workspace.py`, `build/contextentry.py` | Caller-budgeted scalar destinations, ABI2/ABI3 shared control and explicit caller-approved state initialization; actual device completion remains independently qualified |
| Portable scalar layout implementation | `src/merlin/compiler/backend/layout.py`, `lower/c/layout.py` | Installed `merlin.layout` provider; static cast/collapse/expand only |
| Ordinary scalar compilation service | `compiler/backend/scalar.py` | Installed `merlin.scalar`; existing scalar lowering with explicit layout/index width, LLVM IR or selected-target object output |
| Bounded native copy admission | `compiler/lower/llvm/copies.py`, `call.py` | Actual scalar LLVM call admission checks descriptor extents, access direction, copy bounds and nonoverlap; unknown or poison-producing forms refuse |
| Shared LLVM object command construction | `compiler/build/objects.py` | Scalar service and existing `llvmlower.codegen.compile_ll`; caller retains selected compiler/flags and observed runner |
| Observed build processes and host shared-library construction | `compiler/build/process.py`, `shared.py` | Existing codegen callers retain legacy selection wrappers; actual execution/link algorithms and source pins use these owners |
| Bounded source recomputation transform | `compiler/transforms/recomputation.py` | Canonical lowering caller; old module is an exact alias, independent checker retains verification ownership |
| Source multi-output fusion | `compiler/transforms/multioutput.py` | Existing source lowering, fusion discovery and optimization catalog; independent checker and frozen SDK inventories retain their owners |
| Row-normalization discovery and materialization | `compiler/transforms/row_normalization.py`, `row_normalization_source.py` | Existing automatic source route, exact old module identity and independently checked ordered scalar bodies |
| Portable pointwise and reduction tile geometry and transport | `compiler/transforms/pointwise.py`, `reduction.py` | Canonical C lowering consumers; exact semantic-compiler aliases and source-pinned native snapshots |
| Ordered contraction and streamed panel tiling | `compiler/transforms/contraction_k.py`, `contraction_mn.py`, `contraction_mnk.py`, `panels.py` | Canonical C tiling consumer and retained Atlas preparation imports; ordered algorithms and pickle/source identities preserved |
| Shared LLVM lifetime policy and implementation fragments | `src/merlin/compiler/passes/policy.py`, `fragments.py` | Existing LLVM driver and migrated OOT vector providers |
| Shared scalar host transformation library | `compiler/transforms/host.py`, `host_contraction.py`, `host_reduction.py`, `host_broadcast.py`, `host_integer_math.py` | Ordinary scalar lowering and the separately reviewed source release built by `compiler/passes/hostexport.py`; legacy imports resolve to the actual owners |
| Typed native graph handoff | `compiler/build/hostbridge.py`, `graphbridge.py`, `compiler/ir/graph.py` | Retained regional/whole-graph source correspondence and `native-bridge-graph` boundary publication; the generated boundary remains unlinked until a concrete adapter is supplied |
| Shared instruction feature registry | `src/merlin/compiler/selection/features.py`, `shared.py` | Existing feature registrars and OOT code generation |
| Existing-artifact execution | `src/merlin/runtime/service.py`, `spike.py` | `merlin.api.run` and `merlin run` |
| Native Linux execution | `runtime/linux.py`, `linux_worker.py` | Installed `merlin.linux` runner; exact open ELF, argv, observed initial CPU affinity, exit/streams and owned timeout cleanup |
| External workload preparation and result collection | `src/merlin/runtime/external.py` | Public FireSim prepare mode and `merlin collect` |
| Staged semantic application assembly and artifact replay | `src/merlin/compiler/build/application.py` | Explicit `native-compile --application` and `native-verify --application` |
| Serial C dispatch, application ABI and portable handoff | `src/merlin/compiler/lower/c/dispatch.py`, `application.py`, `handoff.py`, `layout.py` | Selected native application region adapters |
| Bounded application cover and completion-aware memory | `src/merlin/compiler/planning/application.py`, `compiler/schedule/arena.py` | Native application assembly and opt-in `runtime.program.build_program` |

Historical `merlin.compile` model and mesh imports remain exact module aliases to
the implementation owners above. They preserve existing imports, record lookup
and supported patch bindings; current callers and source receipts identify the
actual owners. These ownership changes add no backend capability or execution
qualification. Shared scalar lowering stays in `compiler/lower/scalar`, shared
passes and scheduling in `compiler/passes`, `transforms` and `schedule`, and
provider-specific lowering stays behind the installed OOT entrypoints.

Application compilation chooses its complete plan once and passes that same
selection to region compilation and materialization. Independent verification
chooses once from the source again and compares the exact canonical
`selected-plan.json` before exporting region code for replay. A replacement
implementation with identical region boundaries cannot pass merely by updating
artifact hashes. This retains publication, partition, buffer and C replay checks.

`SystemCompiler` accepts explicit component, resource and memory bindings rather
than a manually constructed selection over a future captured graph. Discovery
runs after the frontend has bound immutable parameters. OOT components reuse the
existing source proposal and independent admission hooks. Scalar proposals carry
conditional native workspace obligations; actual lowering and call admission
must discharge them before image publication. The current path supports a
bounded synchronous shared-memory plan. It does not imply that every provider
can supply every proposed region, or that an unsupported lowering can be retried
silently. The complete installed scalar example compiles a saved weighted Torch
export, builds an ELF and runs it through the Linux service; accelerator and
heterogeneous execution retain their independent qualification requirements.

Annotated source requires the original caller to approve every semantic attribute
with exact typed coverage. The compiler preserves that source, the selected
options and the admission identity through discovery and partitioning. Source
approval does not establish an implementation's numerical behavior or runtime
input domain. The built-in scalar routes currently refuse these extra semantic
obligations; they cannot silently absorb an unsupported accelerator region.

Prepared exported state is observed in `compiler/frontend/torch/state.py` and
checked without importing Torch in `compiler/frontend/state.py`. A caller can
explicitly select `StagedProgram(..., freeze_readonly_buffers=True)` for freshly
observed, nonaliasing, readonly BUFFER snapshots. The signature keeps each buffer's
role and target. Unknown effects, writes, aliases, changed payloads and missing
producer evidence refuse; the default remains refusal. This selects snapshot
semantics for later invocations and does not implement mutable model state.

`compiler/quantization/legalize.py` admits supported captured quantization markers
against their actual source arithmetic before invoking the shared lowerer. The
active integer conversion guards NaN before conversion, and preserves the original
rounding, zero-point and saturation order. An imported scalar scale that has
already lost source precision refuses. Actual native CPU boundary checks support
this primitive contract; full-model task quality remains a separate obligation.

The stateful C call contract lives in `compiler/backend/stateful.py`. It binds
caller-owned context storage, serial workspace and checked embedded program bytes
to the selected resource, memory domain and platform artifact. Control emission
checks the selected implementation's actual C ABI and reservations. OOT adapters
own device-specific encoders, layouts and initialization. A callable wrapper and
embedded binary do not by themselves admit an executable image: startup,
initialization, memory visibility and platform loading still need explicit image
contracts and execution evidence.

### Software specification and OOT fusion contracts

The existing typed scalar-body and linalg computation records live in
`target/spec/computation.py`; type/rank comparison lives in `signature.py` and
source-bound instruction model normalization in `instructions.py`. This splits
the former instruction-semantics module without changing its record algorithms,
numerical meanings, unknowns or normalized results. Its historical targetgen
module is an exact alias, including public callable pickle spellings. Compact
expansion observes all three actual owners; Phase 1 startup also pins the physical
compatibility file independently of whether that alias was already imported.
Missing or symlinked startup owners still refuse. Typed computation inspection
does not enable its unsupported executable v1 projection or establish RTL proof.

The semantic-contract migration makes the selected `sw_spec` the compact,
reviewed owner of numerical and instruction state/effect meaning. Opt-in
`merlin.software_spec.v2` now composes named existing records with exact source,
import, tool and format-registry identities. Compatible selections project into
existing v1 consumers. Authored typed computations can be inspected, but their
executable v1 projection explicitly refuses; the broader readable pseudocode
frontend remains future work. Existing reviewed v1 definitions remain the
default and retained paper authority.

`merlin.api.inspect_spec(path, expanded=False)` and
`merlin inspect --spec PATH [--expand]` expose this same boundary without compiling.
The optional `examples/gemmini/target/spec.yaml` is an unreviewed migration
candidate; its external `examples/gemmini/preparation.yaml` retains the original
application methodology. Normalized comparison preserves all target semantics
apart from the explicit review-status change and policy separation.

`targetgen/software_spec.py`, `quantization_spec.py`, `isa_spec.py` and the typed
scalar-body normalization in `instruction_semantics.py` provide existing
infrastructure to consolidate. Number encodings, scheme definitions, application
precision selection and calibrated/runtime qparams have distinct roles. Move
model-specific preparation policy and measurements with their actual consumers;
preserve archival inputs, imported format overlays and independent acceptance.
Explicit source-bound profiles now enter capture recipe selection through
`compiler/quantization/preparation.py`. The strict path withholds changed
numerical intent until its exact alternative is selected. The named legacy
policy and old module identity retain existing behavior. Generate TorchAO
requests only for implemented, version-admitted adapters.
Hardware readout limits constrain implementation offers and do not silently
change the program's requested scheme.

`target/spec/quantization.py` now owns the closed declaration validators; the
legacy quantization controller reexports those same functions. Active numerical
extension inputs preserve their complete floating-element and output-requantization
payloads through legacy and compact intake. Legacy YAML merge overrides and
declaration order are preserved. Structural intake establishes no execution
adapter or hardware correspondence, and generic capture refuses unmodeled
extensions rather than ignoring them.

OOT providers expose implemented fusion patterns and physical lowering recipes;
the reviewed specification supplies numerical/effect meaning, selected SoC facts
supply handoff constraints, and Merlin selects compatible overlapping offers.
Reuse the bounded, paginated `compile/fusion_discovery.py` frontier and existing
realization/catalog mechanisms. The narrow `compiler/planning/fusion.py` adapter
does not yet integrate all those families. Its optional paged OOT hook now uses
the existing discovery frontier and bound cursors, with distinct output, matcher,
proof-session, source and neighborhood limits. The original CPU alternatives and
overlapping raw/fused offers remain available. These limits describe discovery
scope, not hardware incapability or an in-process wall-time guarantee.
For recipe-bearing offers,
`LoweringTicket` now carries the exact selected implementation, pipeline, guards,
ABI, source-to-outline mapping and resource contract through the existing region
invocation. Providers must implement `validate_selected` and `compile_selected`;
the returned acknowledgment binds the actual required artifact kinds, roles and
bytes. Gemmini's admitted raw and fused routes use this path. Legacy proposals
retain explicitly labeled source/options compatibility. Tickets establish
compiler custody and claims, not independent numerical or hardware proof. Full
discovery integration and arbitrary fusion remain separate work.

### Making OOT implementations available

Ordinary compilers register through the installed distribution's
`merlin.backends` entry points. A development checkout can use
`python -m pip install -e /path/to/oot` in the same environment as Merlin;
release and reproducibility checks use the built wheel. These are standard
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
and [local installations](https://pip.pypa.io/en/stable/topics/local-project-installs/),
without a second Merlin installation registry. Native compilers can expose an
adapter to their declared process/toolchain interface; LLVM output is optional.

`merlin inspect --backends` and `api.inspect_backends()` list the selected Python
environment, provider names, distribution versions and declared installation
origins without loading provider code. Duplicate names are ambiguous; origin
metadata is not a source attestation, compatibility check or qualification.
Compilation still explicitly selects a backend and records its actual inputs.

Generated experiment candidates retain run, source, SDK and evaluation custody.
`merlin experiment export` and `merlin-experiment export` reuse the existing
`targetgen.champions.export_champion` owner. The command selects the phase's
frozen/best history and checks the four explicitly supplied provenance,
certification, measurements and ISA-prohibition records. It preserves source
and authored packaging and reports static `merlin.backends` declarations from
the exported `pyproject.toml` without loading provider code. Missing, dynamic
or invalid metadata remain visible. Export performs no installation or remote
publication and does not create an ordinary adapter for a legacy compiler.
An explicit promotion to an installable service must retain the original
receipts; not all generated templates already satisfy that public contract.
`MERLIN_TARGET_PATH` and `provider.yaml` preserve explicit checkout contract/role
discovery, while `backend.toml` describes component composition. These describe
different boundaries and do not replace installed compiler registration.

The installed `merlin.package` service adapts the existing experiment ABI 0.1
process protocol. It accepts explicit `merlin-interface` source and
`package.root`, with optional `package.contract` and `package.timeout` parameters.
It reuses package loading, entrypoint invocation, command-buffer schema and
logical dtype checks. Selected source, prebuilt tool, actual interface and schema
files are observed and rechecked; optional independent reference corpora are not
inspected. Existing declines remain declines. Returned command buffers and opaque
target text do not imply numerical equivalence, callable images, LLVM output or
device execution. Arbitrary model input, ABI 0.2 and deployment requests refuse;
this service creates no experiment source grants or automatic candidate promotion.

Scalar computation lowering is distinct from control lowering and from the C
runtime linked into the executable. Out-of-tree providers that produce compatible
LLVM IR can reuse the shared LLVM conversion boundary without importing the compiler driver. Target instruction
selection and shared-interface-to-target-dialect conversion remain owned by the
selected OOT implementation. The current package invocation still uses its
existing sandbox and exact-input/receipt checks; the new region owner does not
replace those checks with a claim of execution qualification.

The source-bound route uses `bind_source(..., graph_mode="ssa")`, followed by
original-caller proposal admission, `select_execution`, and `PlannedCompiler`
through `merlin.api.compile`. Its current scope is static, flat pure programs,
contiguous regions, serial completion and one shared coherent memory domain.
Scalar and vector providers can share a declared CPU resource; this does not
imply concurrency. Unsupported memory transitions and raw scalar provider-call
boundaries refuse. The existing offload route retains its original outlined
graph contract.

`RegionBackend` explicitly binds each selected provider's compiler, source entry
and options. Every region validates before compilation starts. `plan.json`
records exact source ordinals, rejected alternatives, resource placement, memory
budget, partition fidelity and per-region products. The aggregate stage is
`compiled_regions`, with region roles separated from whole-program executable
roles. Device binaries are admitted without requiring LLVM output. Host/control
lowering, target call-ABI adaptation, linking and deployment admission remain
required before this product can become a runnable heterogeneous image.

For regions with an explicit `merlin.c_call.v1` contract, the same Python compile
call can select `CompileOptions(parameters={"merlin.emit": "c-dispatch"})`.
`merlin.entry` optionally names the generated C entry. This assembles the existing
serial dispatcher against the original partition and its actual compiled
products. Contracts bind ordered tensor shapes/types, resource kind, pointer
alignment, synchronous completion, link inputs and binary payloads. A binary or
LLVM artifact alone does not establish a C calling convention. The initial route
requires stateless wrappers and zero additional host scratch; any uncovered
inline CPU operation refuses before compilation rather than disappearing.

`control/control.json` records calls, the bounded arena, caller-owned buffers,
the selected resource and original artifact receipts. Alignment and buffer
validity are checked before runtime calls. Read-only inputs may share storage;
outputs remain disjoint. The `c_dispatch` stage has no deployment or executable
claim: object compatibility, final linking, backend correctness and observed
execution remain separate obligations. Independent partition verification and
memory-plan checks do not certify a backend's arithmetic or device use.

The installed `merlin.layout` provider is a concrete CPU implementation for one
static cast/collapse/expand operation. It reuses the shared C layout emitter and
preserves storage bits. It is not a replacement for the general scalar LLVM
pipeline. Both use the ordinary compile/artifact service; target instruction
selection remains in OOT providers.

An explicit saved export uses the same service:

```python
from pathlib import Path
from merlin.api import SavedExport, compile

result = compile(
    SavedExport(Path("reshape.pt2"), capture=Path("capture-evidence")),
    backend="merlin.layout", output=Path("compiled"),
)
```

The equivalent CLI is `merlin compile reshape.pt2 --capture-output capture-evidence
--backend merlin.layout --output compiled`. A `.pt2` Path selects this adapter in
Python too; omitting the capture workspace selects `OUTPUT.capture`. A live
PyTorch module, FX graph or function uses `compile(program, example_inputs=(...),
backend=..., output=...)`; an `ExportedProgram` already carries its inputs.
Modules and inputs are copied together before export, preserving shared state
references and the selected training mode. The original archive is frozen before
any later numerical preparation. These framework imports remain lazy.
Configure the pinned model2MLIR checkout and its separate capture interpreter
through the existing environment selectors. Both output directories must be
fresh. The selected producer imports the original export, preserving its trace;
only validated compiler inputs enter the compilation product. Reference arrays
remain in `capture-evidence/reference/`. Immutable parameters and lifted constants
are bound into tensor constants with exact storage bytes; runtime arguments retain
their captured input order. Original staged IR and the transformed input remain
separate artifacts with a checked mapping. The namespaced option
`frontend.max-initialization-bytes` bounds embedded tensor bytes (64 MiB by default).
The current adapter refuses dynamic constraints, multiple/mutation outputs,
mutable buffers and unsupported initializer storage. Stateful applications retain
their explicit state contract; weights do not become user arguments.
Existing full-model and explicit quantization workflows retain their own routes.
Imported semantics, compiled arithmetic and actual target use still need their
respective verification; capture hashes alone establish byte identity.

### Backend collaboration before target lowering

The destination flow obtains implementation candidates from the explicitly
selected OOT backends before committing a device partition. Each backend owns
matching its implemented single-operation and multi-operation patterns. Shared
analyses provide original-source identities, dataflow, live boundaries, effects,
shapes and numerical contracts. Backend matchers reuse these helpers and the
shared pass library; the shared planner does not encode a target's pattern list.

A backend query returns overlapping choices and refusal reasons. A contraction
alone and the same contraction with a supported epilogue may both be candidates.
Other backends may offer the epilogue or entire chain. These choices remain
available until selection; a maximum greedy fusion partition is not the input
to the global planner. Every live intermediate needed elsewhere remains a
boundary value, and unsupported intermediate arithmetic makes a split invalid.

The selected specification and hardware facts constrain independent legality
checks. Editable backend capability records describe implementation claims;
they cannot establish hardware truth or application accuracy tolerances.
Existing `compute_groups`, `TargetOracle`, `output_fusion` and target readout
facets supply useful matching/checking mechanisms. Their legacy selected groups
must not be mistaken for a complete multi-backend optimization decision.

The bounded shared selector compares compatible covers, placements, resource
reservations and boundary operations, including packing, transfer, launch and
synchronization costs. Missing costs stay unknown. Materialization commits only
the selected grouping and boundaries; local lowering then executes the selected
backend pipeline and shared passes. Tensor simplification/fusion, loop/buffer
optimization and command/effect optimization have different IR requirements;
the entire pass library does not run before target lowering.

The existing source-bound proposal admission, `select_execution`, partition
materializer and `PlannedCompiler` are reusable parts of this flow. Universal
backend discovery and complete compile-time orchestration remain migration
work; manually supplied proposals are not automatic backend integration.

The first concrete bridge is `discover_fusion_candidates` in
`compiler/planning/fusion.py`. The selected provider's matcher offers overlapping
bare and fused regions. Shared checks re-open the existing selected target
contract, recheck the strict signed dense body and supported exact readout,
and retain original CPU alternatives. `select_execution` and `PlannedCompiler`
then select, partition and lower those candidates. The implemented scope is
static Generic signed i8/i32 contractions with zero initialization and supported
serialized requantization; an original-caller validator still owns independent
target-compute/overflow admission. Unknown costs, missing full resource
envelopes and unavailable hardware qualification remain explicit. This bridge
does not yet orchestrate every backend from a full-model compile request.

### One public API for compilation, execution and development

The public Python services are `merlin.api.compile`, `merlin.api.inspect`,
`merlin.api.run` and `merlin.api.collect`. The corresponding `merlin compile`,
`merlin inspect`, `merlin run` and `merlin collect` commands adapt the same
requests. `merlin compile` always uses explicit-source grammar; the retained
`merlin-compile` workload/operator command calls `merlin.compile_cli` separately.
The legacy `merlin.compile` package is not a root function. Experiment commands
use `merlin.experiments.definition` and their existing experiment controller.

Compilation accepts MLIR text, a `Path`, an explicit `Source`, supported PyTorch
objects/FX graphs with example inputs, or a saved exported program. `SavedExport`
selects its capture workspace; `StagedProgram` selects compiler-only inputs and
explicit preparation admission. A `str` means source text, not a filename.
Compilation requires one explicitly selected provider, a fresh output directory
and validated `CompileOptions`; unsupported provider options refuse before
emission. `SystemCompiler` and `PlannedCompiler` are explicit advanced provider
objects, as described above. C image linking requires a caller-selected
`CImageLink`; context initialization and positive tensor workspace ceilings are
also explicit. These paths do not replace the legacy default or establish
universal OOT integration, numerical correctness or hardware qualification.

`Compilation` retains an extensible set of artifact kinds with relative paths
and content hashes. LLVM is not required. Its optional `Deployment` identifies
the system, platform, ISA, ABI, hart count and selected memory range. Execution
requires that contract and cannot silently override the compiled hart count.
Provider metadata records claims and production; it grants no verification.

The native application controller now consumes compiler-only staged inputs,
explicit state/parameter contracts and a selected provider's application adapter.
It compiles every selected region, emits a checked serial C dispatcher and bounded
memory plan, and publishes the native program set with a portable C/binary handoff.
Verification replays source partitioning, memory checks and C generation; it does
not establish numerical quality or hardware execution. The bounded first-valid
partitioner does not imply heterogeneous topology or coherence admission. The
selected Atlas backend currently lacks the required application adapter, so that
whole-model route refuses explicitly; the shared controller is not evidence of
Atlas whole-model support. Existing default arena planning remains unchanged.

The new FireSim service only prepares files and collects externally produced
observations. Linux images receive a guest invocation script; bare-metal and
Zephyr images receive a boot description. An external operator owns simulator
setup and execution. Collection binds output bytes to a prepared request and
retains the reported exit code; even a zero exit code does not establish
numerical correctness or timing qualification. The historical `run_on_firesim`
model entry now exposes explicit prepare/collect modes and retains its numerical
gates when collecting UART results. Old launch options and plugin selection fail
before launching work. Research scripts must adopt this handoff lifecycle rather
than treating preparation as a completed experiment.

Ordinary runtime selection freezes the original artifact, deployment and options
as copied identity data. File bytes and request fields are rechecked around
runner validation and execution callbacks; results must name the original digest
and runner. External preparation also keeps the caller's original build receipt
across manifest reloads and compares the final receipt with the initial bytes.
A resealed changed build cannot become the selected input through reload alone.
These are boundary observations, not proof of instructions executed, numerical
correctness or device completion/visibility. Existing output creation and
external-ELF deployment contracts remain supported.

Compilation accepts admitted frontend inputs and returns a durable artifact set.
The selected target bundle names the SoC, resources and hardware facts; backends
name compiler providers; platform selects a supported Zephyr, bare-metal or other
deployment port; runner selects the simulator or physical execution environment.
These settings are different contracts. A provider list alone is not a hardware
topology. Compilation does not execute the resulting program, and execution
consumes existing artifacts without recompiling their source. Legacy commands
may retain their coupled behavior during a documented transition.

| Input | Proposed treatment |
| --- | --- |
| `torch.export.ExportedProgram` or its saved export archive | Preferred PyTorch handoff; retain graph signature, state, constants, constraints and source identities |
| `torch.fx.GraphModule` | Admit supported FX operations with example arguments or complete input metadata, state bindings and explicit constraints |
| Bare `torch.fx.Graph` | Require its referenced module/state, signature and input metadata; never infer absent bindings |
| Supported Python module/function | Capture through the selected frontend with explicit example arguments and keyword inputs |
| Existing capture bundle or admitted MLIR | Reuse current import contracts; Torch capture is unnecessary |

The existing `_pt2_capture_worker.py` and `build_tools/scripts/ingest_applications.py`
already ingest exported programs through model2MLIR. That route currently requires
positional example inputs and refuses keyword examples. A general in-process FX
and export API is still migration work. Preserve the actual export's lifted
constants, mutation/result bindings and shape assumptions; avoid discarding them
by treating only its graph module as the whole program. A saved export archive
and an AOTInductor compiled package may both use `.pt2`; inspect the declared
contents and producer format, not the suffix alone. Existing quantized programs
retain their realized numerical contract unless a new preparation is explicitly
requested and admitted.

Options have typed, discoverable ownership: common compilation, registered pass,
backend, platform, runner and development options. Presets expand into recorded
settings; explicit overrides remain bounded by hardware and numerical constraints.
Unknown or inapplicable options fail at resolution. CLI help and Python discovery
read the same schema. Explicit pipeline descriptions retain repeated pass
occurrences, anchors and order. Per-call options and explicit child-process
environments replace the current process-wide optional-pass mutation.

Run accepts a generated build, an artifact bundle or an existing ELF/device
program with an admitted launch contract. A hash-bound manifest or embedded note
provides platform, ISA, memory, loader, entry/result ABI, constants and hart
requirements. External files require the missing information explicitly; ELF
format alone does not identify Zephyr or a complete invocation protocol. Runtime
hart selection must fit both the hardware's eligible IDs and the compiled image's
worker/boot/barrier contract. Some settings require rebuilding. Host executor
threads and physical RISC-V harts are distinct. An optional loaded session
supports repeated calls with explicit state and completion ownership.

FireSim is an externally managed execution environment. Merlin prepares a
hash-bound workload package and collects the resulting observations; it does not
provision the farm, configure FireSim infrastructure, launch simulations or manage
their lifecycle. The proposed FireSim adapter supports preparation and collection,
not direct execution. For example, `run(..., runner="firesim", mode="prepare")`
returns a prepared workload, not a completed run. An explicit collection operation
matches returned artifacts and results to that request before evaluation.

The handoff includes the existing executable, required inputs/constants, startup
or guest invocation files where applicable, and the expected result protocol.
A bare-metal or Zephyr boot image uses its own startup/harness; it is not a Linux
guest program to invoke with a shell script. An external operator starts the
environment and arranges execution. Infrastructure management remains external
operator tooling. The generic runner interface declares prepare, execute and
collect capabilities, so core orchestration need not special-case a target name.

Each target's examples demonstrate the same public API for a supported kernel
and a supported model route. Keep the model, precision, shapes, platform and
execution prerequisites explicit; use existing exported inputs when appropriate.
Examples contain program construction and user choices, not copied compiler
implementations. For externally managed execution, the example prepares the
workload and shows collection as a separate step. A target with only a bounded
chain implementation must describe that scope rather than claim arbitrary model
support.

Experiments reuse compile, run and independent check services. The everyday
interface selects a study, agent, budget and destination or session; study configuration
holds the same compile/run options, with explicit validated overrides. Existing
planning, preflight, run, resume and status behavior remains available. Resume
checks the study, inputs, target, candidate/library edits, evaluator and tool
identities; it cannot silently change the experiment being continued.

These choices borrow Python ergonomics from
[PyTorch 2](https://pytorch.org/assets/pytorch2-2.pdf), compiler/runtime separation
from [IREE](https://iree.dev/guides/deployment-configurations/), and distinct
compile, executable and execution objects from
[PJRT](https://openxla.org/xla/pjrt/cpp_api_overview).
PyTorch's export API appeared after 2.0; its
[export contract](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/api_reference.html)
is a useful handoff model, not a promise of compatibility with every framework
version. None of these frameworks supplies Merlin's SoC/RTOS or agent-experiment
contracts unchanged.

The [usability review gates](../design/refactor_preservation.md#review-usability-as-part-of-each-migration)
cover inspection, diagnostics, option discovery, reproducible iteration,
portable artifacts, backend contributions, compatibility removal and maintenance.
They assign proposed owners and observable acceptance criteria. Apply them during
each migration; a new directory layout or a documented API alone does not satisfy
them.

### Verification is a separate compiler service

The proposed destination includes `src/merlin/verification/` beside `compiler/`
and `evaluation/`. It promotes the existing `verify/` subsystem to an explicit
architecture owner; it does not invent verification or place its implementation
in the test suite. See the existing
[compiler verification architecture](../design/compiler_verification.md) for
current boundary coverage and limitations.

| Responsibility | Owner and meaning |
| --- | --- |
| Types, SSA, schemas and immediate pass preconditions | Compiler/dialect owners; local structural legality and diagnostics |
| Actual before/after translation validation | Independent verification; semantic refinement within a declared domain |
| Actual schedule/resource/memory and artifact checks | Independent verification using selected facts, semantics and decoded products; each check has explicit scope |
| Lean/SMT invocation, proof assets and evidence qualification | Verification; exact source, model, parser and tool identities |
| Numerical references, task quality, observations and acceptance policy | Evaluation; combines required claims without enlarging their scope |
| Regression examples and negative controls | Tests; exercise the production compiler, verifier and evaluator |

Existing owners include `verification/partition.py`, `outline_ir.py`, `scalar_ir.py`,
`split_reduction.py`, `verify/refine.py`, `verify/lean_checker.py`, `verify/receipts.py`,
`compiler/schedule/compact_verification.py`
and `model_check/program_admission.py`. They check different things. An SMT
proof for a concrete integer IR pair is not a proof of arbitrary shapes,
floating-point behavior, DMA or final machine code. The Lean path preserves
its selected assurance mode and trusted correspondence; kernel-only checking
and native reflection must remain distinguishable. Publication hash checks
establish artifact identity, not semantics.

The compiler invokes the trusted verification service with actual source/output
artifacts, declared assumptions and optional witnesses. The service issues scoped
results, including unsupported, unknown, unavailable and refuted outcomes.
Evaluation applies the independently selected acceptance policy. A serialized
receipt or candidate callback cannot create live acceptance authority.

Verifiers may use reviewed parsers, stable IR/contracts and pinned structural
correspondence implementations. They must not resolve mutable candidate overlays
as their own checker or semantic model. Current verifier fingerprints include
some transformation sources; those dependencies must be deliberately migrated
and requalified, not dropped merely to draw a cleaner import graph. Target
semantic evidence is independently selected, even when it describes the same
instructions as the compiler backend. Required missing evidence leaves that
claim unqualified.

### Shared compiler library available to backend authors

Portable pointwise and reduction tiling live in
`compiler/transforms/pointwise.py` and `reduction.py`; their C consumers import
those owners. Historical semantic-compiler modules are exact aliases, and
native source receipts include the actual algorithms alongside the aliases.
These passes preserve affine maps, padding, seed bytes and ordered boundaries;
they do not select target instructions or authorize numerical reassociation.

Ordered K, M/N and M/N/K contraction tiling and streamed panels live alongside
those passes in `contraction_k.py`, `contraction_mn.py`, `contraction_mnk.py` and
`panels.py`. Their numerical algorithms remain unchanged. Canonical C lowering
uses the real owners; historical semantic-compiler imports and actual Atlas
preparation methods resolve to the same classes and functions. Old record,
exception and callable pickle names remain readable, and active native snapshots
observe the real implementations. Frozen SDK grants and source rosters do not
grow because these files move.

The shared library is broader than `src/merlin/transforms/`. Its reviewed,
versioned public inventory lives in `compiler/schedule/export.py`, with the
implementation closure in `compiler/schedule/library_sources.py` and author
exposure in `targetgen/tool_profile.py` and `tool_profile_shared.py`. The selected
release determines availability; the newest inventory does not enlarge a frozen
experiment's grants. Fresh SDK27 uses shared-library schema v21 and common substrate
V20, naming 133 public modules and 269 source modules, including
analyses, contracts and emitters as well as transformations; this is not a pass count.
The pinned Gemmini paper [shared-library manifest](https://github.com/ucb-bar/gemmini-mlir/blob/7cd030e13360fd0961bd83fa288bd9009ba813df/dependencies/generic-library/shared-library.json)
has 123 public modules, 220 source members and 44 capability entries. Its exact
membership differs from the current checkout and must remain reproducible.

| Library family | Current implementation examples |
| --- | --- |
| Structural, scalar and layout rewrites | `transforms/normalization.py`, `pure_scalar.py`, `unrolling.py`, `tensor_views.py`, `dense_reduction_layout.py`, `dense_row_batching.py` |
| Blocking, fusion and dense schedules | `compiler/schedule/block_schedule.py`, `fusion.py`, `staged_dense.py`, `dense_panel_pipeline.py`, `dense_segmented_pipeline.py` |
| Storage, residency and transfers | `compiler/schedule/temporary_storage.py`, `device_spills.py`, `panel_transfers.py`, `tile_load_coalescing.py`, `dense_rhs_residency.py` |
| Spatial reuse and convolution | `compiler/schedule/rolling_window.py`, `rolling_spatial.py`, `convolution_input_reuse.py`, `horizontal_convolution.py` |
| Configuration, completion and overlap | `compiler/schedule/configuration_motion.py`, `completion_streams.py`, `dense_readout_overlap.py`, `dense_ready_readout.py` |
| Selection, allocation and exploration | `semantic_compiler/`, `transforms/resource_allocation.py`, `pipeline_policy.py`, `pass_exploration.py`, `evaluated_exploration.py` |

Start with [shared compiler pass contracts](shared_compiler_passes.md).
For architecture-based optimization discovery, open
[`joint_optimization_catalog.json`](../design/joint_optimization_catalog.json).
Its `required_traits` describe capability prerequisites; each entry names the
implementation, backend adapter and evidence to request. The generated author
guide lives at
`src/merlin/experiments/phase1/author_skills/pass-library/`.
`phase1/optimization_suggestions.py` joins the catalog to the selected target
traits and emits `OPTIMIZATION_SUGGESTIONS.json` into an author workspace;
`phase1/operation_optimizations.py` checks actual workload prerequisites.
`phase1/resource_prompt.py` adds the selected shared-library contract to the
agent prompt. A suggestion is not semantic legality or verification evidence.

The actual scheduling implementations now live under `compiler/schedule/`;
`compile/scheduling/` retains temporary exact module aliases. Shared tensor and
loop rewrites continue moving into `compiler/transforms/`. Agent prompts and
feedback live in `merlin.experiments` within the same distribution, with optional
engine dependencies. Fresh experiments select SDK27 and freeze its exact source
closure; historical SDK6 through SDK26 replay requires the corresponding frozen
release and explicit source environment. Their metadata and source rosters are
unchanged. Source inventories, editable overlays,
generated guides and their callers migrate together, with one editable owner
for each implementation.

`compiler/schedule/api_catalog.py` and `compiler/passes/composition.py` expose
callable discovery and prerequisites. The
[optimization catalog](../design/operation_optimization_catalog.md) connects
mechanisms to author-facing skills and recipes. These discovery, source-export,
composition and verification paths are preserved functionality, not dead code
because they lack ordinary application imports.

The library implements substantial algorithms an accelerator backend would
otherwise implement itself. Generality comes from explicit hardware/specification
inputs: capacities, layouts, instruction semantics, command effects, resource
lifetimes and encoding adapters. For example, operand retention and overlapping
transfers are reusable algorithms when a backend establishes their required
storage and completion laws. They are not mere pass-name wrappers or universal
optimizations inferred from a target name.

Backend authors and agents may call, configure, compose, modify or replace these
implementations. This is already supported by
`targetgen/compiler_library_fork.py`: an enabled selected library admits sparse
candidate source overrides under `submission/compiler_library/merlin/`.
`merlin.experiments/phase1/compiler_library.py` explains this workflow to the
agent. Unchanged leaves use the selected baseline; candidate edits are hashed
with the backend and run through the ordinary candidate execution path.
Independent checking, approved facts and reference behavior remain outside those
edits. An immutable evaluation snapshot freezes one candidate for measurement;
it does not make the library uneditable during development.

The destination assigns shared rewrites to `compiler/transforms/`, scheduling
algorithms to `compiler/schedule/`, selection to `compiler/selection/`, and
invocation/registration to `compiler/passes/`. These are parts of one public
compiler library with explicit candidate variants, not permanently duplicated
common libraries per backend. Retain and modernize the existing editable overlay
and its source accounting. General improvements can be promoted into a new shared
release; target-specific realization remains in OOT. Migration must preserve each
selected release's source membership, API contracts, author access and checks.
The new `compiler/passes/native.py` and `xdsl.py` only move invocation machinery;
they do not replace this library or automatically expose all of it to candidates.

### Backend output is not restricted to LLVM

The proposed backend `lower` boundary returns a set of typed artifacts and their
consumption requirements. Compatible IR may need further finalization; an object
may need linking; a device binary or command program may instead need embedding,
loading or submission. A product can include several of these plus constants and
runtime bindings. Artifact format and its consumption route are separate fields:
an embedded device binary is data in a host image, not host machine code.

Each artifact contract identifies its exact bytes, format/version, target and
required features, entry or submission ABI, bindings, address/relocation needs,
runtime requirements and completion semantics. Build orchestration validates and
packages the selected products; it does not run every device product through LLVM
or the host linker. Host/control code may still use LLVM while device code follows
its own compiler. Zephyr integration selects embedding/loading, driver and memory
requirements from the system and artifact contracts.

The inspected [Atlas binding](https://github.com/ucb-bar/atlas-mlir/blob/8ee175c8614b517f80048886685a3e4029b98520/python/atlas_mlir_native/binding.py)
already writes `program.bin` from emitted instruction words, with virtual/machine
IR, physical-program metadata and an execution plan. Its inspected manifest
labels qualification as diagnostic only. This is evidence for a non-LLVM product
path, not hardware qualification. The current branch already has
`semantic_compiler/target_binding.py`: `NativeTargetBinding.compile` returns a
manifest and `verify_native_publication` checks binary/plan identity for programs
and ordered program sets. Preserve and integrate that existing native path.
The current `compiler/lower/regions.py` remains
an adapter for the existing textual package protocol; the general artifact
composition and whole-image consumers for that legacy route remain incomplete.

Zephyr configuration consumes both the selected board's hardware facts and the
selected Zephyr tree's expressible options. Its Kconfig discovery has per-build
ownership, so selecting another tree in the same process cannot inherit the
previous tree's vector-state settings. The tree reader does not replace Kconfig
evaluation, board qualification, or execution checks. Temporary old private
configuration entrypoints remain while delivery-script consumers migrate.

Prebuild routing now retains its original group objects and source identity.
Offload materialization checks that source, decisions and stored-argument binding
are unchanged. Group, stream and capsule reports reuse that selection before
the build instead of making fresh grouping decisions afterward. The independent
placement census remains a separate check. Source-group materialization is not
the final image's execution schedule: preparation can still transform the input,
and the existing image route still admits only one device datapath.

Shared boundary analysis enumerates SSA values and nested captures. Existing
source readers retain responsibility for effects, aliases and legal movement;
enumerating a boundary does not authorize a transformation. Matrix offers adapt
the existing explicitly selected provider and preserve Boolean selection;
their cost is unknown unless supplied, and they establish no additional hardware
or numerical capability. This legacy image route does not yet consume the
global heterogeneous selection service described above.

The ignored Zephyr-only `arena_mb` keyword and FireSim driver's `--arena-mb`
option have been removed. Zephyr sizes activation memory from the lowered
program and selected board layout. Baremetal allocation still uses its meaningful
arena setting. Generated drivers no longer advertise an unimplemented RTIO
switch. No CCA, mining, scheduling, or independent verification capability has
been retired by these changes.

All active Merlin implementation lives in the single `src/merlin/` package.
Experiment engines use `merlin.experiments`; design-space analysis uses
`merlin.analysis.dse.{search,pressure,guidance}`. There is no root `merlin/`
source/resource tree or second project-named Python package. The src layout
keeps imports bound to the selected installation, enabling checks outside the
checkout to detect missing packaged code or data.

Shared compiler mechanisms, trusted evaluation and research retain distinct
logical ownership and access; dependency extras enable optional engines without
installing another copy of the code. Installed public resources live under
`src/merlin/resources/`, and native runtime C/header sources under
`src/merlin/runtime/native/`. Active regression cases live under `tests/`.
Target examples, study drivers, captures, retained corpora and reproduction inputs
live under `examples/` and `experiments/`, outside the wheel's declared resource
inventory. Installed research workflows require their selected external inputs.
`packages/` retains historical metadata/inputs and the independent timing observer.
An import prefix or wheel membership alone grants no candidate access; frozen
historical inputs retain their original source environment.

### Compiler and target boundary

| Owner | Implementation |
| --- | --- |
| Merlin | Reusable host code generation, packing, requantization, graph/global optimizations, dispatch, buffer ownership, device compilation orchestration, and runtime infrastructure |
| Target OOT MLIR dialect repository | Target dialect operations, instruction encodings, device kernels/schedules, hardware layout/resource facts, target ABI glue, and target execution support |

An optimization that can be selected independently of the accelerator belongs in Merlin,
including one first measured on a particular device. A mixed optimization uses an explicit
contract: Merlin supplies the generic algorithm and verifies its semantic obligations; the
provider supplies target legality facts, instruction selection and implementation. Generic
OOT prototypes move into Merlin when promoted, with provider delegation replacing duplicated
code. Optional numeric policies retain explicit selection and their original accuracy gates.

Host CPU implementations are shared independently of accelerator selection. Two
targets with the same host ISA, ABI and numerical/effect capabilities reuse the
same host code generation and runtime; a different host selects an explicitly
supported CPU implementation. Device layout and ABI bridges remain target-owned.
See [shared host arithmetic](host_cpu_arithmetic.md). Shared host support does not
admit a target compiler seed, device schedule or protected performance answer.

The OOT dialect is a general compiler backend. Production decisions follow operation
semantics, shapes, layouts, numeric contracts and hardware capabilities, with legality and
resource checks. They must not depend on model names, captured provenance IDs, golden outputs
or benchmark constants. Constant and shape specialization derives from the current input IR.
Experiment drivers may select source-bound candidates; promotion requires general transforms
and cost models tested on independent cases, including tails and refused/fallback cases.
Provenance IDs support traceability and exact source-to-device binding, independently of
the semantic and cost-based decision to apply an optimization.

The catalog defines experiments; `experiments/studies/` still contains checkout-dependent
native engines and resources. Reviewed phase-0 handoff and frozen phase-1/2 inputs preserve
the existing grading authority. Installed commands and synthetic tests do not establish
hardware readiness or qualify a compiler. See [repository structure](repo_structure.md)
for navigation and the remaining qualification boundaries.

## Continuous integration and maintained obligations

The repository has three workflows. `.github/workflows/ci.yml` retains the visible
`pr-fast` workflow and required `fast`, `ruff` and `docs-freshness` check names.
Source and documentation gates run once. `check_test_budget.py` discovers owned
collectors across the repository and counts actual parameterized cases before
execution; errors, collection skips, deselection and totals above 150 refuse the
budget check. The aggregate reports failure, cancellation or skipped required jobs.
The primary workflow waits for the reusable `integration.yml`, which also has
weekly and manual triggers. It builds the main Merlin wheel and source archive,
audits namespace ownership, installs that exact wheel outside the checkout and
checks pinned OOT provider boundaries and wiring. Exact outcomes and release
artifacts are retained together. The independent timing observer is a
separate distribution. These are configured workflows, not evidence that a
remote GitHub run or accelerator execution has occurred.

`release.yml` manually invokes the same qualification workflow. It does not
configure package publication, board deployment or FireSim infrastructure.
Reviewed pinned OOT consumers, native/hardware lanes and complete optional-dependency
execution retain their separate qualification obligations. External actions use full
commit references and repository permissions are read-only. A required package
installation gate fails when its installer is unavailable; optional local
invocation retains its explicit skip behavior.

The permanent suite has a 150-case ceiling, including parameterization: 70
compiler/runtime cases, 50 independent verification cases and 30 contract cases.
Direct test bodies retain independent references and distinct positive/refusal
witnesses. Shared support contains reachable producers, fixtures and references;
retired helper/spelling/Cartesian matrices are not hidden in large loops or
qualification runners. Exact historical paper instruments remain Git/frozen inputs.
Production verifier algorithms, approved facts, numerical thresholds and source
grants are unchanged. Missing Lean, translator or hardware prerequisites remain
explicitly unavailable; collection and host execution do not establish those claims.
Dead-code removal likewise requires real caller and registry review;
removing an unused Zephyr workspace helper does not remove actual tree, SDK,
linker or deployment requirements.

## Data flow

```
External kernels/repos --> integration adapters --> kernel_record / abstraction_candidate / policy_rule
ISA/docs/RTL           --> targetgen            --> target_contract --> dialect_plan --> dialect scaffold
workload_region        --> design_pressure      --> design_pressure --> candidate_contracts
candidate + cost model  --> dse                  --> dse_result --> exploitability_report
```

## Compiler flow (intended)

```
linalg / tensor / scf
  -> contract     (facts, obligations, legality)
  -> schedule     (chosen compiler decisions)
  -> interface    (target-independent HW/SW abstractions)
  -> <target dialect>    (e.g. toynpu)
  -> runtime      (command buffers, dispatch, waits, profiling)
  -> binary / simulator / external runner
```

## Principles

- All Merlin source lives under `src/`; logical compiler/research boundaries and optional dependencies remain explicit.
- Experiment definitions have one catalog; generated state belongs under `out/{runs,artifacts,build}`.
- Target-specific support belongs OOT where qualified; retained reference resources are not
  proof that an exported package is a standalone compiler.
- Coordinate through schemas, not prose.
- Integrations are adapters, never vendored repos.
- Prototype in xDSL; promote to MLIR/C++ only when stable.
- A concept becomes a dialect only when it must survive passes, be verified/transformed, and
  lower. Otherwise it is a schema/YAML artifact first.
