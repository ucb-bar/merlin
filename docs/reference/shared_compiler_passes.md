---
title: Shared compiler pass contracts
kind: reference
status: current
owner: core
last_verified: 2026-10-18
related: [../design/shared_phase1_scheduling.md]
code_refs:
  - src/merlin/transforms/pipeline.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/compile/scheduling/export.py
  - src/merlin/compile/scheduling/library_sources.py
  - src/merlin/compile/scheduling/compact_dense.py
  - src/merlin/compile/scheduling/compact_verification.py
  - src/merlin/compile/scheduling/program_limits.py
  - src/merlin/compile/scheduling/program_contracts.py
  - src/merlin/compile/scheduling/temporary_storage.py
  - src/merlin/compile/scheduling/device_spills.py
  - src/merlin/compile/scheduling/spill_conformance.py
  - src/merlin/compile/scheduling/spill_contracts.py
  - src/merlin/compile/scheduling/copy_elimination.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - src/merlin/compile/scheduling/fusion.py
  - src/merlin/compile/scheduling/fusion_semantics.py
  - src/merlin/compile/scheduling/row_softmax.py
  - src/merlin/compile/scheduling/row_softmax_verification.py
  - src/merlin/compile/scheduling/row_contracts.py
  - src/merlin/llvmlower/cpu_softmax.py
  - src/merlin/llvmlower/automatic_host.py
  - src/merlin/llvmlower/automatic_destinations.py
  - src/merlin/llvmlower/automatic_rounding.py
  - src/merlin/llvmlower/destination_check.py
  - src/merlin/llvmlower/quant_round_check.py
  - src/merlin/compile/scheduling/configuration.py
  - src/merlin/transforms/row_normalization.py
  - src/merlin/transforms/row_normalization_check.py
  - src/merlin/llvmlower/automatic_rows.py
  - src/merlin/model_check/full_build.py
  - src/merlin/llvmlower/codegen_env.py
  - src/merlin/llvmlower/c_runtime.py
  - src/merlin/llvmlower/weight_payload.py
  - src/merlin/llvmlower/blob_weights.py
  - src/merlin/llvmlower/entry_weight_projection.py
  - src/merlin/sched/ir/integer_domain.py
  - src/merlin/compile/scheduling/selected_contracts.py
  - src/merlin/compile/scheduling/dense_contracts.py
  - src/merlin/compile/scheduling/storage_contracts.py
  - src/merlin/compile/scheduling/fusion_contracts.py
  - src/merlin/semantic_compiler/allocation_types.py
  - src/merlin/semantic_compiler/allocate.py
  - src/merlin/verify/lean_resource_geometry.py
---

# Shared compiler pass contracts

The ordinary captured-policy host preparation path automatically legalizes
eligible bounded quantization arithmetic. Compact dense scheduling is a shared
compiler entry point over the existing counted-loop kernel IR. Its encoder and
actual-operand decoder are explicit integration inputs. It has no automatic
interpretation of an opaque backend program.

Ordinary serial host lowering also schedules eligible closed source row
normalization islands. This source-to-source route is separate from the explicit
hardware counted-row entry: it preserves the source's own scalar operations and
continues through the existing CPU toolchain. See the scope below.

After destination selection, eligible bounded integer rounding uses a compact
standard LLVM representation with independent arithmetic reconstruction. See
[host integer rounding](host_integer_rounding.md) for source contracts, budgets,
fallback and qualification scope.

## Abstractions and the research question

Phase 0/1/2 describe a procedure. The interfaces below describe what compiler
authors and search agents can reason about, and what the shared compiler checks.
They extend existing Merlin representations rather than introduce one universal
IR for both source computations and arbitrary target instructions.

| Abstraction | Concrete interface | What becomes reusable | A falsifiable check |
| --- | --- | --- | --- |
| Typed computation and numerical boundaries | Existing dense/Scheduling IR, `ElementwiseSemantics`, and checked source scalar regions | Choose blocking or composition from shapes, layouts and arithmetic instead of operation names | Preserve the complete reduction and conversions; reject a changed mask coordinate or accumulator order |
| Physical resources and execution effects | `DenseResources`, `StorageBank`, `TemporaryObject`, `StorageEffects` and `ConfigurationEffects` | Use the same reuse, allocation and state-cleanup algorithms with different capacities, separate memories and completion behavior | Reject reuse before a pending transfer completes; retain setup after an unknown state change |
| Analyses tied to their inputs | `PassSpec`, `PipelineContext`, `AnalysisStamp` and `PipelineSession` | Compose passes and bounded repairs without reusing facts from an earlier program or hardware selection | Reject stale allocation after a rewrite and invalid explicit pass order |
| Realization checked at the relevant IR level | `DenseEmission`, `ScheduleEncoding`, actual-byte decoders and the source-row inverse checker | Require evidence that the emitted transformation implements its selected alternative | Reject an encoder that restores removed setup, changes an address or omits part of a reduction |

Structural hardware facts and qualified execution semantics have different
authority. RTL can supply widths and capacities; the adapter must separately
establish persistence, completion, arithmetic and decoding. A contract identity
binds that evidence to an analysis; it does not prove the evidence correct.

Phase 1 can choose a deterministic legal alternative through these interfaces.
Phase 2 can search the same bounded alternatives with calibrated costs while
using the same legality and realization checks. A chosen schedule alone is not
a transferable compiler rule. A rule must consume the current operation and
capability inputs and make a checked decision on new inputs.

The research hypothesis is that these interfaces reduce the effort needed to
build a correct compiler and help optimization decisions transfer to unseen
workloads and hardware configurations. The existence of familiar passes does
not establish that hypothesis or its novelty. Evaluation must measure authoring
effort, correctness, held-out whole-model performance and target integration
effort, with and without the interfaces under the same access boundaries and
shared-library version. Calibrated timing proxies are one input to that
evaluation, not the compiler abstraction itself.

Executable examples include
`merlin/tests/infra/test_configuration_elimination.py` (persistent writes,
resets and launch-state changes),
`merlin/tests/infra/test_readout_model_fusion.py` (complete integer graphs and
unsupported physical compositions), and
`merlin/tests/ir/test_row_normalization.py` (complete rows, projected masks,
carried state and source mutations). Their neutral execution models test shared
mechanisms; they do not independently qualify a hardware adapter.

## Inputs and ownership

Select the original hardware configuration and numerical policy. Supply resource
and code-growth budgets, with optional optimization controls. Structural RTL
facts can establish dimensions, widths and capacity; they do not establish
instruction arithmetic, completion semantics or measured performance.

Target repositories retain their encodings, physical mapping, instruction
implementations and execution qualification. Merlin owns candidate search,
ordering, transformations and shared checks. A contract identity binds a chosen
input; naming an identity supplies no authority or missing hardware evidence.

## Ordering and analysis freshness

`PassSpec` names a stage, IR level, prerequisites, produced analyses and
invalidations. `validate_pass_order` rejects duplicate passes, missing producers,
stage inversions and incompatible IR levels. The common pipeline validates an
explicit `PipelineOptions.pass_order` before executing any transformation.
The default order normalizes semantics, performs narrow source buffer reuse,
segments or unrolls ordered loops, schedules dependencies, then recomputes and
checks final storage and dependency analyses.

The narrow source buffer rewrite runs before loop segmentation because its
initializer checker needs the original rectangular nest. Physical device
allocation needs the selected tile representation and belongs after tiling.

`PipelineSession` stamps analyses with the current IR digest, IR level and
selected hardware, numerical, resource, execution and library identities.
Any actual IR mutation invalidates old analyses, including a mutation whose
pass omitted an invalidation declaration. Changing a selected identity likewise
makes an old analysis unusable. A failed action poisons its private session.
An analysis pass that mutates IR is an error.

Selected-contract resolution, compact dense scheduling, finite schedule
compilation, row emission and final-program checks now use the same shared
source-version roster. It includes the block-operation definitions, scalar
admission and witness helpers, semantic value types and IR package/parser
owners. A change detected during a provider callback rejects that compilation;
a change after checking rejects later emission or contract reuse. Reselect and
replan with a stable library. Versions describe source bytes, not loaded-code
attestation, third-party dependencies or a complete transitive import closure.
This roster grants no candidate import access: the separate reviewed
compiler-library contract and existing authoring boundaries still apply.

Compact dense C emission binds the actual checked kernel, index width and
library through an immutable existing `AnalysisStamp`. Editing or deleting its
printable realization report cannot bypass those checks. Source checks run
before and after target C rendering; arbitrary C implementation equivalence
still requires independent qualification.

Ordinary whole-model image reuse also binds the raw `MERLIN_*` control values,
the Python sources in the host lowering and transform families, direct build
owners and selected target-input files. Only a digest of raw controls is stored;
the display snapshot's redaction and long-value summaries cannot establish cache
identity. Changing pass selection, a pass implementation or the selected target
invalidates an older image. A hit reopens the ELF and verifies its recorded hash,
and controls/source identities are rechecked before returning or publishing.
The native ABI implementation and the strict tree reader/selection owners are
also pinned.

`model_check.full_build.build` binds the current complete capture and host-package
trees, and the device-package tree when offload is enabled. It reuses
`model_execution_inputs.strict_tree_sha256`: filenames and every file's bytes
participate, including numerical policy, state, inputs and unknown artifacts.
Hashes stream file contents; weights are not copied or loaded into a second
in-memory snapshot. The caller's submission digest cannot replace reading the
actual package. Host twins need no submission files. Board catalog, host DTS,
explicit target contract/RTL files and an optional reused weights blob have
their own byte identities. Selected/scoped RTL and capability documents are
bound even without backing files; available provider contract and dialect-plan
files are pinned as metadata, without importing provider implementation code.
Read-only fact discovery never extracts RTL. Missing facts remain unknown;
facts appearing or changing during a build require a fresh attempt.

These inputs are read again before publishing a build or returning a hit.
Same-size changes with restored timestamps cannot reuse an older image. Missing
inputs, symlinks and input/cache containment produce a diagnostic before
compilation or reuse. Input-only relinking remains a separate path: it generates
the requested driver inputs and checks regenerated weights and argument calls.
The build retains a relink binding to its existing completed compilation recipe,
generated headers and shared blob/quantization/panel preparation files. Before
reusing those products, relinking reopens the stored build origin, current input
and compiler selections, actual recipe, recorded tools, sources, intermediate
objects, linker script and directly named archive bytes. Newly added preparation
files and headers also invalidate that binding. The ordinary build repairs stale
recorded products by rebuilding. A compiler without a completed recipe can still
produce an executable, but relinking it requires a precise missing-contract
diagnostic rather than executing unverified commands.

Relinking holds the existing build lock so a simultaneous ordinary rebuild
cannot remove its inputs. Its output must be disjoint from the original build,
cache and capture; symlink outputs refuse before deletion. The bound input
dictionary may change each invocation, including carried state, while the
original capture and build remain unchanged. Source and artifact checks bracket
generation, compilation and linking. Each relink emits its own existing-format
`compilation_recipe.json`, recording the actual driver and linker invocations,
generated input-header bytes, direct objects and final executable. Completed
recipes are independently reopened before returning. Failures return no admitted
executable; these receipts alone confer no numerical or hardware authority.

A cached device image is evaluated by the currently selected
`LinkedElfAdmissionService`; a prior acceptance never substitutes for today's
policy. Refusal stops reuse. Source pins, actual ELF bytes and the evaluator's
actual report are revalidated. A host twin issues no accelerator instructions
and does not require device admission. A device relink requires an explicitly
selected service, applies it to the newly linked image, and preserves its source
selection throughout the operation. The full-model checker forwards the caller's
service to every relink; it selects its existing structural service when none
was supplied. Fresh build `seconds` include input and
source validation; a hit reports its current `cache_lookup_seconds` separately
from the retained original build cost. Both exclude the outer file-lock wait.

Legacy records without these bindings rebuild. These checks do not establish
complete toolchain/header, provider-resource or transitive dependency closure,
nor validate a capture's numerical policy by hashing it. Recorded direct archives
are checked, but imported native/system headers and implicit linker/library
resolution still need independent closure. Rechecking complete mutable input
trees and direct tools/objects costs I/O; no stat-only shortcut or performance
gain is claimed. They confer no new
numerical or hardware execution qualification. The focused transaction tests
and real ordinary build/relink/functional-simulator state test run with:

```bash
python -B -m pytest merlin/tests/infra/test_full_model_lowering_cache.py \
  merlin/tests/infra/test_model_check_relink.py merlin/tests/infra/test_model_check_full.py \
  -p no:cacheprovider --basetemp "$TMPDIR/full-model-input-cache"
```

The real execution test skips when the selected native/cross toolchain and
functional simulator are unavailable. It checks every output and carried state
for three calls under each of two changed source programs, plus normal hardware
pin refusal; this small test is not qualification of a complete model corpus.

### Generated weight storage

Ordinary `c_runtime.generate` defaults to streamed weight assembly. It retains
the existing manifest/signature, dead-weight compaction, alias, descriptor and
entry-projection rules; their byte storage uses lazy file/array/zero spans.
Prepared layouts, lifted quantization operands and generated zero stubs occupy
the same offsets and padding as before. Complete shapes and dtypes are checked
when prepared arrays are consumed. Projection and compaction use their existing
algorithms with either byte storage representation, rather than a second packing
policy. State, input streams, numerical operations and runtime dispatch remain
under their existing contracts.

This is host compilation storage, independent of accelerator geometry. The
existing software ABI and weight declarations determine the bytes; the normal
board/window checks still determine whether the final program fits. The 1 MiB
I/O block is a compiler work-buffer budget, not an extracted device transfer
width. Prepared arrays are loaded individually, so peak array storage still
depends on the largest selected array. File-span source identity uses bounded
block hashes, checked when read and reopened after consumption; timestamps alone
never establish content identity. These checks add I/O. They do not establish a
filesystem snapshot or replace the ordinary capture/library freshness checks.

Hash-only generation does not assemble a complete blob in Python memory or
write a temporary weights file. Materializing generation writes bounded chunks,
then independently reopens the actual pending file to check its complete size
and hash before atomic publication. A failed array, source or actual-write check
preserves the previous blob. Explicit immutable `weights_source` reuse keeps
its existing full-content and hardlink checks. No mutable invocation state is
shared, and no cross-invocation hash or weight cache is introduced.

For reproducible comparisons through the same library and verification rules,
`MERLIN_WEIGHT_PAYLOAD_STORAGE=materialized` selects explicit full-payload
materialization; `streamed` is the default. Unknown values refuse before output.
This project control participates in the existing image/cache/recipe identities.
Both choices retain all byte checks, packing algorithms and numerical policy.
The control is an optional compiler-storage choice, not a target capability or
calibrated cost model. Lower compiler heap use alone establishes no model
execution or accelerator latency improvement.

```bash
python -B -m pytest merlin/tests/runtime/test_weight_payload.py \
  merlin/tests/runtime/test_entry_weight_projection.py \
  merlin/tests/runtime/test_blob_weights.py \
  merlin/tests/infra/test_full_model_lowering_cache.py \
  -p no:cacheprovider --basetemp "$TMPDIR/streamed-weights"
```

Shared pass-state types now live in `merlin.transforms.pipeline_state`.
`transforms.pipeline` reexports the identical objects and retains its existing
host verification broker and structured-transform gate. Newly generated
authoring profiles pin the moved state file as a private implementation
dependency without adding it to their public imports. Self-contained frozen
historical profiles retain their original inventories. Instruction declaration
types similarly live in `merlin.sched.isa.types`; installed `sched.isa` retains
its existing backend lookup bridge. Neither move duplicates an algorithm or
changes pass ordering, arithmetic or qualification requirements.

Counted-schedule signed-index interval checking lives in
`merlin.sched.ir.integer_domain`. Both row verification and C emission consume
the same algorithm, bounding intermediate sums and products without expanding
iterations. The emitter preserves its existing exception interface. The shared
checker requires no emitter, ISA loader, backend or numerical framework. An
isolated source-only test verifies one explicit import-closed library leaf with
the existing compiler-library contract.

### Reviewable shared scheduling library

Export all seven shared scheduling families with:

```sh
python -m merlin.compile.scheduling.export --shared-library --output-dir "$TMPDIR/shared-sdk"
```

The destination must be empty and unlinked. The bundle contains 57 declared
Python source owners plus `LICENSE` and `NOTICE`. All algorithm, checker and
numerical implementation files are byte-identical to the ordinary compiler.
Three explicitly recorded namespace initializers omit installed host
orchestration and backend discovery. The manifest records exact membership,
public leaf APIs, namespace replacements, observed external package versions
and a deterministic library digest. Source/resource changes during export
prevent completed manifest publication. Existing standalone dense exports keep
their original format.

From an installed source package, also pass `--license-directory` naming the
explicitly selected directory containing its `LICENSE` and `NOTICE`. Checkout
exports use their own project files. The writer does not search another
checkout or substitute an interpreter/environment license. Ordinary installed
Python source layout is required; zipped/source-free packages are unsupported.

Exporting does not grant authoring access. The host reviews the exact bundle and
selects its existing `CompilerLibraryContract` with an independent review ID,
explicit `public_modules`, the manifest's `(path, module)` member pairs, and
its `review_resources` as additional non-executable members. This pins
`shared-library.json` itself, including its version/dependency observations.
That contract rechecks bytes and direct Python import closure. Every treatment
must receive the same reviewed root, contract, dependency environment and
verification rules. Candidate manifests cannot grant themselves access; no
historical authoring/evaluation grants are widened by this exporter.

The exported entry points are the existing `schedule_compact_dense`,
`compile_schedule`, `compile_row_softmax` and `read_scheduling_contracts` APIs.
They consume current typed computation, selected RTL/software contracts,
numerical policy, resource/code budgets and independently qualified target
adapters. The bundle includes no target implementation, golden outputs,
protected reference compiler, workload schedule or calibration answers.
Unsupported effects, mappings and numerical algorithms retain the existing
fallback/refusal behavior. Backend discovery, whole-model host orchestration,
formal-verifier authority and optional native e-graph build resources are not
public SDK services. External version observations do not qualify dependency
implementations or hardware, and the bundle is not a Python sandbox.

The export regression runs ten cases covering all seven families in an isolated
`python -I -S` process. Only the bundle, selected third-party roots and neutral
test fixtures are available; every imported Merlin module and package path is
checked to originate in the bundle. Existing ordinary entry points emit actual
C/encoded programs, execute independent native primitives, and check complete
outputs/state, rectangular tiling, widths, separate/aliased storage, actual
completion, unsupported async emission and missing drains. These are portable
mechanism tests, not qualification of a real accelerator or complete model.

`repair(stage=..., reason=...)` clears analyses and restarts an already selected
legal fallback within an explicit repair budget. It cannot revive a failed
session, manufacture a fallback or approve an unchecked change. Existing covered
source transformations retain their host verification hooks. `plan_only=True`
remains explicitly unverified.

## Automatic bounded host arithmetic

`prepare_for_lowering` invokes `prepare_automatic_host` when preparation selected
the existing `merlin.quantization_policy.v1` captured policy. Missing policy and
explicit pass ablation preserve the original route. An explicitly selected
alternative rounding lowering keeps ownership of that choice.
The policy's `prepared_mlir_sha256` must bind the exact current input bytes.
A missing or stale binding retains the legal prepared input and writes a
diagnostic instead of applying a policy from another preparation stage.

The supported rewrite is a pure, straight-line tensor `linalg.generic` carrying
an admitted round-to-even, integral clamp and signed integer conversion chain.
The existing matcher checks bounds, widths, uses and safe scale construction.
No approximation, reduction reassociation or numerical admission threshold is
introduced. Unknown scalar effects and unsupported bodies are barriers.
Added operations are bounded; the default budget is 4096.
Buffer generics, unknown tensor encodings and programs with CFG control flow
retain the unchanged route. Unchanged IR is checked structurally without
requiring the arithmetic checker to interpret that control flow.

The independent checker reconstructs the actual scalar output expressions and
checks unchanged surrounding interfaces, attributes, effects and dataflow.
The `automatic_host/report.json` artifact records source/output identities,
selected policy, library digest, actual operation and rounding-site counts,
refusals and checks. This is a conditional arithmetic check at tensor IR;
machine equivalence and hardware support remain separate obligations.
The receipt explicitly reports that this conditional floating-point check is
outside the existing covered integer transformation proof domains.

Use the ordinary optional-pass selector `automatic-host-numerics` to ablate
the rewrite in a paired experiment. No tuning is necessary to enable the
ordinary captured-policy route.
`MERLIN_AUTOMATIC_HOST_NUMERICS=0` also disables it. An explicit optional-pass
selection takes precedence over that environment fallback. Receipts include
the transformation, checker, matcher and pipeline source identities.

## Compact dense loops and operand retention

`schedule_compact_dense` extends `transforms.dense_planning`; it does not expand
the tensor or enumerate dynamic tiles. The existing bounded M/N search chooses
one largest legal K panel per candidate. A physical `k_limit(m, n)` narrows that
panel for separate operand stores, padding and available row windows before
ranking. Unknown costs use the documented traffic-first ranking, without cycle
estimates.

Inputs are a typed `DenseProblem`, `DenseResources`, assigned `DenseLoopStorage`
views and a `DenseLoopAdapter`. Each view declares physical backing, row byte
width, assigned base, usable capacity and alignment. The usable assignment must
already exclude packing scratch, DMA staging and other reservations. The three
live roles may use separate memories or disjoint windows in one backing.

The supported family requires explicit external-buffer disjointness, addressable
accumulator storage, synchronous completion and exact accumulator continuation.
It supports rectangular primitives, different element widths, both M/N outer
orders, peeled panel tails and primitive tails. The first K compute initializes
each output; subsequent computes continue its accumulator in source K order.
Bias seeds, gathers, asynchronous completion and unqualified numerical modes
retain their existing route or require a precise diagnostic.

The emitted program uses the existing `sched.ir.Kernel`, `Loop` and `Call`
objects. M/N output panels retain complete accumulators while successive K
panels load each operand once for its consumers. Partial panels are peeled;
tail dimensions and addresses remain parameterized expressions. Static call
sites are budgeted independently of executed call counts.

The adapter also selects a signed 32- or 64-bit loop index representation.
`CompactDenseResult.emit_c` uses that width through the existing C emitter.
Integer literals and select branches carry the same width, and conservative
intermediate-expression bounds reject possible arithmetic overflow without
expanding dynamic loops. This does not establish instruction argument widths,
physical address windows or pointer ownership; those remain target/runtime
obligations. Omitting the new width option on the existing emitter preserves
its historical emitted text.

`verify_compact_dense` projects actual calls, then uses bounded symbolic domains
to establish complete contraction/output coverage, valid tails and source
coordinates, initialization, source reduction order, physical residency and
absence of intervening overwrites. It checks padded high-water marks against
the current capacities and reconstructs executed commands and requested payload
bytes from actual operands and loop multiplicities. Unknown execution or
configuration effects refuse. No dynamic loop is unrolled by the checker.

The checker requires `z3-solver` and an explicit time budget. One bounded repair
may replace an unproved candidate with a simpler streaming loop and recheck its
actual output. Concrete invariant failures never trigger repair. Missing tools
or an unproved final alternative produce a diagnostic. If no candidate fits,
the scheduling entry returns `None` so the compiler can retain its legal
original route.

The adapter must independently decode actual instruction operands and establish
the selected instruction semantics. Reusing retained requests as a decoder is
not evidence. The shared receipt explicitly leaves final machine emission and
execution qualification unestablished. Tests compile the actual loop program
through the existing C emitter using neutral instruction implementations,
compare complete results to independent ordered arithmetic, and compare executed
commands and byte requests with reconstructed observations. These tests qualify
the shared mechanism's stated scope, not a hardware target.

Existing regular target integrations do not expose this new counted-loop
encoder/decoder contract. They keep their existing legal lowering. Integration
requires that public semantic boundary, actual addressing and tail legality,
completion/continuation qualification, and final emitted-effect checks. It does
not require changing a target's instruction implementation to fit Merlin.

### Selected RTL and software inputs for dense scheduling

The ordinary shared entry now accepts `selected_contracts=selection`, using the
same two pinned documents as configuration cleanup. The optional `dense` section
of `merlin.schedule_effects.v1` has schema `merlin.dense_schedule.v1`:

| Section | Required contents | Owner of the information |
| --- | --- | --- |
| `primitive` | `m`, `k`, `n`, each `{fact: [structural, path]}` | RTL geometry |
| `formats` | `lhs`, `rhs`, `accumulator`, each with `dtype` and fact-backed `bits` | Software format interpretation, RTL widths |
| `storage` | Those three roles, each with `resource`, canonical `backing`, fact-backed `row_bytes`, `origin_bytes`, `alignment_rows` | ISA physical mapping and RTL bounds |
| `external` | `lhs`, `rhs`, `output` ISA byte-memory resource names | Software memory interface |
| `instructions` | `lhs_load`, `rhs_load`, `preload`, `compute`, `store`, each naming its declared ISA instruction | Qualified software operation mapping |
| `semantics` | `layout`, `arithmetic`, `reduction`, `initialization`, `continuation`, `completion`, `accumulator`, `ownership` | Qualified software/runtime behavior |
| `support` | Positive `max_m_blocks`, `max_k_blocks`, `max_n_blocks`; Boolean `partial_output_tiles`, `partial_output_blocks`, `partial_reduction_tiles` | Supported software instruction domain |

For this family, semantics must explicitly select `row_major`,
`signed_integer_mac_modular`, `source_order`, `zero`, `exact`, `synchronous`,
`addressable`, and `invocation_private`, respectively. Each MAC updates the
signed accumulator modulo its declared bit width; no saturation, rounding,
reassociation or intermediate precision change is admitted. The family supports
signed i8/i16/i32 formats whose widths agree with RTL. Unsupported semantics,
including unknown/asynchronous completion, diagnose rather than silently using
a synchronous schedule. This declaration does not validate the original
operation's numerical admission or qualify a target implementation.

Capacities come from the selected ISA's memory resources and their RTL fact
references; units must be bytes. Coordinates in each assigned view use the
existing absolute `base * unit_bytes` mapping into its canonical backing.
`origin_bytes` bounds the resource's interval in that same coordinate system;
it is not an additional offset added by emission. Fact-backed row widths include
physical padding. Different views of one ISA resource cannot disguise its
backing or origin. Distinct resources may explicitly alias one backing.
Unknown capacities, unresolved references, contradictory widths and misaligned
or out-of-window assignments diagnose. Canonical alias mappings still require
independent target/runtime qualification.

The allocation owner supplies already assigned `DenseLoopStorage` windows,
excluding other live state. Optional `reservations` are existing `AddressWindow`
objects in canonical backing bytes; overlapping packing, DMA or other live
reservations refuse. The scheduler derives operand/accumulator byte budgets
from those windows and narrows optional `DenseResources` search controls to the
selected limits. It never allocates an entire local memory merely because its
RTL capacity is known. Simultaneously live views must remain disjoint, even in
one SRAM. The runtime owner must establish external-buffer disjointness and
invocation-private ownership; document labels and identities alone provide no
such evidence.

For example, a lowering integration with checked assignments calls:

```python
selection = read_scheduling_contracts(facts_path, software_path, target=target)
problem = selection.dense.problem(m, k, n)  # Computation extents, RTL primitive.
result = schedule_compact_dense(
    problem, None, assigned_storage, qualified_adapter,
    name=operation_name, dtypes=selection.dense.dtypes,
    static_budget=code_growth_budget, external_buffers_disjoint=True,
    selected_contracts=selection, reservations=live_reservations,
)
if result is not None:
    source = result.emit_c(instruction_set, symbol=operation_name)
```

The adapter must match the selected hardware/numerical identities and carry an
independently selected `qualification_identity`. The contract maps actual
instruction names to projected dense operations. Known ISA memory effects must
exactly match their operation roles, allowing explicitly declared execution
resources. Extra persistent-state effects refuse. This initial logical-coordinate
binding requires integer ISA arguments with one full-width encoded field each,
unit alignment and widths at least the checked C index width. Actual argument
names must match. It conservatively declines narrower encodings even if a
particular shape might fit them. Public target mapping extensions can later
establish those narrower ranges without weakening the check.

Dense compilation now uses the existing `PipelineSession`: admission precedes
blocking, emission invalidates earlier analyses, and realization checks the
emitted loop IR. Invalid explicit `pass_order` is rejected before execution.
A bounded checking-budget repair invalidates analyses and repeats all stages
with the simpler block. The receipt binds current computation, storage,
resources, numerical policy, execution qualification identity, shared source
hashes and both selected input byte hashes. Files and shared implementation
hashes are checked around every stage. `emit_c` rechecks the selected files,
verified kernel digest and implementation hashes before and after rendering.
These checks do not decode arbitrary native machine code or qualify provider
C instruction renderers; those remain explicit integration obligations.

`test_selected_dense_contracts.py` runs paired baseline/optimized emitted C for
three synthetic configurations: 2x3x4 and 3x2x5 primitives, 8/16-bit inputs,
32-bit accumulators, padded separate memories and unequal-width views in one
shared backing. A signed 16-bit case forces accumulator wraparound and compares
against independent exact Python arithmetic with explicit modular reduction.
The same programs also run inside a complete neutral host graph: transpose
packing, dense dispatch, host bias/state arithmetic and position/state updates
over two invocations. Every output, carried state, position, input and guard is
checked. A falsely qualified native implementation that resets each reduction
tile fails the independent reference, even though its projected loop is valid.
Actual command/load counters match reconstructed loop effects. Host
packing and postprocessing counts are reported separately and remain unchanged.
This is a neutral graph experiment, not a real full-model or hardware result.

Reproduce the paired records without downloads:

```sh
python -B -m pytest merlin/tests/infra/test_selected_dense_contracts.py \
  -p no:cacheprovider --basetemp "$TMPDIR/selected-dense" \
  -o junit_family=xunit1 --junitxml "$TMPDIR/selected-dense-results.xml"
```

JUnit properties record scheduling times, per-arm C source bytes, joint native
library size, actual commands, requested load bytes, host packing traffic,
host arithmetic and dispatch counts, and whole neutral graph host-call wall
samples. Native build time is joint; peak compilation memory and accelerator
timing remain unmeasured. Samples include ABI overhead and are insufficient for
a speedup claim. Smaller requested traffic does not establish latency savings.

Existing target providers still need the fact-pinned dense declaration, checked
allocation windows/reservations, qualified logical-operation projection and
source-to-final-program conformance. Ordinary `lower_model` target routing has
not been connected to this selected dense family: it retains its existing
backend route. The integration must be supplied through public target hooks;
no target source or protected reference compiler is imported by this mechanism.

## Executable schedule pipeline

`compile_schedule` connects the existing storage passes, effect checks and final
program checks through `PipelineSession`. Its finite straight-line family uses
the existing `sched.ir.Kernel`, without introducing another schedule IR or pass
manager. The session accepts explicit fingerprint and structural-verifier
callbacks for that representation; xDSL callers retain the original defaults.
Fingerprints, IR level, hardware, numerical policy, resource declarations,
execution contracts and shared-library source identity bind every analysis.
The callbacks establish structural custody, not numerical proof authority.

The default schedule sequence is `semantic_admission`, `storage`, `completion`,
`emission`, then `final_program`. Explicit ordering must contain all selected
passes exactly once and satisfy stages and prerequisites before a target
callback runs. Storage rewrites invalidate all earlier facts and produce newly
checked realization. Failed copy allocation and inserted spills call the same
session's bounded repair mechanism; both consume `ScheduleBudgets.max_repairs`.
Repair clears analyses before allocation and completion are recomputed.
Instruction changes invalidate existing emission and size facts even when a
pass omits an explicit invalidation declaration. Failed actions poison their
private session and return no compiled result.

Selecting `fusion_capabilities` inserts `fusion` between semantic admission and
storage. It constructs checked physical candidates privately; storage selects a
candidate only after allocation succeeds. Fusion allocation fallback consumes
the same repair budget as spills and copy fallback.

Selecting `configuration_adapter` inserts checked configuration cleanup after
admission and before fusion/storage. All later candidates consume its actual
rewritten calls. The default removes redundant pure writes; the paired control
is `ScheduleBudgets(eliminate_configuration=False)`. Configuration, fusion and
storage share the same current hardware identity, final emission and byte checks.

Replacing an analysis also invalidates its consumers transitively, even if the
owning IR is unchanged. Analyses produced together share invalidation: replacing
an encoded byte image cannot leave its old size or conformance result current.
Both explicit-order validation and runtime custody enforce these dependencies.

The target supplies a `ScheduleEncoding` with an encoder, actual-byte call
decoder, `ProgramDecoder`, numerical-policy identity and independent
qualification identity. Configuration and policy must match the selected
`StorageAdapter`. One actual decoded instruction corresponds to one typed call
in this family. The shared pass compares all actual decoded accesses and
completion behavior across both decoders and requires the complete numerical
and memory-effect sequence to match the verified schedule. The decoder cannot
substitute retained requests for actual operands. Independent qualification
must establish each decoder's interpretation, all state effects and executable
instruction semantics; the shared checks cannot establish that a lying decoder
describes its target correctly.

Final size, instruction count, address and branch checks run on the encoded
bytes after storage rewrites and spill insertion. Segmentation requires the
existing qualified continuation contract. Every declared external window is
conservatively treated as published, including inputs, aliases and carried
state. Each final segment is decoded again for numerical and memory conformance,
as well as the existing final-program checks. Receipts retain actual byte
hashes, final counts, requested transfer bytes, pass stamps, selected parameters
and shared source hashes. Timing remains unknown without paired timing evidence.

For an integration that supplies those public contracts:

```python
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule

result = compile_schedule(
    source_kernel, private_objects, selected_banks, storage_adapter,
    selected_encoding, selected_program_limits,
    external_windows=selected_runtime_windows,
    private_ownership=True,
    budgets=ScheduleBudgets(max_repairs=2),
)
checked_segments = result.program.segments
```

These declarations come from selected capabilities and the compiler's ownership
analysis, rather than inferred target names. Unsupported effects, counted loops,
asynchronous instructions, missing ownership and infeasible allocation produce
a diagnostic for the caller's existing legal lowering route. Counted dense
loops and asynchronous storage allocation retain their separately checked
interfaces; this finite encoded family does not silently claim their semantics.
Native C and byte-interpreter tests qualify shared mechanisms only. Existing
full-model target providers still need this public encoder/semantic-decoder
boundary before the schedule pipeline can enter their ordinary model lowering.
This entry is not evidence of that missing target integration or hardware
execution qualification.

## Redundant configuration writes

The ordinary `compile_schedule` entry can now consume both selected documents
through `read_scheduling_contracts(facts_path, software_path, target=...)`.
It reuses `targetgen.isa_spec.resolve_isa`: encoded fields, integer widths and
state capacities resolve from the software ISA's references into the exact RTL
facts bytes. The facts pin and explicit target must match. RTL extraction is
not repeated here and synthetic facts remain synthetic.

The selected `merlin.software_spec.v1` document must have reviewed status and an
ISA declaration. Its additional scheduling section describes canonical state
effects, separately from computation admission or numerical reference engines:

```yaml
scheduling:
  schema: merlin.schedule_effects.v1
  numerical_policy: selected-policy-identity
  instruction_byteorder: little
  configuration:
    declared_instruction_name:
      ordered: true
      pure_write: true
      reads: []
      writes:
        - resource: declared_state_resource
          index: {argument: encoded_field_index}
          value: {argument: encoded_value}
      clobbers: []
```

Every resource must exist in the ISA as state with unit `fields`. Capacity,
when known, is the number of canonical fields and must resolve from an RTL
fact reference. Index/value bindings name integer ISA arguments or use
`{constant: integer}`. Reads require the stated value; clobbers forget a field
without assuming its reset value. State-resource read/write coverage must agree
with the ISA's declared resource effects. A pure setter cannot also read state,
clobber, touch memory or have unordered visibility. The projection checks actual
operand encoding ranges and actual field indices, not instruction spellings or
retained planning annotations. Providers must establish canonical mapping of
physical fields and aliases and complete arithmetic/execution effects; the
loader cannot infer those properties from a resource name.

This initial realization family has fixed-width byte-aligned instructions with
one decoded schedule call per word. Byte order is an explicit software-interface
property. Final verification independently decodes actual words using the
fact-bound ISA and compares all instruction and operand bindings against the
provider decoder. A provider cannot hide changed bytes by returning the intended
register values. Variable-width, packed-bit and opaque framing require their
own qualified integration and are not interpreted by this family.

Pass `selected_contracts=selection` and an independently selected
`configuration_qualification_identity` to `compile_schedule`. It builds and
consumes the configuration adapter automatically. Storage/encoding adapters must
match `selection.hardware_identity` and its numerical policy. This new route
does not accept a second arbitrary configuration callback. Without qualification
it retains the original configuration stream and records
`unavailable_missing_qualification`. Unknown capacities, undeclared instructions
and unknown ordering are conservative barriers. Contradictory declarations,
out-of-range fields or stale input bytes produce diagnostics.

Both selected file identities and the resolved ISA identity enter the existing
pipeline context and report. Input bytes are reopened before and after each
pass, so a changed RTL file or software specification cannot authorize a later
analysis or a successfully returned image. Derived fields are computed from
immutable input bytes. Source hashes for the shared loader and ISA resolver/codec
join the common library identity in every treatment. A review label or a
qualification identity does not establish execution qualification on its own.

`test_selected_schedule_contracts.py` exercises this path with synthetic
fact-bound 32-bit little-endian and 64-bit big-endian operands, capacity and
ordering changes, unknown/reset/nonidempotent instructions and changed final
bytes. Independent native execution checks complete output guards, inputs and
final state. False pulse-on-set semantics fail the independent comparison.
JUnit records scheduling compilation time, actual encoded size, executed
commands and requested payload bytes; memory remains unmeasured and emulator
call time is not accelerator latency. These are whole transfer-program tests,
not full-model or real-hardware integration results.

Existing target providers still need the two pinned input documents, qualified
canonical state semantics and actual operand/final-byte decoding before their
ordinary full-model route can select this extension. They keep their existing
legal route until then; this work makes no target repository changes.

`eliminate_redundant_configuration` uses existing `sched.ir.Kernel` calls and a
selected `ConfigurationAdapter` that decodes actual operands into integer field
reads, writes, clobbers and qualified ordering. A pure write must be independently
qualified as an idempotent persistent-state update with no other effect. The
existing `sets`/`assumes` annotations and instruction names alone grant no deletion.
RTL can identify fields and widths; it does not establish this software-interface
contract. Queue commands, pulses, status acknowledgements and fences are not pure
configuration writes.

Each invocation starts with unknown state. The first write stays. A later pure
instruction is omitted only when every written field already has its exact value.
A partial match retains the entire instruction. Known clobbers invalidate their
fields; unknown operations or ordering invalidate all fields and remain in place.
The pass neither moves instructions nor assumes persistent state between calls.
Loops retain their existing lowering or diagnose; no dynamic expansion is used.

`verify_configuration_rewrite` reconstructs omissions from actual retained calls,
checks identical surrounding instructions/interfaces and proves the same known
state before every retained operation and at return. It accepts no removed-call
witness. In `compile_schedule`, the storage decoder must additionally agree that
each pure setter has no memory or asynchronous effect. Final actual-byte decoding
independently projects configuration effects and rejects reintroduced setup or
altered fields. Explicit ordering requires configuration before fusion/storage,
and repair uses the checked configured input. Completion checks rederive required
register values after fusion, relocation and inserted spills. An exact memory
transfer is not assumed to preserve registers.

If final size checks split the program into launches, the existing `Continuation`
contract must explicitly list every affected field in `preserved_configuration`.
Preserving only memory spaces is insufficient. Unknown configuration effects or
ordering refuse segmentation. Native negative controls demonstrate incorrect
results when registers reset between otherwise memory-preserving launches.

Native paired tests execute actual little-endian 32-bit-field and big-endian
64-bit-field encodings, compare complete outputs and final register state, and
observe commands and requested payload bytes independently. Unknown visibility,
resets, whole-register writes, source misconfiguration, allocation fallback,
corrupted actual output and a falsely declared pulse are covered. These tests
qualify the shared mechanism, not a hardware adapter. No target provider currently
supplies this new boundary automatically. No latency improvement is claimed.

## Private temporary storage

`allocate_temporary_storage` extends the existing semantic graph allocator with
an executable relocation pass over finite straight-line `sched.ir.Kernel`
programs. Each `TemporaryObject` declares one invocation-private backing object,
its virtual source interval, selected byte-addressed bank and alignment. All
typed views of that object share its backing and byte offsets. Separately named
banks can alias one physical memory; distinct physical memories retain their
own capacities. Existing `Reservation` intervals exclude packing scratch, DMA
staging and other unavailable storage across every alias of the backing.

The target's selected `StorageAdapter` decodes actual memory operands,
nonaddress semantics and executable completion instructions. It also rewrites
addresses into actual calls. Every local access must resolve to exactly one
private object, and external accesses need explicit fixed windows. Private
ownership and disjointness from external buffers are required. Complete initial
writes must precede reads, including reads through aliases; a read-modify-write
instruction cannot establish initialization by listing its write first.

Lifetimes extend through completion of actual asynchronous readers and writers.
A token annotation without an executable decoded wait is insufficient. Pending
conflicts, duplicate or missing tokens and a return before completion refuse.
The derived completion events reuse the existing allocation graph and its
independent checker; event positions are ordering facts, not estimated cycles.
Objects read and written by the same instruction remain simultaneously live
unless a different separately qualified transformation explicitly permits reuse.

The pass consumes selected addresses in the emitted calls. Its independent
checker derives assignments again from actual output operands, checks alias
agreement and current reservations/capacities, and compares actual nonaddress
semantics, access extents, external addresses and completion. Native C tests
execute baseline and relocated programs with complete byte comparisons and
independent command/traffic counters, including deferred reads completed by
actual waits. Storage footprint reduction does not imply a traffic reduction.

Infeasible allocation returns `None` for the compiler's existing legal route.
Unknown effects, loops, escaped or persistent storage, nonbyte address mappings
and unsupported completion produce a diagnostic. The entry does not perform
CPU register allocation or infer a target's physical mapping. Device spill
insertion requires the additional exact capabilities described below.
Existing targets need the public effect/address-rewrite adapter and
independent execution qualification before this pass can enter their ordinary
lowering. That integration remains incomplete.

### Selected resource windows and actual storage effects

The same selected RTL/software pair can now drive temporary allocation through
ordinary `compile_schedule`. Add a `storage` section to
`merlin.schedule_effects.v1`, using schema
`merlin.temporary_storage_contract.v1`:

| Section | Required contents | Source of the information |
| --- | --- | --- |
| `banks` | ISA resource name to `backing`, fact-backed `start`, `size`, `alignment`, and `ownership` | RTL geometry plus qualified software physical mapping/ownership |
| `external` | ISA resource name to `backing`, fact-backed `start`, `size`, and `ownership` | Declared runtime address windows and persistence |
| `reservations` | List of `resource`, fact-backed `start`, `size`, and nonempty `purpose` | Packing, DMA staging, persistent or other unavailable intervals |
| `instructions` | ISA instruction name to complete byte accesses, completion, token/waits and exact-copy declaration | Qualified software/ISA effects |

Every memory resource must exist in the selected ISA with unit `bytes`.
Capacities use that ISA's RTL references. Every window must fit the resource's
known capacity. Unknown local capacities diagnose. Coordinates are canonical
backing bytes: `start` is an inclusive lower bound and `start + size` an exclusive
upper bound. This initial family does not translate differently based physical
address spaces or infer units from instruction names.

The existing `StorageBank` now supports an optional `start`, defaulting to zero;
`capacity` remains the exclusive upper address bound. Zero-origin serialized
records retain their previous form. The solver, independent assignment checker,
fixed input/output checks, reservations, temporary alias reconstruction and
spill/copy conformance honor both bounds. A typed alias can have its own lower
bound without excluding those bytes from a wider view. The existing Lean
geometry projection expresses nonzero lower bounds through its existing
translated interval obligations, retaining alignment checks in the original
coordinates. This adds no physical mapping theorem or execution qualification.

Private banks must declare `invocation_private`, and the runtime/lowering owner
must separately supply `private_ownership=True`. A software label cannot
establish actual ownership. External resources must declare `runtime_managed`
or `persistent`, remain fixed and use backings disjoint from private storage.
Distinct external aliases require a qualified canonical projection first;
this family refuses them rather than treating aliased state as independent.
Within one external resource, overlapping fixed accesses retain their original
order. Weights, positions and cache/state updates are not hoisted or relocated.

An instruction declaration has this shape, with names selected as data:

```yaml
accesses:
  - space: {resource: declared_input}
    start: {argument: source_byte_address}
    size: {argument: byte_count}
    writes: false
  - space: {resource: declared_local_view}
    start: {argument: destination_byte_address}
    size: {argument: byte_count}
    writes: true
completion: synchronous
token: null
waits: []
exact_copy: true
```

Access starts/sizes bind an encoded integer argument or `{constant: integer}`.
A typed ISA address argument is admitted for a byte start when its address space
matches the declared resource. Its declared argument alignment and encoding
range are checked by the existing ISA codec. Local starts must be relocatable
arguments. Element-count scaling, strided transfers and opaque pointer framing
need an explicit compatible projection; no multiplier or address conversion is
invented. A memory selector can instead use
`{argument: selector_name, choices: [{value: {fact: [path]}, resource: resource_name}]}`.
Unlisted selector values diagnose. Reads/writes must cover exactly the ISA's
memory effects, with explicit ISA operation semantics. Hidden persistent-state
effects are unsupported in this family.

Asynchronous declarations use `completion: asynchronous`, one numeric `token`
binding and empty `waits`; pure completion instructions use `completion: wait`,
`token: null`, no accesses and numeric `waits` bindings. Those tokens must be
reconstructed from actual instruction operands. Unknown instructions/completion
are barriers. A selected provider's projected accesses, completion, token/waits
and exact-copy claim must agree with the declaration. Actual encoded argument
ranges are checked. A semantic fingerprint retains the ISA operation semantics,
all nonaddress arguments and the provider's nonaddress interpretation. An exact
copy cannot carry additional ISA execution/state effects, and an instruction
with extra execution effects cannot be treated as pure elementwise arithmetic.
Arbitrary
opaque implementation effects are not inferred from that fingerprint.

Use the allocation owner's current objects and runtime premises:

```python
result = compile_schedule(
    kernel, private_objects, None, qualified_storage_adapter,
    qualified_encoding, program_limits,
    selected_contracts=selection,
    private_ownership=runtime_established_private_ownership,
    reservations=packing_and_staging_reservations,
    external_windows=runtime_windows,
)
```

With `banks=None`, geometry is derived automatically. Explicit banks may only
narrow selected windows and strengthen alignment, preserving physical aliases.
Omitting `external_windows` derives the selected fixed windows; supplied runtime
windows must be contained in them. Declared reservations and current runtime
reservations are unioned across canonical aliases, preventing duplicate charges
while preserving each interval's view bounds. Runtime owners still establish
actual invocation ownership and allocation validity; RTL facts alone do not.
Program-size/branch limits remain independently selected inputs at this stage.

`ScheduleBudgets(optimize_storage=False)` retains and independently checks the
original addresses, without relocation, private-copy elimination or spills.
It is a legal baseline only when those original assignments fit the selected
windows and reservations. Otherwise it diagnoses; virtual out-of-capacity
addresses are not a runnable unoptimized route. The enabled arm uses the
existing allocator, completion checks and bounded repairs. Shared implementation
hashes and selected file bytes are revalidated around every schedule pass.

The ordinary final-byte pipeline currently requires synchronous instructions.
Selected asynchronous effects can enter the standalone allocator and its native
completion tests; ordinary encoded compilation rejects them until final-byte
and launch completion support is qualified. No metadata annotation substitutes
for the actual wait. This remains an explicit integration gap.

`test_selected_storage_contracts.py` exercises actual final bytes on independent
neutral native machines with 32-bit big-endian and 64-bit little-endian integer
operands, different alignments, shared aliases, separate memories, nonzero
windows, tails, packing/DMA reservations and deferred completion. Complete
transfer/arithmetic graphs preserve all output bytes, input guards and carried
state over three invocations. A false physical-alias declaration passes projected
checks but fails independent execution, demonstrating the qualification boundary.
Mutations also cover capacities, selectors, ownership, missing initialization,
waits, lying projections, rewritten addresses, final words and stale inputs.

Reproduce the paired records:

```sh
python -B -m pytest merlin/tests/infra/test_selected_storage_contracts.py \
  -p no:cacheprovider --basetemp "$TMPDIR/selected-storage" \
  -o junit_family=xunit1 --junitxml "$TMPDIR/selected-storage-results.xml"
```

JUnit records both compilation times, exact encoded sizes, actual commands,
requested read/write bytes, observed temporary address high-water marks and
complete carried-state hashes. Host packing payload is reported separately and
unchanged. Native banks retain their fixed physical capacity; a lower used
address high-water mark is not lower RSS or smaller hardware SRAM. Traffic and
command counts remain unchanged when this experiment only relocates storage.
Compilation peak memory and accelerator timing remain unmeasured; native-call
wall samples include interpreter/ABI overhead and establish no speedup.

These are shared-mechanism and neutral graph tests. Real full-model target
routing still needs public selected documents, qualified canonical byte-effects
and address rewriting, runtime ownership/windows and final byte decoding.
Ordinary target lowering retains its existing route. No target repository is
modified, and historical model measurements keep their original source pins
and unchanged numerical admission failures.

## Private copy elimination

The ordinary `allocate_temporary_storage` entry defaults to
`eliminate_copies=True`. It identifies complete, synchronous, bit-preserving
private copies through the existing qualified `StorageAdapter`. A destination
that is only read may share its source's bytes when both typed address views
refer to the same physical backing and the source interval is unchanged until
the last destination read. Subrange offsets, other aliases, destination
alignment and every actual view's capacity remain constraints. Copy chains are
forwarded with work bounded by the original call and object budgets.

The rewritten calls are reprojected and allocation, liveness and completion are
recomputed. If extending the source lifetime makes the composition impossible
to allocate, the pass restores the original composition and replans it. A
decoder or invariant failure diagnoses rather than accepting an unchecked
rewrite. `eliminate_copies=False` selects the original fixed-operation family
for a paired comparison. Device spill repair keeps its insertion family fixed;
the outer allocator selects forwarding before that bounded repair.

`verify_storage_dataflow` independently reconstructs the original and final
byte definitions, permits omission only of qualified private copies, and
compares every retained operation's inputs and numerical semantics, physical
alias relations and complete external output and state windows. The receipt counts
eliminated copies, actual calls, and requested bytes from the decoded realized
program. It does not use the optimizer's list of removed calls as evidence.
The existing `verify_temporary_storage` checker retains its narrower role of
verifying relocation with the original operation count; it cannot certify
copy elimination. Native C tests execute both actual allocated programs on
identical input, compare all output bytes to an independent reference, preserve
unwritten output bytes, and count executed calls and byte requests.

This family does not eliminate packing or conversions unless the operation is
independently qualified as an exact byte copy. It does not infer asynchronous
handoffs, different-memory transfers, in-place epilogues, persistent ownership
or executable target support. Source/destination overwrites keep the copy.
Physical arithmetic fusion and target integration remain incomplete. Existing
targets require a public, independently qualified actual-effect decoder and
address rewriter before this mechanism can enter their normal lowering.
Traffic results from synthetic execution are not hardware latency results.

## Physical elementwise fusion

`compile_schedule(..., fusion_capabilities=selected_capabilities)` discovers
adjacent producer-consumer compositions from decoded effects and
`ElementwiseSemantics`. Each declaration contains actual typed scalar MLIR,
the element count, ordered stages, and an explicit byte order for every read
and write. Values are contiguous, with one corresponding scalar value per
element and no broadcast, hidden effects, volatile access or cross-element
dependency. All arguments of one element are read before its result is written.
Shifted or differently typed aliases are refused. Exact in-place aliases retain
that explicitly qualified execution rule.

This family supports total fixed-width integer arithmetic, constants, casts,
comparisons, selection and signed/unsigned min/max. It reuses the existing
scalar eligibility checks and admits additional exact standard scalar classes
under this new composition scope. Floating arithmetic, poison flags, unknown
operations, calls, reductions, side effects and unmodeled metadata remain
barriers. Existing frozen numerical proof domains and authoring grants are
unchanged. No approximation or new numerical policy is selected.

Merlin uses the actual completion-aware lifetime to require a newly initialized,
invocation-private intermediate with no other observer or writer. The complete
typed producer result must be consumed with the same byte layout. One selected
`ReadoutCapability` must apply every stage, and the returned instruction's
decoded readout identity must match it. Capabilities from different readouts
cannot be combined. The target-owned `FusionCapability.emit` either returns one
actual instruction or declines. Selection uses the first compatible qualified
capability in declaration order; it invents no cycle estimates.

`verify_fusion_step` reconstructs the collapsed pair from original and returned
calls without accepting a selected-pair witness. It checks all unaffected
effects, actual input and output byte intervals, explicit layouts, widths and
stage order. Typed scalar expression DAGs must match after substituting the
producer into the consumer. Widening, wrapping arithmetic, clamp order and
narrowing remain observable nodes; algebraic reassociation is not inferred.
The immutable graph cache and scalar work budgets bound compiler space by IR
size rather than element count.

The candidate's objects and lifetimes are then replanned through the ordinary
allocator. An incompatible physical view or insufficient capacity restores the
original composition and recomputes allocation. Scalar operation/source-byte
budgets and `ScheduleBudgets.max_fusions` can retain the original route. Actual
arithmetic mismatches diagnose; they never authorize an unchecked candidate.
If both routes fail allocation, the caller receives the existing precise
unavailable diagnostic. A receipt marked `allocation_fallback` describes a
discarded candidate, not eliminated traffic in the final program.

Allocation/spill verification binds the checked candidate to the realized
schedule; actual-byte decoding then checks the complete final instruction
effects and numerical declarations. Final receipts report actual commands,
encoded size and requested read/write bytes after all rewrites. Tests execute
the emitted byte program in native code and check complete outputs, input
state, intermediate signed wrap, clamp-before-narrowing and actual transfer
counters against an independent vector reference. Separate-memory and aliased
view models, different element widths/byte orders, tails, alignment fallback,
multiple consumers, extra operands, asynchronous barriers and mutation controls
exercise the shared mechanism. Negative controls expose both incorrect target
arithmetic and hidden temporary traffic despite plausible declarations.

The same mechanism also composes a typed identity accumulator readout with an
integer bias/clamp/narrowing epilogue after a completed contraction. Native tests
execute complete two-layer integer graphs through final encoded bytes, including
an observed first-layer activation, both contractions, separate accumulator/scratch
stores, 8/16-bit input elements and i32 wrap before saturation. Counters observe
actual staged requests and result accesses; both intermediate readout stores and
reloads disappear. Missing bias support retains the original materializations.
This scope requires addressable accumulators and contiguous elementwise bias
matrices; it does not infer broadcast bias or packing. A falsely implemented
wider fused addition fails the independent complete-output comparison.

Synthetic tests do not qualify a target. Existing providers must expose a
public, independently qualified scalar/effect projection, complete layouts and
the physical composition emitter, together with final actual-byte decoding.
The initial read-only target integrations do not provide these hooks. Physical
floating readouts, direct contraction-body fusion, noncontiguous layouts,
cross-memory handoffs and asynchronous fusion remain incomplete. Traffic
reductions in these tests are not full-model latency results.

### Fusion selected from RTL and software documents

Ordinary `compile_schedule` can now build its fusion capabilities from an
optional `fusion` section in the same selected scheduling document. Its schema
is `merlin.physical_fusion_contract.v1`, and it requires the selected storage
contract. This reuses `fusion.py`, `fusion_semantics.py`, the allocation pipeline
and final independent ISA-word checks. It introduces no new arithmetic IR or
pass manager.

`elementwise` maps ISA instruction names to these required fields:

| Field | Meaning and owner |
| --- | --- |
| `scalar_ir` | Complete scalar MLIR function defining the software arithmetic, including conversions and numeric boundaries |
| `elements` | One to sixteen encoded integer argument/constant factors defining the logical element count |
| `element_bits` | One RTL fact reference for each memory input/result width, checked against actual scalar MLIR types |
| `stages` | Complete ordered software stage names, with repetitions preserved |
| `byte_orders` | Explicit software byte order of each input/result, independently of instruction word byte order |
| `readout` | Physical readout identity for a composed result; otherwise null |
| `execution` | `contiguous_read_then_write` for this execution family, or an explicit unsupported/unknown declaration |

The execution family means contiguous elementwise memory, no hidden effects,
state, volatility, traps or cross-element dependence, and all inputs of an
element read before writing it. Independent qualification must establish those
properties. Floating arithmetic, unsupported integer operations/metadata and
unknown execution retain the original materializations. Scalar-operation and
source-byte budgets bound parsing and correspondence checks; no tensor elements
are enumerated. Contradictory admitted widths/layouts or a provider projection
that disagrees with the selected semantics diagnose.

The storage contract now supports a byte extent of this form in addition to
its existing argument/constant binding:

```yaml
size:
  elements: [{argument: m}, {argument: n}]
  element_bits: {fact: [facts, accumulator_bits]}
```

This is a bounded product of positive logical dimensions multiplied by the
fact-backed byte width. It is not an arbitrary expression evaluator. Every
factor binds an encoded integer ISA argument or a positive integer constant;
nonpositive dimensions, missing bindings, excessive factors and subbyte widths
refuse. This lets the same memory/effect contract describe shape-dependent
matrix results and packing sizes without capturing benchmark constants. Sizes
are reconstructed from actual operands and compared with the provider's effect
projection before allocation and again through final bytes.

`readouts` maps physical selector names to `applies`, `evidence`,
`max_elements`, and `routes`. `applies` lists stages that this individual
readout can realize. `max_elements` is an RTL fact reference, or null when the
structural bound is unknown. A route explicitly names `producer`, `consumer`
and `result` ISA instructions. Its result must carry that readout and exactly
the concatenated ordered source stages. No union of different physical readouts
can implement one composition. A bound does not qualify arithmetic or completion.

For an integration that has independently qualified its semantics and physical
emitter, the ordinary call is:

```python
result = compile_schedule(
    kernel, objects, None, storage_adapter, encoding, limits,
    selected_contracts=selection,
    fusion_emitters={selector: physical_composition_emitter},
    fusion_qualification_identity=selected_execution_evidence_identity,
    private_ownership=runtime_established_private_ownership,
    external_windows=runtime_windows,
)
```

The target callback retains ownership of physical instruction construction. It
may decline a pair. A returned instruction must match one declared route and
pass the existing actual scalar/memory correspondence, allocation, completion,
final-byte and size checks. Bounds are tested against actual producer extents.
Missing qualification, emitter, structural bound or supported scalar semantics
disables the affected capability and records why. The original route remains
available where legal. Arbitrary parallel `fusion_capabilities` cannot override
a selected fusion declaration. Without the extension, the existing explicit
legacy interface remains separate and makes no selected-fusion claim.

Typed semantics are constructed from the selected document and actual operands.
A provider need not duplicate the scalar function in its projection. If it does
supply a scalar function, its typed DAG, count, layouts, stages and readout must
agree. A reviewed label or qualification string is not independent execution
evidence. The selected instruction/operand codec checks actual emitted words;
it does not infer the behavior of an opaque native instruction implementation.

Selection is deterministic: readouts are visited by sorted selector name, and
routes retain their declared order. The first compatible legal composition is
used within the fusion budget. This is a conservative policy without latency
estimates or calibration claims. Failed physical allocation restores the
original composition, invalidates analyses and replans within the existing
global repair budget. A numerical mismatch is an error, not a reason to emit an
unchecked alternative. `ScheduleBudgets(max_fusions=0)` supplies the paired
baseline through the same selected interfaces and verification rules.

Resolved contracts now also bind the source hashes of their resolution owners.
Changing a resolver after selection requires reselecting both inputs; unchanged
file bytes cannot preserve analyses computed by an older resolver. Changes
during selection or a pass diagnose. Numerical policy, actual IR and storage
budgets remain bound by the existing pipeline stamps. These checks do not
establish transitive third-party dependency closure or broaden authoring grants.

`test_selected_fusion_contracts.py` executes complete neutral two-layer integer
graphs, including rectangular contractions, 8/16-bit input elements, separate
accumulators, shared/separate scratch memories and both complete output tensors.
The selected interface removes both i32 readout stores/reloads: actual command
counts change from seven to five. The fused operation still wraps its i32 sum
before clamping and narrowing. Independent wide arithmetic checks every output,
input preservation and guards. Three recurrent invocations feed the observed
hidden output back as input and change weights/biases between calls; neither
carried state nor mutable weights are hoisted. A false native implementation
that skips wrapping fails the complete-output reference.

Mutation tests cover unsupported execution/arithmetic, capacity boundaries,
scalar widths, stage/conversion order, layouts, undeclared physical routes,
extra ISA effects, final words, stale inputs/resolvers and allocation fallback.
A deliberately restrictive physical view forces fallback and confirms that no
planned transfer elimination is reported as realized. Existing mechanism tests
also cover observers, overwrites, shifted aliases and unsupported completion.

Reproduce the selected-interface pair without downloads:

```sh
python -B -m pytest merlin/tests/infra/test_selected_fusion_contracts.py \
  -p no:cacheprovider --basetemp "$TMPDIR/selected-fusion" \
  -o junit_family=xunit1 --junitxml "$TMPDIR/selected-fusion-results.xml"
```

JUnit records compiler time, exact encoded size, executed commands, actual
requested read/write bytes, selected inputs and shared source hashes. Native
host-call samples include marshalling/interpreter costs and establish no device
latency or speedup. Compiler peak memory, calibrated timing and separate physical
host/accelerator contributions remain unmeasured. These neutral graphs are not
real-model or hardware qualification. Existing full-model records retain their
original source identities, numerical thresholds and admission failures.

Real target lowering still needs the selected two-document binding, qualified
contiguous scalar execution, an actual composition emitter and complete final
byte decoding. Direct contraction-body fusion, broadcast/noncontiguous bias,
floating epilogues, cross-memory handoffs and asynchronous composition remain
unsupported until those explicit contracts and execution evidence exist. No
target implementation is changed to fit the shared pass.

## Exact device register spills

The ordinary `allocate_temporary_storage` entry accepts selected `SpillClass`
capabilities and a `max_spills` repair budget. Each class declares its device
register bank, a separate addressable spill backing, exact byte width,
alignment, transfer emitter and independent qualification identity. It grants
no permission to replace the native toolchain's host CPU register allocation.

When allocation cannot fit, a bounded deterministic search selects unused gaps
in eligible register lifetimes, emits an exact save and reload, and rebases later
uses into a fresh private object. Unknown costs use the documented longest-gap
policy. Each insertion reconstructs actual effects and recomputes liveness,
allocation and completion; previous emission and program-size checks are
invalidated. Spill slots are allocated by the same allocator and can be reused
after their actual last read. Growth and repair budgets bound the search.

The emitted transfer must independently decode as `StorageEffects.exact_copy`:
one equal-sized read/write, no arithmetic or conversion, and actual synchronous
completion. This family supports synchronous straight-line device programs.
Unknown register classes, nonaddressable memory, missing exact transfers,
unsupported widths and asynchronous spill programs keep the existing route or
produce a diagnostic. No solver timeout or invariant failure becomes a legal
unchecked fallback.

`verify_device_spills` reconstructs interval byte provenance from the original
and actual final programs. It accepts no planned spill locations. It checks
every original operation and read in order, physical aliases, current capacity,
alignment and reservations, allowed exact copies, and complete external output
and state intervals. Provenance storage grows with written intervals, rather
than tensor element count. Receipts record observed commands and requested
bytes, with hardware timing and machine-emission qualification explicitly
unestablished. Native tests include a faulty copy implementation that the
independent numerical comparison rejects despite its declared copy effects.

Existing backends need independently qualified register-class and transfer
adapters before this mechanism can enter their ordinary target lowering. A
synthetic execution qualifies the shared mechanism's stated scope only.

### Device spills selected from RTL and software documents

Ordinary `compile_schedule` can derive those existing `SpillClass` capabilities
from the same selected RTL/software pair used for storage and fusion. This
binds the spill planner to the selected device interface; it does not introduce
another register allocator. Add an optional `spills` section under
`merlin.schedule_effects.v1`, alongside the required selected `storage` section:

```yaml
spills:
  schema: merlin.device_spill_contract.v1
  classes:
    vector_values:
      register_bank: register_bytes
      spill_bank: saved_bytes
      width_bits: {fact: [facts, vector_register_bits]}
      alignment: {fact: [facts, transfer_alignment_bytes]}
      store: save_vector
      reload: restore_vector
      execution: synchronous_bit_preserving_bytes
```

Names are identifiers from the selected software/ISA document, not recognized
target instructions. Each class explicitly identifies addressable device
register bytes and a separate invocation-private spill memory. Both banks must
already have selected byte windows, physical backing identities, ownership and
RTL capacities. Aliased register/spill backings and multiple classes for one
register bank reject. This family cannot express nonaddressable accumulator
state or host CPU registers.

`width_bits` and byte `alignment` must reference structural RTL facts, never
literal tuning constants. Widths must contain whole bytes. Either may be `null`
when unknown; that class is then disabled. Software declares `store` and
`reload` separately, so a target can use different instructions in each
direction. Both instructions must have complete, memory-only, exact-copy
effects in the selected ISA/storage declarations, covering their respective
directions. Inexact copies and hidden execution/state effects reject.
Asynchronous transfers or an unknown/unsupported `execution` string disable
the class; they do not acquire synchronous completion from RTL queue facts.

The caller supplies only the target's physical transfer callbacks and an
independently selected qualification identity:

```python
result = compile_schedule(
    source_kernel, private_objects, None, storage_adapter, encoding, limits,
    selected_contracts=selected,
    private_ownership=True,
    spill_emitters={"vector_values": emit_device_transfer},
    spill_qualification_identity=qualified_transfer_evidence,
    budgets=ScheduleBudgets(max_spills=2, max_repairs=2),
)
```

The callback receives actual source/read and destination/write `EncodedAccess`
objects and returns a typed instruction `Call`. Its result must use the selected
store or reload instruction for that direction. Actual decoded accesses,
complete width, bit-preserving effects and synchronous completion must match.
The existing relocation checker subsequently preserves all nonaddress
semantics. Actual final words are independently decoded against the selected
ISA and checked against the realized schedule. No emitter name or private
backend type is inspected.

Missing emitters or independent qualification leave the class disabled.
Feasible programs still compile without spilling; a pressure failure reports
the unavailable class and reason. Supplying arbitrary `spill_classes` together
with selected documents rejects, as do unselected callback keys. The legacy
explicit-capability entry remains available without selected documents. Merely
setting `status: reviewed` or attaching a qualification string does not prove
that a machine implements exact transfers.

Spills still consume the existing global repair budget, recompute liveness,
allocation and completion, and invalidate emission/size results. Selected raw
documents, numerical policy, resource parameters and shared resolver/planner
source identities bind the pipeline analyses. Changes during selection or
compilation reject; prior selected analyses cannot silently survive a changed
resolver. With unknown timing costs, selection remains deterministic
longest-unused-gap. No cycle estimate or calibrated tuning result is implied.
When the transfer requires stronger alignment than the original arithmetic,
replanning strengthens the saved register's alignment as well as the new spill
and reload objects. A nonzero bank window does not waive that constraint.

`test_selected_spill_contracts.py` runs the ordinary path over complete neutral
stateful programs with 24-bit and 128-bit device register values, byte alignment
one and eight, 64-bit little-endian and 32-bit big-endian operands, separate
memories, tails and distinct reload instructions. Three invocations compare all
output bytes, carried inputs and guards against independent byte arithmetic.
A paired existing-explicit-capability arm uses identical inputs, hardware,
shared source versions and transfer implementation; the selected binding
preserves its actual commands and bytes, with no claimed speedup. Disabling
spills under pressure diagnoses; undersized capacity, reservations, aliases,
wrong transfer operands/instructions, changed software, repair limits and
final code growth are exercised. A faulty native spill changes results despite
identical declarations and command/traffic counters, disproving qualification
based solely on those declarations.

Reproduce with the project's Python environment:

```sh
python -B -m pytest merlin/tests/infra/test_selected_spill_contracts.py \
  merlin/tests/infra/test_device_spills.py \
  -p no:cacheprovider --basetemp "$TMPDIR/selected-spill-check"
```

These neutral executions do not qualify a real hardware implementation. Existing
targets still need public selected device-register windows, exact software
transfer/completion semantics, a qualified physical callback, actual-byte
decoding and complete model integration. No target repository was modified to
supply them. Compilation peak memory and accelerator timing remain unmeasured
for this selected binding; broader asynchronous spills and model-level timing
remain unresolved.

## Final encoded programs

`finalize_program` checks actual encoded bytes through the selected target's
independently qualified `ProgramDecoder`. `ProgramLimits` declares instruction
alignment, byte and instruction budgets, memory windows and relative branch
encodings. Decoding must cover every byte exactly once. Every decoded access
must fit a declared window, and every branch must fit its declared encoded
reach and target an instruction boundary inside the program. Unknown structural
facts produce a diagnostic.

An oversized program can be segmented only with an explicit `Continuation`
contract: sequential completed launches, entry at zero, preserved memory spaces
and a launch budget. The caller supplies the original complete output intervals.
The shared algorithm preserves instruction bytes, keeps relative branch regions
closed, and checks future reads and published outputs before allowing volatile
storage to disappear at a launch boundary. It retains partial-overwrite values
conservatively. With branches or asynchronous operations it also retains future
reads conservatively rather than inferring path-specific liveness.

An asynchronous instruction prevents a boundary until an actual decoded drain.
Unknown completion and position-dependent instructions prevent segmentation.
May-pending work is now propagated through the actual decoded branch graph.
Conditional branches include both successors; an explicitly unconditional
relative branch can omit fallthrough. A path that skips its drain and exits
with pending work rejects, even if the entire image fits. The finite boolean
analysis visits instruction states without expanding loop trip counts.
This is a conservative completion check, not a proof of termination or timing.
Every resulting segment is decoded again: its actual normalized effects must
equal the original instruction effects and meet the final budgets. The receipt
records actual counts, sizes and hashes; it does not establish execution timing
or qualify a target decoder or runtime continuation contract. Unsupported cuts
refuse rather than inventing branch relaxation, state saves or launch support.

Neutral tests execute variable-width byte programs before and after segmentation
and cover a distinct fixed-word representation. Existing target integrations do
not supply this final decoder/continuation boundary automatically. Final machine
program checks therefore remain an explicit integration requirement for those
targets.

### Final limits selected from RTL and software documents

An optional `program` section of `merlin.schedule_effects.v1`, using schema
`merlin.final_program_contract.v1`, binds the existing finalizer to the selected
RTL/software pair. `compile_schedule` consumes it automatically before emission
checks and finalization. Passing `None` for its `limits` argument derives the
available constraints. A target/runtime provides the existing qualified
decoder; a separate runtime qualification enables declared continuation:

```python
compiled = compile_schedule(
    kernel, private_objects, None, storage_adapter, encoding, None,
    selected_contracts=selected, private_ownership=True,
    continuation_qualification_identity=qualified_runtime_evidence,
)
```

The same binding works at the existing final-byte entry, including actual
relative branches and asynchronous programs that the finite straight-line
schedule compiler deliberately does not admit:

```python
final = finalize_program(
    actual_image, None, qualified_decoder,
    selected_contracts=selected,
    continuation_qualification_identity=qualified_runtime_evidence,
)
```

The selected `program` section has these exact fields:

| Field | Contents and source |
| --- | --- |
| `schema` | `merlin.final_program_contract.v1` |
| `limits` | RTL references or `null` for `max_bytes`, `max_instructions`, `instruction_alignment` |
| `windows` | ISA byte-memory resource to fact-backed `start`, `size`, software `backing` and `visibility` |
| `branches` | Rule name to actual `instruction`, integer `displacement` argument, fact-backed byte `scale` and software relative `origin` |
| `instructions` | Actual ISA instruction name to `completion`, `relocatable`, `control_flow` |
| `continuation` | `null`, or explicit launch completion, entry, persistence and bounded launch count |

Window coordinates are canonical backing bytes. `visibility` is `private`,
`external` or `unknown`; unknown storage remains observable. A window must fit
known RTL capacity, and must not contradict or widen the existing selected
storage window or its backing/ownership. Aliases must agree on visibility.
The finalizer publishes all external/unknown windows conservatively in addition
to caller outputs. Omitting an output cannot authorize loss of carried state.

Branch displacement bounds come from the selected ISA field's actual width and
the typed integer domain, including signedness. Software declares
`origin: instruction` or `next_instruction`, and RTL supplies byte scale.
`origin: unknown` remains unsupported when that branch occurs. The pass invents
no PC rule, branch relaxation or instruction width. Current selected decoding
supports complete, fixed-width, byte-aligned words from `ResolvedISA`; the
existing explicitly qualified finalizer still supports other decoder framings.

Instruction `completion` is `synchronous`, `asynchronous`, `drain` or `unknown`.
`drain` is a qualified assertion that **all pending work is complete and visible**,
not something inferred from an RTL queue depth or a wait spelling. Storage
completion declarations must agree. Control flow is `fallthrough`,
`conditional_relative`, `relative` (unconditional), or `unknown`. An unknown
control/effect description is a barrier. A return/exit without a represented
branch target cannot be segmented. A fitting unmodified image may retain
unknown completion, reported explicitly; it acquires no completion certificate.

Before each final check, the wrapper independently decodes actual ISA words,
reconstructs declared byte accesses from the existing selected storage operand
bindings, reconstructs relative branches, and checks completion, framing,
fallthrough and relocation against the provider decoder. It consumes no retained
instruction request list. Unknown memory projections diagnose. Segments are
decoded again through the same checks; a decoder that changes its interpretation
for shorter images cannot inherit the whole-image receipt.

Continuation declares `sequential_completion`, `starts_at_zero`, a positive
software `max_segments` budget, `preserved_backings`, and
`preserved_configuration`. The latter maps ISA state resources to RTL field-index
references. Preserving memory does not imply preserving configuration. Actual
encoded state operations must have complete ordered configuration projections,
and every affected field must survive launches. Other nonmemory execution state
needs a separate continuation contract and currently blocks segmentation.
False launch/entry declarations or missing independent runtime qualification
disable segmentation; a fitting unmodified image remains available.

Physical preservation applies to every alias of a backing. The finalizer
canonicalizes spaces for its cut/liveness analysis while retaining each view's
address checks and original bytes. This can remove unnecessary launches that
an analysis over view names alone retained. Partial overwrites never kill an
entire future read. A supplied `Continuation` can restrict the selected memory,
configuration and launch budget, but cannot expand them or assign contradictory
survival to two aliases. The generic explicit finalizer also accepts a complete
`physical_spaces` mapping when no selected contract is used; partial or
unselected alias overrides reject.

Caller `ProgramLimits` may tighten byte/count budgets and branch reach, and
restrict windows. They cannot widen hardware windows or change branch origin
or scale. A missing hardware byte capacity can be paired with an explicit
code-growth budget, but the receipt still says hardware byte capacity is
unknown. Without either bound the pass diagnoses. Missing instruction count
stays unknown, and actual instructions are still counted. Missing instruction
alignment diagnoses; no default alignment is guessed.

Selected raw documents, resolver/finalizer implementations, source controls and
decoder/continuation identities must remain current across decoding and before
return. Edits or callback mutations invalidate the check. These are binding
checks, not independent qualification of a decoder or runtime implementation.
Native negative controls demonstrate wrong outputs or final configuration when
an implementation violates a declared alias or persistence rule.

The tests execute actual little-endian 64-bit-field and big-endian 32-bit-field
programs with different memory models, complete outputs and three invocations.
They cover capacity boundaries, source copy elimination before final counting,
closed forward/backward branches, branch paths that skip drains, completed
asynchronous copies, aliased overwrites, launch/configuration persistence and
stale or inconsistent final decoding. The paired explicit/selected entries use
the same library version, source program, hardware limits and inputs. Observed
launch counts, commands and requested bytes are separate results; fewer launches
do not establish latency improvement.

```sh
python -B -m pytest merlin/tests/infra/test_selected_program_contracts.py \
  merlin/tests/infra/test_selected_program_control.py \
  merlin/tests/infra/test_program_limits.py \
  -p no:cacheprovider --basetemp "$TMPDIR/selected-program-check"
```

No target repository supplies this boundary through this change. Real-target
decoder/launch qualification and complete model integration remain unresolved.
These word-stream checks do not count instructions in arbitrary target C or a
linked native ELF. In particular, a counted-row C program's static call count
does not establish its final native instruction size; ordinary linked-image
admission remains the applicable boundary until a qualified integration is
provided. Compilation peak memory and accelerator timing for these selected
checks are not established by the neutral executions.

## Complete-domain counted row softmax

`compile_row_softmax` consumes a typed `RowSoftmaxProblem`, an explicit
`RowSoftmaxNumerics` source algorithm, chunk resources, selected byte-addressed
banks, four role-to-bank bindings and a `RowSoftmaxAdapter`. The adapter supplies
actual instruction encoding/projection, configuration and qualification
identities, index width, exact source order, addressable state and synchronous
completion. A declaration is conditional on independent qualification; the
neutral test implementations do not establish hardware support.

The source algorithm names its f32/f64 exponential, lane count, exact binary sum
tree and division versus reciprocal multiplication. Each lane starts with its
first exponential and continues in increasing source-column order across every
chunk. `cpu_softmax_row_numerics` exposes the existing pinned CPU f32 algorithm;
the tensor CPU builder and counted scheduler share its eight-lane/tree data.
Finite inputs and the selected RNE, nontrapping, unobserved-status contract remain
explicit prerequisites. No approximation policy or admission threshold changes.

For each row, the emitted counted kernel initializes and completes the **whole
row maximum**, computes **all shifted exponentials**, initializes source lanes,
completes the **whole denominator** and its ordered tree, then normalizes every
column. No chunk normalizes independently. Optional `CausalRowMask(period,
offset)` uses `column <= row % period + offset` in both score passes. Positive
period and nonnegative offset prove every nonempty row has an active element.

The existing allocator places a complete row of exponentials, source lane sums,
maximum and normalization factor. Lifetimes include actual synchronous finish
phases: maximum/factor can reuse bytes, while lane sums/factor cannot overlap
during the final tree. Explicit backing aliases, per-bank alignment and packing
or DMA reservations constrain placement. Workspace is invocation-private;
immutable inputs and outputs require disjoint runtime bindings. Rows reuse
storage only after normalization completes. The source algorithm needs one
full-row exponential allocation; this family refuses if that state cannot fit.

The ordinary shared entry runs `row_semantics` (stage 0), `row_tiling` (2),
`row_allocation` (3), and `row_schedule` (4) through `PipelineSession`.
Prerequisites and explicit order are checked before execution; scheduling
invalidates source, tiling and allocation analyses and produces a fresh
realization. Context stamps bind problem/mask, current shared library sources,
hardware, numerical algorithm, ownership, resources, reservations and execution
qualification. Different chunks use the same interfaces and rules in Phase 2.

Unknown costs select the largest legal chunk. Without tail support, or when
peeled tails exceed the static budget, a bounded descending divisor search tries
at most 64 candidates and retains the primitive granularity as a legal fallback
when it divides the complete domain. An explicit alternative must satisfy the
same limits. Counted loops hold static calls to seven or eleven regardless of
row/column extents; no compiler loop expands the tensor. This policy does not
estimate cycles or imply that smaller chunks improve execution time.

`verify_row_softmax` reconstructs the actual projected calls, phase ordering,
loop bounds, full coverage, affine source/output/state addresses, mask and
numerical identity. It rechecks completed-phase physical ownership and emitted
integer intermediates. Reports contain actual projected command/payload counts
and assigned storage. Instrumented native tests independently observe requested
bytes, commands and touched physical high-water addresses; they compare complete
outputs, input preservation and guards across baseline/tiled programs and an
independent pinned Torch CPU reference. A separate f64 one-lane/direct-division
source exercises different widths, arithmetic order and separate memories.
An intentionally incorrect per-chunk lane reset fails independent output checks.

Reproduce with the native toolchain and pinned reference available:

```sh
python -B -m pytest merlin/tests/infra/test_row_softmax.py \
  merlin/tests/ir/test_cpu_softmax.py -p no:cacheprovider \
  --basetemp "$TMPDIR/row-softmax-check"
```

The entry returns the actual `Kernel`, physical views, realization and pass
records. `result.emit_c(instruction_set, symbol=...)` uses the existing emitter
and checked index width; instruction implementations remain target-owned.
`RowSoftmaxUnavailable` diagnoses unsupported capacity, ownership, numerical or
execution contracts so a caller can retain its original source implementation.
All-masked/arbitrary masks, nonfinite source domains, asynchronous primitives,
unaddressable state, cross-row aliasing and unknown effects are unsupported.

This is an executable shared counted-row path. It does **not** yet automatically
route full-model source softmax to device primitives, integrate a real target's
primitives, compose device QK–softmax–PV/KV execution, or verify final machine bytes.
Those require source admission, public target primitive/projection hooks and a
counted-program final decoder/completion contract. It does not run through the
finite straight-line final-byte pipeline by expanding loops. Native reference
agreement establishes the tested source algorithm, not hardware qualification,
full-model gains, accelerator latency, or calibrated performance.

### Complete rows selected from RTL and software documents

`compile_row_softmax(..., selected_contracts=selected)` now consumes an optional
`row_reduction` section of `merlin.schedule_effects.v1`. It reuses the existing
row planner, state allocator, actual-call checker and counted C emitter. Supply
`None` for `resources`, `banks` and `role_banks` to derive them, or supply
conservative resource controls through the existing typed interfaces:

```python
result = compile_row_softmax(
    problem, source_numerics, None, None, None, qualified_row_adapter,
    selected_contracts=selected,
    private_ownership=True, disjoint_arguments=True,
)
source = result.emit_c(target_instruction_set, symbol="normalize_scores")
```

The selected section has exactly these fields:

| Field | Meaning and authority |
| --- | --- |
| `schema` | `merlin.complete_row_contract.v1` |
| `algorithm` | All `RowSoftmaxNumerics` fields; exact source policy, dtype, exponential, lane order/tree, normalization and numerical premises |
| `element_bits` | RTL fact reference agreeing with the software dtype |
| `limits` | Fact references or `null` for `max_chunk_elements`, `granularity`, `index_bits` |
| `support` | Explicit `partial_chunks`, `completion`, `order`, `state`, and `masks` |
| `roles` | `maximum`, `exponentials`, `lanes`, `normalizer` to selected private storage bank names |
| `external` | `input` and `output` to selected external byte-window resources |
| `instructions` | All seven phase names to their actual instruction/operand bindings |

The source numerical algorithm must equal the declared one, and its
`source_policy` must match the selected numerical policy. The source lane count
and tree are never chosen from hardware vector width. RTL alone cannot establish
the exponential implementation, rounding, masks or reduction equivalence.
Independent execution qualification remains required for the adapter. Supported
execution is synchronous, exact source order and addressable invocation-private
state. An unknown limit, unsupported mask or execution description diagnoses so
the caller can retain its original complete source implementation.

Each phase entry contains `instruction`, `bindings`, `local` and `constants`.
`bindings` names actual typed ISA arguments for `row`, `column`, `count`,
`period`, `offset`, and the applicable `source`/`output` pointers. `local` lists
the phase's state operands in the existing `PHASE_ROLES` order. Each local entry
names an integer byte-address argument and either `selector: null` for an
implicit space or `{argument: ..., value: {fact: [...]}}` for an explicit
memory selector. `constants` fixes every otherwise unused mode/argument to an
integer or, for a typed address, `null`. Every actual operand must be covered
exactly once. No argument or unknown effect is silently ignored.

The selected ISA instruction must declare primitive
`merlin.complete_row.<phase>`, matching logical `bindings`, and exactly
`parameters: {numerical_identity: <source algorithm hash>, completion: ...}`.
Its complete resource reads/writes must agree with the phase's source, state
and output effects. These declarations bind the meaning of existing typed
primitives; they do not synthesize or qualify their target implementation.
The wrapper reconstructs `RowSoftmaxStep` from actual operands and independently
compares it with the provider projection. The existing verifier then checks
complete domains, masks, rounded algorithm identity and physical lifetimes.
Wrong modes, spaces, pointers, counts or instruction names cannot inherit a
retained request as their proof.

Geometry and runtime ownership use `TemporaryStorageContract.allocation_model`,
the same admission used by finite schedule allocation. Bank aliases, view
bounds, alignment and packing/DMA reservations retain their existing checks.
This boundary explicitly converts the row allocator's `byte` unit spelling to
the selected storage contract's `bytes`; it infers no word addressing. Caller
budgets can restrict chunk size, combine granularity or remove tail support,
but cannot expand hardware limits or remap software state roles. Ordinary
complete-row allocation still requires full-row exponential storage.

Current source problem/mask, numerical policy, resources and library sources
are rechecked around every pipeline action. Raw selected documents and resolver
owners must remain current. The returned pipeline receipt records these inputs
and the actual checked kernel. `emit_c` revalidates the kernel, index width,
selected documents and shared source versions before and after rendering.
It uses the existing immutable pass stamp; removing or rewriting the printable
report cannot disable freshness. Changing instructions invalidates emission.
This guards stale emission;
it does not interpret arbitrary target C or establish final machine conformance.

Native paired tests compare one-element chunks and the deterministic largest
legal chunk on the same selected hardware/software and shared library version.
They observe actual primitive commands and logical requested payloads, check all
output bits against independent pinned Torch f32 or ordered `math.exp` f64
references, and preserve inputs/guards. Additional complete neutral attention
tests keep host QK/PV and cache updates identical between arms, check full
probabilities, outputs, KV caches and positions over three invocations, and
reject an implementation that resets denominator lanes at chunk boundaries.
Those surrounding host contractions and cache writes are test integration;
this pass does not optimize or qualify device attention instructions.
Native `.text`, whole-library size and C-source size are recorded separately:
fewer primitive calls need not shrink native code, since the host compiler can
expand primitive bodies. Static call budgets do not replace final linked-image
checks. Host-only call samples include dispatch and surrounding arithmetic and
do not establish an accelerator speedup or a calibrated cost model.

Reproduce with the existing native toolchain and pinned reference:

```sh
python -B -m pytest merlin/tests/infra/test_selected_row_contracts.py \
  merlin/tests/infra/test_selected_row_attention.py \
  merlin/tests/infra/test_row_softmax.py \
  -p no:cacheprovider --basetemp "$TMPDIR/selected-row-check"
```

Limits of this selected C route remain explicit: source/output windows must be
zero-based logical argument windows; general physical pointer relocation is a
target integration requirement. Dynamic integer operands must be at least as
wide as the selected checked 32/64-bit index domain. ISA address operands with
stronger alignment need a separate mapping contract; allocation-bank alignment
is supported independently. Arbitrary masks, asynchronous completion,
nonaddressable state, new approximation policies and whole-row storage spills
are not inferred. No target repository supplies these hooks through this
change. Final machine bytes, real-model routing, compilation peak memory and
qualified accelerator timing remain unresolved for this selected route.

## Automatic source row normalization

Ordinary `lower_model`/`lower_model_file` serial CPU lowering selects
`automatic-host-rows` by default. Disable it with the existing optional-pass
selector or `MERLIN_AUTOMATIC_HOST_ROWS=0`. The selection report records both arms'
identical shared source hashes, source/output byte identities and refusals.
Ordinary lowering writes `automatic-host-rows.json` next to
`model.upstream.mlir`, including the upstream file's actual byte hash. It also
returns the report in `LowerResult.stats["automatic_host_rows"]`. The upstream
hash records product custody after preprocessing; the row inverse check applies
at its original source stage and does not prove the intervening passes or machine
code. Disabled and unsupported selections persist their diagnostic as well.
This pass requires no separately captured quantization policy: the original typed
source is the numerical algorithm and the pass changes no scalar arithmetic.

The matcher admits closed static unencoded f32/f64 tensor islands consisting of
last-axis maximum, subtraction, exponential, complete sum and division, including
checked row-scalar reshape/broadcast views. It retains original scalar regions,
initializers, operation order and attributes. Arbitrary input masks/nonfinite
values retain their original arithmetic. It emits complete-row tensor slices in
one counted loop, then uses the ordinary upstream bufferization and native object
path. It does not replace source exponentials or choose a different sum tree.

An adjacent closed add/subtract/multiply/divide prefix can join the island,
including scalar operands and statically projected singleton broadcasts. The
pass derives each actual input slice from the original typed indexing map,
preserving batch/head/row coordinates and complete columns. Permutations,
repeated axes, nonprojection maps and inconsistent uses remain unsupported.
Scalar splats between a producer's destination and body can join the island when
they have no external observer. Checked identity readouts after normalization can
also join it. This avoids introducing a full-tensor boundary around masks/scales
or leaving a redundant source copy between normalization and its consumer.

Unknown effects, attributes, algorithms, layouts, escaping intermediates and
intervening operations retain the original route. Vector/parallel/explicit
schedules, 32-bit indexing, layouts without a selected index width and source-bound observation/graph
selections retain their existing route. Static code/group budgets bound growth.
The default unknown-cost policy
requires fewer logical row-state elements and preserves existing producer/consumer
pointwise-fusion boundaries, including through views. This prevents the observed
whole-graph regression where isolating a normalization materialized an additional
full tensor. It is a conservative policy, not a timing or physical-allocation
guarantee. Typed ordered multiply/add contractions are recognized as unchanged
boundaries, including through views; their reduction bodies are never moved into
normalization. Other pointwise consumers remain a conservative barrier.
Explicit mechanism tests may disable that cost filter. Selected 64-bit layouts
use the same route, with layout identity and index width bound into the source
analysis context. A native test obtains its layout from the actual selected
compiler, rather than embedding a target layout constant.

The independent checker reconstructs actual loop bounds, mixed-radix row
coordinates, complete slices, original scalar regions and local SSA. Inverting
the checked islands must restore the entire original module, including external
operations, interfaces and carried state. Source admission, scheduling and
conformance use the existing `PipelineSession` with numerical, resource and
library identities. These are structural source checks outside the existing
integer Lean domains; machine equivalence is not claimed.

Reproduce the paired native experiment without model downloads:

```sh
python -B -m pytest merlin/tests/ir/test_row_normalization.py \
  -p no:cacheprovider --basetemp "$TMPDIR/source-row-test" \
  -o junit_family=xunit1 --junitxml "$TMPDIR/source-row-pair.xml"
```

Native tests compare all output bits with the unchanged baseline and independent
ordered arithmetic/libm references, including tails, NaNs and infinities. Complete
QK–normalization–PV graphs additionally check recurrent state, positions and
input preservation over three invocations, with projected masks, scalar scaling,
generic contraction bodies and identity readouts. In these neutral examples,
both masked and unmasked forms reduce
native text size. The unmasked case reduces requested temporary heap, while the
masked case adds one row of scratch despite reducing logical row-scalar state.
This observed whole-graph tradeoff is not a memory or latency improvement claim.
Unsupported producer/consumer fusion boundaries retain their original route. JUnit
properties record compilation wall time, object/text bytes, requested heap payload
and whole native host-call wall samples. Heap payload is not RSS; compile peak
memory and transfer bytes remain unmeasured. Host-call samples include ABI overhead
and are too few for a speedup claim. These neutral CPU graphs establish no KV-cache
ownership, real target support or absolute accuracy for an external model.

## Existing mechanisms and remaining integration requirements

| Concern | Existing mechanism | Requirement beyond the scope above |
| --- | --- | --- |
| Temporary storage | Alias/liveness analyses, source reuse, checked SSA allocation, LLVM arena binding and consumed private straight-line relocation; selected RTL/software windows, aliases, reservations and actual operand/effect checks | Qualified target effect/rewrite adapters and ordinary model integration; general loop and persistent-storage allocation |
| Register pressure | Bounded exact device spill/reload insertion, reusable slots, fresh allocation/completion checks and independent actual byte provenance | Qualified device register/transfer adapters and ordinary target integration; CPU allocation stays with the native toolchain |
| Fusion and transfers | Readout composition admission and connected dispatch; exact private-copy forwarding; consumed total-integer elementwise physical composition with layout/numerical checks, allocation fallback and reconstructed final traffic | Floating/contraction epilogues, noncontiguous and asynchronous handoffs, and qualified full-model target integration |
| Reductions and attention | Complete-domain counted row softmax; automatic narrow CPU source row scheduling and native QK–softmax–PV graphs with carried state; ordered source loops and complete CPU reductions | Qualified source-to-device connection, target hooks/final decoding, device QK–softmax–PV/KV composition and broader source algorithms |
| Configuration | Checked persistent-state cleanup through ordinary shared schedule compilation and actual final bytes; existing configuration checker/hoist primitive | Independently qualified target state decoder and ordinary model integration |
| Final program limits | Actual decoded instruction size, branch/address checks and closed-region segmentation through declared interfaces | Independently qualified target decoder and runtime continuation integration; ordinary targets retain their existing linked-image admission |
| Persistent state | Runtime state binding, immutable weight hashing/reuse and recurrent relinking | Qualified address windows, ownership and persistence for each device runtime |
| Whole-model coverage | Source routing/census and compiled CPU fallback machinery | Complete realized coverage and unchanged numerical gates for each selected full model |

An analysis, selected composition, metadata assignment or small executed kernel
does not establish these remaining integrations. Actual full outputs and carried
state need paired execution and independent references under unchanged policies.
Functional simulator instruction counts, traffic reductions, compilation cost
and qualified latency measurements are different results and must be reported
separately.
