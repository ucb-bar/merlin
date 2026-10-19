---
title: Shared Phase 1 scheduling
kind: design
status: current
owner: core
last_verified: 2026-10-19
related: [../reference/shared_compiler_passes.md]
code_refs:
  - src/merlin/transforms/pipeline.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - src/merlin/compile/scheduling/device_spills.py
  - src/merlin/compile/scheduling/row_softmax.py
  - src/merlin/compile/scheduling/program_limits.py
  - src/merlin/model_check/program_admission.py
---

# Shared Phase 1 scheduling

Merlin owns reusable scheduling algorithms. Target backends supply primitive
geometry, storage available after reservations, supported emission forms and
instruction legality. The algorithm never selects a target or model by name.

## Backend-independent inputs

Shared passes must consume typed operation semantics, layouts, numerical
boundaries, aliases and declared memory/execution effects. Resource and
instruction capabilities are explicit configuration inputs. Backend Python
classes, emitter function names, private command-buffer keys and a particular
target dialect spelling are not the shared input contract.

Run a pass at an IR level that still represents the decisions it changes.
Fusion and tiling need computation semantics; a final opaque command buffer
cannot supply those semantics automatically. Later verification needs actual
effects reconstructed from emitted operations, rather than estimates copied
from the selected plan. A backend may expose those through supported operation
interfaces or an explicit projection at its integration boundary.

An unknown operation is a conservative barrier until its relevant semantics
and effects are available. Preserve a legal unoptimized route when possible;
otherwise diagnose the missing contract. Do not recognize a backend by name,
guess effects from its instruction spelling, or require it to use today's
emitter structure. Generality means a stable semantic interface with explicit
coverage, not automatic interpretation of arbitrary unknown IR or binaries.

The shared compiler can be developed and tested without editing a target
repository. Tests should exercise neutral representations, different resource
geometries and unavailable capabilities. Existing target integrations are
consumers and validation examples, not definitions of the shared API.

## Relation to established compiler abstractions

IREE's [Flow/Stream/HAL separation](https://iree.dev/reference/mlir-dialects/Stream/)
distinguishes dispatchable tensor computation, explicit asynchronous scheduling
and device execution. Its [Stream passes](https://iree.dev/reference/mlir-passes/Stream/)
propagate completion timepoints and reuse storage under timeline ordering.
Merlin's corresponding integration contracts must retain computation semantics
before dispatch and actual completion dependencies before storage reuse. A
source-level last use alone cannot establish asynchronous completion.

XLA's [lowering pipeline](https://openxla.org/xla/hlo_to_thunks) separates layout
assignment, fusion, copy insertion, scheduling and buffer assignment. Layout
conflicts can require real copies; ownership can also require copies before
in-place writes. This motivates separate layout, alias and realized-effect
checks rather than assuming every copy is redundant. Its
[rematerialization pass](https://openxla.org/xla/hlo_passes) also illustrates a
different legal choice: recompute a pure intermediate to shorten its storage
lifetime. Admitting that choice in Merlin would require current effects,
numerical legality, resource bounds and an explicit cost policy.

These transformations and abstractions are established compiler techniques.
Their presence does not by itself establish a novel contribution. Merlin's
research question is whether explicit resource facts and independently qualified
software semantics can make reusable transformations easier to compose and tune
for a newly generated backend, while retaining checked whole-model behavior.
This requires held-out workload/target evidence and comparisons with equivalent
shared libraries and verification rules. It cannot be inferred from a list of
pass names or a successful tuning run.

IREE's [tuning interface](https://iree.dev/reference/tuning/) already separates
dispatch knobs, validity constraints and Transform dialect tuning specifications.
Merlin Phase 2 should likewise choose among legal bounded alternatives and
parameters through the same checked pass interfaces. Pass ordering remains
constrained by prerequisites, current analyses and invalidations; tuning cannot
remove those obligations. Existing handwritten algorithms and future
agent-proposed choices must use the same original input contracts and admission
rules. This is a design comparison, not a claim that all corresponding passes
or cross-target integrations are implemented.

## Dense operand reuse and command budgets

`merlin.transforms.dense_planning` searches a bounded M/N block domain. Each
candidate retains its output block in private accumulators and reuses loaded
A/B panels. K advances in source reduction order. A target may allow partial
output tiles and blocks explicitly. One largest legal K panel is considered
per M/N choice; this is a bounded initial candidate family, not exhaustive
schedule search.

```python
from merlin.transforms.dense_planning import (
    DenseProblem, DenseResources, choose_dense_block,
)

problem = DenseProblem(
    m=64, k=64, n=64,
    tile_m=16, tile_k=16, tile_n=16,
    lhs_bytes=1, rhs_bytes=1, accumulator_bytes=4,
)
resources = DenseResources(
    operand_bytes=4096, accumulator_bytes=8192,
    max_m_blocks=8, max_n_blocks=8, max_k_blocks=1,
)
choice = choose_dense_block(
    problem, resources, static_budget=64,
    price=adapter.static_command_count,
    admits=adapter.is_legal,
)
```

The adapter responsibilities are explicit:

| Input or action | Owner |
| --- | --- |
| Primitive tile and operand/accumulator widths | Selected target facts |
| Usable capacity after banks and live reservations | Target adapter |
| Supported partial tiles and block limits | Target emitter |
| Layout, numerical and instruction legality | Target adapter |
| Candidate search and deterministic ranking | Merlin |
| Static command count or conservative upper bound | Target emitter |
| Selected command lowering and realization checks | Target adapter |

An unavailable candidate returns `None`; the caller keeps its legal streaming
or loop fallback, or reports that this emitter cannot implement the request.
Search rejects a domain above its candidate budget before enumerating it.
It does not allocate tensors or expand dynamic iterations.

Ranking minimizes requested A/B payload bytes, then load requests and static
commands. Payload counts exclude bias and protocol/bus overhead. These are
traffic estimates, never invented cycle savings. Capacity calculations use
padded tile extents even when the logical edges are smaller.

`merlin.transforms.dense_ir.dense_schedule_op` records the selected problem,
primitive and block shapes, element widths, storage footprints/capacities and
static command bound as `schedule.block_contraction`. Its verifier recomputes
the footprint and checks bounds. The operation executes no command. Target
realization checks remain required; this metadata alone does not prove
aliases, asynchronous completion, or preservation of numerical boundaries.

Both regular Gemmini implementations consume these exact shared sources in
their ordinary compiler entrypoints. Their constraints differ: the generated
backend supports bounded loop bodies; the handwritten backend retains its
straight-line emission limit. A shared decision algorithm does not promise
identical instruction sequences or equal supported shape ranges.

## Shared emission conformance

`DenseEmission` is the common observation interface for this candidate family.
Each adapter walks its emitted operations and reconstructs separate LHS, RHS
and bias load counts, output tile stores, static command sites and transfer
address high-water marks. Dynamic transfer counts include loop multiplicities;
static command counts count each site once. The high-water marks are exclusive
byte offsets relative to the assigned regions, rather than device addresses.

```python
from merlin.transforms.dense_planning import DenseEmission, verify_dense_emission

# Reconstruct from emitted operations, not from the chosen block's estimates.
observed = DenseEmission(
    static_commands=static_sites,
    lhs_loads=dynamic_counts["lhs"], rhs_loads=dynamic_counts["rhs"],
    bias_loads=dynamic_counts["bias"], output_tiles=dynamic_counts["output"],
    operand_high_water=operand_end, accumulator_high_water=accumulator_end,
)
verify_dense_emission(choice, observed, static_budget=command_budget)
```

The shared check requires each transfer role to match the selected plan and
both storage bounds and the command bound to hold. Equal total load counts do
not excuse replacing an A load with a B load. Static prices may be conservative
upper bounds. Missing storage observations refuse validation. Target adapters
also check their address spaces, operand ownership and transfer extents.
Both regular backends record the checked observation in their command-buffer
schedule report. These checks run during ordinary lowering, without an agent.

This is a small common adapter interface, not the complete scheduling adapter.
It does not prove alias safety, asynchronous completion, compute addresses,
epilogue semantics or eliminated intermediate transfers. The complete
lifetime/effect and fusion realization contracts remain required before those
optimizations can be admitted.

The target-free conformance tests evaluate every candidate for three distinct
rectangular primitive geometries, with output/reduction tails and optional bias.
An enumerated traversal checks all output elements, increasing K order and
actual edge payloads against a direct integer contraction. Separate mutation
tests reject role substitutions, missing transfers, storage overflow and command
overruns. Both compiler entrypoints also reject corrupted target transfers.
Synthetic geometries exercise abstraction coverage; they do not qualify new
RTL configurations or numerical modes.

## Synchronous schedule lifetimes

`compile.scheduling.block_schedule.check_residency` now checks both operand
residency and accumulator lifetimes. `schedule_contraction` and
`schedule_convolution` invoke it before returning a schedule. No model selection
or additional user annotation is required for this existing representation.

| Schedule state | Required invariant |
| --- | --- |
| Loaded operand | The expected block, gathered rows, row offsets and sufficient columns remain resident. |
| Resident weights | A preload establishes the weight block; each compute agrees with its reduction index and dimensions. |
| Pending preload | Exactly one compute consumes it before another preload or store. |
| Accumulator rows | Each row belongs to the output and reduction prefix being updated. |
| Reduction | Initialization precedes increasing source K; later updates accumulate. |
| Output store | Every output tile is stored exactly once, after its complete reduction, from its own accumulator. |

The checker derives the reduction sequence and edge dimensions from the
contraction. For convolution the sequence is kernel taps followed by channel
blocks, including channel tails. It does not assume that all targets use the
same primitive size or that one physical accumulator must hold an output for
the entire program. Rows can be reused after their previous output is stored.
Tests exercise such reuse with a smaller accumulator than the original layout,
as well as all six loop orders, matrix tails and gathered convolution.

```python
from merlin.compile.scheduling.block_schedule import check_residency

# Recheck an adapter's transformed BlockSchedule before lowering it.
check_residency(transformed_schedule)
```

State scales with represented rows and output tiles, without allocating model
tensors or enumerating tensor elements. Unknown operations, evident output
aliasing, or shared operand/accumulator backing without an alias model are
refused. This is the synchronous abstract schedule contract. It does not prove
that asynchronous hardware has completed a transfer or that distinct external
views have different backing storage. Such targets need qualified effect and
completion adapters; physical byte mappings cannot be inferred from row counts.

The regular dense backends now project bounded, unseeded emitted regions into
this representation. Their adapters decode actual source addresses, row strides,
local addresses, accumulator flags and readout widths. They reconstruct operand
identities from emitted loads, rather than copying the planned block identities.
The generated adapter expands runtime loops only after checking an explicit
operation budget. Unsupported bias seeds, outer batch effects or large dynamic
regions report `unavailable`; existing legal lowering remains available.

`DenseEmission` remains a transfer/budget report. A separate `lifetime` report
records this synchronous projection's status and scope. A checked local lifetime
does not establish numerical policy, external-view aliasing or hardware
completion. Those obligations remain explicitly unestablished, so this result
alone cannot authorize cross-region fusion or a new local handoff.

## Physical epilogue composition

Compute-group formation asks `TargetReadout.applies_composition` about the
complete candidate stage set on the selected unit. Stages offered by distinct
readouts or units cannot be combined into a fictitious path. Existing seed
routes may license a bias before the readout. Existing numerical checks still
control operation order, conversions, rounding and reassociation.

Phase 0 projects existing readout declarations into the operation capability
registry with configuration, scalar ABI, stage routes and evidence references.
Projection leaves execution support `unknown`. A positive observation must
identify the same explicit configuration and carry probe evidence. An
unordered `applies` list never supplies an ordered-stage proof.

This is a physical composition gate, not a general fusion search engine.
Connected dispatch planning still owns cross-kernel handoffs and its checked
realization report. A regular Gemmini direct output handoff remains disabled
on the currently tested runtime: its local-store instruction traps.

## Fresh Phase 1 releases and standalone packages

Select `--common-substrate merlin.common_compiler_substrate.v5` for a fresh
Phase 1 release that includes the checked block scheduler. V5 adds exactly
`merlin.compile.scheduling.block_schedule` to the V4 public source inventory.
Its isolated source closure supports importing that module without the broader
compile package, RTL derivation helpers or a local Merlin installation.
V4 adds the dense planner and decision IR to V3. V4 and V5 retain V3's checker
inventory and formal coverage policy. Historical V1/V2/V3/V4 releases keep
their original inventories; existing grants are not silently expanded.

Standalone compilers can export the exact canonical sources:

```sh
python -m merlin.compile.scheduling.export --output-dir /path/to/backend/mlir_oot
```

The command prints the release schema and both source hashes. Package both
`common_dense_planning.py` and `common_dense_ir.py` in the compiler's manifest.
The planner requires only Python's standard library; the decision IR uses
xDSL. Two independent backends and rectangular synthetic geometries exercise
the same planner; an isolated test imports only the published source closure.
Add `--include-block-schedule` to export the byte-exact canonical checker as
`common_block_schedule.py`. This opt-in export returns schema v2; the original
two-file v1 export is unchanged. Standalone copies contain no target-specific
fork of the shared scheduling or lifetime algorithms.

## Reusable pass catalogue and phase ownership

Shared passes belong in Merlin. Phase 1 target implementations provide explicit
adapters and integrate the passes into their compiler pipeline. Phase 1 builds
and qualifies that compiler; ordinary model compilation executes its passes.
Phase 2 may search and calibrate the same legal choices. It must not be needed
to repair missing correctness mechanisms in a claimed Phase 1 workload scope.

The catalogue separates implemented shared mechanisms from target integration.
The [pass reference](../reference/shared_compiler_passes.md) defines each admitted
domain, ordinary compiler entry, fallback and executable example. Neutral
execution tests establish those mechanisms; hardware qualification remains a
separate obligation.

| Capability | Shared mechanism | Target inputs and responsibilities | Current boundary |
| --- | --- | --- | --- |
| Temporary memory | Alias-aware liveness, lifetime packing and allocation verification | Physical backing, units, alignment, banks, packing and DMA reservations | Consumed private straight-line relocation and source/host reuse are tested in ordinary recurrent models. General persistent/loop allocation and qualified device adapters remain open. |
| Register spilling | Pressure analysis, exact spill placement, slot reuse and recomputed liveness/allocation/completion | Register classes, exact spill/reload conversions, legal storage and instructions | Bounded synchronous device spill repair executes in neutral recurrent models. Async spilling and computed expression slicing are unsupported; real-device hooks remain unqualified. Native CPU allocation stays with its toolchain. |
| Looped tiling | Bounded candidate search, ordered loops, affine addressing and tail domains | Primitive shapes, capacity, layouts and legal accumulator continuation | Compact dense source lowering executes checked counted loops in ordinary recurrent models. Arbitrary target lowering and broader numerical reductions remain unsupported. |
| Fusion and transfers | Compatible physical composition, allocation fallback and exact private-copy elimination | Ordered arithmetic/conversions, layouts, backing and supported handoffs | Total-integer pointwise composition and copy elimination execute in ordinary recurrent models. Floating/contraction epilogues and asynchronous handoffs remain unsupported. |
| Configuration cleanup | Persistent field tracking and elimination checked against actual effects | Ordered idempotent writes, resets, clobbers and state persistence | Ordinary neutral models execute fewer setup instructions with equal complete results and settings. Pulse/unknown semantics remain barriers; real-target state adapters need qualification. |
| Reductions and attention | Complete-domain row planning, masks and explicit intermediate lifetimes | Accumulation type, reduction order, conversions and supported numerical algorithms | Counted full-row softmax and narrow source scheduling execute inside complete attention graphs with carried state. General device QK/PV/KV composition remains open; independent tile softmax is never a whole-row substitute. |
| Numerical lowering | Ordered conversions, explicit quantization boundaries and contract checks | Arithmetic domains, rounding, saturation, scaling and qualified instruction semantics | Existing typed policies and checks are partial coverage, not equivalence for every available hardware instruction. |
| Program size | Final decoded instruction/byte/address/branch checks and bounded legal segmentation | Encoded instruction widths, branch reach, IMEM limits, relocations and launch semantics | Final-byte checks and qualified closed-region segmentation execute in neutral tests. Ordinary build/cache/relink checks the selected linked section; it never installs hypothetical segments or supplies missing target continuation. |
| Persistent state | State ownership, weight/KV placement, rebinding and address-range planning | Loader/runtime ABI, supported DMA windows, address arithmetic and persistence | Existing model state binding and verified weight reuse do not establish arbitrary device addressability or resident-session timing. |
| Whole-model placement | Source operation census, legal route selection and realized coverage accounting | Supported compositions and actual device/compiled-CPU implementations | Existing pre-build routing must still prove each requested model's coverage, linking and numerical behavior. |

These mechanisms are parameterized by semantics and resource/execution contracts,
not model names or target names. RTL can supply structural geometry and encoding
facts. It cannot by itself establish a software allocator policy, model accuracy,
completion across an unqualified interface, a numerical approximation contract
or a measured cost. Missing evidence leaves the corresponding implementation
unavailable, with an explicit supported fallback or diagnostic.

Phase 1 must provide correct allocation, bounded whole-region code generation,
numerical lowering, execution ordering, placement and state/address handling for
its declared scope. Spilling is required when that scope exceeds register
capacity and no other legal schedule/fallback suffices. Supported fusion is a
Phase 1 implementation obligation; Phase 2 chooses among qualified fusion,
tiling and placement alternatives using measured feedback.

For comparisons across Phase 1 implementations, freeze the common source
version and expose the same shared passes and verification rules to all
treatments. Record declared parameters separately from measured tuning results.
Target-specific emitter implementations and protected tuning results remain
subject to their own grants. A compiler calling a common planner still needs
emission and execution checks; neither the library call nor an advertised
capability proves that the decision was realized.

An illustrative ordering is:

```text
source coverage and numerical legality
  -> placement and supported composition planning
  -> looped tiling and full-domain reductions
  -> buffer/register allocation, spills and DMA staging
  -> dependency scheduling and lifetime verification
  -> target instruction emission
  -> final instruction size, branch and address checks
  -> linking, runtime binding and execution qualification
```

These stages may need bounded iteration. Spilling introduces accesses and
temporaries, so liveness, allocation and scheduling must be recomputed. Final
program size must be checked after spills and branch repair. Register spills
and branch relaxation may operate on machine IR rather than tensor MLIR.
Generic algorithms still belong in Merlin; the target adapter supplies exact
encodings, instruction effects and the legal repair/reload sequences.

### Pass prerequisites and invalidation

The implemented common `PassSpec`/`PipelineSession` infrastructure records
prerequisites, produced analyses, invalidations, stages and IR levels. Structured
transforms, compact dense/row compilation and finite schedule compilation use
this infrastructure at their respective representations. There is no claim
that one global pass manager can interpret arbitrary target output.

Analyses bind the current IR, hardware, numerical policy, resources, execution
and shared-library identities. A mutation invalidates prior analyses even if a
pass omitted an invalidation declaration. Replacing a produced analysis also
invalidates its dependent checks. Invalid explicit ordering refuses before
execution; failed sessions cannot issue successful results. Selected fact and
software documents are reopened around relevant callbacks and consumption.

| Stage | Required inputs | Changes that require recomputation |
| --- | --- | --- |
| Composition selection | Operation semantics, numerical boundaries, target legality and effects | Placement, conversion or graph changes |
| Tiling and operand retention | Legal compositions, layouts, usable capacities and emission budget | Composition, layout or resource changes |
| Physical allocation and spills | Selected tiles, aliases, lifetimes and address restrictions | Added transfers, new temporaries or changed retention |
| Dependency and lifetime verification | Actual allocated accesses and completion rules | Reordering, spilling, address reuse or instruction substitution |
| Emission conformance | Selected schedule and actual emitted operations | Any subsequent instruction or control-flow change |
| Final program checks | Final instructions, addresses and branch displacements | Spill insertion, segmentation, relaxation or linking changes |

Fusion selection before allocation is provisional. Allocation must either
realize its local-storage requirements or select a legal fallback and invalidate
the affected schedule. Spill insertion likewise returns to allocation and
dependency scheduling. These repair loops need explicit iteration and code-size
budgets; failure must terminate with a fallback or diagnostic.

The schedule pipeline implements bounded repair: spill insertion restarts
allocation, and failed fusion allocation restores the legal original
composition and replans. Completion, emission and final size checks consume
the repaired representation. Instruction changes require fresh final checks.

Executable tests reject consumers before producers, stale IR/contract/library
analyses and exhausted repair budgets. They also reconstruct mutated actual
output independently: correctly ordered planning does not establish correct
emission. See `test_shared_transforms.py`, `test_schedule_pipeline.py`,
`test_device_spills.py`, `test_selected_fusion_contracts.py` and
`test_selected_program_control.py` for those boundaries.

## Qualification and remaining scope

A regular Gemmini 64x64x64 paired Spike check compares streaming and blocked
code on identical inputs. Both must pass every output, repeated invocation,
input immutability and guard check. Final ELF audits compare custom instruction
sites with emitted primitives. This establishes a kernel-level result.
Functional Spike `rdcycle` values are instruction-model counts and cannot be
reported as accelerator performance cycles.

Earlier model and target-integration measurements remain with their original
compiler revisions and evidence. They do not qualify a new backend or this
branch's complete-model numerics. Shared tests establish the declared planner
and verification scope; end-to-end integration requires separate execution
and numerical checks with identical inputs and frozen policies.
Ordinary neutral dense, pointwise, storage, spill and complete-row models now
compare full outputs and carried state through these shared interfaces. Their
records separate compilation cost, static code, executed commands, requested
payload and functional simulator observations. They do not qualify real-target
latency or physical bus traffic. General device attention, broader fusion and
qualified Gemmini/MX/Atlas integration still require their actual public
execution, numerical, decoding, ownership and continuation contracts.
