---
title: Shared Phase 1 scheduling
kind: design
status: current
owner: core
last_verified: 2026-10-17
---

# Shared Phase 1 scheduling

Merlin owns reusable scheduling algorithms. Target backends supply primitive
geometry, storage available after reservations, supported emission forms and
instruction legality. The algorithm never selects a target or model by name.

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

The following is a capability roadmap, not a claim that every row is complete:

| Capability | Shared mechanism | Target inputs and responsibilities | Current boundary |
| --- | --- | --- | --- |
| Temporary memory | Alias-aware liveness, lifetime packing and allocation verification | Physical backing, units, alignment, banks, packing and DMA reservations | Existing SSA resource allocation and host arena reuse cover narrower representations; general device VMEM/DRAM allocation remains open. |
| Register spilling | Pressure analysis, spill placement, slot reuse, liveness and rescheduling | Register classes, exact spill/reload conversions, legal storage and instructions | A general device spill rewrite is not delivered here. Native CPU register allocation remains the selected toolchain's responsibility. |
| Looped tiling | Bounded candidate search, ordered loops, affine addressing and tail domains | Primitive shapes, capacity, layouts and legal accumulator continuation | Shared dense planning and loop segmentation exist; arbitrary target lowering and affine emission verification are not established by a tile example. |
| Reductions and attention | Full-domain reduction planning, masks and explicit intermediate lifetimes | Accumulation type, reduction order, conversions and supported numerical algorithms | Existing CPU softmax and attention preparation do not qualify independent per-tile softmax as a whole-row reduction. |
| Numerical lowering | Ordered conversions, explicit quantization boundaries and contract checks | Arithmetic domains, rounding, saturation, scaling and qualified instruction semantics | Existing typed policies and checks are partial coverage, not equivalence for every available hardware instruction. |
| Program size | Code budgets and final size diagnostics; bounded unrolling and legal segmentation planning | Encoded instruction widths, branch reach, IMEM limits, relocations and launch semantics | Existing command-site budgets bound selected emitters. Final branch repair and segmentation still require target emission support. |
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

The ordering above is a design contract, not a claim that one pass manager
already enforces every stage. The existing common transformation pipeline has
a fixed Python order and per-change checks. Dense adapters currently enforce
selection, emission and realization checks through their ordinary entrypoints.
Neither mechanism yet manages the complete catalogue's analysis dependencies.

The shared pipeline should give each pass explicit prerequisites, results and
invalidations. A dependent pass must consume analyses for the current IR,
selected configuration and numerical policy. An old successful check cannot
approve a changed program. A deterministic dependency order supplies the
default pipeline; an incompatible explicit order must fail with a diagnostic.

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

Tests for that future orchestration must reject a consumer before its producer,
reject stale analyses after a mutation, and verify recomputation after spills
and fusion fallback. Adapter tests must separately mutate emitted operations,
since a correctly ordered plan can still be emitted incorrectly. The existing
dense corruption tests cover that last obligation within their bounded scope.

## Qualification and remaining scope

A regular Gemmini 64x64x64 paired Spike check compares streaming and blocked
code on identical inputs. Both must pass every output, repeated invocation,
input immutability and guard check. Final ELF audits compare custom instruction
sites with emitted primitives. This establishes a kernel-level result.
Functional Spike `rdcycle` values are instruction-model counts and cannot be
reported as accelerator performance cycles.

One full pretrained ResNet-50 pair now passes all 1000 output logits against
the independent integer reference with fixed quantization parameters. Its
functional instruction counts decrease 2.84% inside routed groups and 0.13%
for the full model. This is one evaluation input, not a dataset accuracy or
hardware timing qualification. TinyLlama's complete prefill/decode pair
now completes 128 prefill tokens and 32 teacher-forced decode steps per variant
with identical complete outputs and carried state. Functional counts decrease
7.45% inside routed groups and 0.30% for the full session. The frozen integer
reference and both compiled variants still fail the existing absolute floating
model quality gate on this input. Compiler/reference drift also remains.
Completion and paired equivalence do not satisfy those numerical gates.
The complete lifetime/effect verifier, broader capability-driven fusion and
Atlas integration remain subsequent work through these shared interfaces.

## Bounded storage during model validation

`model_check.full_build.relink` regenerates the input argument table and verifies
the derived weight size and SHA-256 without writing another checkpoint blob.
It reuses the original weight object and checks an unchanged argument-table
call before recompiling the driver. A mismatch refuses relinking.

Saved-model and stage builds accept an optional `weights_source` pointing to
an existing generated blob. Each stage still derives its own layout and full
weight bytes; reuse requires identical size and SHA-256. The runtime generator
then creates a hard link, which requires a compatible filesystem. Ordinary
weight regeneration replaces the destination atomically, preserving any other
stage's shared inode. Different stage blobs require separate storage.

These services are independent of accelerator capabilities and live in Merlin.
They reduce build storage and I/O, without changing target schedules or model
numerics. An execution adapter must preserve binary output bytes and select
the machine declared by the final ELF before complete-state comparisons count
as successful model validation.

Catalog preparation uses this same packed-weight representation. Packed data
stays in `blob_weights.npz`; trailing entry arguments carry its shape and type.
The reference evaluator validates the storage against that argument ABI before
binding it. Missing data or a shape/type mismatch refuses evaluation.

The [catalog storage measurement](../evidence/shared_catalog_preparation_20261016/summary.json)
on full TinyLlama decode changes prepared device IR from
about 2 GiB of text to 3.5 MiB, with the packed bytes stored separately. The
observed preparation run decreases from 13m41s to 55s and peak RSS from 6.9 GiB
to 2.1 GiB. Original reference IR, all 155 group bindings and the five kernel
interfaces are byte-identical. These are compiler preparation measurements;
they establish no inference speedup or numerical qualification.
