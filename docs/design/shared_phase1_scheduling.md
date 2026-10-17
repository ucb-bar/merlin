---
title: Shared Phase 1 scheduling
kind: design
status: current
owner: core
last_verified: 2026-10-16
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

Select `--common-substrate merlin.common_compiler_substrate.v4` when generating
a fresh Phase 1 release. V4 adds only the shared planner and decision IR to the
V3 public module inventory. It retains V3's checker inventory and formal
coverage policy. Historical V1/V2/V3 releases keep their original inventories;
existing grants are not silently expanded.

Standalone compilers can export the exact canonical sources:

```sh
python -m merlin.compile.scheduling.export --output-dir /path/to/backend/mlir_oot
```

The command prints the release schema and both source hashes. Package both
`common_dense_planning.py` and `common_dense_ir.py` in the compiler's manifest.
The planner requires only Python's standard library; the decision IR uses
xDSL. Two independent backends and rectangular synthetic geometries exercise
the same planner; an isolated test imports only the published source closure.

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
remains an acceptance gate. Separate floating-model quality reporting is
required; capture, routing census and kernel checks do not satisfy that gate.
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
