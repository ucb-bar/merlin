---
title: Pass catalog first — Gemmini storage allocation
kind: design
status: draft
owner: merlin-experiments
last_verified: 2026-10-13
related: [compiler_verification, fresh_compiler_origin, compiler_authoring_prompts, architecture]
code_refs:
  - src/merlin/targetgen/tool_registry.py
  - src/merlin/targetgen/generate_bundles.py
  - src/merlin/targetgen/generate_prompt.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/levels.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/resource_boundaries.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/resource_frontiers.py
  - src/merlin/perf/deps/liveness.py
  - examples/gemmini/platform/hardware.yaml
  - examples/gemmini/target/software-spec.yaml
  - examples/gemmini/target/contracts/target_contract.yaml
  - examples/gemmini/phase0/recipe.yaml
---

# Pass catalog first: Gemmini storage allocation

## Scope and intended result

Start with one catalog entry, `register_allocation`, and one Gemmini Phase 1
authoring experiment. The agent reads a static catalog and independently writes
an allocation stage in its generated Gemmini OOT compiler. For this target the
stage places resident tensor values in scratchpad and accumulator storage; it
does not assume an Atlas-style tensor register file or replace host register
allocation performed by LLVM. Treat “register allocation” as the catalog role,
with local-storage allocation as its Gemmini realization.

The first milestone adds the catalog and its controlled visibility only. It
does not add a broker that chooses passes, new constraint diagnostics, or new
Phase 0 test generation. Existing public capsules and execution grading are
used to evaluate the first run. Later milestones independently add the other
two ideas: IR-based broker diagnostics and constraint-derived Phase 0 tests.

The catalog-only setup is now implemented as the opt-in `EL4-P` level and
`gemmini-pass-catalog` experiment. See the
[setup and launch guide](../../examples/gemmini/phase1/PASS_CATALOG.md) and
[installed catalog](../../merlin/contract/pass_catalog.yaml). The YAML below
records the original proposed vocabulary; the installed schema is authoritative.
No authoring campaign has been run. Target-specific implementation belongs in the candidate's
OOT repository; reusable catalog infrastructure belongs in Merlin.

The objective is safe storage and correct execution, not a required pass name.
An agent may fuse allocation with another stage or use another valid strategy.
For the demonstration, ask it to expose an allocation stage so its decisions
can be inspected, without making that stage's spelling a correctness gate.

## What the Atlas compiler teaches us

Reviewed source: `~/Projects/atlas-mlir`, commit
`d211b6f5db7e1e78fa34b990c268e0b41af11d04`. This was source inspection,
not a build or execution qualification. The relevant files are:

| File in atlas-mlir | Observed behavior | Portable requirement |
| --- | --- | --- |
| `include/Atlas/AtlasVirtualAllocation.h` | `VirtualAllocationPlan` maps virtual values to tensor/scalar registers, MXU resources and DMA placements; comments require rebuilding after source values/order change | Allocation is tied to the current IR and resource declarations |
| `lib/AtlasVirtualAllocation.cpp` | `colorValues` computes CFG liveness, constructs interference, and uses stable degree-first greedy coloring | Allocate simultaneous live storage, not the total number of SSA values |
| Same file | BF16 occupies pairs; FP8 and pack operations change reservations; results cannot alias operands under its conservative model | Explicit footprint, alignment, reservations and operation alias rules are required |
| Same file | Exhausted coloring reports failure; no spill implementation appears in `colorValues` | A limited allocator must refuse unsupported pressure rather than emit unsafe assignments; greedy failure alone does not prove infeasibility |
| `lib/AtlasRegisterAllocationVerification.cpp` | Independently recomputes liveness; rejects missing, duplicate, foreign/stale assignments, bounds violations, misaligned pairs, reserved registers and interference | Check the proposal against IR and declarations independently of allocator internals |
| Same file | Also checks argument staging, ABI, DMA writes and helper contents | Lowering introduces storage uses/clobbers beyond ordinary SSA liveness; declare them or exclude the affected program |
| `test/register-allocation-verification.cpp` | Starts from allocator placements, mutates public assignments, and checks rejection | Test valid proposals and targeted corruptions separately |

Atlas's hardcoded register ranges, temporary choices, mixed-format partitioning
and DMA layouts are reference implementation details. None become Gemmini facts
or constants in the generic catalog. The reference informs reviewer requirements;
the Phase 1 agent receives independently declared hardware/software inputs and
the generic entry, not Atlas source, placements, test answers or IR dumps.

## Proposed first catalog entry

Proposed destination: `merlin/contract/pass_catalog.yaml`, with a small schema
and loader. Field names here are proposed vocabulary, not existing fact paths.
Keep conditions descriptive in the first iteration: the agent reads them and
reasons from admitted declarations. Automated applicability evaluation is deferred
to the diagnostics milestone.

```yaml
schema: merlin.pass_catalog.v1
roles:
  - id: register_allocation
    summary: Assign virtual resident values to finite physical storage.
    consider_when:
      - Values require explicit placement in declared register files or local memories.
    inputs:
      hardware:
        - storage_spaces_with_capacity_and_address_units
        - alignment_and_reserved_regions
        - legal_storage_classes_and_access_geometry
      software:
        - operation_reads_writes_aliases_and_completion_semantics
        - value_footprint_and_layout_rules
      ir:
        - definitions_uses_shapes_types_and_operation_order
    unknown_policy: Report missing declarations; do not invent defaults.
    output:
      - value_to_space_offset_extent_assignments
      - explicit_refusal_for_unsupported_or_unresolved_programs
    obligations:
      - Assign every required value exactly once, accounting for declared aliases.
      - Stay within each declared space and outside reserved regions.
      - Satisfy declared footprint, alignment and access geometry.
      - Keep interfering storage disjoint unless declared semantics permit aliasing.
      - Preserve numerical results and required effect order.
    strategies:
      - liveness_based_reuse
      - linear_scan_or_interference_based_placement
      - spill_recompute_or_reschedule_if_supported
    verification:
      storage_safety:
        method: independent_allocation_check
        scope: current_IR_assignments_and_declared_resource_model
        rollout: later_diagnostics_milestone
      numerical_semantics:
        method: existing_execution_comparison
        scope: executed_cases_under_declared_numerical_contract
        rollout: first_catalog_milestone
      resource_geometry:
        method: lean_resource_checks
        scope: explicitly_encoded_bounds_alignment_and_disjointness_obligations
        rollout: deferred_until_existing_bridge_is_located_and_qualified
```

Validate the schema, unique role IDs and obligation fields. Test that no target
name, fixed capacity, reference-code path, or solution is required by the entry.
It provides requirements and possible strategies, not an allocator template.
Verification method declarations express intended obligations; their presence
must never count as evidence that a check or proof has run.

## Milestone 1: catalog-only agent experiment

1. Audit admitted Gemmini inputs. The hardware selector requires extracted
   arrays, memories, timing and source consistency. The software specification
   independently declares numerical and transfer semantics. The target contract
   explicitly leaves memory capacities and mesh geometry to extracted facts.
   Identify the selected scratchpad and accumulator spaces, address units,
   capacity, row geometry, footprint conversion, alignment, reservations and
   completion rules. Record every absent field as unknown with its provenance.
   Do not derive hardware facts from a handwritten compiler or validation model.
2. Add the single-entry catalog, schema and loader. Audit bundle visibility
   before landing the file: existing bundle grants of `merlin/contract/` must
   not silently expose it to control arms. Land the access restriction and file
   together. Reuse `tool_registry.py`, `generate_bundles.py` and readiness checks.
3. Introduce a separate opt-in catalog-only experiment variant through Phase 1
   level/arm registration. The only extra authoring information is the static
   catalog. Add an instruction through `generate_prompt.py` to inspect applicable
   roles against admitted facts and report missing information. Do not introduce
   a planning broker in this milestone.
4. Launch a fresh Gemmini candidate with the existing build/entrypoint contract.
   Ask it to create an allocation stage before physical command/address emission:
   map virtual resident values to declared scratchpad/accumulator regions, reuse
   dead storage, handle reservations, and refuse unsupported cases explicitly.
   Begin with straight-line sequential operations, static sizes and known effects.
   CFGs, asynchronous reuse, spills and implicit aliases require later support.
5. Require a review artifact containing before/after IR, assignments and the
   actual pipeline invocation. Validate that the production entrypoint executes
   the stage; a separately demonstrated allocator is insufficient. This initially
   uses the existing compiler execution path and manual inspection, not a new
   general per-pass runner.
6. Build the candidate and run existing public Gemmini capsules with numerical
   comparison and the existing grading ladder. A software comparison is not a
   hardware qualification; record unavailable tiers separately. Freeze the
   candidate, inputs, catalog digest, prompt and transcript for attribution.

**Exit criteria:** the catalog validates, only the opt-in variant sees it, the
agent authors an allocation stage wired into a buildable Gemmini compiler, and
existing execution checks establish the measured behavior. Report any gap in
independent allocation safety checking rather than claiming it is proved.

Compare catalog-only against the same baseline inputs, budget and seeds. Inspect
whether the agent uses the entry and derives storage requirements correctly;
measure build success, existing capsule results and unsafe/refused cases. A run
without a stage is still evaluated by behavior, then reported as missing the
requested inspectable demonstration.

## Worked Gemmini example to use in the campaign

Use independently declared load, contraction and readout semantics to construct
a sequential region: load operands A and B into scratchpad, compute an i32
accumulator C, read C to the output, then load D and E and compute/read F.
A and B remain resident until their final hardware read completes. C remains
resident until readout completes. D/E and F can reuse the corresponding dead
regions if the declared completion/effect model permits it. Preserve an operand
into the second computation in a second case to make premature reuse observable.

Let selected facts provide usable capacities `C_s` and `C_a`; let footprint rules
provide row extents `r_A`, `r_B`, `r_C`, etc. Require a legal aligned placement
for the simultaneously live scratchpad operands and a separate placement for
the accumulator. Do not compare i8 scratchpad bytes directly with i32 accumulator
rows, or count tensor values as interchangeable one-slot registers. Select legal
shapes from extracted geometry; no fixed example capacity represents the device.

Expected agent output is a mapping from virtual values to space/offset/extent,
used by command emission, plus an explicit explanation/refusal when necessary
declarations are missing. The eventual independent checker recomputes lifetimes
from declared semantics, not lifetimes supplied by this mapping.

## Milestone 2: integrate broker diagnostics

After the catalog-only run, add an answer-free diagnostic through the existing
Phase 1 broker/tool registration. It accepts the current generic xDSL/MLIR IR,
its operation-semantics declarations and optionally proposed assignments. Unknown
operation effects, dynamic sizes or unsupported control flow produce an explicit
unsupported/unknown result, never a successful partial check.

Derive required values, footprints and liveness from actual IR. Distinguish
read-before-write reuse from operations that require separate operands/results.
Track mutable storage versions/aliases only when declared; the initial subset
can reject them. Require sequential completion, or explicit completion points,
before releasing storage. Existing `perf/deps/liveness.py` analyzes decoded
architectural instructions; reuse its concepts where appropriate, but do not
treat it as a ready-made SSA tensor-storage analysis. Audit other liveness
infrastructure before introducing a new implementation.

Return capacity/pressure witnesses and, for proposed allocations, coverage,
bounds, alignment, reservation and overlap violations. Each diagnostic names
the values, operations/program point, footprint, space and source fact. Give the
agent evidence and possible remedies; do not prescribe a pass because the total
number of virtual values exceeds capacity. Alignment/fragmentation can invalidate
a layout even when the sum of live footprints fits.

Test broker round trips, unknown facts/effects, safe reuse, true pressure and
bad proposals. Bind verification to digests of current IR, assignments, hardware
facts, semantic declarations and checker version; mutate each input and ensure
old success cannot be reused. Ensure diagnostics cannot read golden allocations
or expose reference implementations. Add this as a separate opt-in variant and
compare it with catalog-only under the same experimental conditions.

## Milestone 3: integrate constraint-derived Phase 0 tests

Extend existing `phase0/resource_boundaries.py` and `resource_frontiers.py` where
their geometry/sweep contracts apply; audit their ability to represent lifetime
patterns instead of writing a parallel boundary generator. Emit capsules through
the existing Phase 0 route from selected facts and software declarations. Keep
capacity exhaustion separate from a semantically executable hardware workload:
an oversized resident working set may require tiling, spilling or rescheduling.

| Generated case | Constraint exercised | Failure it should catch |
| --- | --- | --- |
| Below usable capacity | Bounds and ordinary placement | Incorrect offsets or footprint conversion |
| Exactly at usable capacity, when a legal aligned case exists | Boundary inclusivity | Off-by-one bounds or unnecessarily rejected legal layouts |
| Above simultaneous capacity | Pressure handling | Unsafe emission rather than a supported remedy or explicit refusal |
| Many sequential values with small peak footprint | Lifetime-based reuse | Allocating by total value count or leaking dead regions |
| A value retained across several operations | Live-through storage | Premature reuse/clobber |
| Multiple outputs retained until declared consumption | Output lifetimes and coverage | Omitted output or overlapping live results |
| Aligned footprint and reserved-region boundaries | Geometry and reservations | Misalignment or use of unavailable addresses |

Every test records its constraint, fact provenance, lifetime pattern and intended
fault. If exact-capacity shapes are unattainable under geometry/alignment, record
that limitation and use the nearest legal boundaries. For the initial allocator,
unsupported over-capacity cases are refusal tests, not passing execution cases;
later supported remedies must pass numerical and hardware constraints. Positive
capsules compare outputs under declared numerical semantics. None asserts a pass
name, a reference allocation or a specific strategy.

Test deterministic generation, sensitivity to changed capacity/geometry facts,
and unknown-field handling. Test on independent shapes and a second target's
declarations before claiming generality. Compare catalog + diagnostics against
catalog + diagnostics + generated tests; preserve hidden-answer boundaries and
freeze separate bundle identities instead of changing historical runs.

## Independent verification and Lean follow-up

Use an independently owned checker to validate required-value coverage, bounds,
alignment, reserved regions and overlap under independently recomputed lifetimes.
Seed negative proposals: overlap live A/B, move a region past capacity, remove C,
misalign a base, use a reserved region, duplicate an assignment and reuse a
success result after IR/effects/geometry changes. Pair them with accepted reuse
cases so the checker cannot pass qualification by rejecting every proposal.

`lean_resource_geometry.py` was not found in this Merlin checkout. Locate its
repository/revision and inspect its contract before planning an actual adapter.
If available, translate only the declared resource obligations it supports,
record assumptions and exact input digests, and require a completed Lean check
before reporting a proof. No placeholder verification method counts as success.

Storage safety, numerical preservation and hardware-model fidelity are separate
claims. Allocation verification covers storage under declared effects/geometry;
execution comparison supplies evidence on tested numerical cases; selected RTL
and source-consistency evidence qualify the hardware assumptions. Neither a
geometry proof nor execution on a handful of inputs proves the entire compiler.

The final demonstration should retain the complete chain: selected declarations
→ generated public capsule → current IR diagnostic → agent proposal → independent
verification → emitted program → execution comparison, including deliberately
broken proposals and stale-result rejection. This completes the storage example
before extending the catalog to scheduling or timing.
