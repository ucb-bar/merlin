---
title: Generalization from bounded Phase 1 execution to large compiler inputs
kind: design
status: draft
owner: merlin-experiments
last_verified: 2026-10-08
related: [component_compiler_convergence, fresh_compiler_origin, component_phase2_workflow, component_final_evaluation]
code_refs:
  - src/merlin/experiments/phase0/component_coverage_plan.py
  - src/merlin/experiments/phase0/component_coverage_inputs.py
  - src/merlin/experiments/phase0/resource_boundaries.py
  - src/merlin/experiments/phase1/component_qualification.py
  - src/merlin/targetgen/component_program.py
---

# Generalization with bounded execution

Phase 1 should produce a functional general compiler without executing large
models in its authoring loop. A finite small test suite cannot guarantee
correctness for arbitrary shapes or graph interactions. The protocol therefore
needs distinct execution, compilation, legality and transfer evidence. Passing
one category must never silently discharge another.

This protocol includes proposed extensions. Existing coverage and qualification
enforce immutable guard and withheld membership. Versioned source-derived
budgets now bound logical reference work and data for integer component DAGs
and movement before allocation; unknown numerical paths are refused. The
separate source-only roster and ordinary large compilation path now preserve
their required denominator and join fresh compiler qualification. Independent
static proof producers for the complete checks below remain missing; linking
alone leaves those obligations UNKNOWN and full qualification incomplete.

## Freeze the domain and budgets before authoring

Disclose supported operations, ranks, layouts, numerical rules, shape bounds,
index types, allocation limits and rejection behavior. Select them from the
minimal semantic contract and independently extracted hardware facts. Freeze
public generators, test budgets, coverage classes and private seed commitments
before authoring. Do not derive these selections from final validation shapes,
layer frequencies, reference programs or historical performance.

Define a small execution budget in actual work and storage, including reference
generation, rather than imposing a small limit on every dimension. A long thin
contraction can exercise many reduction tiles with bounded total work. Tail and
multiple-tile cases must also fit the execution budget. Address-width and very
large footprint cases can belong to compilation and legality checking instead.
An obligation exceeding a budget is missing coverage until assigned to an
appropriate declared mode; it cannot disappear from the denominator.

## Three complementary Phase 1 evidence classes

| Class | What runs | What it establishes |
| --- | --- | --- |
| Bounded execution | Ordinary frontend, lowering, link and runtime on small independent inputs | Complete numerical outputs, real instructions, ownership, publication and synchronization for those executions |
| Scale compilation | The same ordinary pipeline through the final linked artifact on large independently generated shapes | Actual lowering/link availability and discharged static legality obligations; no numerical or timing claim |
| Bounded graph interaction | Small independent multi-operation graphs through the same pipeline and runtime | Composition, aliasing, reuse, observation, fallback and numerical behavior for those graphs |

### Bounded execution and metamorphic checks

Cover one and multiple tiles, tails on each axis, rectangular shapes, alternate
layouts and numerical stress. Exercise split reductions, padding/cropping,
batch decomposition, reused versus freshly prepared inputs and legal fusion
versus unfused execution against an independent oracle. Derive the relation's
preconditions from the numerical contract. Floating point reassociation and
quantized reduction are not universally equivalent; never assume exact
equality or widen an original tolerance to make a relation pass.

Keep complete outputs and guard regions. Changing seeds, values and graph
structure helps expose lookup implementations, but randomized testing remains
bounded evidence. Public development variants are tuning feedback. Private
transfer variants remain protected certification inputs, with their feedback
policy and attempt limits fixed before authoring.

### Large shapes without large numerical execution

Generate shape-only source IR independently of validation. Compile it through
the ordinary frontend and backend to the linked ELF, without materializing huge
weights or golden tensors. Bind each result to the exact source, compiler,
ordered stages, toolchain, emitted IR, objects and final artifact. Compilation
must not select a test-only implementation or stop at a backend's plan JSON.

An independent checker must reopen the emitted program and verify obligations
such as:

- Every intended iteration/output is covered, with tails and reductions accounted for.
- Index arithmetic, strides, byte extents and encoded fields stay within their actual widths.
- Declared tile footprints, simultaneous live allocations and double buffers fit the selected stores.
- Lifetime, alias, dependency and transfer ranges agree with the generated program.
- Large admitted inputs can stream or tile across capacity limits; exceeding resident capacity alone cannot justify refusing an input inside the declared supported domain.
- All required symbols link, and every executable section passes the instruction policy.

Use structural IR, independently checked bounds or explicit discharged solver
obligations. A candidate's claim that its own schedule is legal is insufficient.
Checks apply only to the supported representation and proven domain; an opaque
or unproved case stays UNKNOWN. Code size and compilation cost must be recorded
to expose exhaustive shape expansion. Legitimate specialization from current
IR remains allowed, with its legality and resource checks.

A scale-compilation pass proves neither large-input numerical behavior nor
physical asynchronous correctness. Those remain separate obligations.

### Graph structure without full models

Generate small graphs from independently reviewed operation semantics. Include
chains, forks/joins, shared producers, multiple observed outputs, views and
aliases, repeated invocation, immutable weight reuse and buffer recycling.
Include supported host/device transitions and explicit unsupported cases.

Vary graph depth, consumer count, layouts and numerical stress within the
execution budget. Test contraction/requantization/activation chains, residual
joins, and attention primitives where independently selected semantics require
them. Do not copy a validation block, topology, weight or shape roster.

Check lifetime and dataflow on deeper synthetic graphs without expensive
numerical execution where an independent structural checker can discharge the
obligation. Numerical error propagation still needs its own bounded execution
and numerical contract; local tolerance passes do not guarantee a final output
will pass the original whole-model gate.

## Phase 2 tests performance regimes and preserves correctness

Phase 2 needs separately generated multi-size components, including streaming,
long reductions and multiple tiles, plus composition/reuse scenarios. Keep
calibration and held groups separate by shape and regime. Fix coverage strata
and scoring policy independently of validation layer frequencies. A tiny case
does not establish performance in a capacity, bandwidth or reuse regime it
never exercises.

Use analytical bounds to identify compute, traffic, command, packing, dispatch
and materialization costs, then check predicted rankings against admitted
measurements. Roofline alone cannot establish scheduling, overlap, contention,
host cost or final cycle parity. A prediction outside its qualified domain is
an uncertainty or missing measurement, not an admitted timing result.

Every promoted compiler snapshot must retain the Phase 1 execution, graph and
scale-compilation obligations. Phase 2 may repair generic correctness defects
within its authorized compiler edit surfaces, with requalification. It cannot
relax the numerical gate or use final validation failures as tuning feedback.

## Final evidence and fairness

Freeze the compiler and all admitted dependencies before compiling protected
validation workloads. Evaluate their original complete outputs and matched
hardware/timer boundaries against the frozen handwritten reference. The
reference remains private final evidence, never an authoring dependency.

If final validation exposes a missing interaction, that campaign failed. Any
later development campaign must disclose the observation; reusing the same
now-observed workloads cannot be described as an untouched final holdout.

These mechanisms are shared experimental infrastructure, with the same
disclosed access rules for compared arms. Their development history must be
reported. Extracted hardware facts and semantic tests are legitimate inputs;
reference schedules, validation shapes and privately selected winning rules
are excluded. Neither publication upstream nor a generic API makes hidden
reference knowledge an admissible input.

## Required implementation extensions

### Ordinary compilation transport

The core `merlin.targetgen.compile_only_execution.compile_source_only` now calls
the ordinary scoped package commands, selected stock LLVM translation/object
compiler and shared linker. A typed static original ABI preserves every declared
input/output slot; a retained pointer reference keeps the actual kernel reachable
through linker garbage collection without allocating operands or output buffers.
The complete linked artifact goes through the explicitly selected instruction
policy before any execution, which this role never attempts.

Its report reopens exact package/contract file membership, original source, tool
and build selections, LLVM/object/ELF products and every actual invocation. Failed
attempts retain unqualified records. Semantic coverage, index bounds, resource
legality and complete-output coverage remain UNKNOWN. The independently issued
source-only roster now binds the same live hardware, minimal software and original
descriptor to budgeted copy/contraction sources and their complete ordered ABI.
Source failures retain every required member and refuse partial issuance. The
mandatory fresh-origin grading join retains separate compilation and static
denominators; independent static checkers remain required. Caller-written ABI
or report JSON cannot establish
those obligations. Numerical and physical evidence remain separate.

1. Extend the existing source-derived integer reference budgets to every admitted numerical/frontend path, and add distinct evidence modes before data or goldens are allocated. Process and simulator limits need separate controls.
2. Add an ordinary compile-through-link grading role and source-bound scale checks, with a denominator distinct from numerical execution.
3. Generate independent semantic graph interactions and numerical-contract-aware metamorphic relations, with committed private transfer variants.
4. Requalify both static and execution obligations on every promoted Phase 2 snapshot, and qualify predictions by size and reuse regime.

Current resident-size generation supplies boundary cases; it explicitly does
not prove allocation placement, live ranges, scheduling or emitted-code
coverage. Current parameter-axis interactions are also distinct from graph
interaction coverage. Implementing these extensions must preserve those limits
rather than relabel existing results as stronger evidence.
