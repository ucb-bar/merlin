---
title: Shared transformations, ISA generation and Lean verification
kind: guide
status: current
owner: ir
last_verified: 2026-10-12
related: [verified_compiler_roadmap, compiler_verification, phase0_specification, compiler_authoring_prompts, aws_phase0_phase1]
code_refs: [src/merlin/transforms, src/merlin/verify, src/merlin/targetgen, packages/merlin-experiments/src/merlin_experiments/phase1/brokers]
---

# Shared transformations, ISA generation and Lean verification

The common substrate supplies executed xDSL transformations, an optional
instruction dialect generator and an independent host Lean checking route.
It provides a small, explicitly modeled foundation. It does not establish a
complete tensor-to-device compiler or a whole-program SCF, memory, LLVM or RTL
equivalence theorem. The [capability roadmap](../design/verified_compiler_roadmap.md)
records the remaining pass groups and their required evidence.

## Select a fresh experiment release

Fresh `corpus prepare` selects `merlin.common_compiler_substrate.v2`. Direct
bundle generation selects it with
`--common-substrate merlin.common_compiler_substrate.v2`. Supply an explicit
`--lean-toolchain-manifest` and the reviewed `--host-index-bits` of the selected
host ABI. Missing formal tools cannot satisfy covered obligations. V1 retains
its original source membership and formal registry; it cannot request the new
V2 options or acquire the V2 pass observer through a resumed run.

The profile records the actual common source bytes, automatic checker sources,
formal coverage policy and selected toolchain. Earlier frozen bundles retain
their original interpretation; preparing a new profile does not revise an old
run or prove the worker's execution readiness.

| Availability in a new common-substrate release | Additional author capability |
| --- | --- |
| Every EL | Common transformations and analyses; host-owned automatic preservation checks |
| EL4 | Existing RTL-informed authoring inputs and RTL-check feedback |
| EL5, extending EL4 | Declarative ISA instruction dialect generation |
| EL6, extending EL5 | Interactive Lean proof inspection and checking |

Automatic checks for common transformations are shared infrastructure, rather
than an EL6-only benefit. Exact source grants determine which author APIs are
available. The host checker and its trusted toolchain need not be exposed to an
author merely because the author can request a common transformation.

Phase 2 inherits the exact original Phase 1 compiler tool profile, target/spec
inputs, runtime grants and proof policy. It cannot silently add ISA generation,
Lean authoring or a different proof toolchain after Phase 1 admission.

## Prepare the formal toolchain on the worker

Install a reviewed Lean and Lean-MLIR revision independently of the compiler
candidate. Build the dependency library's `LeanMLIR.Framework.Basic` with its
own pinned dependency lock. Use that project's declared `lean-toolchain`;
Merlin does not select a floating release or download dependencies during
verification. One exercised dependency is Lean-MLIR revision
`b8c4a771319a044854a4bbb49b8821381ef86ff8`, which declares
`leanprover/lean4:nightly-2025-12-01`.

Prepare the shared mathematical library and its file inventory explicitly:

```sh
python -m merlin.verify.lean_cli prepare-toolchain \
  --executable "$LEAN_EXECUTABLE" \
  --library-root "$LEAN_STANDARD_LIBRARY" \
  --library-root "$LEAN_MLIR_BUILD_LIBRARY" \
  --library-root "$LEAN_DEPENDENCY_BUILD_LIBRARY" \
  --dependency-revision "$REVIEWED_DEPENDENCY_REVISION" \
  --work-root "$TASK_SCRATCH" \
  --output "$FORMAL_TOOLCHAIN_OUTPUT"
```

Repeat `--library-root` for every dependency's compiled import directory and
`--runtime-file` for additional required native loader files. The command
compiles only Merlin's packaged foundation, pins the exact executable and
complete import products, and writes `lean-toolchain.json`. Existing output
is refused. Compilation uses one Lean worker with explicit memory and wall
budgets. The record is `prepared_unqualified`: it does not qualify a worker,
infer native-loader dependencies or prove that an operator-selected library
was built from its claimed source revision.

Stage any required native SONAME aliases as ordinary files while preserving
their lexical loader names before inventorying them. Frozen snapshots preserve
the declared file tree and executable permissions; do not grant whole toolchain
directories. The selected platform's ELF interpreter and system-library closure
need independent deployment qualification. Host checks clear loader-injection
environment variables. Verify the manifest and execute a supported actual
before/after check on the allocated worker before starting authoring. Then pass
the manifest to release preparation; a declaration alone is not execution proof.

## Request a verified common transformation

The experiment-facing API is
`merlin.transforms.client.apply_transform(workspace, source_ir, options)`.
It submits actual structured IR and documented parameters through the original
host broker. The request cannot choose alternative semantics, coverage,
toolchains or a saved proof result. The broker resolves its own archived input
snapshot, applies transformations and checks each actual before/after pair.

Common options select shape normalization, integer simplification, buffer reuse,
tile iteration counts by original loop depth and explicit dependency orders.
V2 also accepts `eliminate_redundancy`, `hoist_invariants`, `unroll_factors`
by original loop depth and `max_added_operations`. The broker bounds each
unroll factor to 64 and total added operations to 4,096. These selectors default
off; code size follows the explicit factor and growth budget, rather than the
input trip count.
The host ABI determines index width; a request cannot change it. This is a
parameterized mechanism: no hardware dimension or traversal preference is
implied by choosing a tile size.

| First supported transformation | Concrete behavior and boundary |
| --- | --- |
| Shape/view normalization | Eliminate identical memref casts and resolve static dimensions; no general reshape/layout inference |
| Integer simplification | Fold scalar signless wrapping add/sub/mul and neutral operands; no floating reassociation or poison flags |
| Ordered loop segmentation | Emit bounded outer/inner `scf.for` loops with tail bounds and forwarded iter_args; preserve reduction order |
| Storage reuse | Reuse identical static contiguous integer allocations with nonoverlapping explicit lifetimes and structurally recognized complete initialization |
| Sequential dependency planning | Derive SSA edges and preserve all effects conservatively, including unknown operations; no inferred asynchronous overlap |
| V2 scalar CSE/DCE | Reuse guarded upstream elimination for exact, total scalar integer operations; preserve opaque operations and their consumers |
| V2 invariant hoisting | Move total integer computations with loop-external operands; preserve effects and loop-carried variants, including zero-trip loops |
| V2 ordered unrolling | Emit a bounded repeated bulk and exact tail, or inline a bounded small loop; preserve body-effect order and carried state |

The callable input domain is already bufferized `arith/scf/memref/func` IR.
Loop code size depends on IR size and rank, rather than the number of tensor
elements or tiles. Escaping allocations and unsupported layouts remain explicit
unknowns in storage planning. Dependency planning treats unknown configuration
and device operations conservatively.

Standalone `apply_common_pipeline` calls transform a private clone and dispatch
covered changes to a supplied host verification hook. An explicit `plan_only`
call can inspect unverified transformed IR; it creates no accepted proof.
Experiment authors use the broker route. A permissive callback, success string
or serialized certificate is not host-issued verification authority.

## Keep proof scope precise

Each change carries the actual before/after IR, their digests, parameters and
typed witnesses. Structural validation checks correspondence before Lean checks
the mathematical obligation. The loop recognizer checks bounds, IV substitution,
carried values, tail limits, preserved body and surrounding IR.

The initial modeled obligations are:

- Scalar bit-vector arithmetic under the admitted wrapping semantics.
- Ordered iteration partition/index arithmetic and fold segmentation, paired
  with structural correspondence of the emitted loops.
- Disjoint live intervals for allocations assigned the same slot, paired with
  checked initialization, lifetime and rewrite structure.
- Supported scalar scheduling equivalence; general opaque state transitions
  and asynchronous protocols remain outside the first formal model.
- V2 total pure-function CSE/DCE and invariant-motion identities, with exact
  scalar operation and SSA guards checked structurally.
- V2 ordered unroll partition, index bounds and state-fold identities, with
  actual body cloning, substitutions and exact yields checked structurally.

The exporter, structural projection, selected primitive semantics and Lean
kernel/toolchain are explicit assurance boundaries. Descriptor validity,
allocation failure, target instruction/RTL correspondence, final host ABI and
machine-code execution need separate evidence. A mathematical interval theorem
does not prove hardware capacity or memory safety for an arbitrary program.

The policy is strict within the host's fixed coverage: failed proofs, missing
tools, timeouts and unknown covered obligations refuse acceptance. Candidate
declarations cannot narrow that coverage. Outside the modeled domain, retain
independent checks and report the unproved scope. Do not label a tactic failure
as a demonstrated numerical counterexample.

## Analyze dependencies and assign explicitly declared resources

`merlin.transforms.analyses.analyze_ir(module, index_bits=...)` derives stable
SSA identities, alias roots, known/unknown effects, captured uses, lexical live
intervals and conservative dependencies from the actual typed IR. Views keep
their underlying buffers live; escaping values remain live through the scope.
Distinct external memory arguments may alias. Separate fresh allocations are
disjoint under the source's valid-allocation semantics. Unknown effects form
ordering barriers rather than implied independence.
Effect traits are software declarations: target instruction correspondence
still requires the admitted ISA semantics and independent hardware checks.

Pressure and working-set fields are static **whole-value logical footprint
subtotals**. They are not access-range/cache footprints, physical representation
sizes, peak runtime heap usage or cycle estimates. Unknown sizes remain explicit.
Index size requires a reviewed width; no host width or unknown dtype is guessed.
`validate_analysis` regenerates the facts for the current IR and rejects stale
results. Analysis itself requires no solver or simulator.

`merlin.transforms.resource_allocation.allocate_instruction_resources` adapts
one sequential, single-block function to the existing generic finite-domain
allocator. Provide a `ResourceBinding` for every actual SSA value and explicit
`StorageBank` geometry, units, alignments, reservations and any fixed boundary
addresses. The admitted OOT representation must establish each binding's extent;
a logical tensor byte count does not prove a physical layout.

Known effectful instructions require explicit sequential completion constraints.
Their operand resources remain pinned through scope completion. Result-free
state events extend operand lifetimes without consuming physical storage.
Unknown effects, pointer/alias representations, dynamic sizes, region control
flow and multiple simultaneous results refuse this allocation model. Its
ordered epochs establish no hardware latency or asynchronous-overlap claim.

`emit_resource_allocation` records a typed assignment in cloned IR.
`validate_allocated_ir` checks that the actual program is unchanged, rebuilds
its resource graph and lifetimes, then independently checks the actual recorded
addresses, alignment, capacity, backing-store aliases, reservations and boundary
constraints. A saved solver status is insufficient. The OOT must consume and
check those addresses during terminal instruction lowering; metadata alone
does not demonstrate generated instruction use or RTL conformance.

These additional APIs require their exact leaves in the selected common source
inventory; fresh V2 includes them. Earlier frozen inventories retain their membership. Their structural
and allocation checks are not new Lean proof receipts or a complete memory
safety theorem; extending formal coverage remains a separate admission step.

Repeated successful obligations can use a bounded process-local cache after
revalidating the toolchain inventory. The key includes actual IR, generated
query, semantic model and scope. A cache hit creates a fresh host completion;
saved summaries cannot seed the cache or discharge an obligation.

The worker's installed xDSL and optional Z3 are external Python dependencies,
outside this common source inventory. Use the project's locked dependencies;
the allocation solver requires the `semantic-search` extra. A successful tiny
Lean readiness check establishes the broker's modeled scalar proof route, not
allocation solver availability or the complete compiler. Missing Z3 returns
`tool_unavailable`, which cannot qualify an allocation.

## Observe changes and select development tests

V2 broker results include `pass_observations` from the actual before/after IR:
operation and loop counts, known logical allocation bytes, refreshed storage
and dependency facts, unresolved aliases, block-scoped liveness and logical
pressure, and invalidated analyses. Pressure is a known whole-value logical
subtotal by lexical block, rather than a physical or runtime storage peak.
Dynamic work, physical traffic
and cycles remain `UNKNOWN` where they are not measured. These records describe
the change; they do not establish a performance ranking.

Phase 2's ordinary `evaluation-submit-impact` action uses a completed original
cheap evaluation job and validated LLVM/object/ELF producer chains to select
expensive development checks. It omits a member only when its relevant emitted
artifacts are demonstrably unchanged. Missing joins, failed jobs and unknown
impact include the member; missing complete cheap evidence selects the full
roster. A partial development selection cannot promote a candidate. Promotion
still requires the complete ordinary gate on its immutable compiler revision.

## Select logical host graph optimizations

`lower_model` and `lower_model_file` accept an explicit
`merlin.llvmlower.graph_optimizations.GraphOptimizationContract` binding the
actual source SHA-256 and entry function. Select `propagate_layout=True` and
declare `independent_element_order_unobserved=True` for layout propagation.
Floating tensors additionally require
`floating_environment_unobserved_nontrapping=True`. The first selected domain
is positive static pointwise tensors and permutations.

Select `fusion_groups` as immutable tuples of contiguous nodes from the
outlined logical graph. This route reuses the existing logical host dispatch
planner and emitter. Independent checks compare complete typed interfaces,
actual scalar regions, symbolic logical output coordinates, opaque-observer
operands and expanded actual source against the emitted driver. Unsupported
views, reductions, opaque regions and undefined calls refuse this selection.

`minimum_transpose_bytes_saved`, `minimum_dispatches_saved` and
`maximum_added_operations` are explicit structural cost contracts. Code growth
is checked before fused-body cloning and afterward. These proxies do not
predict cycles. Reports retain `timing=UNKNOWN` and do not claim Lean or
hardware conformance. Physical formats, device placement, transfers and
scheduling remain OOT responsibilities; ordinary lowering selects this route
only when the caller supplies a contract.

## Keep the authored ISA minimal

The software spec optionally contains `isa` with schema `merlin.isa_spec.v1`.
Author typed signatures, operand units and bindings, mathematical primitive
references, explicit memory/state effects and resource names. Reference the
selected facts for instruction fields and hardware-dependent values. Reuse
existing numerical policies rather than introducing compiler algorithms into
the spec.

Resolved facts bind the declaration to one selected facts identity. Encoding
resolution refuses absent, inconsistent, overlapping or uncovered fields.
Missing effects remain unknown. Resolving an encoding does not prove that the
declared mathematical operation matches the RTL's behavior.

EL5 and EL6 can run:

```sh
python -m merlin.targetgen.isa_codegen_cli \
  --software-spec "$PUBLIC_SOFTWARE_SPEC" \
  --facts "$PUBLIC_RTL_FACTS" \
  --target "$TARGET" \
  --out "$OOT_WORKSPACE/build/instruction-dialect"
```

The generated package contains typed xDSL instruction definitions, local
verifiers, encoding/decoding mechanics, semantic/effect records and documentation.
It belongs in the candidate's OOT workspace. It supplies neither a tensor-to-ISA
lowering nor a device schedule, tiling policy or allocation strategy.

In EL6, `python -m merlin.verify.lean_cli inspect` reports the fixed formal domain.
`check --before ... --after ... --entry ... --toolchain ... --work-root ...`
checks actual supported scalar IR. An optional resolved ISA model supplies
admitted primitive meanings; it cannot turn unknown stateful instructions into
pure arithmetic. CLI output is a diagnostic summary, not replayable broker
authority.

## Validate each assurance separately

Use structural verification and fault mutations for every covered transformation.
Use independent numerical execution on connected public programs, tails,
multiple outputs, shared producers and repeated calls. Compile and link larger
programs without executing their enormous loops to check code/resource scaling.
Instruction conformance and native target tests remain independent of Lean.

Report preservation, numerical agreement, placement completeness, RTL legality,
build/link success and measured performance separately. A proof or generated
dialect cannot replace those other gates. Derive development tests from public
semantics, selected hardware constraints and allowed development programs;
compiler answers and private validation programs remain outside author inputs.
