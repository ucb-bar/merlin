---
title: Shared transformations, ISA generation and Lean verification
kind: guide
status: current
owner: ir
last_verified: 2026-10-19
related: [verified_compiler_roadmap, compiler_verification, phase0_specification, compiler_authoring_prompts, phase0_phase1_setup]
code_refs: [src/merlin/transforms, src/merlin/verify, src/merlin/targetgen, src/merlin/llvmlower/graph_optimizations.py, src/merlin/llvmlower/host_graph_policy.py, src/merlin/llvmlower/host_optimization_request.py, src/merlin/llvmlower/layout_conformance.py, src/merlin/llvmlower/layout_propagation.py, src/merlin/llvmlower/lower.py, src/merlin/runtime/backends/spike_model.py, src/merlin/compile/baremetal_model.py, packages/merlin-experiments/src/merlin_experiments/phase1/brokers, packages/merlin-experiments/src/merlin_experiments/phase1/feedback/compiler_profile_child.py]
---

# Shared transformations, ISA generation and Lean verification

The common substrate supplies executed xDSL transformations, an optional
instruction dialect generator and an independent host Lean checking route.
It provides a small, explicitly modeled foundation. It does not establish a
complete tensor-to-device compiler or a whole-program SCF, memory, LLVM or RTL
equivalence theorem. The [capability roadmap](../design/verified_compiler_roadmap.md)
records the remaining pass groups and their required evidence.

## Select a fresh experiment release

Fresh `corpus prepare` selects `merlin.common_compiler_substrate.v3`. Direct
bundle generation selects it with
`--common-substrate merlin.common_compiler_substrate.v3`. Supply an explicit
`--lean-toolchain-manifest` and the reviewed `--host-index-bits` of the selected
host ABI. Missing formal tools cannot satisfy covered obligations. V1 retains
its original source membership and formal registry; it cannot request the new
V2 options or acquire the V2 pass observer through a resumed run. V1 and V2
cannot acquire the V3 authored-transformation checking action through resume.

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

## Use the tooling inside actual compiler calls

The original Phase 1/2 package box stages the existing public transformation
channel in each isolated compiler-call workspace. `MERLIN_COMMON_WORKSPACE`
selects that channel. The source-pinned client can request `apply_transform` or
`check_transform` using the same frozen source grants, formal coverage,
semantics and selected toolchain as its author. Requests cannot choose another
owner or grant themselves proof authority. The host binds the compiler/input
identity and aggregate deadline, then reaps proof children before staging ends.

Budgeted whole-model grading reconstructs this exact original profile executor
in a fresh interpreter through a parent-issued sealed readonly descriptor.
The child reopens the frozen inputs, package sandbox, selected environment and
any original live resource boundary. Foreign or incomplete owners refuse.
This does not reconstruct arbitrary alternative component executors. An unset
`MERLIN_MODEL_BUDGET_S` retains the inline route. See
[the transport scope](../design/public_compile_only_feedback.md#ordinary-compiler-execution-with-the-common-profile).

Caller-selected artifact ABI 0.2 profiles also retain structural emission in
this box, with fresh profile/source checks before output publication. This is
not a numerical adapter or permission to treat opaque artifacts as LLVM.
Frozen compiler scopes continue to refuse Python bytecode/cache writes.

An accepted per-call IR pair is not a proof that the compiler used that pair
in its final artifact. Under the active original V3 scope, ordinary lowering
with `emit_transformation_trace` retains its actual input/emission observation;
the object entry checks the raw checkpoint chain and binds a freshly produced
object. The entry symbol comes from the original build/capability contract,
not from the trace. Absent traces retain historical reuse behavior and grant
zero formal credit; unconsumed traces also grant none. Unsupported edges remain
unproved in partial reports; covered failures reject the build. The join covers
one current observed source using its host-issued text identity. Distinct
emissions with identical code retain separate source checks. Trusted host
artifact copies register their exact path and byte pin before a read can
recover that identity. Plain copied/serialized text cannot select a pending
source, and ambiguous artifact paths refuse. A host rewrite invalidates the
path's previous ownership before writing; its latest observed issued producer
owns that path. Equal-byte plain or untraced rewrites carry no prior identity.
An intervening observed writer prevents an older emission from registering.
Issuance belongs to one process
and live original scope; saved or reconstructed identifiers grant no authority.
Failed or incomplete object consumers retain their claim until scope exit,
keeping legacy artifact caches unavailable for retries of that emission.
Later consumers of a completed origin recheck its original source, package,
emission and invocation pins and observe a freshly built object. They never
restore proof authority from a saved report or a byte-only cache. Ordinary
oracle tiers therefore rebuild native objects; the independent proof checker
can reuse only its own qualified successful checks.
Whole-model workers share a completed object only within the same observed
origin; untraced builds retain their original byte-based reuse. The host ledger
retains at most 4,096 pending/completed sources, 8,192 original/alias records,
8,192 observed writer paths and 256 MiB
of issued text objects per scope. These are host resource ceilings, not tensor
capacities. No per-alias text copy or tensor-data copy is required.
Explicit compile-only proof transport does not duplicate the ordinary check.
This custody establishes neither the compiler's internal pass
history nor terminal lowering, dispatch, host optimization or machine semantics.
Numerical and RTL qualification remain independent.

## Select reusable host optimizations

The ordinary object producer reuses Merlin's logical layout/fusion checks,
upstream allocation hoisting/lifetime passes and checked LLVM arena binder.
An original host caller can supply `HostOptimizationSelection`; its context
declares permissions and its request contains only bounded choices and the
SHA256 of the actual source. A finished LLVM artifact cannot request tensor
fusion or structured buffer passes. Device operations are not rewritten as
host computation.

For a fresh V3 experiment, a compiler can emit the optional command-buffer
`host_optimization_request` defined in the original command-buffer schema.
The same normal and compile-only object paths consume it. It supplies neither
a pass pipeline nor semantic/runtime permissions. The original typed build
service or unchanged frozen capability contract selects the entry symbol; the
frozen profile selects index width.
No candidate manifest, filename, target name or proof report grants permission.

Permissions are an optional `host_optimization` block in the original reviewed
software specification, which is already bound by `semantic_inputs`. Omission
admits no requested host optimization and does not add a new required input.
For example, an operator who has independently established the required runtime
premises can select:

```yaml
host_optimization:
  schema: merlin.host_optimization_admission.v1
  permitted_stages: [llvm]
  permitted_optimizations: [static_arena]
  sequential_execution: true
  successful_allocations: true
  non_reentrant_entry: true
```

These booleans are reviewed premises, not evidence produced by this block.
Unspecified premises default false. Logical/structured host selection also
requires original `data_layout`; logical graph changes require
`independent_element_order_unobserved`, and floating graph changes require
`floating_environment_unobserved_nontrapping`. Requests cannot modify scales,
rounding, approximation bounds or the required numerical gate. Phase 2 inherits
these exact bytes from Phase 1.

Each selected build preserves the original source, selected IR, raw translation,
later LLVM and final object identities in `kernel.host_optimization.json`.
Raw compiler stack reports remain immutable; their canonical measurements use
separate files. Conditional structural checks and artifact custody do not issue
a Lean proof, numerical pass, deployment receipt or cycle estimate. Existing
numerical and instruction-policy checks still apply to the resulting artifact.

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
compiles Merlin's packaged `Foundation`, `ResourceGeometry` and `StateEffects`
into one product root, pins the exact executable and
complete import products, and writes `lean-toolchain.json`. Existing output
is refused. Compilation uses one Lean worker with explicit memory and wall
budgets. The record is `prepared_unqualified`: it does not qualify a worker,
infer native-loader dependencies or prove that an operator-selected library
was built from its claimed source revision.

Bit-vector obligations using `bv_decide` additionally require an explicitly
selected `--sat-solver` and `--allow-native-reflection`. Supply the solver's
required native loader files with `--runtime-file`; stage them as ordinary files
under the same lexical RUNPATH layout before preparation. The solver may have
its own small prefix: the Lean executable and import libraries are reused.
The original manifest pins every selected file, the solver and this permission.
Verification passes its exact frozen path to Lean and never uses PATH or a
sibling fallback. Missing solver or permission leaves the covered proof
unavailable before tactic execution.

Lean's selected BV tactic checks LRAT through native reflection. Its receipt
uses `lean_native_reflection` and states that the original Lean compiler and
native checker are trusted. `Lean.ofReduceBool` and `Lean.trustCompiler` are
admitted only under this explicit original mode; other unapproved axioms refuse.
Kernel reduction
receipts retain `lean_kernel`. These are different assurance scopes. Neither
proves machine code, physical memory behavior, numerical execution or runtime
qualification. Phase 2 inherits the same original selection.

Stage any required native SONAME aliases as ordinary files while preserving
their lexical loader names before inventorying them. Frozen snapshots preserve
the declared file tree and executable permissions; do not grant whole toolchain
directories. The selected platform's ELF interpreter and system-library closure
need independent deployment qualification. Host checks clear loader-injection
environment variables. Verify the manifest and execute a supported actual
before/after check on the allocated worker before starting authoring. Then pass
the manifest to release preparation; a declaration alone is not execution proof.

Private host tool inventories may batch a maximal closed subdirectory below an
original library root only when its complete plain-file tree exactly equals
the selected manifest membership. Unpinned parent metadata stays excluded.
The freezer checks every selected byte before copying and against the pending
snapshot before sealing. Extra files, aliases, public overlap or unresolved
ownership retain individual private leaves. This batches private storage and
masks without exposing a directory to the author; EL6 public grants retain
their exact file membership. Large policies also require the worker setup in
[the bubblewrap argument guide](bubblewrap_argument_budget.md).

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

## Check a transformation authored in the OOT

V3 also admits
`merlin.transforms.client.check_transform(workspace, before_ir, after_ir, entry="...")`
on the original Phase 1 channel and inherited Phase 2 broker. Submit complete
actual before/after IR. The request cannot provide semantics, purity claims,
resource limits, proof tactics, a different toolchain or saved receipts.

The original host reconstructs the primitive registry from the frozen software
ISA and selected raw hardware facts, checking both byte identities. Without
an ISA declaration it retains the standard builtins domain. A scalar proof
checks the complete typed function for every input bit pattern. Host-selected
simplification and bit-vector tactics share one bounded work budget. Failing
a covered proof cannot fall through into an unsupported-domain success.

An exact sequential permutation can instead enter the state model. The host
checks complete typed SSA correspondence, conservative aliasing and every
actual adjacent exchange. Lean derives commutation from disjoint read/write
frames and proves preservation of the surrounding sequential context.
Static structured loops aggregate effects without expanding their iterations.
Distinct caller pointers may alias; distinct named device resources are also
conservatively overlapping unless the original reviewed model proves otherwise.

ISA effects alone do not establish totality, atomic completion or visibility.
Effectful OOT operations require a separate original reviewed admission of their
complete sequential state abstraction. An asynchronous DMA command is not
automatically treated as a completed memory update. Unknown effects, unresolved
access bounds, poison arithmetic and unsupported observations refuse this proof.
Standard memory checks explicitly assume defined accesses and successful fresh
allocations; these are IR semantics, not deployment or allocation guarantees.

`verify_resource_assignment` is an internal host API for the actual emitted SSA
resource annotation. It freshly reconstructs capacity/alignment, shared backing
reservations, lifetimes, boundary addresses and instruction-port obligations.
Lean checks the resulting geometry under the original host's representation
and constraints. The author message cannot supply new bank capacities or
overwrite permissions. Physical representation, constraints matching hardware,
and final ISA consumption remain independently checked.

Only live Lean completion under the original host selection can be accepted. Serialized
reports describe results and do not create authority. A standalone checked
pair does not establish that a compiler's final object implements that pair;
artifact custody and independent numerical/RTL gates remain necessary.

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
kernel/toolchain are explicit assurance boundaries. Native-reflection receipts
also trust the originally admitted Lean compiler and native LRAT checker;
they are not kernel-only proofs. Descriptor validity,
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

For ordinary model compilation, select original versioned software permissions
without manually preparing a source hash:

```python
from merlin.llvmlower.host_graph_policy import read_host_graph_policy

policy = read_host_graph_policy(
    software_path, target=selected_target, entry="forward",
    minimum_transpose_bytes_saved=1, maximum_added_operations=8192,
    physical_layout=True,
    maximum_layout_assignments=100_000, maximum_layout_permutations=16,
)
# Pass graph_optimizations=policy to compile_saved_model or spike_model.build.
```

The selected `merlin.software_spec.v1` file must have `status: reviewed` and the
selected target. Its existing permission block is shared with the Phase 2
original-executor route:

```yaml
host_optimization:
  schema: merlin.host_optimization_admission.v1
  permitted_stages: [logical_host]
  permitted_optimizations: [propagate_layout]
  independent_element_order_unobserved: true
  floating_environment_unobserved_nontrapping: true
```

These are operator-supplied semantic premises. A review label, dtype or RTL fact
does not prove them. Omitted permissions remain false. This host-only pass needs
no accelerator ISA or storage declaration: it changes tensor indexing while
retaining scalar arithmetic, and makes no hardware resource or cost inference.

With `physical_layout=True` (the policy default), Merlin forms static typed
pointwise/transpose islands from actual SSA and invokes the existing physical
layout planner. The candidate domain contains identity and the bounded group
generated by source permutations. Each generic operation keeps its scalar body
and selects exact input/output axis orders; lower-rank and unit-broadcast
operands retain their declared coordinate maps. Shared producers are emitted
once. Conversions restore canonical tensors at opaque boundaries, preserving
snapshots before unknown effects. Generic operations reading their initialized
destination remain boundaries.

The selected plan is realized as actual tensor IR. Independent checks reconstruct
complete logical output coordinates, scalar work by lexical control scope,
conversion membership/maps/payloads, and definition-before-use dominance.
Reporting a selected plan alone does not satisfy these checks. Host memory
capacity, banks and allocator behavior remain unknown; compiler-owned axis
permutations grant no accelerator residency or storage capacity. Conversion
costs describe logical tensor materialization, not physical bus traffic or cycles.

The controls bound different parts of this process:

| Control | Meaning |
| --- | --- |
| `maximum_layout_assignments` | Maximum assignments explored by the existing planner. |
| `maximum_layout_permutations` | Maximum source-generated axis permutations admitted per island. |
| `maximum_added_operations` | Limit on actual IR construction, including cloned scalar regions. |
| `minimum_transpose_bytes_saved` | Required reduction in actual transpose-output payload bytes. |
| `physical_layout=False` | Select the existing demand-driven layout rewrite instead. |

An exhausted planning/domain budget retains the original island. A snapshot
that would become an unsafe alias also retains its original materialization.
If the complete selection then lacks the required savings, the automatic
policy keeps the exact original source. Construction-budget exhaustion also
retains the original source under the automatic policy; an explicit source
contract receives a diagnostic. Malformed source and failed independent checks
remain errors.

The policy binds the actual source at the existing post-preparation logical
hook, preserving captured quantization and the selected native host pipeline.
Original automatic host passes remain enabled. `enabled=False` provides a paired
baseline under the same controls. Missing permission or insufficient transpose
byte savings retains exact input source; no partially rewritten source is used.
Unknown permission fields, malformed input, stale software/library bytes and
failed conformance remain errors. The software snapshot, policy, derived source-bound contract
and selection status are retained and rechecked through saved-model execution.
This selection accepts one policy or one explicit contract, never both.

`lower_model`, `lower_model_file`, `spike_model.build` and `compile_saved_model`
also accept `graph_optimizations=` with an explicit
`merlin.llvmlower.graph_optimizations.GraphOptimizationContract` binding the
actual source SHA-256 and entry function. Select `propagate_layout=True` and
declare `independent_element_order_unobserved=True` for layout propagation.
An explicit contract selects this planner with `physical_layout=True`; its
default remains the existing demand-driven rewrite.
Floating tensors additionally require
`floating_environment_unobserved_nontrapping=True`. The first selected domain
is positive static pointwise tensors and permutations. The original software
contract must establish these permissions; RTL dimensions cannot supply them.

For example, NHWC → NCHW → broadcast bias → ReLU → captured quantization →
NHWC can perform its arithmetic directly in NHWC. Every scalar operation,
conversion, rounding and clamp remains in its source order. Static singleton
bias axes can use checked views. This does not fuse arithmetic stages or
change the numerical policy.

Scalar `linalg.generic` yields may carry `prov.*` trace annotations. Surviving
scalar bodies retain their exact annotations; provenance IDs do not select
optimizations. Unknown yield attributes or properties keep the operation opaque.

The checked family retains shared pointwise producers. A consumer can read the
existing layout through composed indexing maps while directly writing its new
layout. Independent conformance rejects increased scalar evaluation requests
for any typed operation in a lexical control scope. Reports separate these
static tensor-domain counts from materialization bytes; neither estimates cycles
or includes dynamic invocation multiplicity. This conservative policy requires
no calibration and avoids duplicating arithmetic to reduce a byte proxy.

Select `fusion_groups` as immutable tuples of contiguous nodes from the
outlined logical graph. This route reuses the existing logical host dispatch
planner and emitter. Independent checks compare complete typed interfaces,
actual scalar regions, symbolic logical output coordinates and expanded actual
source against the emitted driver. External declarations, calls and structured
single-block regions can surround checked pointwise computations. Their typed
interfaces, ordered opaque operations, operands, results, nested captures and
terminators remain unchanged. Unknown operations are barriers, including
operations with unknown executable metadata. An independent observation check
rejects moving a materialized tensor read past an unknown effect. Arbitrary
reshapes, slices, reductions and unstructured control flow are not interpreted
by this selected family. Flatten/expand bias views require an adjacent static
pair and at most one nonunit source axis; other views remain boundaries.

`minimum_transpose_bytes_saved`, `minimum_dispatches_saved` and
`maximum_added_operations` are explicit structural cost contracts. Code growth
is bounded during layout construction and before fused-body cloning. These
proxies do not predict cycles. Reports retain `timing=UNKNOWN` and do not claim
Lean or hardware conformance. Accelerator encodings and device ABI remain OOT
responsibilities; this route selects no device placement or schedules.
Ordinary lowering selects this route
only when the caller supplies a contract or software policy. With an explicit
source contract, if no legal rewrite meets the
explicit minimum savings, selection fails with a diagnostic; omit the contract
or use `propagate_layout=False` to retain the source layout. Failed checks never
publish partially rewritten IR.

The source hash binds the exact input to the logical-host hook, after ordinary
preparation. A contract for an earlier capture is rejected when preparation
changes its bytes. Callers must bind their reviewed prepared source; the
compiler does not silently retarget permissions or hashes. Existing host
optimization requests use this same hook under their original permissions.

The lowering directory retains `host-graph-input.mlir`,
`host-graph-optimized.mlir` and `host-graph-optimization.json`. The report binds
implementation source hashes, current numerical/resource context and ordered
source-analysis, rewrite and conformance records. Rewriting invalidates the
source-coordinate analysis. Ordinary bare-metal builds bind the report in the
compilation recipe; saved-model receipts expose all three artifact identities
and recheck source, implementation and contract custody before and after
execution. These are observations, not machine-code equivalence proofs or new
authoring access grants.

Phase 1 and Phase 2 use the same original software permissions and independent
verification rules. Comparisons using `HostGraphPolicy` must expose the same
library version, controls and budgets to every treatment; tuning cannot change
semantic permissions or infer capacity from a layout label. The existing
`HostOptimizationRequest` v1 preserves its exact serialized fields and selects
the demand-driven rewrite. Version 2 exposes the same checked planner through
the ordinary source-bound command-buffer request in both phases:

```python
from merlin.llvmlower.host_optimization_request import HostOptimizationRequest, REQUEST_SCHEMA_V2

request = HostOptimizationRequest(
    source_sha256=actual_source_sha256,
    schema=REQUEST_SCHEMA_V2,
    propagate_layout=True,
    physical_layout=True,
    maximum_layout_assignments=100_000,
    maximum_layout_permutations=16,
).record()
```

Place this record in the command buffer's `host_optimization_request`. Both
versions reject unknown or missing fields; v1 rejects all three new fields.
Version 2 requires them even when the planner is disabled. Its search bounds
are positive integers up to 1,048,576; they bound compiler work and supply no
hardware capacity or performance evidence. The planner choice requires
`propagate_layout=True` and the same original host layout and numerical
permissions as the existing rewrite. Current source, choices and original
permissions remain bound throughout the ordinary build. Tuning cannot widen
those permissions or change the numerical admission thresholds.

Validate the mechanism with:

```sh
python -B -m pytest merlin/tests/ir/test_host_graph_optimizations.py \
  merlin/tests/ir/test_host_layout_planning.py \
  merlin/tests/ir/test_host_layout_realization.py \
  merlin/tests/ir/test_host_layout_execution.py \
  merlin/tests/ir/test_layout_propagation.py \
  merlin/tests/infra/test_host_layout_model.py \
  -p no:cacheprovider --basetemp "$TMPDIR/host-layout-check"
```

The model test requires the ordinary native lowering and functional simulator.
It compares complete recurrent history, float state, position, inputs and bias
against an independent ordered reference. It inspects actual lowered IR,
allocation sites, raw LLVM call bodies and final native artifacts. Native
inlining is allowed and recorded separately from logical boundaries.
Functional instruction counts and logical transpose bytes do not measure
accelerator latency or physical bus
traffic. These neutral host models establish neither real-accelerator support
nor general full-model speedups.

## Prefetch qualified copies across private banks

The ordinary scheduling pipeline can issue an existing asynchronous input copy
earlier while work uses a different physical backing. It preserves the copy's
instruction, token and wait. This first family has one outstanding transfer,
explicit byte-addressed windows and invocation-private temporaries. It creates
no asynchronous instructions and changes no arithmetic or reduction order.

Three inputs establish different facts:

- Selected RTL facts supply memory windows, alignment and structural queue and
  in-flight bounds. Missing bounds remain unknown.
- Original software semantics supply exact byte-copy effects, actual completion
  tokens/waits and the specific instruction pairs permitted to overlap.
- Independent execution qualification establishes those overlap semantics over
  **all selected operand/resource-selector combinations**. Disjoint addresses,
  queue depth and a review label alone do not establish safe engine overlap.

The existing `merlin.temporary_storage_contract.v1` optionally admits this exact
`prefetch` member. The example instruction names must match the selected ISA;
they are not builtin pass selectors:

```yaml
prefetch:
  schema: merlin.async_prefetch.v1
  max_in_flight: {fact: [facts, copy_inflight]}
  queue_capacity: {fact: [facts, copy_queue_entries]}
  independence: memory_disjoint
  pairs:
    - {launch_instruction: load_async, crossed_instruction: vector_update}
    - {launch_instruction: load_async, crossed_instruction: copy}
```

Capacities require structural fact references or `null` for unknown; software
literals are rejected. An unresolved/unknown bound or a known zero disables
prefetch. Each launch form must declare asynchronous `exact_copy: true` with
one equal-sized read/write and no additional execution/state effects. Crossed
forms must be synchronous, memory-only, and independently projected as exact
copies or total integer elementwise operations. Unknown effects and
configuration operations are barriers. A new target requiring additional
observable engine state must model it before using this family.

Supply `prefetch_qualification_identity=` from the original qualified runtime
through `compile_schedule`, `compile_source_pointwise`, or the frozen
`SourcePointwiseLowering` selection. A declaration or identity string is not
qualification evidence. The ordinary model route receives that selection in
`shared_source_lowerings`; callers retain the existing provider, numerical and
complete-output gates.

`ScheduleBudgets` defaults to `prefetch_distance=1` and `max_prefetch_moves=8`:
with the qualified capability present, a copy may advance by one preceding
operation automatically. Missing capability/qualification leaves legacy order
unchanged. Set `prefetch_distance=0` for a paired baseline; a larger finite
distance permits more independent work between issue and the original wait.
Phase 2 may tune these controls under identical software, qualifications,
hardware limits, library versions and verification rules. These controls do
not increase queue capacity or authorize new instruction pairs.

Candidate construction precedes allocation. Independent checks verify the
original-call permutation and every crossed dependency using actual physical
backings; distinct view names cannot hide aliases. Recomputed liveness keeps
pending reads and writes alive through completion. If candidate allocation
fails, bounded repair restores the original order and replans. Unknown overlap,
shared-bank work, multiple pending transfers and unsupported pairs retain the
legal original order. Invalid original waits/dependencies, stale declarations,
failed checks or exhausted repair budgets remain precise diagnostics.

The pointwise source checker defers copied provenance until its named wait;
pending destinations cannot be read and sources cannot be overwritten. Arithmetic
remains synchronous. Final checks independently decode actual emitted bytes,
including issue order, operands and completion; reports record actual launch
positions/accesses. Partial waits cannot stand in for the current final
decoder's global drain. Compact dense lowering remains synchronous and does
not inherit this capability.

Prefetch preserves command count and requested copy bytes; it offers a legal
issue-order alternative. Report compilation cost, code size, logical transfer
requests and host packing separately. Neither earlier issue nor functional
instruction counts establish accelerator overlap, physical bus traffic or a
whole-model speedup. Such claims require an appropriate paired timing source.

Check the selected mechanism and unchanged exported dependency closure with:

```sh
python -B -m pytest merlin/tests/infra/test_copy_prefetch.py \
  merlin/tests/infra/test_shared_scheduling_export.py \
  -p no:cacheprovider --basetemp "$TMPDIR/copy-prefetch-check"
```

The export test uses an isolated interpreter and a neutral native executor;
it does not qualify any accelerator implementation.

## Fuse an exact integer dense output path

The dense source route can keep a complete reduction in i32 and store its
final result as i8 through one declared integer readout. For example, a
single-use `maxsi(acc, 0)`, `minsi(value, 127)`, `trunci` epilogue can run in
the store path. This separates the accumulator format from the output format:
the planner's accumulator capacity, initialization, reduction order and exact
continuation remain i32. A truncation without a clamp has different semantics
and cannot inherit this readout.

The original dense software declaration can include this optional member:

```yaml
output_readout:
  schema: merlin.dense_integer_readout.v1
  scalar_ir: |
    module { func.func @readout(%x: i32) -> i8 {
      %zero = arith.constant 0 : i32
      %high = arith.constant 127 : i32
      %positive = arith.maxsi %x, %zero : i32
      %bounded = arith.minsi %positive, %high : i32
      %result = arith.trunci %bounded : i32 to i8
      func.return %result : i8
    } }
  output:
    dtype: i8
    bits: {fact: [facts, output_bits]}
  instruction: store_i8
```

`store_i8` is an example selected ISA instruction, not a builtin opcode. It
must be distinct from the original full-width store, have explicit integer
operand bindings, read only the accumulator and write only the output. The
width reference must resolve to eight bits in the selected facts. The scalar
function must be a supported total unary integer computation from i32 to i8;
floating operations and unknown effects are refused. Its expression must
match the original source computation, including operation order, widths,
signed comparisons and constants.

Supply `SourceDenseLowering.readout_qualification_identity` from independent
qualification of that implementation. The declaration, instruction spelling,
width fact and an epilogue-stage label do not establish execution semantics.
With this declaration and qualification available, the ordinary source route
attempts fusion by default. Set `fuse_output_readout=False` for the paired
baseline while keeping the same facts, software specification and library.
The direct shared scheduling API uses `output_readout=` and
`readout_qualification_identity=` on `schedule_compact_dense`; source checking
additionally binds the original `output_operation` through
`schedule_source_dense`.

Source matching requires a same-block, single-use consumer of the complete
accumulator tensor. The pointwise computation must not read its initializer
or captured runtime values. Exact typed integer constants may be captured;
the checker clones and checks their current values rather than treating every
capture as invariant. Unknown intervening operations are barriers. Missing
capability or qualification, additional accumulator consumers and unsupported
epilogues retain the dense contraction and ordinary host epilogue. Malformed
declarations, stale inputs and incorrect realized stores are errors.

When selecting the shared logical descriptor bridge, also supply
`LogicalBorrowedABI.readout_format_qualification_identity` for the final i8
pointer format, matching the selected readout qualification. The bridge must
use one-byte output elements in its descriptor guards, scratch budget, C
signature and packing fallback. Its accumulator allocation remains four-byte
i32 storage. This is separate from borrowed-buffer completion, noescape and
nofree qualification; neither fact implies the other.

The actual projected store must reconstruct the checked readout identity from
its instruction and operands. The verifier checks final output width, tails,
complete reduction and accumulator ownership, and accounts requested output
bytes at the final width. These checks condition on independently qualified
instruction implementations; a selected plan or host source rewrite alone
does not prove that a backend eliminated its intermediate transfers.

The mechanism stays within the existing 61-source shared library. Phase 1 and
Phase 2 must receive the same version, source hashes, controls and verification
rules. Tuning may select a legal readout; it cannot change its scalar semantics,
hardware widths, qualification or numerical thresholds. Trusted model-binding
services are not added to the authoring import grant.

Current validation covers source matching, rectangular primitives, tails,
separate accumulator/output widths, missing qualification and mutations of
store semantics or checked context. Run the focused source tests with:

```sh
python -B -m pytest merlin/tests/infra/test_dense_integer_readout.py \
  merlin/tests/infra/test_pointwise_constant_captures.py \
  -p no:cacheprovider --basetemp "$TMPDIR/dense-integer-readout-check"
```

Ordinary complete-model paired execution is a separate qualification step.
This scope establishes no real accelerator support, timing improvement,
floating-scale reassociation or nearest-even readout implementation. Ordered
f32 scaling needs the existing full-domain numerical proof and original
qualified rounding/scale contract before it can use this path.

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
