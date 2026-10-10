# AGENT.md — merlin/python/merlin/targetgen/contract

## Purpose

Experiment-ABI contract layer.

## Modules

- `compile.py` — Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.
- `build_recipe.py` — Pure compile/link recipe; the runtime backend re-exports this same class.
- `build_service.py` — Typed host-only build service and isolated pure target-package loader; no backend discovery, execution or reference imports.
- `execution_service.py` — Explicit pinned functional transport; no backend discovery or correctness, ISA, hardware timing, or counter authority.
- `elf_admission.py` — Source-pinned linked-artifact admission before explicit simulator dispatch; rejected binaries are not executed or assigned numerical/timing results.
- `readback_policy.py` — Explicit invocation-only full-value output transport and selected build-byte receipt; never source this choice from a candidate command buffer or ambient environment.
- `interface_emit.py` — ``merlin_iface`` interface-grammar: emit a Merlin command buffer as contract text, and
- `schemas.py` — Fail-closed JSON-Schema validation against the ``merlin/contract/schemas/`` bundle.
- `toolchain.py` — MLIR toolchain resolution for the experiment ABI (env-overridable).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

`materialize.py` owns public copying, explicit-ceiling cohort publication, admission
validation, coverage and leases. It never selects evaluator adapters. Use core
`materialize_public_cohort` with an explicit ceiling; default evaluated selection is
`merlin_experiments.corpus.admission.public_capsules_for`, not a lazy core export.

`linalg_iface` inventories returned/data-consumed tensor constants, splats and fills
as payloads; destination-only initializers keep their prior representation. Ordered
`returns` bind actual SSA inputs/results, including repeated and multiple results;
unsupported multi-block joins stay unavailable. This structural inventory grants
no semantic owner, lowering implementation or correctness/resource/runtime proof.

`build_recipe.named_object_paths` shares deterministic object naming across
contract and layer builds. Equal basenames from caller and provider sources must
never overwrite one object; imported objects are reserved and link order is
retained. Unique basenames keep their original object names and commands.

`HarnessBuildRecipe.header_dependencies` declares direct harness headers for
ELF-cache identity. Exact bytes and first-match include-root resolution are
checked; a missing, unreadable or shadowed declaration disables caching. This
does not establish a complete transitive compiler-header closure.

The opt-in `ReadbackPolicy` changes only the trusted harness readback format.
Default calls retain their old build/cache path. A selected full-value build is
cache-free, requires a renderer that explicitly accepts the policy, and records
the unchanged command buffer, selected codec/recipe/source, generated harness,
object and ELF bytes. Recheck those pins after execution before any numerical
result. This is not complete toolchain closure or a numerical-support grant.

New readback build receipts bind the actual selected kernel object member as
well as its bytes. A reader never guesses a transform suffix or substitutes the
original object after stack repair. The named member must stay inside the build
owner and match the complete receipt identity. Historical unnamed receipts keep
their original fixed-name convention; their bytes are never rewritten.
The binary alternative is separately selected and stages its own length-aware
packer plus the existing range helper; a raw byte console is archived before
strict parsing. B64 and absent-policy behavior remain unchanged.

The opt-in coherent-memory policy stages no serial codec and uses a distinct
v2 build receipt. Execution requires an explicitly supplied trusted memory
reader, selected-engine revalidator and provider request ABI: bound physical output storage is admitted
before launch; normal exit, exactly one DONE, no serial substitutes, complete
logical values and unchanged build bytes are required afterward. Optional
evaluators own the memory decoder; core never imports a grader to select one.
An output-readback admission does not prove numerical or compiler correctness.

The separately selected packet-memory policy stages all three generic codec
headers and uses a distinct v3 build receipt. Its trusted reader checks the
ELF-bound bounded arena and complete logical output frames, not output padding.
Existing raw-memory, binary, B64 and absent-policy build identities stay unchanged.

An explicit `FunctionalExecutionService` can accompany `BuildOnlyService` through
the ordinary compile/full-output path. It runs only supplied pinned callbacks;
it never resolves a selected backend or upgrades an engine name into RTL timing.
Its source pins establish attribution and drift checks. Independently evaluated
semantic, physical-effect and hardware-runtime qualification belongs to the
private experiments owner, and cannot be issued by this transport.

The functional transport freezes exact function/code, bound-method owner and
stdlib partial bindings, rechecking them before and after running and parsing.
This shallow selection check proves neither mutable reachable state, import
closure nor source-to-bytecode correspondence.

An explicit `RecordedProcessExecution` fixes the native executable, direct ELF
operand, working directory, environment and stdout or merged-stream selection.
The functional service requires its exact fixed runner and source membership;
ordinary execution reopens the actual closed process and captured bytes before
and after parsing/output validation. Saved receipts, unrelated subprocesses or
matching console values cannot replace that execution. Legacy callbacks retain
their diagnostic route. Exact argv/input attribution does not prove loader or
ISA semantics, transitive dependencies, isolation, callback/source equivalence,
descendant cleanup, runtime qualification, effects, hardware or timing. The
native timeout bounds the selected subprocess; it is not an OS deadline or a
process-tree cleanup guarantee, and captured output has no aggregate byte limit.

An optional exact `PreparedProcessReadbackPlan` derives a values-free object
roster and complete file budget from the original typed direct harness ABI and
repeat/counter/phase plans. Only this explicit selection admits one whole-token
request operand and one output operand beside the single ELF operand. Its
private preparation files, original command buffer, selected sources and actual
ELF are rechecked before dispatch; the output must be absent before execution
and complete afterward. Actual invocation inputs, product bytes, tool, argv,
environment identity and captured stream are reopened at consumption. Source
pins do not prove imports or bytecode. The external source-owned loader and
decoder still owe exact ELF symbol resolution, packet grammar, original values,
histories, completion, isolation, resource and physical correspondence. This
transport issues no stage, runtime, cold/warm, measurement or timing authority.
Absent selection preserves the original unsupported-options refusal and serial
route. V1 supports one bounded exact-size readback file, not inherited channels
or arbitrary prepared commands.

The build-only renderer must be an actual Python function or bound method whose
inspected file and current bytes belong to the selected source roster. Unwrap
only exact stdlib `functools.partial`, never supplied wrapper/source attributes.
Validate before invocation and recheck afterward. The ordinary model adapter
pins its forwarding wrapper as well as its selected provider files. Direct file
membership does not prove bytecode, closure/import dependencies, genericity or
independent runtime qualification.

The optional `LinkedElfAdmissionService` executes after linking and before any
simulator dispatch. Reopen its actual report and exact ELF bytes before and after
execution. Refusal is completed evaluation data with `execution=not_attempted`,
never a fabricated console, output roster, timing result or completed process.
This source-bound transport does not issue instruction or physical authority.
It freezes the selected evaluator's actual function/code, bound owner and exact
stdlib partial bindings, using the functional transport's shallow selection
mechanism. Recheck through completed output decoding; same-file substitutions
cannot replace the preexecution policy. Mutable reachable state, import closure
and source-to-bytecode equivalence remain unproved.

`compile_only.CompileOnlySourceAbi` contains static original tensor declarations,
with provenance supplied by an independent source producer. Its retained
pointer reference links the ordinary kernel and dependencies without allocating
tensor storage, rendering numerical readback or executing the ELF. The shared
linker accepts it only with an explicit pure build service and no input values,
packing, warm profile or readback authority. Link success proves no output-store,
index, resource, numerical or physical obligation.
An explicit empty original input tuple or logical input map represents a zero-input
program; it is distinct from absent input data. The original output roster stays
nonempty and complete, with every output pointer and tensor type checked.

`source_observation` keeps the original straight-line reader contract when no
control-flow plan is selected. An explicit `ControlFlowObservationPlan` freezes
the whole LLVM input, entry, layout declaration and every reader budget. Its
separate versioned `observe_control_flow` method receives the complete bounded
static CFG, including cyclic joins and dead blocks, through the ordinary prepared
source-verifier call. The actual graph and input invocation are retained. Layout
declarations, callback attribution and graph inventory do not prove same-object
DataLayout/storage correspondence, source semantics, dynamic paths, output
coverage, ownership, effects, resources, stages or runtime. Unsupported module
metadata remains refused; do not strip it or silently reinterpret old readers.

`pointer_storage` declares original logical pointer storage only from explicit
software choices and original static tensor types. No contiguous/noalias/order,
endian, alignment or extent default comes from candidate output. The shared
pointer renderer may enforce the exact selected call binding. Source provenance,
actual allocation, lifetime, resource capacity and runtime effects remain
independent obligations; the optional experiment owner freezes a live selection
before authoring and exact public projection.

An explicitly selected `ExecutionDeadline` accompanies pure build and functional
services through ordinary translation, object, harness, link and execution.
Each subprocess receives the declining remainder and later stages refuse after
expiry. Completed in-process work is checked at stage boundaries; Python callbacks
are not preempted. Partial console and interrupted records remain diagnostics,
never completed values or a timer qualification. Unselected calls retain their
existing timeout behavior.

`compile_only.require_pointer_entry` checks the actual nonvariadic external C
entry, void result, plain pointer parameters and matching block signature.
Compile-only and explicit component numerical execution share this boundary;
matching arity alone cannot admit an integer parameter or another convention.
Parameter/result attributes without a supported calling contract refuse.
Preserved standard metadata supplies no body, device or effect proof.

Ordinary component execution and compile-only transport accept an optional exact
`CompilerLibraryContract` and canonical root together. Reopen their reviewed
member bytes before commands and at subsequent stage boundaries; retain the
same selection and actual member dependencies in transport reports and static
proofs. The fresh author view and qualified grader must use that same live
selection. An absent pair preserves the original import restrictions. Approved
direct imports and source-byte attribution do not establish semantic generality,
transitive dependency isolation, runtime correctness, effects or hardware timing.
