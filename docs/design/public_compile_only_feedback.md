---
title: Public scale compilation through the original brokers
kind: design
status: current
owner: ir
last_verified: 2026-10-13
related: [verified_compiler_tooling, verified_compiler_roadmap, phase2_test_justification]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase0/public_compile_sources.py, packages/merlin-experiments/src/merlin_experiments/phase1/feedback/compile_only.py, packages/merlin-experiments/src/merlin_experiments/phase1/feedback/compiler_profile_child.py, packages/merlin-experiments/src/merlin_experiments/phase1/brokers/compiler_call_channel.py, packages/merlin-experiments/src/merlin_experiments/phase2/public_compile_only.py, src/merlin/targetgen/compile_only_execution.py, src/merlin/verify/lean_emission_chain.py]
---

# Public scale compilation through the original brokers

The original Phase 1 selfcheck accepts `--compile-only --capsules all` or a
comma-separated list of original public member names. It derives complete static
tensor ABIs from retained Phase 0 tensor sources, invokes the same package
commands, stock LLVM translation, object compiler and no-data linker, and reopens
the actual products and invocation records. No reference tensors, captured
weights, numerical harness or simulator are used.

`phase0.public_compile_sources` preserves arbitrary public compositions exactly.
For a single original integer contraction/copy supported by the existing generic
source producer, it also constructs type-only siblings at extent ratios 8 and 64.
These fixed source-metadata probes inspect code growth; they do not recommend a
target schedule or use validation-model shapes. Existing public resource-frontier
members retain their source types and derivation identity. Unsupported sources,
dynamic shapes, captured values and exhausted metadata budgets remain unavailable.

For faster feedback, `--compile-only --scale-probe --capsules PUBLIC_ID` stops
after ordinary package lowering and stock LLVM translation. It builds no object,
links no program and executes no simulator. Retained public integer interface
matmul, movement and NHWC convolution sources gain a fixed scale ladder (8 and
64), including tails on either side of an unambiguous selected hardware edge.
The producer preserves original dtypes, convolution stride/padding/dilation,
input joins and complete output geometry. Unsupported layouts, floating
precision and compositions remain exact original sources. Existing linalg
sources retain their ordinary source producer. No values or reference outputs
are allocated, and no private model shape supplies a case.

Scale feedback reports observed LLVM function/code bytes, inline assembly and
call sites, source/product hashes and aggregate wall time. These lexical counts
describe compiler resource cost, not instruction effects or correctness. A
typed original-host `DeviceIRBudget` can refuse static expansion before object
compilation; clients cannot supply or weaken its limits. Without such a closed
host selection, feedback explicitly reports `resource_budget.mode=not_selected`.
Later failures retain already-emitted byte counts as unverified partial output.
Every numerical, device-resource, index, output-coverage and static semantic
obligation remains unproved; this diagnostic alone never establishes compiler
or whole-model qualification. Counted-loop source lowering is still a backend
responsibility, while Merlin supplies reusable loop transforms and observations.

A required scale resource gate also demands complete shape coverage for each
selected public member. Interface sources must retain the original, 8-fold and
64-fold geometric extents and both tails at the selected positive hardware edge.
The ordinary linalg producer supplies its original/8-fold/64-fold geometric
domain, without claiming hardware-tail coverage. Larger actual tensor ABIs and
distinct source identities remain bound to the observed rows. An unsupported
exact-only source, omitted large shape or missing selected tail stays incomplete
even when all available small emissions fit the budget.

The request cannot choose source paths, ABI, index width, resource constraints,
toolchain, compiler-library admission, instruction policy or proof results.
The original archived bundle and immutable snapshot select the common service.
Its original import-admission scope permits exactly the common and optional
public leaves from that Phase 1 profile. The package box reapplies those public
snapshot grants and the existing final answer masks; host checker files and
the original author workspace are never mounted. Phase 1 makes one bounded copy
of compiler source (32 MiB/8,192 files), excluding generated build trees. Products
are private temporary files, removed after their verified hashes are reported.

Phase 2's ordinary retained-revision queue adds
`kind=corpus-compile-only-feedback` only for a newly admitted Phase 1 profile.
It uses the same original public source derivation and compiler tools. Immutable
revision, input, runtime-source and profile identities remain bound to the job;
authors can continue editing another revision while the original queue owns
the in-flight source. Existing CPU/RAM/disk/deadline and cleanup grants apply.
Compile-only work consumes zero simulator-engine tokens; its assigned CPU and
memory/disk/deadline bounds still apply, and it cannot dispatch a simulator.
No second grader or compiler implementation is introduced.

Feedback includes actual source, ABI, object, linked entry, ELF, stack-limit and
invocation identities, code sizes and compile wall time. A successful link earns
no numerical, RTL, device-resource, index-bound or output-coverage pass. Those
obligations remain `UNKNOWN`; `all_pass` is false and timing cycles are absent.
The original complete numerical/performance gate and champion selection are
unchanged. Unsupported target build or instruction-policy services refuse the
observation. CPU protocol controls qualify this transport, not a full-model
compiler, deployment or performance result.

An optional original-host emission selection can additionally check a bounded
`emit_transformation_trace` chain emitted by the same immutable compiler. Its
first checkpoint must be the exact original input and its last checkpoint the
exact target artifact consumed by the observed object producer. Each edge is
checked freshly by the original broker, without accepting candidate receipts.
Covered failures reject; unsupported edges retain independent-check obligations.
The trace establishes a checked relationship, not the compiler's internal history
or LLVM/object semantics. Reopening transport records cannot recreate a live
formal completion, and no numerical/RTL credit follows from this chain.

The active original V3 scope also connects traced ordinary interface lowering
to the actual object entry through `lean_emission_context`. It retains the actual
input/emission observation, checks the raw checkpoint chain at object production,
and binds the fresh object consumed there. The entry is selected by the original
build/capability contract, never by trace metadata. Absent traces preserve
historical reuse with zero formal credit; unconsumed traces also grant none.
Unsupported edges remain unproved in partial reports; covered failures reject
the build. Custody covers one selected observed source, not other same-byte
emissions. The explicit compile-only proof transport avoids a second ordinary
check. This scope does not prove terminal lowering, dispatch execution,
host optimization or machine code, and leaves numerical/RTL gates independent.

## Ordinary compiler execution with the common profile

Fresh original Phase 1 archives select the same frozen profile for selfchecks,
round and continuous checkpoints, final grades, and Phase 2 development and
paired measurements. The original host admission scope permits only the exact
common public leaves and the originally selected optional author tools. Actual
compiler subprocesses receive those snapshot leaves at their recorded public
destinations, including split Python source roots. Automatic checker sources and
private grading inputs remain outside the compiler's mount view.

Package invocations isolate one input and declared output in a fresh private
workspace. Relocated grader copies must have the same bounded source inventory;
declared builds run against the grader's private copy and reopen its source
identity afterward. Threaded numerical and paired measurement work retains the
same original context. Historical bundles without a common profile retain their
existing execution behavior.

An exec-based whole-model budget child reconstructs the original execution
scope through `compiler_profile_child`. The original parent issues a bounded,
sealed readonly descriptor binding the frozen archive/snapshot/profile,
compiler source, parsed target, explicit sandbox inputs, selected interpreter,
direct host sources and process environment. The child reopens these selections
and any selected live resource boundary before entering the original compiler
scope; the parent rechecks them and the returned transport identity afterward.
Foreign or incomplete execution owners refuse rather than fall through to an
unboxed compiler. This route requires the exact original profile executor;
it does not reconstruct arbitrary alternative execution services. The inline
route remains available when `MERLIN_MODEL_BUDGET_S` is unset. Neither transport
identity nor process completion establishes whole-model correctness, deployment
or performance; the original complete grade remains required.

The same original package box preserves an explicitly caller-selected artifact
ABI 0.2 profile for structural emission. It revalidates the typed profile and
compiler source before publishing output, without mounting the profile or
converting artifact bytes to LLVM. ABI 0.2 still has no numerical/certification
adapter. LLVM compile-only and numerical qualification cannot consume this
structural permission as an execution admission. Frozen source scopes continue
to refuse Python cache writes; the core scratch-copy opt-in is separate.


### Checks requested inside a compiler invocation

The original host now stages the existing `.qa_channel` protocol in each
isolated compiler-call workspace. `MERLIN_COMMON_WORKSPACE` names that workspace;
the compiler passes it to the same public `apply_transform` or `check_transform`
client. The channel admits only those original actions. Its host service selects
the same frozen semantics, formal coverage and proof toolchain; request data
cannot replace them or assert a successful proof.

The protected host selection binds the compiler source inventory, actual input
bytes, profile/archive/snapshot identities and aggregate invocation deadline.
Replies bind the original request ID and source hashes and remain descriptive.
No saved reply reconstructs live checking authority or qualifies numerical/RTL
execution. The proof process and selected dependencies remain outside the
compiler namespace. On completion or cancellation the dedicated host owner
reaps its proof children before private staging is removed; source identities
are reopened afterward. Pending file, request-byte and request-count limits
bound this channel independently of simulator work.
