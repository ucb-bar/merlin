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

The first delivery supplies executed common xDSL transformations, an optional
instruction dialect generator and an independent host Lean checking route.
It provides a small, explicitly modeled foundation. It does not establish a
complete tensor-to-device compiler or a whole-program SCF, memory, LLVM or RTL
equivalence theorem. The [capability roadmap](../design/verified_compiler_roadmap.md)
records the remaining pass groups and their required evidence.

## Select a fresh experiment release

New bundle preparation can select
`--common-substrate merlin.common_compiler_substrate.v1`, together with an
explicit `--lean-toolchain-manifest` and the reviewed `--host-index-bits` of the
selected host ABI. These options are available on ordinary `corpus prepare`
and bundle generation. Missing formal tools cannot satisfy covered obligations.

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

Repeated successful obligations can use a bounded process-local cache after
revalidating the toolchain inventory. The key includes actual IR, generated
query, semantic model and scope. A cache hit creates a fresh host completion;
saved summaries cannot seed the cache or discharge an obligation.

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
