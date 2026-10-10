# Merlin repository

Merlin generates target compilers through three phases: derive capsule tests (0), build and
certify a functional compiler (1), and optimize target performance (2).

## Ownership

- `src/merlin/`: canonical Python source. Keep shared compiler, scheduling, capture,
  target/toolchain and verification primitives independent of research orchestration.
- `packages/`: optional research distributions. Do not duplicate core implementations.
- `experiments/`: the experiment catalog and versioned definitions; start here to run a study.
- `merlin/`: schemas, tests, native runtime, and legacy engines/resources during migration.
  `merlin/python/merlin` is a compatibility symlink, not another source tree.
- `build_tools/`, `docs/`, `third_party/`: build/gates, durable docs, and opt-in upstream dependencies.
- `out/{runs,artifacts,build}/`: generated state. Required source/reference fixtures belong elsewhere.

## Invariants

Read `CLAUDE.md`, local instructions, and `docs/reference/architecture.md` before changes.
Target-specific facts and implementations belong in OOT support packages. Evaluated compiler
candidates have stricter import/access rules than trusted support plugins.
Shared execution tooling may bind target contract/facts data without importing a
support implementation. Neutral tensor transport and hardware-legality checks
must not supply kernels, packing, schedules or preferred lowering choices.

**Compiler ownership rule:** all target-specific implementations belong in the target's OOT
MLIR dialect repository: dialect operations, instruction encodings, device kernels/schedules,
hardware layout and resource facts, target ABI glue, and target execution support. Merlin owns
reusable host code generation, packing, requantization, graph/global optimizations, dispatch,
buffer ownership, device compilation orchestration, and runtime infrastructure. A transform
that can be selected independently of the accelerator belongs in Merlin even if its first
performance evidence comes from one target. Keep the generic algorithm and contract in Merlin;
keep the target's legality facts, instruction selection and implementation in OOT. Optional
numeric or performance policies remain explicit, with their original correctness gates.
Do not add a shared implementation in OOT or a target-specific branch in core. When promoting
an OOT prototype, move its generic part to Merlin and delegate from the provider.

**Generalization rule:** the OOT dialect is a compiler backend, not a workload-specific
kernel generator. Production optimization and lowering must derive applicability from
operation semantics, shapes, layouts, numeric contracts and declared hardware capabilities.
Model names, captured provenance IDs, golden outputs and benchmark-specific constants must
not drive production decisions. Ordinary constant/shape specialization is valid when derived
from the current input IR with explicit legality and resource checks. Provenance IDs remain
valid for traceability and exact source-to-device binding, not optimization strategy selection.
Capture-specific selections belong in experiment drivers; promote their successful strategies
into general passes and cost models. Check additional shapes, tails and independent cases at promotion,
and retain precise fallback/refusal when the optimization cannot be proved.

Preserve corpus identities, hidden answers, frozen compiler/certificate attribution, and native
phase grading semantics. Register access identities before relocating graders or oracles.
Never infer dead runs from age or a directory suffix; use leases and explicit retention pins.
Do not modify historical evidence during migrations.

## Publication authorization

Do not open a pull request in any repository without explicit user approval.
An instruction to upstream or push changes does not authorize a new PR.
Use the user-authorized branch or direct-main workflow, preserving reviewed
changes and keeping published history intact.

## Before pushing

Fetch the remote and review **every commit that the push would publish**, including commits
introduced by a merge. Inspect the commit list, each commit's diff and size, and the final
range diff. Each commit must contain one coherent, reviewable change and follow the message
convention in `CLAUDE.md`. Split mixed or unnecessarily large commits and reword unclear or
nonconforming messages while they are still unpublished. Run the relevant checks and confirm
the remote branch is an ancestor before pushing. Do not push an unreviewed commit series,
even when the final tree passes tests or a push has been requested urgently.

Never force-push a shared or published branch to repair history. If poor history has already
been published, report it and prepare a separate reviewed remedy before changing that branch.

## Verification

Run the relevant behavior tests, source-layout/access checks, and
`python build_tools/scripts/check_structure.py`. Regenerate code-derived docs after changes.
Release checks must inspect actual wheels/sdists and install outside the checkout without optional
research packages. Report unavailable hardware separately from passing tests.
