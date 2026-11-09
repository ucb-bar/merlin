# Merlin repository

Merlin derives capsule tests and develops target compilers. The joint experiment
combines correctness and performance authoring using the native Phase 1 engine;
historical Phase 0/1/2 names remain in the module layout and separate workflows.

## Ownership

- `src/merlin/`: canonical Python source. Keep shared compiler, scheduling, capture,
  target/toolchain and verification primitives independent of research orchestration.
- `src/merlin/experiments/` and `src/merlin/analysis/dse/`: experiment orchestration
  and design-space analysis in the same namespace; optional dependencies stay lazy
  and compiler/research access boundaries remain explicit.
- `src/merlin/resources/` and `src/merlin/runtime/native/`: installed public data
  and target-independent native runtime sources, respectively.
- `packages/`: historical project metadata/inputs and the independent `merlin-timing` observer distribution.
- `examples/`: small walkthroughs and target-owned public inputs.
- `experiments/`: the catalog, study definitions and unbundled reproduction inputs;
  start here to select a study. These inputs are separate from installed engines.
- `tests/`: the permanent suite, capped at 150 collected cases including parametrization.
- `build_tools/`, `docs/`, `third_party/`: build/gates, durable docs, and opt-in upstream dependencies.
- `out/{runs,artifacts,build}/`: generated state. Required source/reference fixtures belong elsewhere.

There is one installed package at `src/merlin/`; the root `merlin/` tree is gone.
The src layout prevents the working directory from supplying an accidental second
package. See `docs/reference/repo_structure.md` for actual implementation owners
and retained compatibility modules.

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

## Readability

Follow `CONTRIBUTING.md` for module ownership and comment conventions. Keep one
coherent concept understandable in its owner. Review decomposition near 400
handwritten lines and ordinarily keep new or substantially rewritten modules
under 600; avoid both mechanical splits and monoliths just below a CI ceiling.
Preserve real verification, source-identity and ABI boundaries when splitting.

## Verification

Run the relevant behavior tests, source-layout/access checks, and
`python build_tools/scripts/check_structure.py`. Regenerate code-derived docs after changes.
Release checks must inspect actual wheels/sdists and install outside the checkout without optional
research dependencies. Report unavailable hardware separately from passing tests.
