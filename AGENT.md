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
Preserve corpus identities, hidden answers, frozen compiler/certificate attribution, and native
phase grading semantics. Register access identities before relocating graders or oracles.
Never infer dead runs from age or a directory suffix; use leases and explicit retention pins.
Do not modify historical evidence during migrations.

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
