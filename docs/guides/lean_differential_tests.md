---
title: Lean regression tests and semantic differential evaluation
kind: guide
status: current
owner: verification
last_verified: 2026-11-10
related: [lean_compiler_stack, lean_bounded_pass_profiles, compiler_verification]
code_refs:
  - merlin/tests/data/lean_suite.json
  - build_tools/scripts/run_lean_tests.py
  - build_tools/scripts/check_lean_guard.py
  - src/merlin/verify/lean/differential.py
  - src/merlin/verify/lean/mlir_execute.py
---

# Lean regression tests and semantic differential evaluation

The executable regression suite contains **200 pytest cases** in 30 files. Its
explicit file roster is [lean_suite.json](../../merlin/tests/data/lean_suite.json).
The runner rejects an empty suite or a collection above 200 cases:

```sh
.venv/bin/python build_tools/scripts/run_lean_tests.py --collect-only -q
.venv/bin/python build_tools/scripts/run_lean_tests.py -q
```

Configure the pinned Lean SDK, compiler Python and LLVM translator using the
[compiler-gate tutorial](lean_bounded_pass_profiles.md#configure-the-tools).
Set `MERLIN_LEAN_REQUIRED=1` when qualifying the suite: absent tools then fail
explicitly. An ordinary developer run may skip tests whose tools are missing.
Use the SDK's concrete `bin` directory; elan shims without a default toolchain
cannot resolve the pin from a temporary graph-proof working directory.

## Retained coverage

The suite checks tensor DCE, Q/DQ, draft MACC, ordered unrolling and tiling,
private buffer reuse, configuration-write elision, configuration hoisting and
host graph fusion. Representative positive checks use real Lean checks and
actual compiler boundaries. Negative cases cover corrupted arithmetic, loop
carry changes, rejected configuration changes, forged evidence, stale checker
sources, worker crashes/timeouts, missing verdicts and product quarantine.
Replay cases reconstruct fresh proofs rather than accepting saved statuses.
Coverage controls distinguish source exclusions from accepted proof jobs.

The former large Python suite and its parameter grids were removed. The current
suite intentionally samples fewer shapes, numerical formats, source grammars,
frontend integrations and native execution cases. Passing it establishes these
retained regressions, not the broader historical test population. Dated evidence
retains its original source version and denominator; removed test modules remain
readable at 04ed0ae25, the parent of the consolidation commit.

The sixteen-kernel and complete-model campaigns run separately, outside the
roster ([gate overview](lean_compiler_stack.md#coverage-and-qualification-status)).
The kernel campaign's per-case test is slow-marked and skipped unless
`MERLIN_RUN_KERNEL_HANDOFF=1`, so it never enters a 200-case run.

## Full proof build and audit

The Python case ceiling does not reduce the Lean semantics or soundness proofs.
The 68 Lean-native regression modules remain part of the library build. Continue
to run the complete build, statement-lock check and axiom/declaration audit:

```sh
.venv/bin/python build_tools/scripts/check_lean_guard.py
```

The guard builds `MerlinVerify`, `merlin-lean` and `merlin-lean-audit`; verifies
frozen files and pinned theorem statements; and audits every current source
module. The separate packaged graph proof tree is compiled and audited by its
actual gate checks. See the [gate overview](lean_compiler_stack.md) for claims
and the upstream trust boundary.

## Differential evaluation

The reusable `merlin.verify.lean.differential` driver and native execution
adapter remain available for separately authored evaluations. They compare
finite evaluations of the Lean semantics and the selected native MLIR pipeline.
Such evaluations require matching parser/execution-engine bindings and explicit
LLVM/runtime tools; they supply finite semantic-fidelity evidence, not universal
upstream compiler verification. The reduced Python suite does not retain the
former exhaustive per-rule native differential matrix.

The former upstream elementwise-fusion differential suite was removed with the
rest of the superseded tests. Its helper module had already been deleted during
the upstream-verification cleanup, leaving an import that prevented collection.
Merlin-owned host graph fusion remains covered by acceptance, deliberate
corruption and fresh-process replay cases.
