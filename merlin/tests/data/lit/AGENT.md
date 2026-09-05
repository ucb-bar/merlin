# AGENT.md — merlin/tests/data/lit

## Purpose

The **static verification layer**: `lit` + `FileCheck` tests that run one compiler pass on one module
and assert what it did. Milliseconds, no simulation. See `docs/design/compiler_verification.md`.

## Invariants

- Files here are **tracked, hand-authored, and target-independent**. They exercise the core dialects
  (`contract` / `schedule` / `interface` / `runtime`) and the frozen `merlin_iface` grammar, whose op
  names are defined in-tree and therefore have a derivation source.
- **Never CHECK a generated target dialect's op mnemonics.** Those are invented per backend-generation
  run. That experiment was run, corroborated over 383 runs, and removed — see
  `merlin/python/merlin/targetgen/rtl_check_compiler.py`. Per-target derived tests are generated into
  `out/artifacts/verify/lit/<target>/`, never tracked here.
- Run via `merlin/tests/ir/test_lit_suite.py`, or directly:
  `third_party/llvm-build/bin/llvm-lit -sv merlin/tests/data/lit`.
- A missing tool makes the suite `unsupported`, never green.
