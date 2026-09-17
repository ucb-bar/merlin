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
- **Negative controls use `--split-input-file -verify-diagnostics`, not `not ... | FileCheck`.** The
  `not` form passes whenever an error appears anywhere in the output, so it keeps passing after the
  constraint it names stops being enforced — any other diagnostic satisfies it. `-verify-diagnostics`
  binds each `expected-error` to a LINE and to its message text and additionally fails on any
  diagnostic that was not expected. Write the message by RUNNING the case and copying what the tool
  says; a guessed message fails against correct output (that has happened twice here).
- **A constraint the grammar cannot express is pinned in `iface/unchecked_by_irdl.mlir`, not omitted.**
  Those cases are malformed modules that are ACCEPTED, each naming the layer that does catch it. Their
  purpose is to stop a green suite being read as "everything malformed is rejected". If one starts
  failing, IRDL grew the expressiveness — move the case into `iface/invalid.mlir`.
