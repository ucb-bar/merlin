# AGENT.md — merlin/python/merlin/verify

## Purpose

Compiler-pass verification: the **static** (lit/FileCheck) and **formal** (SMT) layers that sit beside
the capsule oracle ladder. The bench grades *outcomes*; this package verifies *passes*. Model, results
and working log: `docs/design/compiler_verification.md`.

## Modules

- `tools.py` — locate FileCheck / llvm-lit / mlir-opt / mlir-translate; `None` is a first-class answer.
- `smt_ops.py` — the one `smt` op xDSL does not ship: the solver scope the SMT-LIB exporter requires.
- `smt_export.py` — xDSL `smt` module -> upstream `mlir-translate --export-smtlib` -> z3.
- `smt_semantics.py` — a semantics for the `interface` dialect, given by lowering it to `smt`.
- `refine.py` — translation validation: assert the negation of the refinement relation, solve.
- `proofs.py` — audit `contract.prove` tokens as verified / asserted / unattributed.
- `witness.py` — turn a solver counterexample into a schema-valid witness the bench can grade.
- `faults.py` — the seeded fault corpus, one knob each, applied to real pass output.
- `evaluate.py` — run every fault past every layer; produce the detection matrix.
- `plots.py` — the four figures, each generated from a JSON record, never from a literal.

## Invariants

- **A layer that cannot run must never look like one that ran clean.** Missing tool, missing solver,
  solver timeout: all report an explicit state. `unknown` is never `verified`; `abstracted` is never a
  pass. This is the single rule the whole package is shaped around.
- **Multiply at the DATA's width, never the accumulator's.** Bit-blasted multiplier area scales as
  terms x width^2, so declaring an i8 element at a 32-bit accumulator width and constraining it down
  spends 16x the partial-product area for the same product. Measured 2026-09-05: refuting
  `swapped_matmul_operands` at 8x8x8 took 439 s that way and 26 s with a 16-bit multiply. Elements are
  declared at their own dtype width; `Encoder.sign_extend` widens via `smt.bv.concat` (the high half
  of `ashr(x, w-1)` is all sign bits, so concat IS sign extension). A product too wide for its
  accumulator raises rather than truncating.
- **Verification cost is not refutation cost, and the cheap one proves nothing about the other.** For
  a correct program both sides of the query are syntactically identical, so z3's rewriter collapses
  it in preprocessing without bit-blasting a single multiplier; `unsat` in seconds at a shape where
  `sat` never returns is the normal case, not a surprise. Never cite a scaling curve measured on
  correct programs as evidence that fault detection is tractable at that shape.
- **`unsat` is the PASS** for a refinement query, and `sat` must carry a counterexample. If a model
  ever comes back empty, the exporter's trailing `(reset)` is being handed to z3 — see `smt_export`.
- **Never CHECK a generated target dialect's op mnemonics.** They are invented per backend-generation
  run and have no derivation source; that experiment was run over 383 real runs and removed. Check the
  stable in-tree dialects (`contract`/`schedule`/`interface`/`runtime`) or the decoded instruction
  stream, which `targetgen/rtl_check_compiler.py` already owns.
- **Emitting a check is not verifying anything.** A check that was generated but never run records
  `unmeasured`; only a green suite records `verified`.
- Floats are verified structurally, never bit-exactly: reassociation is a legal backend choice, so a
  bit-exact float refinement would reject *correct* backends. `smt_semantics` raises rather than
  pretending.
- Verdicts are written to the shared log (`MERLIN_VERIFY_LOG`) that `check_pass_obligations.py` reads,
  via `xdsl_dialects.lowering.passes.record_verification`. Recording never gates a run.

## Gotchas

- Run python as `PYTHONPATH=<repo>/merlin/python .venv/bin/python`: the shared venv may resolve
  `merlin` from a different worktree.
- xDSL ships the `smt` dialect but **not** `smt.solver`, and has no extract/concat/extend op. Checked
  0.68 and 0.70 — identical. Upgrading does not help; `smt_ops` supplies the scope and
  `smt_semantics.in_range` constrains element widths with the shift identity instead.
- Keep queries quantifier-free: take extents CONCRETE from the IR's own types.
