# AGENT.md — src/merlin/verify

## Purpose

Compiler-pass verification: the **static** (lit/FileCheck) and **formal** (SMT/Lean) layers that sit beside
the capsule oracle ladder. The bench grades *outcomes*; this package verifies *passes*. Model, results
and working log: `docs/design/compiler_verification.md`.

## Modules

- `llvm_symbols.py` resolves a unique defined function through type-preserving,
  address-only LLVM aliases. Unknown expressions, cycles and declarations refuse;
  resolving an entry never removes the original helper bodies from auditing.
- `input_copy_loops`, `input_copy_integer` and `input_copy_batch` derive bounded
  original CFG domains and execute their constant control/address instructions in
  batches. Sibling loops reuse depth axes. Count-only observation leaves tensor
  and opaque results unknown; it supplies no memory/effect admission. The input
  transport proof additionally requires disjoint reads/writes, original byte
  origins and initialized scratch. Forwarding tracks actual source write epochs;
  every copy chain must end in a decoded input transfer or unchanged publication
  of decoded result bytes. CPU result reads require the intervening command fence.
  Scalar copy limits remain in force; command/control traversal shares the existing
  constant-control bounds. Batching bounds lanes per invocation, expanded work per
  region, actual vector instructions, processed payload bytes and aggregate storage.
  No numerical comparison or device-work gate is waived.
- `host_call_alignment` reopens actual host LLVM/object inputs and selected stock
  LLVM tools. It observes original descriptor call sites and proves low address
  bits structurally. Kernel assumptions and allocator names are not authority;
  unknown origins retain one-byte alignment. This proves no bounds, nonaliasing,
  mandatory execution, numerical or final-image property.

- `tools.py` — locate FileCheck / llvm-lit / mlir-opt / mlir-translate; `None` is a first-class answer.
- `smt_ops.py` — the one `smt` op xDSL does not ship: the solver scope the SMT-LIB exporter requires.
- `smt_export.py` — xDSL `smt` module -> upstream `mlir-translate --export-smtlib` -> z3.
- `smt_semantics.py` — a semantics for the `interface` dialect, given by lowering it to `smt`.
- `refine.py` — translation validation: assert the negation of the refinement relation, solve.
- `proofs.py` — audit `contract.prove` tokens as verified / asserted / unattributed.
- `counterexamples.py` — turn a solver counterexample into a capsule the bench grades, by writing a
  explicitly experiment-declared SMT sidecar entry (prefix `CX`, `source_role: smt_counterexample`)
  that Phase0's explicit `load_profile(..., smt_profile=...)` merges. Core does not discover that
  output path or import the experiment catalog. Going through the generator is the point: registration,
  dedupe, `MANIFEST.yaml` provenance, scrubbing and `golden.yaml` all come for free, and the input
  VALUES land under `out/artifacts/verification/` rather than in a capsule directory. An earlier
  `witness.py` wrote the directory itself, got none of that, and put a counterexample-inputs file
  where no ignore rule matched it — in a public repo. It is gone; do not reintroduce that shape.
- `faults.py` — the seeded fault corpus, one knob each, applied to real pass output.
- `evaluate.py` — run every fault past every layer; produce the detection matrix.
- `scalar_ir.py` — byte-bound QF_BV translation validation of a pure scalar-integer before/after
  MLIR pair (first consumer: `perf.whole_model_chunks.chunk_forward`); everything else abstains.
- `outline_ir.py` — byte-bound structural check of `outline_dispatches` output: expands the emitted
  private kernel calls and compares typed SSA graphs; a syntactic proof, not a matmul theorem.
- `split_reduction.py` — SMT proof or counterexample for one signed split-K accumulation at
  declared operand/partial/output widths; an algebraic obligation, not an emitted-kernel check.
- `merlin.verify.plots` — optional analysis-owned figures under
  `packages/merlin-analysis/src/merlin/verify/`, each generated from a JSON record, never a literal.
- `merlin.verify.replay` / `merlin.verify.replay_layers` — optional analysis-owned historical replay,
  under `packages/merlin-analysis/src/merlin/verify/`; install the analysis `replay` extra.

## Invariants

- `lean_contract.py` owns public result data and strict modeled coverage; it cannot issue
  accepted evidence. `lean_export.py` projects actual pure integer SSA under explicit
  primitive semantics. `lean_checker.py` runs the selected Lean kernel and owns live
  host completions. Deserialized receipts, candidate callbacks and status strings are
  never acceptance authority. Reuse only bounded process-local successful proof checks
  bound to actual IR, semantic model, scope and revalidated toolchain products.
- The common transformation broker chooses the frozen semantics, index width and
  toolchain. Lower experiment levels receive automatic checking through that broker;
  the author-facing Lean API is an explicitly separate treatment. Do not broaden grants.
- Loop/storage checks combine parameterized arithmetic/lifetime lemmas with trusted
  structural correspondence. They do not prove complete SCF/memory execution, LLVM,
  physical resource allocation, ISA encoding, numerical policy or RTL behavior.
  Unsupported effects, poison flags and metadata remain explicit gaps.
- `lean_obligations.py` projects V2 total-scalar rewrites and ordered unrolling
  only after exact IR correspondence passes. Generic function/fold lemmas do not
  model timing, allocation failure or arbitrary memory/RTL execution. The
  invocation inventory commitment is written after closed-file byte validation;
  a saved commitment cannot seed the process-local completion ledger.
- `lean_checked.py` reconstructs actual emitted resource geometry and actual
  authored sequential reorderings before live receipt acceptance. V3's
  `lean_isa_semantics.py` selects original frozen software/facts; candidate
  messages cannot set primitive meanings, bank geometry or atomic completion.
  `StateEffects` derives commutation from read/write frames under admitted
  sequential semantics. `ResourceGeometry` checks actual projected intervals
  and address relations; neither supplies a physical/RTL proof.

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
- **A detection rate is only worth as much as its denominator.** `replay.py` exists because the first
  answer to "would this have caught real bugs?" was seven hand-picked fixes, six caught — a
  demonstration chosen after the layers existed, by someone who knew what they check. Never cite that
  shape of number. The replacement fixes the population before drawing (every `fix(` commit touching an
  observed path), draws with a recorded seed, uses shuffle-then-take so a larger sample EXTENDS the
  smaller one rather than replacing it, reports `unreplayable` commits instead of dropping them, and
  reports the rate over fixes that PREDATE the layers separately — a fix that shipped with its own
  regression test is caught by that test, not by the layer.
- **A shadowed package that will not import is not a detection.** The replay pins old files over a copy
  of all package owners, which can fail to import because a sibling moved. Exit 1 alone is not proof:
  a trusted live bootstrap must record executed assertion/numeric checks. Import, setup, runtime and
  all-skipped failures are unavailable. Missing expected numeric outputs are explicit rejections.
  New records name `executed_checks_v2`; old exit-code-only records are not equivalent. The lit layer
  remains unqualified in replay until its child commands provide isolated execution receipts;
  standalone lit remains available. Do not silently present a partial instrument as all five layers.
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

- Core owns the `merlin.verify` namespace initializer; optional analysis contributes figures and
  historical replay. Run tests with the intended checkout's interpreter; the shared test configuration
  pins that checkout's core and extension source roots. Historical replay instead explicitly pins
  its composite frozen source root and refuses fallback to live or editable source owners.
- xDSL ships the `smt` dialect but **not** `smt.solver`, and has no extract/concat/extend op. Checked
  0.68 and 0.70 — identical. Upgrading does not help; `smt_ops` supplies the scope and
  `smt_semantics.in_range` constrains element widths with the shift identity instead.
- Keep queries quantifier-free: take extents CONCRETE from the IR's own types.

`lean_sat_runtime` binds generated BV obligations to an explicit original SAT
executable/file inventory. Native reflection requires independent original
opt-in, uses the exact frozen solver path and labels the broader Lean compiler
and native LRAT TCB. Successful selected BV checks use `lean_native_reflection`,
not kernel-only assurance. `Lean.ofReduceBool` and `Lean.trustCompiler` are
admitted only in this original mode; other reported axioms retain the fixed
policy, and kernel reduction remains separately labeled `lean_kernel`.
Identifiers are lexed structurally, excluding comments and strings; a generated
query cannot override the solver. Saved documents remain descriptive, not live
completion authority. Native loader closure and deployment stay independent.

`lean_emission_products` issues immutable text carriers only within the original
host process and live scope. Lookup uses exact object identity; equal-byte
strings, serialized IDs and reconstructed objects cannot select a source.
Trusted artifact copies retain exact path/byte association, with ambiguous
paths refusing. Failed claims remain pending, and object sharing stays within
one observed origin. Host rewrites invalidate previous path ownership; intervening
writers prevent stale registration. Completed sources may feed later ordinary
consumers only with a fresh source check and actual object observation, bypassing
legacy byte caches. Bound pending/completed text, alias and writer records; never copy tensor
data or grant an author new source permissions through this bookkeeping.

Observed source/object transports inspect only their live emission issuer before
entering the existing fresh custody operation. That inspection is not admission:
the operation must verify the original owner before any candidate/native work.
Source publication remains after independent operation-exit verification and its
own fresh locked publication check. Pure retention inside that same locked check
must not repeat the entire payload scan. Never issue aliases before final custody
or reuse this bookkeeping path across commands, threads, or unlocked work.
