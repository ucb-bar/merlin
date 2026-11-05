# AGENT.md — merlin/lean

## Purpose

`MerlinVerify`: Lean 4 checkers, with machine-checked soundness theorems, for supported
Merlin-owned compilation boundaries outside the OOT target compiler. Upstream MLIR/LLVM
passes are trusted; do not add verification for their implementations. Shared MLIR operation
semantics and lemmas used by Merlin checks remain in scope. See
[`verification_scope.md`](../../docs/design/verification_scope.md). The broader design is
[`docs/design/merlin_lean_verification.md`](../../docs/design/merlin_lean_verification.md). Read it
before changing anything here.

## Build

```sh
export PATH=$HOME/.elan/bin:$PATH          # elan-managed toolchain (lean-toolchain pins v4.35.0)
cd merlin/lean
lake exe cache get                         # Mathlib v4.35.0 oleans (decompress from ~/.cache/mathlib)
lake build
```

Packages and build products go under `out/build/lean/` (`lakefile.toml`), per the repo's one-root
output convention. Never commit anything from there. `out/build/lean/packages` must be a real
directory at the `lake-manifest.json` revisions; if Mathlib modules are missing, `lake exe cache get`
restores them. `lake build --no-build MerlinVerify merlin-lean merlin-lean-audit` must report every
target up to date before tests are meaningful.

## Running the tests

Configure the tools as in the
[tutorial](../../docs/guides/lean_bounded_pass_profiles.md#configure-the-tools)
(`MERLIN_COMPILER_PYTHON`, `MERLIN_LEAN_SDK_BIN`, `MERLIN_VALIDATE_WORKERS=1`, single-threaded
Lean/BLAS) and set `MERLIN_LEAN_REQUIRED=1`, so a missing tool fails instead of skipping. Two
non-obvious inputs:

- The configuration-elision and loop-transform unit tests compare against an original producer
  checkout at `repo_root().parent / "merlin-graph-proof-intake"`: a detached sibling worktree at
  `4499eebb959c0f92cefd92cc6d80646ba66474ed` (the revision the qualification workers pin; reachable
  from `origin/feat/megaphase`). Without it they fail with an unavailable producer or
  `No module named 'merlin.transforms'`.
- `test_lean_checked.py` skips unless `MERLIN_LEAN_TOOLCHAIN` names a toolchain manifest. An
  SDK-only manifest fails with `unknown module prefix 'Merlin'`; the manifest also needs Merlin's
  Lean library. `python -m merlin.verify.lean_cli prepare-toolchain` with that library as an extra
  `--library-root` is the likely route, not yet confirmed.

## Layout

| Path | What | Owner |
|---|---|---|
| `MerlinVerify/Core/AST.lean` | **G0-frozen** AST, mirrors `merlin/contract/schemas/lean_ir.schema.json` | lead |
| `MerlinVerify/Core/Json.lean`, `Core/Print.lean` | JSON decoder, generic-MLIR printer (round trip, D9) | WP-2 |
| `MerlinVerify/Sem/{Outcome,Float,Value,Refine,Program}.lean` | **G0-frozen** semantic core: outcomes, parametric floats, values, refinement, generic evaluator | lead |
| `MerlinVerify/Sem/Ops.lean` | **G0-frozen** dispatcher by dialect | lead |
| `MerlinVerify/Sem/Ops/{Arith,Linalg,Func}.lean` | op semantics: scalar `arith`/`math`, `linalg.generic`/`fill`/`index`, `func.call` | WP-4a |
| `MerlinVerify/Sem/Ops/{Tensor,LinalgNamed}.lean` | `tensor.*`, named linalg ops via their OpDSL generic form | WP-4b |
| `MerlinVerify/Check/Types.lean` | **G0-frozen** `ClaimType`, `Verdict`, `Checker` | lead |
| `MerlinVerify/Check/{Alpha,Corr,Step,C1,Synth,Module}.lean` | per-boundary refinement checkers (C1, module) | WP-6 |
| `MerlinVerify/Check/{Typing,Progress,Support}.lean` | value types and the progress checker (`checkOps_progress`) | WP-6 |
| `MerlinVerify/Check/Rules*.lean`, `Check/Affine.lean` | per-family typing rules with safety proofs; affine bounds | WP-6 |
| `MerlinVerify/Check/{Rulebook,Chain}.lean` | assembled rulebook, `supportCheck`, end-to-end `chain_refines` | WP-12 |
| `MerlinVerify/Rules/<Family>.lean` | rewrite-rule families and their theorems | per family WP |
| `MerlinVerify/Driver/` | composed drivers: W8A8 chain/bounded/IEEE/numerical, bf16 f32 accumulation, scalar square, source profile | per driver WP |
| `MerlinVerify/Tests/` | `example`/`#guard` regression files for checkers and rules; part of the library build | per family WP |
| `MerlinVerify/Spec/` | reviewed specifications (`Annotations`, `Dequant`, `Quantize`). Barrier-authored. | per spec WP, human review |
| `MerlinVerify/Claims/` | reviewed claim files; not created yet (claim statements are pinned by the guard's `PINNED` list) | review only |
| `MerlinVerify/Assumptions/` | the ONLY place an `axiom` may appear, each listed in the ledger; not created yet, still enforced by the guard and `Audit.lean` | review only |
| `Main.lean`, `Audit.lean` | the `merlin-lean` checker and `merlin-lean-audit` executables (`lakefile.toml`) | lead |
| `Differential.lean`, `W8A8*Main.lean` | drivers run with `lake env lean --run` by `src/merlin/verify/lean/{differential,*_policy}.py` | per driver WP |
| `src/merlin/verify/lean/mlir_export.py` | authoritative exporter: the pipeline's MLIR (via `torch_mlir`) → `merlin.lean.ir.v1` | WP-2 |

The compiler gates for loops, buffers, configuration writes and host fusion use a second source tree,
`src/merlin/verify/lean/Merlin/` (see its `AGENT.md`). It has no Lake project: each gate compiles the
exact source closure it binds with `lean -o` into a per-check library.

**G0-frozen** files change only by an interface change request to the lead. Additive lemmas about
frozen definitions go in a file you own, never in the frozen file.

## Rules (design §1.2 — enforced by `build_tools/scripts/check_lean_guard.py`)

1. No `sorry`, `admit`, new `axiom` outside `Assumptions/`, `native_decide`, `@[implemented_by]`,
   `@[extern]`, `unsafe`, or `partial def` anywhere under `MerlinVerify/`. `#print axioms` of every
   claim ⊆ {`propext`, `Classical.choice`, `Quot.sound`} plus reviewed `Assumptions/` entries.
   `bv_decide` adds a native axiom per proof on v4.35.0, so it is effectively banned too.
2. Certificates are hints. A checker recomputes every fact its soundness proof needs.
3. Unsupported is `stuck`/`abstracted`, never skipped and never verified.
4. Write the soundness statement before the checker.
5. Do not change the Merlin pass you are verifying except to add certificate-recorder calls; a suspected
   pass bug is reported, not fixed in passing.
6. Do not weaken or delete tests.
7. Floats are parametric (`Sem/Float.lean`). Never prove a float reassociation or constant fold.

## Gotchas

- `Check/AlphaFast.lean` installs a proved equality with `@[csimp]`, so compiled
  `alphaOps` calls use persistent indexes while the original definitions and
  theorem statements remain unchanged. Import it through `Check/Step.lean` or
  directly in a caller; `implemented_by` remains forbidden.
- `Float32.round` rounds half away from zero; MLIR `math.roundeven` rounds half to even.
- `/out/**` un-ignores nested `README.md`/`AGENT.md`; `/out/build/lean/**` is re-ignored so Mathlib's
  own docs are never swept into a commit.
- The evaluator (`Sem/Program.lean`) supports single-block regions and acyclic call graphs at G0;
  anything else is `stuck`.
