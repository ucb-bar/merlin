# AGENT.md — merlin/lean

## Purpose

`MerlinVerify`: Lean 4 checkers, with machine-checked soundness theorems, for every Merlin-owned
compilation boundary outside the OOT target compiler. The design is
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
output convention. Never commit anything from there.

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
| `MerlinVerify/Check/*` (other) | checker framework: graph diff, rules, congruence, `Defined₀`, C1 | WP-6 |
| `MerlinVerify/Rules/<Family>.lean` | rewrite-rule families and their theorems | per family WP |
| `MerlinVerify/Spec/` | reviewed specifications (policy, entry, annotations). Barrier-authored. | per spec WP, human review |
| `MerlinVerify/Claims/` | pinned top-level theorem statements | review only |
| `MerlinVerify/Assumptions/` | the ONLY place an `axiom` may appear; each listed in the ledger | review only |
| `exporter/` | C++ `merlin-mlir-export` (MLIR's own parser → `merlin.lean.ir.v1`) | WP-2 |

**G0-frozen** files change only by an interface change request to the lead. Additive lemmas about
frozen definitions go in a file you own, never in the frozen file.

## Rules (design §1.2 — enforced by `lean-guard` once WP-3 lands)

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

- `Float32.round` rounds half away from zero; MLIR `math.roundeven` rounds half to even.
- `/out/**` un-ignores nested `README.md`/`AGENT.md`; `/out/build/lean/**` is re-ignored so Mathlib's
  own docs are never swept into a commit.
- The evaluator (`Sem/Program.lean`) supports single-block regions and acyclic call graphs at G0;
  anything else is `stuck`.
