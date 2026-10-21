# `prepare-fix-bool-fptosi`: bounded f32 checkpoint

**Status:** implemented conditional program proof and policy composition; proposed
IEEE comparison instance proved; A2/program-policy/model review remains pending.
See [the human review packet](A2-f32-review.md) for the exact proposal and open decisions.
No human approval is inferred from a passing checker or test.

## Final claims

The accepted boundary carries `policy "frontend-float-to-bool"`. The authoritative
before/after MLIR is exported by torch-mlir and decoded by Lean. No Python
Candidate flags are proof facts. The compiler pass, W8A8, frozen evaluator, A2
primitive and `SPEC_LOCK` were not changed in this checkpoint.

`FrontendBoolProgram.intent?` inventories all definitions, including nested block
arguments, rejects duplicate IDs, computes source eligibility/types, and replaces
every eligible f32-to-i1 signed or unsigned cast. Its generated +0 IDs start above
the entire source inventory. `check` requires exact `alphaOps` correspondence of
the whole materialized program and target, existing `moduleCheck` acceptance for
all defined functions, modeled comparison keys, and no other top-level changes.
It checks source operands, positive-zero bits/type, predicate 13, empty fast-math,
result replacement/uses, fresh SSA, regions, attributes and unrelated operations.
Unsupported cases refuse acceptance rather than reducing the denominator.

The final program theorem is:

```
check before after = true
  => exists intended, intent? before = some intended
     and, for every defined function f,
       FuncRefinesC opSem True True after f intended f
```

This is `FrontendBoolChain.accepted_intent`, derived from the existing
`FrontendBoolProgram.check_sound` and MLIR evaluator `moduleCheck_sound`.
The reference is the reconstructed comparison program, not the original
potentially poison-producing cast. Conditional refinement allows an unmodeled
reference execution to remain stuck; it does not establish full SmolVLA support.

`FrontendF32Comparison.accepted_site_exec` connects an occurrence of a generated
site in an accepted intent program to the bit-level A2 value in `evalOps`, followed
by an arbitrary continuation. It discharges `UneCorrect` with the proposed f32
instance. Occurrence, source-environment lookup, result type and source/zero
freshness remain explicit local premises. `model_site` covers actual `opSem`,
including poison; `model_tensor` proves the coordinate extension, including
independently poison coordinates. Existing `linalg.generic` semantics executes
scalar region bodies. There is no separate global PyTorch source interpreter or
proof identifying such an interpreter with the materialization. A2's primitive
file explicitly says it is not a completed source-program transformer; human
review must also approve the proposed program policy.

The older `FrontendBoolFptosi.Candidate.check_sound` is a schematic initialized
f32 sanity lemma, with its restricted comment. The integrated checker does not
use Candidate booleans.

## What happened to `UneCorrect`

Its exact premise is that comparison with f32 positive zero under the exact
no-fast-math UNE key returns false for either zero encoding and true for all other
encodings. Poison is handled outside this premise by eager evaluator propagation.
The frozen arbitrary `FloatModel` cannot imply the law: `alwaysFalse_not_une`
constructs a countermodel.

There **is** a concrete host adapter in `Differential.lean`. It widens f32 inputs
through `Float32.toFloat`, an opaque extern with no logical equation in Lean
4.35.0. This precise conversion bridge prevents the host proof; float unpacking,
NaN tests and comparison themselves have logical models. The earlier report's
claim that the repository had no concrete adapter was incorrect.

The new **proposed**, non-default `FrontendF32Comparison.model(base)` overrides
only f32 at `uneKey` with the complement of ordered equality on Lean's existing
IEEE binary32 unpack model. `une_zero` proves the law over all 2^32 encodings;
`model_correct` proves `UneCorrect` for that instance. All NaN encodings, either
infinity, both zeros, subnormals, fractions and ordinary values are included.
The proof uses no new axiom and does not change the frozen semantics or host
adapter. It does not prove equality with the host adapter or compiler execution.
Adopting the instance as the approved MLIR interpretation awaits review. The exact
alternative host bridge and smallest proposal are in the review packet.

## Pipeline integration

`FrontendBoolChain.chain_after_repair_refinesC` proves:

```
accepted repair before -> after
and ChainRefinesC(opSem, ctxOk, D, f, after :: suffix)
  => FuncRefinesC(opSem, ctxOk, D, final, f, Intent(before), f)
```

Context and input domains are explicit. `chain_after_repair_strict` adds
`FuncNotStuck` of **Intent(before)** on that domain to obtain strict refinement.
Source support cannot replace that premise. Successful, nonpoison final execution
also needs intent definedness; that claim is not implemented here for SmolVLA.

`compose_chain` replays the actual boundary checker, checks byte continuity,
function coverage, argument types/shapes, index widths, late witness freshness
and handoff binding, then writes a separate `policy_composition` record. It binds
the exact repair-before, final-after, A2 and composition proof source hashes and
audits theorem axioms. Frontend receipts additionally bind and audit the proposed
comparison source, accepted-site and tensor theorems. Tampered model status or
proof metadata is refused on replay. Multiple repair checkpoints are explicitly
unsupported. Prefix boundaries are recorded but do not move the reference back
to `Intent(original capture)`.

A nested `policy_composition.verdict = verified` means conditional refinement of
the final program against the materialized policy at that exact checkpoint. Its
A2-review/model fields still state pending obligations. The ordinary
`refinement_verdict` and overall chain verdict remain `abstracted` across a policy
repair. The original-source claim, support requirement and gate were not weakened
to obtain a passing chain.

## Format and cast inventory

| Production case | This checkpoint |
| --- | --- |
| f32 to i1, `arith.fptosi`, default source attributes | Checked policy materialization; proved comparison instance proposed |
| f32 to i1, `arith.fptoui`, default source attributes | Same f32 bool policy; capture-intent convention needs review |
| f16, bf16, f64 to i1, either cast | Production rewrites; Lean explicitly refuses; no coverage extension |
| Nondefault cast attributes/properties, duplicate IDs, nesting beyond fuel 64 | Explicit refusal |
| Cast to integer widths other than i1 | Production leaves it alone; unrelated operations must correspond |
| No repaired site | Dedicated repair checker refuses |

The production pass handles all four `AnyFloat` formats and both cast spellings.
f32 coverage is not verification of the entire pass. Direct tensor-typed casts
are unsupported; supported tensor cases execute f32 scalar casts inside regions.

## Qualification: execution tests, separate from proofs

- Actual SmolVLA capture: full authoritative MLIR before/after verifier, export,
  and Lean checker acceptance; one repaired site in `forward`, wider casts retained.
- Independent synthetic signed/unsigned f32 cases and a two-site case including an
  unused unsigned cast. Retaining even the unused cast must fail the Lean check.
- Ten MLIR-valid output mutations reach and are rejected by the actual Lean
  checker: ordered-ne, wrong predicate, fast-math flags, wrong source, negative
  zero, wrong zero/type, retained cast, bypassed comparison, wrong replacement
  use, and an extra definition. A duplicate same-scope SSA name is rejected by
  the authoritative compiler; it is not counted as a valid Lean mutation.
- Pinned PyTorch **2.10.0+cpu**, CPython **3.12.14**, NumPy **2.5.4** actually ran.
  The A2 oracle compares 18 f32 bit patterns: signed zeros, signed fractions,
  signed minimum subnormals/normals, signed maximum finite values, signed
  infinities, positive/negative quiet/signaling NaNs, and signed ones.
- Native differential tests compile the repaired signed-capture fixture and compare NaN, infinities,
  signed zeros, fractions and ordinary values with the independent bit oracle and
  pinned PyTorch. Poison is inspected structurally in the original LLVM cast;
  undefined numeric outputs are never used as an oracle. These samples do not
  prove a universal comparison law or the host widening bridge.
- Supplemental historical run: PyTorch **2.14.1+cu130** also passed the nine native
  tests. It remains supplemental and is not substituted for the pinned oracle.

Commands (run from this separate checkout, with the named environments):

```sh
export MERLIN_COMPILER_PYTHON=/home/jeremy/Projects/model2MLIR/.venv/bin/python
export MERLIN_TORCH_PYTHON=/tmp/merlin-pv-frontend-pytorch210/bin/python
export MERLIN_LEAN_REQUIRED=1
export PYTHONPATH=/tmp/merlin-pv-frontend/src
export MERLIN_CLANG=/usr/bin/gcc
export MERLIN_LLVM_LLC=/home/jeremy/Projects/llvm-project/build/bin/llc
export MERLIN_MLIR_INSTALL=/home/jeremy/Projects/llvm-project/build
/home/jeremy/Projects/merlin/.venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_spec_frontend_torch.py \
  merlin/tests/ir/test_bool_cast_poison.py -q -rs
# 10 passed, 0 skipped; 13.43 s
```

Other measured versions: Lean **4.35.0**; Mathlib
`56a005855ea17e0b0958f73ec801ae140b0a1679`; torch-mlir **20261001**;
xDSL **0.73.0**; LLVM **23.0.0git** at
`a47bddccec30255619bb8c37fa59700e661d4e66`; GCC **16.2.1 20260810**.
The compiler interpreter contains supplemental PyTorch 2.14.1; the oracle above
runs in the separate pinned CPU interpreter.

Final regression command, using the exports above and additionally
`PATH=/home/jeremy/.elan/bin:$PATH`:

```sh
/home/jeremy/Projects/merlin/.venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_frontend_bool.py \
  merlin/tests/ir/test_lean_spec_frontend_torch.py \
  merlin/tests/ir/test_lean_chain.py \
  merlin/tests/ir/test_lean_guard.py \
  -k 'not actual_smolvla_exported_program' -q -rs
# 26 passed, 1 deselected, 0 skipped; 150.34 s
```

The deselected full SmolVLA case passed in the preceding run against this same
checker build; all six current frontend tests are therefore measured, including
the new two-site case. The first combined run had **19 passed and 7 failed**:
all seven failures were guard-test setup failures because `lean` was absent from
`PATH`. After adding the Elan bin directory, those seven passed independently
(**4.72 s**) and again in the final regression. No prerequisite failure was
converted into a skip. Across the current suites, **36 distinct tests passed**
(frontend 6, chain 13, guard 7, pinned A2 oracle 1, native bool tests 9).

Build and guard results:

- `lake build merlin-lean` passed, followed by the full guard's
  `lake build MerlinVerify merlin-lean merlin-lean-audit` build.
- Audit of the comparison/composition imports: **4,092 theorems, 0 violations**.
  Full guard audit: **7,566 theorems, 0 violations**. All new claim theorems depend
  only on `propext`, `Classical.choice`, and `Quot.sound`.
- Full `build_tools/scripts/check_lean_guard.py` **fails the lock check** for
  exactly: added `MerlinVerify/Spec/FrontendIntent.lean`, changed
  `schemas/lean_chain.schema.json`, and changed `schemas/lean_receipt.schema.json`.
  No frozen semantic or pinned-statement mismatch was reported. `SPEC_LOCK`
  was deliberately not regenerated. This is an outstanding review/admission
  blocker, not a passing guard result. New theorem statements are not added to
  its pinned list; proof-source hashes and theorem audits do not replace review.
- Ruff check/format and `git diff --check` passed.
- The refreshed actual receipt was replayed through `_validate_receipt`, including
  round trips, the actual Lean checker and comparison-proof bindings: **no errors
  (`[]`)**. The actual snapshots/exports retained the hashes below.

## Actual artifact bindings

The generated receipt is
`stages/0003_prepare-fix-bool-fptosi/receipt.json` in this checkout. Large MLIR and
JSON artifacts are regenerated by the tracked test and are not committed.

| Evidence | SHA-256 |
| --- | --- |
| Before MLIR | `f3749b75b3c196c132d21ae0b493a722a0009cb0845f49a55a0813dcb48bbc83` |
| After MLIR | `45c2b10259217eacd15b8bf9509c00a6dd3dbe57165de4a5137393e7e828043e` |
| Authoritative before export | `874bd9a354eb0d71c6cf8799e7ef2b282826d4e81ab4503f1bf1b0856ef5c193` |
| Authoritative after export | `f8272ae7f051110c2ff06235f9a3460401c609ef95b598df489e0fb297e28464` |
| Unchanged A2 primitive | `6f84e09fbf40206461924ff8dc72ce8842bbce9e13abaecb0b0d87e46c8f8e32` |
| Checker executable | `87a93c9255b2645115887a9631d370e01d8f7caf4e233a2e873dd08c9a0be9a5` |
| Comparison proposal source | `832292eb06545561559de52fac5e8d42f63f187f4b11ac35a5aade0a6fbcef56` |
| Composition theorem source | `8e5bf2350757713030255f7544d8de1f7c56126a5f2772fe70d492df512b74b8` |
| Unchanged SPEC_LOCK | `a805dc3f0f78d2a95803463808f7d02f90ed0327b76124d4d14118f61328bc95` |

## Remaining obligations and checkpoint disposition

| Obligation | Status |
| --- | --- |
| Actual-program acceptance to materialized-intent evaluator refinement | Proved and integrated |
| Every generated f32 site's scalar bool result, poison and continuation | Proved for proposed IEEE instance, with explicit local occurrence/environment/freshness premises |
| Poison-aware coordinate lifting | Proved for proposed IEEE instance |
| Universal f32 `UneCorrect` | Proved for proposed IEEE instance; false for arbitrary `FloatModel` |
| `UneCorrect Differential.hostModel` | Unproved; exact opaque widening bridge identified |
| Conditional repair-policy plus ordinary suffix composition | Proved and integrated at exact checkpoint |
| Human approval of A2 primitive, capture convention, program policy and comparison instance | Pending; review packet provided |
| Global identification with a separate source-program A2/PyTorch interpreter | Not proved; no such completed transformer in A2 |
| Full SmolVLA intent support/definedness; strict/defined final execution | Unsupported here; no unconditional end-to-end claim |
| Multiple repair checkpoints or lifting policy across prefix | Unsupported |
| f16/bf16/f64 proof | Out of scope; explicitly refused |
| Frozen lock admission of new A2/schema/theorem artifacts | Pending review; lock deliberately unchanged |

This bounds the frontend checkpoint. The remaining work requires specification or
model review and, for full execution, a larger source/support proof. Further
format expansion is deferred so shared-pass/W8A8 verification can resume without
claiming these obligations are closed. No W8A8 changes were made here.
