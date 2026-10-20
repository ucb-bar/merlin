# `prepare-fix-bool-fptosi`: frontend policy boundary

## Claim and proof boundary

The claim is `policy "frontend-float-to-bool"`. It compares the actual after module with a Lean-reconstructed intent program for the actual before module. It does **not** claim equivalence to `arith.fptosi`/`arith.fptoui` to `i1`: the original conversion can produce poison. The independently authored A2 file, `MerlinVerify/Spec/FrontendIntent.lean` (cherry-picked from `pv/specs` at `4d5965876`), defines `f32ToBool` by an f32 bit test: both signed zeros are false, all other f32 encodings including NaNs are true, and poison remains poison. A2 labels that file pending human review; this work does not approve it.

`FrontendBoolProgram.intent?` reads the authoritative exported **before** AST in Lean. It inventories definitions through nested regions, refuses duplicate SSA IDs and unsupported source sites, replaces every eligible f32-to-i1 cast, assigns fresh zero IDs, and checks each generated comparison's interpreted key and modeled attributes. `FrontendBoolProgram.check` then compares the reconstructed program with the authoritative exported **after** AST using exact `alphaOps` correspondence over the whole module and the existing `moduleCheck` on every defined function. This checks operand identities, result types and uses, attributes, properties, comparison predicate 13 (`une`), empty fast-math flags, +0 f32 bits, cast removal, SSA freshness, nested regions and all unrelated operations. The Python tests do not supply proof facts.

`check_sound` proves that acceptance implies both exact whole-module `alphaOps` correspondence and `FuncRefinesC` of every defined after function against the reconstructed intent under the existing Lean `Sem.Ops.opSem`/`denote` evaluator. `replacement_exec` and `accepted_site_exec` connect the generated, provenance-bearing comparison to A2's `f32ToBool` in the actual evaluator: for initialized f32 or poison, the replacement binds A2's value to the original cast result ID before *any* following operations. `pointwise_a2` proves the poison-aware tensor extension at every index; `linalg.generic` executes those scalar regions through the frozen evaluator. The accepted-site theorem exposes the site-occurrence, source-environment, fresh-ID and comparison-law premises rather than hiding them. `check_sound` is conditional refinement where the reconstructed source is modeled; it is not a proof that every possible SmolVLA execution is fully modeled or defined.

The legacy `FrontendBoolFptosi.Candidate` theorem remains a schematic f32 scalar sanity result and is **not** used by the integrated checker. Its comment now states that restriction accurately. The actual checker uses decoded before/after programs and `FrontendBoolExecution.checked_site_then_tail`, not Candidate flags.

## Float interpretation and review status

`UneCorrect M` is exactly: for every f32 bit pattern `x`, `M.cmp` at the no-fast-math `arith.cmpf une` key with positive zero returns false iff `x` has an all-zero magnitude field. `FrontendBoolFptosi.alwaysFalse_not_une` constructs a valid frozen `FloatModel` that violates this law, so the premise cannot be discharged for *all* current models. This repository has no concrete Lean IEEE comparison interpretation or theorem that proves `UneCorrect` for one. No axiom, invented law, frozen semantic change, or `SPEC_LOCK` change was made. Native samples check examples; they do not prove the universal premise. A2's specification also still requires human approval. Thus the receipt's `verified` verdict certifies the checked policy-program boundary with these named obligations, not unconditional IEEE/PyTorch correctness.

## Format and cast inventory

| Production `fix_bool_fptosi` case | Lean boundary status |
| --- | --- |
| `f32 -> i1` via `arith.fptosi`, default attributes | Accepted and covered by A2 `f32ToBool`, conditional on `UneCorrect` |
| `f32 -> i1` via `arith.fptoui`, default attributes | Accepted with the same bool intent and condition |
| `f16`, `bf16`, `f64 -> i1`, either signed or unsigned cast | Production pass rewrites; Lean refuses because A2 supplies no bit-level intent for these formats |
| Non-default cast attributes/properties, unsupported nesting or malformed SSA | Lean refuses explicitly |
| Casts to `i8`, `i32`, or other non-`i1` integer widths | Production pass leaves them alone; this bool-boundary checker requires at least one repaired site |

The production pass selects `AnyFloat`, which comprises f16, bf16, f32 and f64 in the supported dialect, and handles both `fptosi` and `fptoui`. The f32-only denominator is **not** the whole production pass. The pass and A2 spec were not edited here.

## Actual boundary, mutation and oracle evidence

The checked-in `SY_model_smolvla/capsule.interface.mlir` has one float-to-i1 site in `forward`; wider float-to-integer casts remain. The test runs the production pass, verifies the full xDSL module, parses and verifies both generic-form texts with the authoritative torch-mlir compiler bindings, exports both to Lean JSON, and runs `merlin-lean check-frontend-bool` over the full 31 MB exports. The receipt records the exact before/after byte hashes, export hashes, A2 source hash, `SPEC_LOCK` hash, checker binary hash, theorem axioms, and oracle version. The synthetic boundary also receives a replayed receipt. The recorded artifacts and receipts are under `out/artifacts/verification/merlin-passes/frontend/stages/`; the regression regenerates them from the checked-in capture.

The final actual-boundary receipt is `stages/0003_prepare-fix-bool-fptosi/receipt.json` in this checkout. Its verdict is `verified` for `forward`, with both `check_sound` and `accepted_site_exec` audited to only `propext`, `Classical.choice`, and `Quot.sound`. Byte bindings:

| Evidence | SHA-256 |
| --- | --- |
| Before MLIR | `f3749b75b3c196c132d21ae0b493a722a0009cb0845f49a55a0813dcb48bbc83` |
| After MLIR | `45c2b10259217eacd15b8bf9509c00a6dd3dbe57165de4a5137393e7e828043e` |
| Authoritative before export | `874bd9a354eb0d71c6cf8799e7ef2b282826d4e81ab4503f1bf1b0856ef5c193` |
| Authoritative after export | `f8272ae7f051110c2ff06235f9a3460401c609ef95b598df489e0fb297e28464` |
| A2 `FrontendIntent.lean` | `6f84e09fbf40206461924ff8dc72ce8842bbce9e13abaecb0b0d87e46c8f8e32` |
| Lean checker executable | `29d4f28ba42971198aeb7359be135e1e22ca6f0d059894e3012a668c5630a562` |
| Unchanged `SPEC_LOCK` | `a805dc3f0f78d2a95803463808f7d02f90ed0327b76124d4d14118f61328bc95` |

The 12.7 MB MLIR snapshots and 32.3 MB JSON exports are generated artifacts in the separate checkout; they are not committed to the branch. The checked-in regression recreates them from the tracked SmolVLA capture and verifies the generated pair with the compiler and Lean.
An independent regeneration from the tracked capture produced exactly one rewritten site and byte-for-byte matched both recorded MLIR hashes above.
The actual receipt was replayed from its bound snapshots and exports using `_validate_receipt` after the final build; replay returned no errors (`[]`).

`test_lean_frontend_bool.py` verifies each mutated **after** module with torch-mlir before exporting it to the actual Lean checker. The checker rejects ordered-ne and wrong predicates, `nnan` fast-math, a wrong source operand, negative zero, a wrong float type, a retained cast, a bypassed comparison, an incorrect `linalg.yield` use and an extra SSA definition. A duplicate same-scope SSA name is invalid MLIR and is rejected by the compiler parser/verifier; it is not counted as a Lean mutation. Separate tests confirm f32 unsigned acceptance and explicit refusal for f16, bf16 and f64 signed/unsigned cases.

The native differential tests in `test_bool_cast_poison.py` compile the repaired synthetic module and compare outputs with an independent f32 bit-pattern oracle and PyTorch `to(torch.bool)` for NaN, both infinities, +0, -0, positive and negative fractions, and ordinary positive and negative values. The original cast is inspected as a poison source, never assigned a numeric oracle result.

The local run used Lean 4.35.0, Mathlib revision `56a005855ea17e0b0958f73ec801ae140b0a1679`, torch-mlir `20261001`, xDSL `0.73.0`, PyTorch `2.14.1+cu130`, LLVM `23.0.0git` (`llc`), and GCC `16.2.1`. `test_bool_cast_poison.py`: **9 passed, 0 skipped**. The integrated Lean boundary suite: **4 synthetic/receipt tests passed**, and the actual SmolVLA case passed in a separate full-boundary run. `test_lean_chain.py` plus the A2 oracle test without a configured pinned interpreter: **13 passed, 1 skipped**; that skip is exactly `test_lean_spec_frontend_torch.py`, which requires `MERLIN_TORCH_PYTHON`. Forcing that test to use the available PyTorch `2.14.1+cu130` fails its explicit `2.10.0+cpu` version assertion. The pinned A2 oracle was therefore **not** measured here; the separate native differential used PyTorch 2.14.1 and has no skip.

## Integration limits

The ordinary refinement chain theorem cannot compose a policy repair against its poison-producing before program. `compose_chain` now replays a frontend policy receipt and records its label, while marking an ordinary source-refinement chain containing that link `abstracted` instead of silently treating it as `refine_conditional`. A future policy-aware chain theorem is needed for end-to-end composition across other preparation boundaries. The existing `FuncRefinesC` conclusion is also conditional on modeled execution; no full-model support or definedness claim for SmolVLA is made here.

The explicit request to leave `SPEC_LOCK` frozen was honored. The receipt binds the new theorem implementation by checker-executable hash and audits its axioms; it does not claim that the new theorem statement was added to `SPEC_LOCK`.

The remaining conditions for an unconditional frontend-intent claim are: a reviewed A2 specification, a justified concrete IEEE `FloatModel` with a Lean proof of `UneCorrect`, and a policy-aware chain theorem if this repair is to be composed as an end-to-end source claim. The local tensor and continuation theorems do not assert that all surrounding SmolVLA operations are supported by the frozen evaluator.
