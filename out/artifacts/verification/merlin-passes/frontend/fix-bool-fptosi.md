# `fix_bool_fptosi`: integrated f32 frontend-intent proof

**Status (2026-10-21): the actual-program f32 repair proof is complete and integrated.**
The user approved the documented f32 source policy and direct IEEE instance.
No unproved comparison-law premise remains in `check_sound_ieee`. Full SmolVLA
execution support and campaign lock admission remain separate, explicit limits.
See [the reviewed policy](A2-f32-review.md), [approval record](f32-policy-approval.json)
and [admission request](lock-admission.md).

This work changes neither the compiler pass, W8A8, frozen evaluator, A2 primitive
nor SPEC_LOCK. It stays on `pv/frontend` in `/tmp/merlin-pv-frontend`; main is untouched.
It proves a frontend-intent repair, not ordinary equivalence/refinement of the
original numeric casts and not correctness of PyTorch capture or native codegen.

## Final theorem

`MerlinVerify.Rules.FrontendBoolSimulation.check_sound_ieee` states:

```text
check before after = true
  => for every base FloatModel, context, function name and argument list:
       policyDenote before (context with M := FrontendF32Comparison.model base) f args
       = denote opSem (context with M := FrontendF32Comparison.model base) after f args
```

The trust boundary includes authoritative MLIR export/round trip, Lean decoding,
the existing evaluator and external-environment interpretation, the kernel and
standard logical axioms, and the existing assumption ledger. Other float entries
remain parametric in `base`; native code generation is not proved here.

This is exact equality of evaluator outcomes, including successful results,
poison, UB and stuck diagnostics. It is stronger than conditional refinement.
It does not assert that every surrounding operation successfully executes.

`policyDenote` inventories the original source AST and runs the existing Lean
MLIR evaluator with `FrontendBoolPolicy.sem`: eligible scalar f32-to-i1 signed or
unsigned capture casts directly call A2's independent `f32ToBool`; every other
operation keeps frozen `opSem`. The same interpretation runs through nested
regions and internal calls. It neither constructs nor reads the after program.
The contract makes both signed zeros false, every other initialized f32 encoding
true (including all NaNs), and poison poison. Tensor-region execution lifts it
coordinate-wise. The approved capture convention identifies these casts as bool
intent; it does not reinterpret arbitrary handwritten numeric casts.

### What acceptance recomputes

The actual authoritative before/after exports are decoded in Lean. Source
eligibility, operand/result types, source identity, no-fast-math UNE predicate,
positive-zero bits/type, SSA freshness, removal and replacement uses are derived
from those ASTs. No Python Candidate Boolean or assertion is a proof premise.

`structural` retains complete source-driven intent materialization and exact
`alphaOps` correspondence, modeled-key checks, duplicate-source-ID refusal and
outside-function accounting. Every site is checked, including unused casts.
Unrelated operations, attributes, properties, types, regions and uses must
correspond. Unsupported formats, attributes, multiblock regions, excessive
inventory nesting (budget 64) or absent repairs cause explicit refusal.

The independent executable simulation checks every source/target function,
block and region with an SSA correspondence. `site_step`, mutual `ops_sound` /
`regions_sound` / `region_sound`, and `evalFunc_sound` connect acceptance to the
actual evaluator. They derive lookup, occurrence, freshness and replacement-use
facts internally; the final theorem needs no caller-supplied local-site facts.
Calls preserve symbol lookup, external declarations and evaluation fuel.
`FrontendBoolContext` proves that region contexts preserve the caller's model,
external environment and index width while changing only the loop index.

The old C1 module filter is replaced on this path by this complete direct
source-policy execution proof, which also covers internal calls. Whole-boundary
structural requirements remain. The older materialization and conditional
scalar/site theorems remain available, but are not the final execution claim.
`Candidate.check_sound` is only an initialized-f32 schematic sanity lemma.

## `UneCorrect`: precise premise, now discharged

The general `check_sound` exposes exactly:

```text
forall x : BitVec 32,
  M.cmp uneKey f32 x positive_zero
    = (if x.toNat % 2^31 = 0 then false else true)
```

The key is `arith.cmpf`, predicate 13, fast-math none
(`fm1:0;rm4:none;pred13:attr 13 : i64`). It quantifies initialized bit patterns;
poison follows separately from the frozen eager propagation rules.
`alwaysFalse_not_une` shows the law is false for arbitrary unrestricted models.

The approved direct instance overrides only that key at f32 with the complement
of ordered equality on Lean 4.35's existing IEEE binary32 unpack model. Every
other entry delegates to `base`. `une_zero` and `model_correct` prove the law for
all 2^32 encodings by exponent/significand/sign cases, including every quiet or
signaling NaN payload/sign, infinities, signed zeros, subnormals and fractions.
`check_sound_ieee` applies that proof to the whole program. Dependencies are only
`propext`, `Classical.choice`, `Quot.sound`; no axiom or native decision is added.

The historical `Differential.hostModel` widens with `Float32.toFloat`, an opaque
extern without a logical equation in Lean 4.35. Its exact comparison bridge remains
unproved and is **not a dependency** of the approved direct-instance theorem.
The review packet gives the precise alternative obligation. Native samples do
not prove this bridge or any universal floating law. Floating exception flags,
traps and signaling-NaN exception effects are outside the value/poison model.

## Pipeline and receipts

`FrontendBoolChain.chain_after_source_policy` composes the accepted repair with
proved ordinary suffixes, on declared context/input domains, into:

```text
RefinesC(final, policyDenote(exact repair.before))
```

Its explicit `UneCorrect` premise is discharged by the approved direct instance.
The strict variant additionally requires policy support on that same domain.
A prefix does not transport policy back to the original capture. The composer
refuses multiple repair checkpoints. Ordinary original-source refinement and the
overall ordinary chain verdict remain abstracted across the repair.

`check.py` routes the production boundary to the new checker. Receipts bind the
actual snapshots, round-tripped exports, A2, approved interpretation definitions,
review record, proof sources, axiom audits, checker binary and unchanged lock.
Replay reruns the actual checker and operational qualifiers; forged proof,
approval or qualification metadata is rejected. Composition also checks artifact
continuity, function coverage, typed domains, index widths and handoff bytes.

`FrontendBoolQualified.support_sound` derives source-policy support and strict
refinement on the actual source signature's typed domain. `defined_sound` derives
identical successful nonpoison source-policy/target results on typed nonpoison
inputs. Its additive `FrontendBoolDefined` book proves f32 comparison totality
without any numerical law, reusing existing generic-body and definedness proofs.
The qualifiers use index64 explicitly; the equality theorem has no such restriction.

## Actual execution limits

| Program | Repair equality | Support | Definedness |
| --- | --- | --- | --- |
| Actual full SmolVLA `forward` | Accepted | Unsupported | Unsupported |
| Synthetic tensor generic, two signed/unsigned repairs + integer continuation | Accepted | Proved | Proved |
| Synthetic scalar `mask` callee | Accepted | Proved | Proved |
| Synthetic caller with `func.call` | Accepted; call execution samples pass | Support checker refuses call typing | Unsupported |

Actual SmolVLA's first support refusal is
`tensor.empty: result of unmodelled type`. The export identifies an unchanged
dynamic allocation: before result ID 1033, operand 1029; after result ID 1034,
operand 1030; result `tensor<?xi64>`. `vtyOf` admits static shapes only, and the
frozen `Sem.Ops.Tensor.sem` also returns `stuck "tensor.empty with dynamic sizes"`
for runtime-sized empties. This is a genuine surrounding-model limitation.
Full defined execution requires reviewed dynamic-tensor typing/semantics and
further support qualification; no such frozen change is authorized here.

No changed bool site is removed from the denominator to hide this refusal.
Full-program equality under the frozen evaluator is proved; successful full-model
execution is not claimed. The positive tensor and call fixtures ensure that
execution qualification actually reaches and observes repaired results, including
poison, instead of relying solely on a surrounding stuck outcome.

## Format/cast denominator

The production dispatch is `AnyFloat` input and scalar integer result width 1,
for both `arith.fptosi` and `arith.fptoui`. It is broader than four float classes
in installed xDSL 0.73.0; [the inventory](format-inventory.json) lists all 18.

| Case | This proof |
| --- | --- |
| f32 to i1, either spelling, supported attributes | Whole-program A2 equality, all bit patterns/poison |
| f16, bf16, f64 to i1, either spelling | Production rewrites; actual Lean checker explicitly refuses |
| f80, f128, tf32, eight float8 classes, two float6 classes, float4 | Production `AnyFloat` gate includes them; no proof/MLIR/native qualification here |
| Wider integer result or nonfloat input | Pass leaves it; correspondence still required |
| Nondefault/unsupported source attributes, duplicate IDs, nesting exhaustion | Refused |
| Tensor values | Supported via scalar f32 casts in tensor regions; no direct tensor-cast rule |

The inventory's zero-construction/printing probe is not a pass correctness test:
f80/f128 lack that xDSL implementation, and f8E8M0FNU has no zero encoding (its
`FloatAttr(0.0, ...)` prints a positive value). These are unqualified paths,
reported without changing the compiler or expanding this f32 policy.
f32 verification is not verification of every production format.

## Qualification evidence: tests versus proofs

- Full tracked SmolVLA capture, not a reduced fixture: authoritative compiler
  verifier, before/after export and Lean acceptance, one repaired site. Refreshed
  actual receipt replay returns `[]`; separate support refusal is retained.
- Independent signed/unsigned f32 and two-site fixtures, including an unused
  unsigned cast that must also be repaired.
- Ten compiler-MLIR-valid mutations reach and are rejected by **Lean**: ordered-ne,
  wrong predicate, fast-math, wrong operand, negative zero, wrong zero/type,
  retained cast, bypassed comparison, wrong replacement use and extra definition.
  Duplicate same-scope SSA names fail the compiler verifier; they are not counted
  as compiler-valid Lean negatives.
- Actual source-policy and target evaluator executions: 18 IEEE bit patterns plus
  poison, through two repairs, integer `andi`/`xori` continuations, tensor lifting
  and an internal call. Python's expected results are an independent bit-contract
  transcription. These are samples, separate from universal program theorems.
- The pinned PyTorch oracle asserts **2.10.0+cpu** and actually runs 18 patterns.
  Native repaired code compares **23** patterns against independent bits and that
  pinned oracle, retaining prior samples and adding signed signaling NaNs,
  subnormals, minimum normals and maximum finite values. The original cast is
  inspected as a poison source, never used as a numeric oracle.
- PyTorch **2.14.1+cu130** native results remain supplemental evidence.

Measured versions: CPython **3.12.14**, NumPy **2.5.4** in the pinned oracle;
Lean **4.35.0** (`eb6cafb7290e8bd5f2937522d8d804f58f02bafc`);
Mathlib `56a005855ea17e0b0958f73ec801ae140b0a1679`;
torch-mlir **20261001**, xDSL **0.73.0**, LLVM **23.0.0git**
`a47bddccec30255619bb8c37fa59700e661d4e66`, GCC **16.2.1**.
The compiler interpreter's newer PyTorch is not substituted for the pinned oracle.

### Reproduction

From this checkout, build first:

```sh
export PATH="$HOME/.elan/bin:$PATH"
(cd merlin/lean && lake build MerlinVerify merlin-lean merlin-lean-audit)
export PYTHONPATH="$PWD/src"
export MERLIN_COMPILER_PYTHON=/home/jeremy/Projects/model2MLIR/.venv/bin/python
export MERLIN_TORCH_PYTHON=/tmp/merlin-pv-frontend-pytorch210/bin/python
export MERLIN_LEAN_REQUIRED=1
export MERLIN_CLANG=/usr/bin/gcc
export MERLIN_LLVM_LLC=/home/jeremy/Projects/llvm-project/build/bin/llc
export MERLIN_MLIR_INSTALL=/home/jeremy/Projects/llvm-project/build
/home/jeremy/Projects/merlin/.venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_frontend_bool.py \
  merlin/tests/ir/test_lean_spec_frontend_torch.py \
  merlin/tests/ir/test_lean_chain.py merlin/tests/ir/test_lean_guard.py -q -rs
/home/jeremy/Projects/merlin/.venv/bin/python -m pytest \
  merlin/tests/ir/test_bool_cast_poison.py -q -rs
/home/jeremy/Projects/merlin/.venv/bin/python build_tools/scripts/check_lean_guard.py
```

Replace absolute tool paths on another host. Large generated MLIR/JSON and binaries
are uncommitted; the tracked tests regenerate them. Do not copy old verdicts onto
rebuilt artifacts. [The compact evidence](qualification-20261021.json) binds the
actual run and exact source/tool/artifact hashes; its absolute paths attribute this
host and do not imply laptop availability.

## Final gate results and outstanding obligations

- Full library/checker/audit build passed (1,443 jobs).
- Final frontend/chain/guard/oracle regression: **29 passed, 0 skipped**, 263.81 s
  (frontend 8, chain 13, guard 7, pinned oracle 1), including actual SmolVLA.
- Final pinned native suite: **9 passed, 0 skipped**, 12.45 s, 23 edge patterns.
  Supplemental PyTorch 2.14.1+cu130: **9 passed, 0 skipped**, 13.77 s.
  These are **38 distinct tests**; supplemental/rerun passes are not added to that count.
- Full audit: **7,860 theorems, 0 violations**. The full guard exits 1 for the
  three exact admission differences in the admission request; no frozen-semantic or existing
  pinned-statement mismatch is reported. SPEC_LOCK was not regenerated.
- Ruff checks/format checks and `git diff --check` passed.
- A supplemental trial omitted the compiler-interpreter environment and recorded
  **6 passed, 3 skipped**; it was corrected and rerun to the nine-pass result above.
  A receipt refresh initially reused an existing replay directory; replay was
  rerun in a fresh directory. Neither failed setup is counted as qualification.


| Obligation | Status |
| --- | --- |
| Acceptance to actual whole-program execution against direct A2 | Proved and integrated |
| Source eligibility, SSA/environment/freshness/use facts | Recomputed and proved from actual IR |
| Poison, tensor-region lifting, surrounding operations and internal calls | Proved |
| Universal f32 UNE law under direct IEEE instance | Proved, no floating axiom |
| Human approval of this f32 policy/capture convention/direct model | Recorded from user; scoped exclusions preserved |
| Ordinary suffix composition against exact repair source policy | Proved and integrated |
| Supported synthetic typed support/definedness | Proved and qualified |
| Actual full SmolVLA support/defined execution | Unsupported: unchanged dynamic tensor allocation |
| Historical host-adapter widening bridge | Unproved, unused by final theorem |
| Original numeric-cast equivalence; capture/codegen correctness | Not claimed |
| f16/bf16/f64 or other format proof | Out of scope and unsupported |
| Multiple policy checkpoints / backwards policy transport | Unsupported |
| Lock/schema/new statement admission and implementation review | Awaiting integration-owner review; lock unchanged |

The remaining full-model execution and admission obligations must not be described
as completed by the f32 repair theorem. This closes the bounded frontend proof
work and leaves a precise integration handoff; no W8A8 work was changed here.

## Follow-up: exact remaining runtime obstruction (2026-10-21)

The request to finish remaining work was followed by a request for a plain-English
explanation of the protected-model change. That reply is not recorded as approval.
No frozen model, compiler, W8A8, real guard or SPEC_LOCK changes have been made.

`FrontendBoolRuntimeGaps` now supplies an executable actual-AST classifier and a
proved obstruction to successful execution. `collected_stuck` proves every reported
operation is stuck if stepped. `neverSucceeds_sound` scans unconditional block
steps up to a terminator and proves the actual function cannot return `.ok` for
any context/arguments/results when accepted. This permits earlier UB and does not
assert that every execution reaches the reported operation.

The checker accepts this stronger obstruction for **both complete actual SmolVLA
exports**. Each contains one refused dynamic tensor.empty, two scf.for and two
scf.if operations. The six scf.yield terminators already belong to the evaluator;
they are not six additional dispatch gaps. The full after AST contains 51,781
operations. No changed bool site or surrounding operation was removed to obtain
these results. This explains why full-model successful execution cannot be proved
against the existing frozen evaluator by adding only a typing rule.

[Runtime evidence](runtime-refusals-20261021.json) binds the exact exports and new
proof/fixture source hashes. The new suite has **5 passed, 0 skipped**, 29.49 s:
compiler-valid independent dynamic-empty/if/for cases, a nested-only witness that
must not imply unconditional failure, and both full actual before/after exports.
Theorem dependencies remain only the three standard logical axioms. The primary
frontend checker binary is unchanged, so the preceding 38-test f32 qualification
remains attributable to those same bytes; the five new tests are separate.

The concrete [runtime extension proposal](runtime-extension-review.md) separates
modeling/poison decisions from the already approved f32 bool contract. NaN remains
an initialized value, not poison. The exact unapplied
[admission patch](admission-candidate/admission.patch) and
[review record](admission-candidate/review.json) contain three file-hash admissions
and eight new theorem pins. The candidate passes the authoritative lock comparison;
that result is not permission to apply it or human admission. Existing frozen
statement hashes are unchanged. The generator refuses unrelated drift.

Follow-up full-library/checker build passed. The full guard audited **7,943
theorems with zero violations**, then failed for the same three admission
differences. No frozen semantic or existing statement drift was reported.


## Additive runtime integration follow-up (2026-10-21)

The authorized candidate is implemented in commit `0843cf2eb`. Its complete review packet is
[`runtime-candidate-review.md`](runtime-candidate-review.md), with exact statements, source/
artifact bindings and qualification evidence under [`runtime-candidate/`](runtime-candidate/README.md).
It proves acceptance-to-successful-execution for the actual 46-operation mask-selection cut
and retains whole-program repair-policy equality under the extended interpretation. The proof
includes actual count-provider execution, dynamic allocation, both loops, reshapes and returns.
It is not a full SmolVLA progress theorem. Runtime review/admission, cut-domain establishment
within the full model, and other non-exhaustively inventoried obligations remain separate.

The fresh full build passed (1457 jobs); all 62 relevant tests passed, zero skipped. Audit:
9005 theorems, zero violations. Native comparisons and pinned PyTorch 2.10.0+cpu tests remain
execution evidence, not proofs of universal float laws. Protected semantics, A2 and SPEC_LOCK
are unchanged; guard admission still has its same three pre-existing entries. The unrelated
baseline structure ceiling failure is recorded in the candidate packet. No f16/bf16/f64 proof
or active interpretation/receipt change is included.
