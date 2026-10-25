# Scalar-square whole-program composition and separate claim forms

Operational typing for the actual TinyLlama capture is now discharged separately in
[`../typing/review/README.md`](../typing/review/README.md). This structural composition report
continues to leave numerical laws, source definedness, and multiplication-policy approval pending.


The new Lean theorem `ScalarSquareClaims.source_equal` composes the local numerical rule
through SSA environments, nested regions and function calls to compare the **original power
source** with the checked target. It is conditional on an undischarged numerical law and
explicit source-side operational typing obligations. Neither actual captured program has
an instantiated source-preservation claim. The native counterexamples remain valid.

A different theorem, `approved_multiplication_policy`, concerns an explicitly approved
multiplication reference. It does not assert original-source equivalence and does not use
the power/multiplication numerical law. No policy approval has been granted or synthesized.

This extends the [earlier boundary packet](../review/README.md). That packet and all native
counterexamples are retained unchanged as historical evidence. Its stated composition gap
is addressed by the conditional theorems described here; its numerical finding is unchanged.

## Conditional original-source equality

In `merlin/lean/MerlinVerify/Rules/ScalarSquareClaims.lean`, the theorem has this form:

```text
ScalarSquareProgram.check source target = true
Numerical F
SourceTyping source F indexBits
F ctx
Inputs source indexBits entry args
--------------------------------------------------------
denote opSem ctx target entry args
  = denote opSem ctx source entry args
```

The right-hand side is the original source, not the multiplication reference. `F` is the
explicit admissible-context predicate. It may bind a chosen floating model and state
typed/defined external-call contracts. It need not admit every possible external environment.
Nested operations must use callbacks only in `F` and on their typed argument domains.

`Numerical F` requires `ScalarSquareContract.Law ctx.M site.kind site.attrs` for every
`ctx` satisfying `F` and every eligible `Site`. Thus it is a family-wide law for the admitted
f32/f64 and absent/none/contract cases, stronger than just the finite list of captured sites.
It is not a law over every unconstrained `FloatModel`. `numerical_nonempty` provides a
constant-zero model witness to satisfiability; this witness is not an IEEE/native model.
For the tested native implementation the exact numerical law is refuted, including for
ordinary finite inputs. It stays a theorem argument and is never installed as an axiom.

`SourceTyping` is a source-only obligation. It contains no whole-program source/target
equivalence premise. Its components are:

| Obligation | Meaning |
| --- | --- |
| `Arguments` / `HasType` | Runtime scalar/tensor arguments have the declared supported types at the selected index width |
| `Invariant` / `Good` | Every newest lexical SSA binding has its runtime type and the exact literal-two producer fact, when applicable |
| `ResultFacts` | Whenever an original source step and result binding succeed, its results have those types and producer facts |
| `PrimitiveTyping` | An unchanged source operation on the current operands cannot distinguish callbacks that agree on the declared typed region/call domains |
| `ReadyOps`, `ReadyRegions`, `ReadyBlocks` | These local obligations hold along successful source prefixes and at every typed nested-region invocation |
| `SourceTyping` | Function-body obligations hold for each original function and every call fuel, using the original source call semantics |

`PrimitiveTyping` is an explicit semantic typing obligation, not an executable success flag.
The new module proves it for regionless operations without callees and for a direct
`func.call` whose actual arguments satisfy the callee domain. It does not automatically
discharge the obligation for every structured operation in either captured model. The
checker’s type inventory is diagnostic, not a proof of `SourceTyping` or definedness.

## How composition works

`ScalarSquareSSA.lean` proves that binding block arguments and operation results preserves
the runtime invariant. Bindings shadow both types and producer facts. A selected rewrite
therefore reads the actual base and exact positive-two exponent from the actual runtime
environment. The local operation theorem applies to all typed base bits, including NaNs,
and handles poison separately. The source and target bind equal result lists to the same
SSA results before continuing.

`ScalarSquareComposition.lean` proves mutual composition for operation lists, region
closures and blocks. The structured-operation premise requires agreement only on typed
region arguments in admissible contexts. It does not assume that a region with f32 block
arguments must agree on arbitrary malformed values. Captured outer values remain in scope;
block arguments shadow them and facts do not leak between siblings.

`ScalarSquareProgram.lean` independently checks the function inventory and the normalized
body of each function, in addition to the original complete-output/site checker. Function
names, visibility, declaration status and order are checked, and lookup correspondence
and equal function counts are proved. Function bodies start from empty environments.

`calls_equal` inducts on the evaluator’s call fuel. Typed calls use the induction hypothesis;
external declarations use the same `ctx.ext` interpretation on both sides. Unknown symbols
and exhausted call fuel produce equal error outcomes, not successful claims. The final
denotation uses the shared evaluator’s `number of functions + 1` fuel on both programs.
Recursive programs are covered only by this bounded evaluator; a strict claim still needs
definedness at the chosen entry/domain.

All added proofs use only `propext`, `Classical.choice` and `Quot.sound`. The retained
proof-audit report includes elaborated statements and audits the new driver’s import
closure: 1,542 project theorems audited, zero violations, with 18 elaborated statements retained.
No frozen semantics, theorem pins, SPEC_LOCK, shared Main/CLI/lakefile or shared
checker executable was changed.

## Claim A: strict source preservation

`StrictSourcePreservation` and `strict_source_preservation` additionally require:

1. `ContextDomain F indexBits`: at least one admissible context exists and every admissible
   context has the chosen index width. Choosing `F = False` cannot establish this claim.
2. `EntryDomain source indexBits entry`: the entry is a defined function and the typed input
   domain is inhabited. Unsupported argument types cannot manufacture an empty-domain claim.
3. `FuncNotStuck ... source entry` on that exact context/input domain.

Together with the checker, numerical law and source typing, these prove strict refinement
against the **original source** under the existing `Refines ResultsRefine` definition.
That definition excludes `stuck` but allows source UB to be refined by anything. It also
allows poison values in an `ok` result. This must not be described as absence of UB/poison.

The separate `source_success` theorem takes an actual source outcome `.ok results` and
proves the target returns the identical `.ok results`. This excludes a stuck/UB outcome
for that invocation; output poison would require a further poison-free premise if a paper
claim intends to exclude it.

**Current status:** numerical law undischarged/refuted natively; source typing, context/input
domain witnesses and source definedness not discharged. This claim is not instantiated.

## Claim B: approved multiplication policy

`policyMeaning ctx source entry args` is explicitly:

```text
denote opSem ctx (ScalarSquare.reference source) entry args
```

It retains dynamic/other powers and changes only the checked literal-two sites. This is a
different numerical specification. `approved_multiplication_policy` requires a separate
`PolicyDecision` and `Authorizes` proof, inhabited context and entry/input domains, and
`NotStuck` of this policy meaning. The approval record binds the exact original source,
exact multiplication reference, context predicate (including numerical/external contracts),
index width and entry, and records a nonempty reviewer identity and affirmative decision.

`Authorizes` checks consistency of review data. Lean does not authenticate human identity or
establish the external authority of a reviewer. Actual approval must come from the review
process. The evidence runner writes only `policy-review.pending.json` records, with
`approved=false`, no reviewer, no selected model/context interpretation and undischarged
definedness. There is no command-line approval switch. `unapproved_rejected` proves that a
false decision cannot satisfy `Authorizes`.

This policy theorem follows complete target/reference equality through the full shared
program semantics, including calls and nested regions. It never invokes `source_equal` and
never concludes equality to the original power source. It needs no pow-square numerical
law and no `SourceTyping` premise about the original power execution. Policy definedness
remains a separate obligation. `policy_success` provides the corresponding `.ok results`
form, with an explicit authorization premise and no numerical-law premise.

**Current status:** approval, context/input-domain witnesses and policy definedness are
pending. This claim is not instantiated. Human approval alone would not supply a missing
definedness proof.

## Actual captures and retained evidence

| Capture | Sites | Function/body correspondence | Unsupported runtime value types |
| --- | ---: | --- | ---: |
| SmolVLA | 66 | checked | 5 |
| TinyLlama | 45 | checked | 0 |

Both actual captures have one `forward` function, with evaluator call fuel 2. SmolVLA’s
unsupported values are IDs 1033, 1034, 1037, 1040 and 1043, all `tensor<?xi64>`. Both entries’
argument types are supported by the current type inventory. Zero unsupported value types
on TinyLlama is not a proof of operation support, operational typing or successful execution.

Separate f32/f64 regressions use three functions, a two-call chain and a `linalg.generic`
region, with rewrites both in the callee and inside the caller’s nested region. Mutated
callees, call operands, function visibility, added/dropped functions and forged approval/
definedness reports are rejected. Kernel-checked examples cover lexical producer shadowing,
wrong runtime argument types, false approvals, empty context domains and the distinction
between equality of stuck outcomes and strict refinement.

Final qualification: nine new tests passed, and four slow replay tests passed (the two
composition receipts and two original boundary receipts). Source-layout tests: 18 passed.
Ruff, artifact layout, generated-document freshness and frozen-file lock checks passed.
The structure gate retains its earlier unrelated `zephyr_model.py` module-size failure.

[Run summary](summary.json) binds the receipts and pending policy-review records.
[SmolVLA receipt](SY_model_smolvla/receipt.json) and
[TinyLlama receipt](SY_model_tiny_llama/receipt.json) bind source/import bytes, compiled
modules, original receipts, reports and proof audits. Replay repeats the original boundary
replay, reruns the program checker, re-audits the proof environment/statements, and checks
the claim metadata. Rehashing a forged green/approved report does not make it replay.

[Counterexample preservation record](counterexamples-preserved.json) binds all 44 prior
native evidence files, together with the prior summary and manifest, to their earlier
recorded digests. Their contents are checked both before and after regeneration. No prior
counterexample was removed, weakened, regenerated or reclassified as a proved numerical law.

## Reproduction and review

From the repository root:

```sh
.venv/bin/python build_tools/scripts/verify_scalar_square_composition.py \
  --compiler-python /home/jeremy/Projects/model2MLIR/.venv/bin/python
```

This regenerates only composition receipts and qualification results. It reads the existing
scalar-square captures/receipts and native evidence, without rewriting them. To replay one
composition receipt, use a new output directory:

```sh
.venv/bin/python -m merlin.verify.lean.scalar_square_composition replay \
  out/artifacts/verification/merlin-passes/frontend/scalar-square/composition/SY_model_smolvla/receipt.json \
  out/artifacts/verification/merlin-passes/frontend/scalar-square/composition/review-replay-001 \
  --compiler-python /home/jeremy/Projects/model2MLIR/.venv/bin/python
```

Review the context/input domains, primitive callback typing and result-fact obligations,
definedness strength (NotStuck versus successful/poison-free execution), and the exact
separation of the two claims. Do not approve the refuted native exact-bit law. Review any
multiplication policy as an explicit change of numerical specification. No approval,
theorem pin or lock update is implied by this packet.
