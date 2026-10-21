# Remaining runtime-model and admission decision

Status: **proposal, not approved or activated**. The documented f32 bool policy
and direct IEEE comparison instance already have user approval; this is a new,
separate decision. Compiler pass and W8A8 remain outside the edit scope.

## Why a new decision is required

The existing user instruction forbids changing frozen semantics or SPEC_LOCK.
`merlin/lean/AGENT.md` requires an interface-change review for the frozen model;
plan RULES reserve SPEC_LOCK admission to the integration owner. The latest request
to finish remaining work does not explicitly revoke those constraints.

The f32 repair theorem is complete. Full successful SmolVLA execution is a
broader model obligation: the actual after export has 51,781 operations, including
one dynamic `tensor.empty`, two `scf.for` and two `scf.if`. Frozen `tensor.empty`
rejects dynamic sizes, and the dispatcher has no SCF family. Six `scf.yield`
terminators are already recognized by the generic evaluator and are not counted
as six additional dispatch gaps.

`Rules/FrontendBoolRuntimeGaps.lean` recomputes witnesses from decoded IR and proves
that each classified operation is stuck if stepped. Its unconditional block scan
also proves that a classified function can never return `.ok`, regardless of the
context or input values; earlier UB is permitted. This is a semantic obstruction,
not a Python shape guard or only a missing typing rule. The actual-boundary
qualification records which of these stronger claims the checker accepts.

## Decision A: admit the completed f32 work

The exact, unapplied patch is `admission-candidate/admission.patch`.
`prepare_admission.py` extracted elaborated statements from the actual Lean
modules, generated a candidate lock and checked it with the authoritative guard.
It changes exactly three file hashes and adds eight statement pins; existing
statement hashes, frozen model files and the real SPEC_LOCK/guard remain unchanged.
`admission-candidate/review.json` binds the base and candidate bytes.

Applying that patch would admit the existing A2 file bytes and receipt/chain
schemas, and pin the completed frontend theorems. This is not approval of A2's
other bool-to-f32/linear contracts, which remain unused by this f32 claim.
A passing candidate comparison is mechanical consistency, not human admission.

## Decision B: implement a separately reviewed runtime extension

Proposed smallest vocabulary needed by this boundary:

1. **Runtime-shaped `tensor.empty`.** Resolve every static result dimension from
   the type and each dynamic dimension from one runtime index operand, in order.
   Require the operand count and index width to match. For initialized nonnegative
   sizes, return a functional tensor of the resolved shape with poison contents,
   preserving ledger A7. Malformed forms and negative dimensions remain explicitly
   unsupported. No allocation contents are assumed initialized.
2. **`scf.if`.** For an initialized scalar i1, execute exactly the selected
   single-block branch with the same context and captured environment. Return its
   yielded values; preserve `.ub`/`.stuck`. Both branches are checked, even if one
   is unreachable for a particular sample. All result counts/types must correspond.
3. **`scf.for`.** Initially support the signed-index, positive-step form used here,
   one single-block body and explicit loop-carried values. Compute iteration count
   in mathematical integers; execute IVs `lb + j*step` in order, pass IV followed
   by current carried values to the region and return its final yield. Check the
   index width and representability of loop controls/advancement. Unknown
   attributes, unsignedCmp, malformed regions and unsupported control cases refuse.
   Do not model overflow as accidental host/Nat wraparound.
4. **Dynamic intermediate typing.** Use an additive shape-pattern relation and
   progress/definedness checker that bridges existing static `VTy` facts. Keep
   frozen `Check/Typing.lean` unchanged if that interface suffices. Prove actual
   region capture, branch/loop-carried typing, scalar-index width, bounds and
   initialization facts. Reuse existing static-op rules only through proved
   conversions. Never turn dynamic shapes into arbitrary unchecked static shapes.

### Poison/control decision needing explicit review

The existing model has no dynamic-allocation or SCF poison convention. The
proposed rule treats a consumed poison branch/loop control or shape extent as UB,
while ordinary poison data remains per element. This follows the existing style
for poison-consuming assertions/indices, but is a new modeling choice, not a
proved comparison law or a fact inferred from native tests. If this choice is not
approved, these cases must stay unsupported and the universal typed-poison input
support claim cannot simply be made green.

The first implementation must conservatively reject unsupported control/extent
forms and prove their absence or discharge their preconditions on the actual
boundary. It must not silently narrow the declared input domain. Successful
nonpoison whole-model execution may also need additional numeric/bounds reasoning;
this proposal does not promise defined outputs for arbitrary NaN weights or an
arbitrary FloatModel's numeric conversions.

### Exact prospective edit scope

- Frozen `MerlinVerify/Sem/Ops/Tensor.lean`: runtime-shaped empty handling.
- New frozen-family `MerlinVerify/Sem/Ops/Scf.lean` plus import/dispatch in frozen
  `MerlinVerify/Sem/Ops.lean`.
- Additive frontend/runtime typing and proof modules, existing frontend region
  locality proof, actual-boundary qualification, receipts and reports.
- Guard statement pins and SPEC_LOCK **only after** the completed extension and
  its qualification are reviewable. The current admission candidate would be
  regenerated with the exact additional reviewed hashes; it is not a blanket
  authorization to refresh unrelated lock drift.

Keep Core AST/runtime values, FloatModel, A2 primitives, compiler passes and W8A8
unchanged. Add no axiom or floating algebra law. Preserve original-source versus
repair-policy claims and ordinary-chain refusal. If further model gaps arise,
identify them before broadening this vocabulary.

## Source basis and qualification required

The pinned LLVM checkout is `a47bddccec30255619bb8c37fa59700e661d4e66`.
Its `mlir/include/mlir/Dialect/SCF/IR/SCFOps.td` specifies signed default bounds,
positive step, loop-carried arguments/results and selected if branches. Its tensor
definition supplies dynamic extents as index operands. The current online
[SCF reference](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)
additionally spells out trip count and an advancement no-overflow condition; it
must not be substituted for an unchecked claim about the pinned compiler version.
The [tensor reference](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-tensoremptyop)
describes runtime dimensions and unspecified contents. These are specification
sources, not formal proof of native compiler fidelity.

Before activation: prove static conservativity, model all new contexts explicitly,
reprove frontend program simulation/locality, qualify compiler-valid independent
SCF/dynamic examples and actual SmolVLA, and run guard/audit and affected shared
regressions. Native samples must use defined controls and avoid treating UB or
uninitialized data as numerical oracles. No test failure or unknown case becomes
verified by changing labels.
