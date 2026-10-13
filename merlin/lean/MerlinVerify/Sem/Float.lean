/-
G0-FROZEN. Floating point is a PARAMETER (design D5).

Every float operation is interpreted by a `FloatModel` with no laws. Theorems quantify over all
models, so a claim proved here holds for IEEE-754 on the target in particular, and a rewrite that
reorders, reassociates or constant-folds float arithmetic cannot be proved — as intended.

Operations are identified by their MLIR op name plus the op's attributes that affect numerics
(fast-math flags, rounding mode, comparison predicate), passed as `FloatOpKey`. Two ops with different
keys are different uninterpreted functions, so a rewrite that changes `fastmath` is not provable.

If a rule genuinely needs an IEEE fact, it takes a `FloatLaws`-style hypothesis declared in
`Assumptions/` and listed in the ledger; there are none at G0.
-/
import MerlinVerify.Core.AST

namespace MerlinVerify

/-- Bit pattern of a value of float format `k`. -/
abbrev FBits (k : FKind) := BitVec k.width

/-- What identifies an uninterpreted float function: the op name and every numerics-relevant
attribute, printed canonically by the semantics (e.g. `("arith.addf", "fastmath=none")`). -/
structure FloatOpKey where
  op : String
  modifiers : String
  deriving DecidableEq, Repr, Hashable

structure FloatModel where
  /-- Same-format operations: `arith.addf/subf/mulf/divf/remf/maximumf/minimumf/...`,
  `math.*` (exp, tanh, erf, rsqrt, sqrt, roundeven, absf, ...), `arith.negf`, `arith.select`
  is NOT here (it is not a float operation). Arity is the length of the argument list. -/
  op : FloatOpKey → (k : FKind) → List (FBits k) → FBits k
  /-- Float comparison (`arith.cmpf`); the predicate is part of the key. -/
  cmp : FloatOpKey → (k : FKind) → FBits k → FBits k → Bool
  /-- Format conversion (`arith.extf`, `arith.truncf`). -/
  convert : FloatOpKey → (src dst : FKind) → FBits src → FBits dst
  /-- Float → integer (`arith.fptosi`, `arith.fptoui`). `none` is poison (out of range / NaN). -/
  toInt : FloatOpKey → (k : FKind) → (w : Nat) → FBits k → Option (BitVec w)
  /-- Integer → float (`arith.sitofp`, `arith.uitofp`). -/
  ofInt : FloatOpKey → (w : Nat) → (k : FKind) → BitVec w → FBits k

end MerlinVerify
