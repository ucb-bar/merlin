/-
G0-FROZEN. Claim types, verdicts and the shape of every checker (design D7, §5).

A checker is a Lean function plus a proof that `verified` implies its claim. A certificate is
an untrusted hint inside `Input`: a wrong certificate can only make `check` return `abstracted`.
-/
import MerlinVerify.Sem.Program

namespace MerlinVerify

/-- What a verified boundary means (design D7). The paper's wording is generated from this. -/
inductive ClaimType where
  | equiv
  | refine
  /-- Equivalence modulo an argument/result map (design §4.6). -/
  | equivArgs
  /-- `after ⊑ P(before)` for the named, reviewed policy spec transformer `P`. -/
  | policy (name : String)
  | erasure
  | partition
  /-- Refinement under an environment where new external symbols meet their contracts (D8). -/
  | contractSeam
  | safety
  deriving DecidableEq, Repr

/-- Checker verdicts, mapped onto `xdsl_dialects.lowering.passes.VERDICTS` (design D12):
`verified`, `refuted` (only with an executed counterexample), `abstracted` (rejected or
unsupported, with a reason). `unmeasured` is a Python-side state: the checker never ran. -/
inductive Verdict where
  | verified
  | refuted (witness : String)
  | abstracted (reason : String)
  deriving DecidableEq, Repr

def Verdict.isVerified : Verdict → Bool
  | .verified => true
  | _ => false

structure Checker (Input : Type) where
  name : String
  claimType : ClaimType
  /-- The meaning of `verified` for this input. Statements of `Claim` are pinned (design §8.6). -/
  Claim : Input → Prop
  check : Input → Verdict
  sound : ∀ i, check i = .verified → Claim i

end MerlinVerify
