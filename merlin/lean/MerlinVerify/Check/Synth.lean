/-
UNTRUSTED script synthesis for C1. A certificate is a hint (design D2): this module builds one from
the two programs by greedy alignment, and `c1Check` re-checks every step. A wrong or incomplete
synthesis can only make the boundary `abstracted`, never `verified`.

Current alignment: ops that are alpha-equivalent are `same`; a source op with no counterpart is
offered to the DCE rule. Rules that need certificates from the pass (rewrites that create ops)
add their own synthesis or read the pass's certificate.
-/
import MerlinVerify.Rules.Dce

namespace MerlinVerify.Check

open MerlinVerify

def synthDce (ρ : Rho) : List Op → List Op → List Step
  | [], [] => []
  | _ :: ss, [] => .site Rules.Dce.rule 1 0 :: synthDce ρ ss []
  | [], _ :: _ => [.same]
  | s :: ss, t :: ts =>
    match alphaOp ρ s t with
    | some ρ' => .same :: synthDce ρ' ss ts
    | none => .site Rules.Dce.rule 1 0 :: synthDce ρ ss (t :: ts)

/-- Build the C1 input for function `fn` with a synthesized DCE/identity script. -/
def synthC1 (src tgt : Module) (fn : String) : C1Input :=
  let script := match src.func? fn, tgt.func? fn with
    | some fS, some fT =>
      match fS.body, fT.body with
      | .mk [.mk argsS opsS], .mk [.mk argsT _] =>
        let ρ0 := (argsS.map (·.id)).zip (argsT.map (·.id))
        match tgt.func? fn with
        | some f => match f.body with
          | .mk [.mk _ opsT] => synthDce ρ0 opsS opsT
          | _ => []
        | none => []
      | _, _ => []
    | _, _ => []
  { src, tgt, fn, script }

end MerlinVerify.Check
