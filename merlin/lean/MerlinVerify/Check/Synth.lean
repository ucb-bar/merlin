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

/-! ### Diagnostics (reports only; no proof depends on them) -/

/-- Why `runScript` refuses, at the first failing step (`none` when it accepts). -/
def explainScript : Nat → Rho → List Step → List Op → List Op → Option String
  | _, _, [], [], [] => none
  | k, ρ, .same :: st, s :: ss, t :: ts =>
    match alphaOp ρ s t with
    | some ρ' =>
      if s.isTerminator then
        if ss.isEmpty && ts.isEmpty && st.isEmpty then none
        else some s!"step {k}: ops or steps after the terminator"
      else explainScript (k + 1) ρ' st ss ts
    | none => some s!"step {k}: source {s.name} and target {t.name} differ (operands, attributes, types or regions)"
  | k, ρ, .site R nS nT :: st, ss, ts =>
    if (ss.take nS).length = nS ∧ (ts.take nT).length = nT then
      match R.check ρ (ss.take nS) (ts.take nT) with
      | some ρ' => explainScript (k + 1) ρ' st (ss.drop nS) (ts.drop nT)
      | none =>
        let names := (ss.take nS).map (·.name)
        some s!"step {k}: rule {R.name} refuses {names} (a used result, or an op that is not effect-free)"
    else some s!"step {k}: rule {R.name} needs more ops than remain"
  | k, _, [], _, _ => some s!"step {k}: ops remain after the script ends"
  | k, _, _, _, _ => some s!"step {k}: the script continues past the ops"

/-- Why `c1Check` refuses `i` (`none` when it accepts). -/
def explainC1 (i : C1Input) : Option String :=
  match i.src.func? i.fn, i.tgt.func? i.fn with
  | some fS, some fT =>
    match fS.body, fT.body with
    | .mk [.mk argsS opsS], .mk [.mk argsT opsT] =>
      if fS.isDecl || fT.isDecl then some s!"@{i.fn} is a declaration"
      else if argsS.map (·.ty) != argsT.map (·.ty) then some "argument types differ"
      else if !freshIds [] (argsT.map (·.id)) then some "target arguments reuse ids"
      else if !(calleesOps opsS ++ calleesOps opsT).all (fun c => isDeclIn i.src c && isDeclIn i.tgt c) then
        some "a call to a function that is not a declaration in both modules"
      else explainScript 0 ((argsS.map (·.id)).zip (argsT.map (·.id))) i.script opsS opsT
    | _, _ => some "a body outside the single-block fragment"
  | _, _ => some s!"no function @{i.fn} in both modules"

end MerlinVerify.Check
