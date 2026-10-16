/-
UNTRUSTED script synthesis for C1. A certificate is a hint (design D2): this module builds one from
the two programs by greedy alignment, and `c1Check` re-checks every step. A wrong or incomplete
synthesis can only make the boundary `abstracted`, never `verified`.

Current alignment tries alpha-equivalent prefixes, closed-expression CSE and adjacent motion,
proved operation expansions, and DCE.
Rules that need certificates from the pass (rewrites that create ops)
add their own synthesis or read the pass's certificate.
-/
import MerlinVerify.Rules.Dce
import MerlinVerify.Rules.Cse
import MerlinVerify.Rules.Motion
import MerlinVerify.Rules.Generalize
import MerlinVerify.Rules.Commutative

namespace MerlinVerify.Check

open MerlinVerify

/-- Prefer the unchanged prefix when both next operations still match. This prevents a valid
identity boundary with repeated constants from being mistaken for a CSE boundary. -/
def samePrefix? (ρ : Rho) : List Op → List Op → Option Rho
  | s :: s' :: _, t :: t' :: _ => do
    let ρ' ← alphaOp ρ s t
    let _ ← alphaOp ρ' s' t'
    return ρ'
  | _, _ => none

/-- The synthesizer proposes a single-op rule; the C1 checker independently rechecks it. -/
def localRewrite? (ρ : Rho) (s t : Op) : Option (Rule × Rho) :=
  [Rules.Generalize.rule, Rules.Commutative.rule].findSome? fun r =>
    (r.check ρ [s] [t]).map fun ρ' => (r, ρ')

def synthLocal (ρ : Rho) : List Op → List Op → List Step
  | [], [] => []
  | _ :: ss, [] => .site Rules.Dce.rule 1 0 :: synthLocal ρ ss []
  | [], _ :: _ => [.same]
  | s :: ss, t :: ts =>
    match samePrefix? ρ (s :: ss) (t :: ts) with
    | some ρ' => .same :: synthLocal ρ' ss ts
    | none =>
      match Rules.Cse.check ρ ((s :: ss).take 2) [t] with
      | some ρ' => .site Rules.Cse.rule 2 1 :: synthLocal ρ' (ss.drop 1) ts
      | none =>
        match Rules.Motion.check ρ ((s :: ss).take 2) ((t :: ts).take 2) with
        | some ρ' => .site Rules.Motion.rule 2 2 :: synthLocal ρ' (ss.drop 1) (ts.drop 1)
        | none =>
          match localRewrite? ρ s t with
          | some (r, ρ') => .site r 1 1 :: synthLocal ρ' ss ts
          | none =>
            match alphaOp ρ s t with
            | some ρ' => .same :: synthLocal ρ' ss ts
            | none => .site Rules.Dce.rule 1 0 :: synthLocal ρ ss (t :: ts)
  termination_by ss ts => ss.length + ts.length
  decreasing_by
    all_goals simp_all
    all_goals omega

/-- Build a C1 input with a synthesized local-rewrite script. -/
def synthC1 (src tgt : Module) (fn : String) : C1Input :=
  let script := match src.func? fn, tgt.func? fn with
    | some fS, some fT =>
      match fS.body, fT.body with
      | .mk [.mk argsS opsS], .mk [.mk argsT _] =>
        let ρ0 := (argsS.map (·.id)).zip (argsT.map (·.id))
        match tgt.func? fn with
        | some f => match f.body with
          | .mk [.mk _ opsT] => synthLocal ρ0 opsS opsT
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
