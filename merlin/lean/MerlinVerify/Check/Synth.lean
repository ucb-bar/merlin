/-
UNTRUSTED script synthesis for C1. A certificate is a hint (design D2): this module builds one from
the two programs by greedy alignment, and `c1Check` re-checks every step. A wrong or incomplete
synthesis can only make the boundary `abstracted`, never `verified`.

Current alignment tries alpha-equivalent prefixes, closed-expression CSE and adjacent motion,
proved operation expansions, dequantization sites, and DCE; `synthC1Live` first erases the ops
that are dead in the source.
Rules that need certificates from the pass (rewrites that create ops)
add their own synthesis or read the pass's certificate.
-/
import MerlinVerify.Rules.DequantSite
import MerlinVerify.Rules.Dce
import MerlinVerify.Rules.Cse
import MerlinVerify.Rules.Motion
import MerlinVerify.Rules.MotionIndependent
import MerlinVerify.Rules.Generalize
import MerlinVerify.Rules.Commutative
import MerlinVerify.Rules.IntegerFold
import MerlinVerify.Rules.ReshapeCompose
import MerlinVerify.Rules.IntegerIdentity
import MerlinVerify.Rules.ReshapeMixed

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

/-- Keep an unchanged first operation when the second pair already has a checked local rewrite.
This preserves its SSA mapping before considering larger sites that erase intermediate values. -/
def rewritePrefix? (ρ : Rho) : List Op → List Op → Option Rho
  | s :: s' :: _, t :: t' :: _ => do
    let ρ' ← alphaOp ρ s t
    let _ ← localRewrite? ρ' s' t'
    return ρ'
  | _, _ => none

/-- A candidate always consumes a nonempty source fragment; targets may contain zero or more
operations. Rules recheck every proposed fragment independently of this synthesizer. -/
structure SiteCandidate where
  rule : Rule
  sourceTail : Nat
  targetCount : Nat

def multiRewrite? (ρ : Rho) (ss ts : List Op) : Option (SiteCandidate × Rho) :=
  -- Preserve intermediate correspondences before trying a fold that maps only the last result.
  let candidates : List SiteCandidate :=
    [⟨Rules.Cse.rule, 1, 1⟩, ⟨Rules.Motion.rule, 1, 2⟩, ⟨Rules.MotionIndependent.rule, 1, 2⟩, ⟨Rules.DequantSite.rule, 0, 2⟩,
      ⟨Rules.ReshapeCompose.rule, 1, 1⟩, ⟨Rules.ReshapeMixed.rule, 1, 1⟩,
      ⟨Rules.IntegerIdentity.rule, 1, 1⟩, ⟨Rules.IntegerIdentity.rule, 1, 0⟩] ++
    (Rules.IntegerFold.candidateTails ss ts).map (fun n => ⟨Rules.IntegerFold.rule, n, 1⟩)
  candidates.findSome? fun c =>
    if c.sourceTail + 1 ≤ ss.length ∧ c.targetCount ≤ ts.length then
      (c.rule.check ρ (ss.take (c.sourceTail + 1)) (ts.take c.targetCount)).map fun ρ' => (c, ρ')
    else none

/-- Greedy alignment. A source op for which `drop` holds is erased before any alignment is tried. -/
def synthLocalWith (drop : Op → Bool) (ρ : Rho) : List Op → List Op → List Step
  | [], [] => []
  | s :: ss, [] =>
    if drop s then .site Rules.Dce.rule 1 0 :: synthLocalWith drop ρ ss [] else
    match multiRewrite? ρ (s :: ss) [] with
    | some (c, ρ') => .site c.rule (c.sourceTail + 1) c.targetCount ::
        synthLocalWith drop ρ' ((s :: ss).drop (c.sourceTail + 1)) ([].drop c.targetCount)
    | none => .site Rules.Dce.rule 1 0 :: synthLocalWith drop ρ ss []
  | [], _ :: _ => [.same]
  | s :: ss, t :: ts =>
    if drop s then .site Rules.Dce.rule 1 0 :: synthLocalWith drop ρ ss (t :: ts) else
    match (samePrefix? ρ (s :: ss) (t :: ts)).orElse
        (fun _ => rewritePrefix? ρ (s :: ss) (t :: ts)) with
    | some ρ' => .same :: synthLocalWith drop ρ' ss ts
    | none =>
      match multiRewrite? ρ (s :: ss) (t :: ts) with
      | some (c, ρ') => .site c.rule (c.sourceTail + 1) c.targetCount ::
          synthLocalWith drop ρ' ((s :: ss).drop (c.sourceTail + 1)) ((t :: ts).drop c.targetCount)
      | none =>
        match localRewrite? ρ s t with
        | some (r, ρ') => .site r 1 1 :: synthLocalWith drop ρ' ss ts
        | none =>
          match alphaOp ρ s t with
          | some ρ' => .same :: synthLocalWith drop ρ' ss ts
          | none => .site Rules.Dce.rule 1 0 :: synthLocalWith drop ρ ss (t :: ts)
  termination_by ss ts => ss.length + ts.length
  decreasing_by
    all_goals simp_all
    all_goals omega

def synthLocal (ρ : Rho) : List Op → List Op → List Step := synthLocalWith (fun _ => false) ρ

mutual
  /-- Every id an op reads, including from its regions (an over-approximation: region-local ids too). -/
  def usesOp : Op → List ValueId
    | .mk _ _ operands _ _ regions _ => operands ++ usesRegions regions

  def usesRegions : List Region → List ValueId
    | [] => []
    | (.mk bs) :: rs => usesBlocks bs ++ usesRegions rs

  def usesBlocks : List Block → List ValueId
    | [] => []
    | (.mk _ ops) :: bs => usesOps ops ++ usesBlocks bs

  def usesOps : List Op → List ValueId
    | [] => []
    | op :: ops => usesOp op ++ usesOps ops
end

/-- Ids that reach a terminator or an op the DCE rule may not erase, computed backwards. -/
def liveIds (ops : List Op) : List ValueId :=
  ops.foldr (fun op live =>
    if op.isTerminator || !Rules.Dce.erasable op || op.results.any (fun r => live.contains r.id) then
      usesOp op ++ live
    else live) []

/-- Erasable ops none of whose results is live: what a dead-code pass removes. Aligning first
without them keeps a dead op from being matched with an identical live one later in the block. -/
def deadIn (ops : List Op) : Op → Bool :=
  let live := liveIds ops
  fun op => !op.isTerminator && Rules.Dce.erasable op && op.results.all (fun r => !live.contains r.id)

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

/-- Erasable ops nothing in `ops` reads: what upstream `cse` erases as it reaches them (not their
now-unused producers, which it keeps). -/
def unusedIn (ops : List Op) : Op → Bool :=
  let used := usesOps ops
  fun op => !op.isTerminator && Rules.Dce.erasable op && op.results.all (fun r => !used.contains r.id)

/-- `synthC1` with every source-dead op erased first (an alternative script; still rechecked). -/
def synthC1Live (src tgt : Module) (fn : String) : C1Input :=
  let script := match src.func? fn, tgt.func? fn with
    | some fS, some fT =>
      match fS.body, fT.body with
      | .mk [.mk argsS opsS], .mk [.mk argsT opsT] =>
        synthLocalWith (deadIn opsS) ((argsS.map (·.id)).zip (argsT.map (·.id))) opsS opsT
      | _, _ => []
    | _, _ => []
  { src, tgt, fn, script }

/-- `synthC1` with every unused source op erased first. -/
def synthC1Unused (src tgt : Module) (fn : String) : C1Input :=
  let script := match src.func? fn, tgt.func? fn with
    | some fS, some fT =>
      match fS.body, fT.body with
      | .mk [.mk argsS opsS], .mk [.mk argsT opsT] =>
        synthLocalWith (unusedIn opsS) ((argsS.map (·.id)).zip (argsT.map (·.id))) opsS opsT
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
