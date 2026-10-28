/- Complete checked execution of actual six-op mask application. The producer of
the kept-position tensor remains a separate obligation; no diagonal is presumed. -/
import MerlinVerify.Rules.W8A8AttentionMaskApply
import MerlinVerify.Rules.W8A8AttentionAnnotations

namespace MerlinVerify.Check.PolicyW8A8AttentionMaskApply

open MerlinVerify Rules W8A8AttentionMaskApply

def Correct (maskShape shape : List Nat) (maskMap : AffineMap) (input keepId output : Nat) (actual : List Op) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (keepValues scoreValues : Idx → Val)
    (keep : Idx → Bool) (scores : Idx → FBits .f32),
    env.get? keepId = some (.tensor ⟨maskShape, .int 1, keepValues⟩) →
    env.get? input = some (.tensor ⟨shape, .float .f32, scoreValues⟩) →
    (∀ point, Idx.InBounds maskShape point → keepValues point = W8A8AttentionMaskBody.boolean (keep point)) →
    (∀ point, Idx.InBounds shape point → scoreValues point = W8A8OrderedBody.value (scores point)) →
    evalOps W8A8SourceSemantics.opSem ctx call env (actual ++ [W8A8AttentionTemplate.terminator output]) =
      .ok [.tensor (result maskShape shape maskMap keep scores)]

def check (maskShape shape : List Nat) (maskMap : AffineMap) (input keepId output : Nat) (actual : List Op) : Bool :=
  decide (actual.length = 6 ∧ input ≠ keepId ∧ ∀ op ∈ actual,
    W8A8AttentionAnnotations.Allowed op ∧ op.isTerminator = false ∧ op.successors = []) &&
  geometryCheck maskShape shape maskMap &&
  alphaOps [(0, input), (1, keepId)]
    (operations maskShape shape maskMap ++ [W8A8AttentionTemplate.terminator 7])
    (actual.map W8A8AttentionAnnotations.clean ++ [W8A8AttentionTemplate.terminator output])

theorem check_sound {maskShape shape : List Nat} {maskMap : AffineMap} {input keepId output : Nat} {actual : List Op}
    (checked : check maskShape shape maskMap input keepId output actual = true) :
    Correct maskShape shape maskMap input keepId output actual := by
  simp only [check, Bool.and_eq_true] at checked
  have headers := (of_decide_eq_true checked.1.1).2.2
  intro ctx call env keepValues scoreValues keep scores maskSupplied scoresSupplied maskInitialized scoresInitialized
  let canonical : Env := [(0, .tensor ⟨shape, .float .f32, scoreValues⟩), (1, .tensor ⟨maskShape, .int 1, keepValues⟩)]
  have correspondence := ParametricAlpha.alphaOps_sound W8A8SourceAlpha.laws call call
    [(0, input), (1, keepId)]
    (operations maskShape shape maskMap ++ [W8A8AttentionTemplate.terminator 7])
    (actual.map W8A8AttentionAnnotations.clean ++ [W8A8AttentionTemplate.terminator output]) ctx canonical env checked.2
    (by
      intro source target corresponding
      cases source with
      | zero =>
        change some input = some target at corresponding
        have same := Option.some.inj corresponding
        subst target
        change some (.tensor ⟨shape, .float .f32, scoreValues⟩) = env.get? input
        exact scoresSupplied.symm
      | succ source =>
        cases source with
        | zero =>
          change some keepId = some target at corresponding
          have same := Option.some.inj corresponding
          subst target
          change some (.tensor ⟨maskShape, .int 1, keepValues⟩) = env.get? keepId
          exact maskSupplied.symm
        | succ source =>
          change none = some target at corresponding
          cases corresponding)
    (fun _ _ => rfl)
  have normalized := W8A8AttentionAnnotations.eval_clean ctx call actual
    (fun op member => ⟨(headers op member).1, (headers op member).2.1⟩) env [W8A8AttentionTemplate.terminator output]
  have executed := template_execute ctx call canonical maskShape shape maskMap (geometryCheck_sound checked.1.2)
    keepValues scoreValues keep scores rfl rfl maskInitialized scoresInitialized
  exact normalized.symm.trans (correspondence.symm.trans executed)

end MerlinVerify.Check.PolicyW8A8AttentionMaskApply
