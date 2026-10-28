/- Whole-segment SSA/attribute correspondence to the source softmax template.
This does not claim template execution success or an admitted captured mask. -/
import MerlinVerify.Rules.W8A8AttentionTemplate
import MerlinVerify.Rules.W8A8AttentionAnnotations

namespace MerlinVerify.Check.PolicyW8A8AttentionTemplate

open MerlinVerify Rules W8A8AttentionTemplate

def Correct (shape : List Nat) (width input output : Nat) (actual : List Op) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (argument : RVal), env.get? input = some argument →
    evalOps W8A8SourceSemantics.opSem ctx call [(0, argument)]
      (operations shape width ++ [terminator 16]) =
    evalOps W8A8SourceSemantics.opSem ctx call env (actual ++ [terminator output])

def check (shape : List Nat) (width input output : Nat) (actual : List Op) : Bool :=
  decide (shape ≠ [] ∧ actual.length = 16 ∧ ∀ op ∈ actual,
    W8A8AttentionAnnotations.Allowed op ∧ op.isTerminator = false ∧ op.successors = []) &&
    alphaOps [(0, input)] (operations shape width ++ [terminator 16])
      (actual.map W8A8AttentionAnnotations.clean ++ [terminator output])

theorem check_sound {shape : List Nat} {width input output : Nat} {actual : List Op}
    (checked : check shape width input output actual = true) : Correct shape width input output actual := by
  simp only [check, Bool.and_eq_true] at checked
  have headers := (of_decide_eq_true checked.1).2.2
  have matched := checked.2
  intro ctx call env argument supplied
  trans evalOps W8A8SourceSemantics.opSem ctx call env
    (actual.map W8A8AttentionAnnotations.clean ++ [terminator output])
  · apply ParametricAlpha.alphaOps_sound W8A8SourceAlpha.laws call call [(0, input)]
      (operations shape width ++ [terminator 16])
      (actual.map W8A8AttentionAnnotations.clean ++ [terminator output]) ctx [(0, argument)] env matched
    · intro source target corresponding
      cases source with
      | zero =>
        change some input = some target at corresponding
        have same := Option.some.inj corresponding
        subst target
        change some argument = env.get? input
        exact supplied.symm
      | succ source =>
        change none = some target at corresponding
        cases corresponding
    · intro _ _
      rfl
  · exact W8A8AttentionAnnotations.eval_clean ctx call actual
      (fun op member => ⟨(headers op member).1, (headers op member).2.1⟩) env [terminator output]

end MerlinVerify.Check.PolicyW8A8AttentionTemplate
