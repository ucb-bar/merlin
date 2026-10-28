/- Bind the conservative numerical scale class to actual checked SSA calibration producers.
This proves a scale precondition only; activation finiteness and model laws remain separate. -/
import MerlinVerify.Check.PolicyW8A8SourceFacts
import MerlinVerify.Rules.W8A8FiniteLaws

namespace MerlinVerify.Check.PolicyW8A8FiniteParameters

open MerlinVerify MerlinVerify.Sem.Ops Rules
open ConstantFacts (Fact model)

def NormalScale (facts : List Fact) (op : Op) : Prop :=
  ∀ env, BindingFacts.Valid model facts env →
    ∃ dataId scaleId zeroId scale s,
      op.operands = [dataId, scaleId, zeroId] ∧ env.get? scaleId = some (.tensor scale) ∧
      Dequant.float32 (scale.get []) = some s ∧ W8A8FiniteLaws.scaleCheck s = true

def Correct (types : List (ValueId × Ty)) (facts : List Fact) (op : Op) : Prop :=
  PolicyW8A8SourceFacts.CalibrationCorrect types facts op ∧ NormalScale facts op

def normalScaleCheck (facts : List Fact) (op : Op) : Bool :=
  match op.operands with
  | [_, scaleId, _] => match PolicyW8A8SourceFacts.value? facts scaleId with
    | some (.tensor scale) => match Dequant.float32 (scale.get []) with
      | some s => W8A8FiniteLaws.scaleCheck s
      | none => false
    | _ => false
  | _ => false

theorem normalScaleCheck_sound {facts : List Fact} {op : Op}
    (checked : normalScaleCheck facts op = true) : NormalScale facts op := by
  intro env valid
  unfold normalScaleCheck at checked
  split at checked
  · next dataId scaleId zeroId operands =>
    split at checked
    · next scale bound =>
      split at checked
      · next s decoded =>
        exact ⟨dataId, scaleId, zeroId, scale, s, operands,
          PolicyW8A8SourceFacts.value?_sound valid bound, decoded, checked⟩
      · cases checked
    · cases checked
  · cases checked

def check (types : List (ValueId × Ty)) (facts : List Fact) (op : Op) : Bool :=
  PolicyW8A8SourceFacts.checkParameters types facts op && normalScaleCheck facts op

theorem check_sound {types : List (ValueId × Ty)} {facts : List Fact} {op : Op}
    (checked : check types facts op = true) : Correct types facts op := by
  simp only [check, Bool.and_eq_true] at checked
  exact ⟨PolicyW8A8SourceFacts.checkParameters_sound checked.1, normalScaleCheck_sound checked.2⟩

#print axioms normalScaleCheck_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8FiniteParameters
