/- Integer constants contribute bounds from their actual modular bits at the declared
width. Negative signed results remain unbounded; their textual spelling cannot supply a
nonnegative interval. The existing constant rule proves execution and initialization. -/
import MerlinVerify.Check.BoundedFloatFacts
import MerlinVerify.Check.FiniteProducers

namespace MerlinVerify.Check.BoundedFloatFacts.IntegerProducers

open MerlinVerify Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem

def interval {width : Nat} (bits : BitVec width) : Option IndexRanges.Interval :=
  if 0 ≤ bits.toInt then some ⟨bits.toInt.toNat, bits.toInt.toNat + 1⟩ else none

theorem interval_sound {width : Nat} (bits : BitVec width) :
    Uniform (interval bits) (.scalar (.some (.bits width bits))) := by
  unfold interval
  split
  · next nonnegative =>
    refine ⟨bits.toInt, rfl, ?_, ?_⟩ <;> dsimp <;> omega
  · trivial

def rule (ib : Nat) (Γ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match facts, ConstantFacts.value? ib op with
  | [], some (.scalar (.some (.bits _ bits))) =>
    match (FiniteFacts.Producers.book ib).rule (Refined.mapEnv (fun info => info.floating.numeric) Γ) op [] with
    | some [fact] => some [⟨fact.base, ⟨FloatFacts.fromClass fact.info, interval bits⟩⟩]
    | _ => none
  | _, _ => none

theorem rule_sound (ib : Nat) : ∀ Γ op facts results,
    rule ib Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args facts →
    ∃ values, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values results := by
  intro Γ op facts results checked ctx call env args width valid inputs
  unfold rule at checked
  split at checked
  · next w bits decoded =>
    cases inputs
    split at checked
    · next fact selected =>
      cases checked
      obtain ⟨eligible, computed⟩ := ConstantFacts.value_spec decoded
      obtain ⟨values, executed, holds⟩ := (FiniteFacts.Producers.book ib).sound
        (Refined.mapEnv (fun info => info.floating.numeric) Γ) op [] [fact] selected ctx call env [] width
        (valid.map (source := model) (target := FiniteFacts.model)
          (fun info => info.floating.numeric) (fun _ _ holds => holds.1.1)) .nil
      have ordinary : W8A8SourceSemantics.isSource op.name = false := by
        simp [W8A8SourceSemantics.isSource, eligible.1, W8A8SourceContract.name]
      have actual : sourceSem ctx call op [] (evalRegions sourceSem call env op.regions) =
          .ok [.scalar (.some (.bits w bits))] := by
        rw [eligible.2.2.2.1, evalRegions, W8A8SourceSemantics.unchanged ordinary,
          ConstantFacts.context_independent eligible width call, computed]
      rw [actual] at executed
      cases executed
      cases holds with
      | cons head tail =>
        exact ⟨_, actual, .cons ⟨head.1, FloatFacts.finiteMap.embed_sound _ _ head.2, interval_sound bits⟩ .nil⟩
    · cases checked
  · cases checked

def book (ib : Nat) : Refined.Book model sourceSem ib where
  rule := rule ib
  sound := rule_sound ib

#print axioms interval_sound
#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.BoundedFloatFacts.IntegerProducers
