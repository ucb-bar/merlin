/- Learn numerical classes from actual constant bits and preserve them through splats.
The ordinary definedness rule still proves successful typed execution; no arithmetic identity
or output hint can establish a numerical class. -/
import MerlinVerify.Check.FiniteFacts
import MerlinVerify.Check.PolicyW8A8Defined

namespace MerlinVerify.Check.FiniteFacts.Producers

open MerlinVerify MerlinVerify.Sem.Ops Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem

def constantRule (ib : Nat) (Γ : Refined.FEnv Class) (op : Op) (fs : List (Refined.Fact Class)) :
    Option (List (Refined.Fact Class)) :=
  match fs, ConstantFacts.value? ib op with
  | [], some (.scalar value) =>
    match (PolicyW8A8Defined.book ib).rule (Refined.baseEnv Γ) op [] with
    | some [fact] => some [⟨fact, classify value⟩]
    | _ => none
  | _, _ => none

theorem constantRule_sound (ib : Nat) : ∀ Γ op fs rs,
    constantRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs checked ctx call env args width valid inputs
  unfold constantRule at checked
  split at checked
  · next value decoded =>
    cases inputs
    split at checked
    · next fact selected =>
      cases checked
      obtain ⟨eligible, computed⟩ := ConstantFacts.value_spec decoded
      obtain ⟨vs, executed, holds⟩ := (PolicyW8A8Defined.book ib).sound
        (Refined.baseEnv Γ) op [] [fact] selected ctx call env [] width valid.base .nil
      have ordinary : W8A8SourceSemantics.isSource op.name = false := by
        simp [W8A8SourceSemantics.isSource, eligible.1, W8A8SourceContract.name]
      have actual : sourceSem ctx call op [] (evalRegions sourceSem call env op.regions) =
          .ok [.scalar value] := by
        rw [eligible.2.2.2.1, evalRegions, W8A8SourceSemantics.unchanged ordinary,
          ConstantFacts.context_independent eligible width call, computed]
      rw [actual] at executed
      cases executed
      cases holds with
      | cons head tail => exact ⟨_, actual, .cons ⟨head, classify_sound value⟩ .nil⟩
    · cases checked
  · cases checked

def splatRule (_ : Refined.FEnv Class) (op : Op) (fs : List (Refined.Fact Class)) :
    Option (List (Refined.Fact Class)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧
      op.name = "tensor.splat" ∧ op.regions = [] then
    match fs, Sem.Ops.Tensor.resultTensorTy op with
    | [⟨⟨.scalar ty, true⟩, cls⟩], some (shape, elt) =>
      if ty = elt then some [⟨⟨.tensor shape elt, true⟩, cls⟩] else none
    | _, _ => none
  else none

theorem splatRule_sound (ib : Nat) : ∀ Γ op fs rs,
    splatRule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs checked ctx call env args _width _valid inputs
  unfold splatRule at checked
  split at checked
  · next conditions =>
    obtain ⟨modelled, dialect, named, regions⟩ := conditions
    split at checked
    · next ty cls shape elt result =>
      split at checked
      · next same =>
        subst elt
        cases checked
        obtain _ | ⟨head, tail⟩ := inputs
        cases tail
        rename_i value
        cases value with
        | tensor _ => cases head.1.1
        | scalar value =>
          refine ⟨[.tensor (Tensor.full shape ty value)], ?_,
            .cons ⟨⟨hasTy_full head.1.1, ?_⟩, full_holds head.2 shape ty⟩ .nil⟩
          · have ordinary : W8A8SourceSemantics.isSource op.name = false := by
              simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]
            rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary,
              tensor_sem_of modelled dialect]
            simp only [Sem.Ops.Tensor.sem, Op.blind_name, named, resultTensorTy_blind, result]
          · intro _ point bounds
            change Idx.InBounds shape point at bounds
            simpa only [RVal.Defined, Tensor.full, Tensor.canon, bounds, ↓reduceIte] using head.1.2 rfl
      · cases checked
    · cases checked
  · cases checked

def book (ib : Nat) : Refined.Book model sourceSem ib :=
  ({ rule := constantRule ib, sound := constantRule_sound ib } : Refined.Book model sourceSem ib).orElse
    { rule := splatRule, sound := splatRule_sound ib }

#print axioms constantRule_sound
#print axioms splatRule_sound
#print axioms book

end MerlinVerify.Check.FiniteFacts.Producers
