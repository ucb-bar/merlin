/- Learn exact scalar f32 bits from actual constants and preserve elementwise facts through
splats. Neither a result type nor an unvalidated annotation supplies literal evidence. -/
import MerlinVerify.Check.FloatFacts
import MerlinVerify.Check.FiniteProducers

namespace MerlinVerify.Check.FloatFacts.Producers

open MerlinVerify Rules Definedness MerlinVerify.Sem.Ops

local notation "sourceSem" => W8A8SourceSemantics.opSem

def constantRule (ib : Nat) (Γ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match facts, ConstantFacts.value? ib op with
  | [], some (.scalar (.some (.flt .f32 bits))) =>
    match (FiniteFacts.Producers.book ib).rule (Refined.mapEnv Info.numeric Γ) op [] with
    | some [fact] => some [⟨fact.base, ⟨fact.info, some bits⟩⟩]
    | _ => none
  | _, _ => none

theorem constantRule_sound (ib : Nat) : ∀ Γ op facts results,
    constantRule ib Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args facts →
    ∃ values, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values results := by
  intro Γ op facts results checked ctx call env args width valid inputs
  unfold constantRule at checked
  split at checked
  · next bits decoded =>
    cases inputs
    split at checked
    · next fact selected =>
      cases checked
      obtain ⟨eligible, computed⟩ := ConstantFacts.value_spec decoded
      obtain ⟨values, executed, holds⟩ := (FiniteFacts.Producers.book ib).sound
        (Refined.mapEnv Info.numeric Γ) op [] [fact] selected ctx call env [] width
        (valid.map Info.numeric finiteMap.project_sound) .nil
      have ordinary : W8A8SourceSemantics.isSource op.name = false := by
        simp [W8A8SourceSemantics.isSource, eligible.1, W8A8SourceContract.name]
      have actual : sourceSem ctx call op [] (evalRegions sourceSem call env op.regions) =
          .ok [.scalar (.some (.flt .f32 bits))] := by
        rw [eligible.2.2.2.1, evalRegions, W8A8SourceSemantics.unchanged ordinary,
          ConstantFacts.context_independent eligible width call, computed]
      rw [actual] at executed
      cases executed
      cases holds with
      | cons head tail => exact ⟨_, actual, .cons ⟨head.1, head.2, rfl⟩ .nil⟩
    · cases checked
  · cases checked

def splatRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧
      op.name = "tensor.splat" ∧ op.regions = [] then
    match facts, Sem.Ops.Tensor.resultTensorTy op with
    | [⟨⟨.scalar ty, true⟩, info⟩], some (shape, elt) =>
      if ty = elt then some [⟨⟨.tensor shape elt, true⟩, info⟩] else none
    | _, _ => none
  else none

theorem splatRule_sound (ib : Nat) : ∀ Γ op facts results,
    splatRule Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args facts →
    ∃ values, sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values results := by
  intro Γ op facts results checked ctx call env args _width _valid inputs
  unfold splatRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next ty info shape elt result =>
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
            rw [regions, evalRegions, W8A8SourceSemantics.unchanged ordinary, tensor_sem_of modelled dialect]
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

end MerlinVerify.Check.FloatFacts.Producers
