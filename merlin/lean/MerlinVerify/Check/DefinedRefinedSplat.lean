/- A splat preserves every proved elementwise fact of its initialized scalar, including
integer intervals. The scalar value and actual result type/shape establish the tensor facts. -/
import MerlinVerify.Check.DefinedReadView
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Check.Definedness.Refined.Splat

open MerlinVerify Sem.Ops

variable {Info : Type} {M : Model Info}

def rule (_ : FEnv Info) (op : Op) (facts : List (Fact Info)) : Option (List (Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧ op.name = "tensor.splat" ∧ op.regions = [] then
    match facts, Sem.Ops.Tensor.resultTensorTy op with
    | [⟨⟨.scalar elt, true⟩, info⟩], some (shape, resultElt) =>
      if elt = resultElt then some [⟨⟨.tensor shape resultElt, true⟩, info⟩] else none
    | _, _ => none
  else none

theorem rule_sound (elementwise : Elementwise M) (ib : Nat) : ∀ Γ op facts results,
    rule Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds M ib Γ env → FactsHold M ib args facts → ∃ values,
      opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values ∧ FactsHold M ib values results := by
  intro Γ op facts results checked ctx call env args _ _ inputs
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next elt info shape resultElt result =>
      split at checked
      · next same =>
        subst resultElt
        cases checked
        obtain _ | ⟨head, tail⟩ := inputs
        cases tail
        rename_i value
        cases value with
        | tensor _ => cases head.1.1
        | scalar value =>
          refine ⟨[.tensor (Tensor.full shape elt value)], ?_, .cons ?_ .nil⟩
          · rw [regions, evalRegions, tensor_sem_of modelled dialect]
            simp only [Sem.Ops.Tensor.sem, Op.blind_name, named, resultTensorTy_blind, result]
          · apply tensorFact_of_cells elementwise rfl rfl
            intro point bounds
            change Idx.InBounds shape point at bounds
            simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte] using head
      · cases checked
    · cases checked
  · cases checked

def book (elementwise : Elementwise M) (ib : Nat) : Book M opSem ib where
  rule := rule
  sound := rule_sound elementwise ib

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.Definedness.Refined.Splat
