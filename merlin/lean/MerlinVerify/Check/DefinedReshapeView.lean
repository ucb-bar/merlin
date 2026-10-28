/- Static casts and reshapes retain elementwise facts by proving their actual reads.
Reassociation and element counts come from the operation, including empty tensor shapes. -/
import MerlinVerify.Check.DefinedReadView
import MerlinVerify.Check.RulesTensor

namespace MerlinVerify.Check.Definedness.Refined.ReshapeView

open MerlinVerify Sem.Ops

variable {Info : Type} {M : Model Info}

def reshapeRule (_ : FEnv Info) (op : Op) (facts : List (Fact Info)) : Option (List (Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧
      (op.name = "tensor.collapse_shape" ∨ op.name = "tensor.expand_shape") ∧ op.regions = [] then
    match facts, Sem.Ops.Tensor.resultTensorTy op, Sem.Ops.Tensor.reassociation? op with
    | [⟨⟨.tensor sourceShape elt, true⟩, info⟩], some (shape, resultElt), some groups =>
      if elt = resultElt ∧ reshapeValid op groups sourceShape shape = true then
        some [⟨⟨.tensor shape resultElt, true⟩, info⟩]
      else none
    | _, _, _ => none
  else none

theorem reshapeRule_sound (elementwise : Elementwise M) (ib : Nat) : ∀ Γ op facts outputs,
    reshapeRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds M ib Γ env → FactsHold M ib args facts → ∃ values,
      opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values ∧ FactsHold M ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _ _ inputs
  unfold reshapeRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next sourceShape elt info shape resultElt groups result reassociation =>
      split at checked
      · next geometry =>
        obtain ⟨rfl, valid⟩ := geometry
        cases checked
        obtain _ | ⟨head, tail⟩ := inputs
        cases tail
        rename_i value
        cases value with
        | scalar _ => cases head.1.1
        | tensor tensor =>
          have size := reshapeValid_size valid
          let output := Tensor.canon shape elt fun point =>
            tensor.get (Sem.Ops.Tensor.delinearize tensor.shape (linearIndex shape point))
          have view : ReadView tensor output := by
            intro point bounds
            change Idx.InBounds shape point at bounds
            refine ⟨Sem.Ops.Tensor.delinearize tensor.shape (linearIndex shape point), ?_, ?_⟩
            · apply delinearize_inBounds
              rw [head.1.1.1, ← size]
              exact linearIndex_lt shape point bounds
            · simp only [output, Tensor.canon, bounds, ↓reduceIte]
          refine ⟨[.tensor output], ?_, .cons (readView_fact elementwise head rfl rfl view) .nil⟩
          rw [regions, evalRegions, tensor_sem_of modelled dialect]
          unfold reshapeValid at valid
          rcases named with named | named <;>
            simp_all only [output, Sem.Ops.Tensor.sem, Op.blind_name, resultTensorTy_blind,
              reassociation?_blind, head.1.1.1, head.1.1.2.1, ↓reduceIte,
              beq_self_eq_true, Bool.and_self]
      · cases checked
    · cases checked
  · cases checked

def castRule (_ : FEnv Info) (op : Op) (facts : List (Fact Info)) : Option (List (Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧ op.name = "tensor.cast" ∧ op.regions = [] then
    match facts, Sem.Ops.Tensor.resultTensorTy op with
    | [⟨⟨.tensor sourceShape elt, true⟩, info⟩], some (shape, resultElt) =>
      if sourceShape = shape ∧ elt = resultElt then some [⟨⟨.tensor shape resultElt, true⟩, info⟩] else none
    | _, _ => none
  else none

theorem castRule_sound (elementwise : Elementwise M) (ib : Nat) : ∀ Γ op facts outputs,
    castRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → EnvHolds M ib Γ env → FactsHold M ib args facts → ∃ values,
      opSem ctx call op args (evalRegions opSem call env op.regions) = .ok values ∧ FactsHold M ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _ _ inputs
  unfold castRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions⟩ := header
    split at checked
    · next sourceShape elt info shape resultElt result =>
      split at checked
      · next geometry =>
        obtain ⟨rfl, rfl⟩ := geometry
        cases checked
        obtain _ | ⟨head, tail⟩ := inputs
        cases tail
        rename_i value
        cases value with
        | scalar _ => cases head.1.1
        | tensor tensor =>
          refine ⟨[.tensor tensor.canonize], ?_, ?_⟩
          · rw [regions, evalRegions, tensor_sem_of modelled dialect]
            simp only [Sem.Ops.Tensor.sem, Op.blind_name, named, resultTensorTy_blind, result,
              head.1.1.1, head.1.1.2.1, beq_self_eq_true, Bool.and_self, ↓reduceIte]
          · exact canonize_refined elementwise (tensors := [tensor]) (.cons head .nil)
      · cases checked
    · cases checked
  · cases checked

def book (elementwise : Elementwise M) (ib : Nat) : Book M opSem ib :=
  ({ rule := castRule, sound := castRule_sound elementwise ib } : Book M opSem ib).orElse
    { rule := reshapeRule, sound := reshapeRule_sound elementwise ib }

#print axioms reshapeRule_sound
#print axioms castRule_sound
#print axioms book

end MerlinVerify.Check.Definedness.Refined.ReshapeView
