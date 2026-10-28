/- Read-only tensor views preserve initialized cells and every proved elementwise fact.
Each operation must establish its actual coordinate mapping and output shape/type. -/
import MerlinVerify.Check.DefinedElementwise

namespace MerlinVerify.Check.Definedness.Refined

open MerlinVerify

/-- Each observed output cell reads an actual, in-bounds input cell. -/
def ReadView (input output : Tensor) : Prop :=
  ∀ point, Idx.InBounds output.shape point → ∃ source,
    Idx.InBounds input.shape source ∧ output.get point = input.get source

theorem tensorFact_of_cells {Info : Type} {M : Model Info} (elementwise : Elementwise M)
    {ib : Nat} {output : Tensor} {shape : List Nat} {elt : Ty} {info : Info}
    (shapeEq : output.shape = shape) (typeEq : output.elt = elt)
    (cells : ∀ point, Idx.InBounds output.shape point →
      (⟨⟨.scalar elt, true⟩, info⟩ : Fact Info).Holds M ib (.scalar (output.get point))) :
    (⟨⟨.tensor shape elt, true⟩, info⟩ : Fact Info).Holds M ib (.tensor output) := by
  refine ⟨⟨⟨shapeEq, typeEq, ?_⟩, ?_⟩, ?_⟩
  · intro point bounds
    exact (cells point (shapeEq.symm ▸ bounds)).1.1
  · intro _ point bounds
    exact (cells point bounds).1.2 rfl
  · exact (elementwise _ _).2 (fun point bounds => (cells point bounds).2)

theorem readView_fact {Info : Type} {M : Model Info} (elementwise : Elementwise M)
    {ib : Nat} {input output : Tensor} {sourceShape targetShape : List Nat} {elt : Ty} {info : Info}
    (holds : (⟨⟨.tensor sourceShape elt, true⟩, info⟩ : Fact Info).Holds M ib (.tensor input))
    (shape : output.shape = targetShape) (type : output.elt = elt) (view : ReadView input output) :
    (⟨⟨.tensor targetShape elt, true⟩, info⟩ : Fact Info).Holds M ib (.tensor output) := by
  apply tensorFact_of_cells elementwise shape type
  intro point bounds
  obtain ⟨source, bounded, read⟩ := view point bounds
  rw [read]
  exact ⟨⟨holds.1.1.2.2 source (holds.1.1.1 ▸ bounded), fun _ => holds.1.2 rfl source bounded⟩,
    (elementwise _ _).1 holds.2 source bounded⟩

#print axioms tensorFact_of_cells
#print axioms readView_fact

end MerlinVerify.Check.Definedness.Refined
