/- Complete projection-loop semantics and exact signed accumulation. The operation's
actual shapes/maps determine the loop box; the scalar body and each cell's reduction
cardinality are proved independently of any policy approval or float interpretation. -/
import MerlinVerify.Rules.W8A8ProjectionLayout

namespace MerlinVerify.Check.PolicyW8A8Projection
open MerlinVerify Sem.Ops.Linalg Rules

def reference (layout : W8A8ProjectionLayout.Layout) (left right : Idx → Option (BitVec 8))
    (initial : Idx → Option (BitVec 32)) : Tensor :=
  Tensor.canon layout.outputShape (.int 32) fun cell => W8A8IntegerBody.value 32
    (W8A8TensorTrace.accumulate
      (W8A8IntegerTensor.coordinate layout.outputShape (W8A8ProjectionTrace.outputMap layout.mask))
      (W8A8IntegerTensor.step
        (fun point => left (W8A8IntegerTensor.coordinate layout.leftShape layout.leftMap point))
        (fun point => right (W8A8IntegerTensor.coordinate layout.rightShape layout.rightMap point)))
      initial (allPoints layout.ranges) cell)

def Correct (layout : W8A8ProjectionLayout.Layout) (body : Region) : Prop :=
  ∀ ctx call env left right initial,
    genericSem ctx layout.maps layout.iterators
      [.tensor (W8A8IntegerTensor.tensor 8 layout.leftShape left),
        .tensor (W8A8IntegerTensor.tensor 8 layout.rightShape right)]
      [.tensor (W8A8IntegerTensor.tensor 32 layout.outputShape initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env body) =
      .ok [.tensor (reference layout left right initial)]

def check (layout : W8A8ProjectionLayout.Layout) (body : Region) : Bool :=
  W8A8ProjectionLayout.check layout && PolicyW8A8Body.check body

theorem check_sound {layout : W8A8ProjectionLayout.Layout} {body : Region}
    (checked : check layout body = true) : Correct layout body := by
  simp only [check, Bool.and_eq_true] at checked
  have valid := W8A8ProjectionLayout.check_sound checked.1
  intro ctx call env left right initial
  exact W8A8IntegerTensor.checked_generic_sound (ib := ctx.indexBits) valid.1 checked.2 call env left right initial

/-- Zero initialization gives the mathematical signed dot product for every actual cell.
Captured bytes may include -128, and every reduction prefix stays in the signed i32 range. -/
theorem check_zero_sound {layout : W8A8ProjectionLayout.Layout} {body : Region}
    (checked : check layout body = true) (ctx : Ctx) (call : CallSem) (env : Env)
    (left right : Idx → BitVec 8) : ∃ result : Tensor,
    genericSem ctx layout.maps layout.iterators
      [.tensor (W8A8IntegerTensor.tensor 8 layout.leftShape (some ∘ left)),
        .tensor (W8A8IntegerTensor.tensor 8 layout.rightShape (some ∘ right))]
      [.tensor (W8A8IntegerTensor.tensor 32 layout.outputShape (fun _ => some 0))]
      (fun c => evalRegion Sem.Ops.opSem c call env body) = .ok [.tensor result] ∧
    result.shape = layout.outputShape ∧ ∀ cell, Idx.InBounds layout.outputShape cell → ∃ bits : BitVec 32,
      result.get cell = .some (.bits 32 bits) ∧ bits.toInt = W8A8Accumulation.dot
        (W8A8IntegerTensor.pairs
          (W8A8IntegerTensor.coordinate layout.outputShape (W8A8ProjectionTrace.outputMap layout.mask))
          (fun point => left (W8A8IntegerTensor.coordinate layout.leftShape layout.leftMap point))
          (fun point => right (W8A8IntegerTensor.coordinate layout.rightShape layout.rightMap point))
          (allPoints layout.ranges) cell) := by
  refine ⟨reference layout (some ∘ left) (some ∘ right) (fun _ => some 0),
    check_sound checked ctx call env _ _ _, rfl, ?_⟩
  have parts := checked
  simp only [check, Bool.and_eq_true] at parts
  have valid := W8A8ProjectionLayout.check_sound parts.1
  intro cell bounds
  let a := fun point => left (W8A8IntegerTensor.coordinate layout.leftShape layout.leftMap point)
  let b := fun point => right (W8A8IntegerTensor.coordinate layout.rightShape layout.rightMap point)
  let pairs := W8A8IntegerTensor.pairs
    (W8A8IntegerTensor.coordinate layout.outputShape (W8A8ProjectionTrace.outputMap layout.mask))
    a b (allPoints layout.ranges) cell
  have count : pairs.length = layout.reductionSize := by
    dsimp only [pairs, W8A8ProjectionLayout.Layout.reductionSize]
    rw [valid.2.2.1] at bounds ⊢
    exact W8A8ProjectionTrace.mapped_pair_count valid.2.1 bounds a b
  refine ⟨W8A8Accumulation.accumulate pairs, ?_, ?_⟩
  · simp only [reference, Tensor.canon, bounds, ↓reduceIte]
    change W8A8IntegerBody.value 32
      (W8A8TensorTrace.accumulate
        (W8A8IntegerTensor.coordinate layout.outputShape (W8A8ProjectionTrace.outputMap layout.mask))
        (W8A8IntegerTensor.step (some ∘ a) (some ∘ b)) (some ∘ (fun _ => 0))
        (allPoints layout.ranges) cell) = _
    rw [W8A8IntegerTensor.initialized_cell]
    rfl
  · apply W8A8Accumulation.exact_of_length
    rw [count]
    have bound := valid.2.2.2
    omega

#print axioms check_sound
#print axioms check_zero_sound
end MerlinVerify.Check.PolicyW8A8Projection
