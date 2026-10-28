/- Exact actual last-axis maximum tensor, using its checked scalar region and
source-derived row order. All-masked and empty rows retain negative infinity. -/
import MerlinVerify.Rules.W8A8RowTrace
import MerlinVerify.Rules.W8A8ReductionRows
import MerlinVerify.Rules.W8A8TranscendentalModel

namespace MerlinVerify.Rules.W8A8AttentionMaximum

open MerlinVerify Sem.Ops.Linalg W8A8MaskedMaximum

def row (width : Nat) (input : Idx → FBits .f32) (cell : Idx) : List (FBits .f32) :=
  (List.range width).map (fun column => input (cell ++ [column]))

theorem generic_sound {cap : Nat} {ctx : Ctx} {actual : Region}
    (shape : List Nat) (width : Nat) (iters : List IterKind)
    (geometry : Check.StructuredOk
      [identityMap (shape.length + 1), W8A8RowTrace.outputMap shape.length] iters.length
      [.tensor (shape ++ [width]) (.float .f32), .tensor shape (.float .f32)] (shape ++ [width]))
    (accepted : Check.PolicyW8A8ReductionBody.check .maximum actual = true)
    (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (inputValues initial : Idx → Val) (input : Idx → FBits .f32)
    (initialized : ∀ point, Idx.InBounds (shape ++ [width]) point →
      inputValues point = W8A8OrderedBody.value (input point))
    (seeds : ∀ cell, Idx.InBounds shape cell → initial cell = W8A8OrderedBody.value negativeInfinity)
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → Admissible cap (input point)) :
    genericSem ctx [identityMap (shape.length + 1), W8A8RowTrace.outputMap shape.length] iters
      [.tensor ⟨shape ++ [width], .float .f32, inputValues⟩] [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) fun cell =>
        W8A8OrderedBody.value (W8A8SoftmaxRow.rowMaximum (row width input cell)))] := by
  let coordinate := W8A8IntegerTensor.coordinate shape (W8A8RowTrace.outputMap shape.length)
  have maskLength : (W8A8RowTrace.mask shape.length).length = (shape ++ [width]).length := by
    simp [W8A8RowTrace.mask]
  have selectedShape : W8A8ProjectionTrace.select (W8A8RowTrace.mask shape.length) (shape ++ [width]) = shape := by
    rw [W8A8RowTrace.select_mask]
    simp
  have indices (point : Idx) (bounds : Idx.InBounds (shape ++ [width]) point) :
      (evalMap (W8A8RowTrace.outputMap shape.length) point >>= toIdx shape) = some (coordinate point) ∧
        Idx.InBounds shape (coordinate point) := by
    simpa only [selectedShape, W8A8RowTrace.projection_map] using
      W8A8OrderedCells.projected_index maskLength bounds
  have typedInput : Check.RVal.HasTy ctx.indexBits (.tensor ⟨shape ++ [width], .float .f32, inputValues⟩)
      (.tensor (shape ++ [width]) (.float .f32)) := by
    refine ⟨rfl, rfl, ?_⟩
    intro point bounds
    change Check.Val.HasTy ctx.indexBits (.float .f32) (inputValues point)
    rw [initialized point bounds]
    simp [W8A8OrderedBody.value, Check.Val.HasTy, Scalar.conforms]
  have initialTyped : W8A8OrderedCells.typedState ctx.indexBits shape initial := by
    intro cell bounds
    rw [seeds cell bounds]
    simp [W8A8OrderedBody.value, Check.Val.HasTy, Scalar.conforms]
  have reads (point : Idx) (bounds : Idx.InBounds (shape ++ [width]) point) :
      readAt (.tensor ⟨shape ++ [width], .float .f32, inputValues⟩) (identityMap (shape.length + 1)) point =
        .ok (W8A8OrderedBody.value (input point)) := by
    have actualRead := readAt_identity (t := ⟨shape ++ [width], .float .f32, inputValues⟩) bounds
    change readAt (.tensor ⟨shape ++ [width], .float .f32, inputValues⟩)
      (identityMap (shape ++ [width]).length) point = .ok (inputValues point) at actualRead
    simpa only [List.length_append, List.length_singleton, initialized point bounds] using actualRead
  have executed := W8A8ReductionTensor.generic_sound geometry typedInput accepted call env
    (W8A8RoundedReal.transcendentalFallback (W8A8UnarySign.model fallback)) interpretation input initial coordinate
    initialTyped reads (fun point bounds => (indices point bounds).1) (fun point bounds => (indices point bounds).2)
  have exactRows :
      Tensor.canon shape (.float .f32)
        (W8A8TensorTrace.accumulate coordinate (W8A8ReductionCells.step .maximum input) initial
          (allPoints (shape ++ [width]))) =
      Tensor.canon shape (.float .f32) (fun cell =>
        W8A8OrderedBody.value (W8A8SoftmaxRow.rowMaximum (row width input cell))) := by
    apply Tensor.canon_ext
    intro cell bounds
    have exactTrace := W8A8ReductionCells.initialized_cell .maximum coordinate input
      (allPoints (shape ++ [width])) initial cell negativeInfinity (seeds cell bounds)
    have rowMembers := W8A8ReductionRows.words_all coordinate input (allPoints (shape ++ [width])) cell
      (Admissible cap) (fun point member => members point ((mem_allPoints _ _).mp member))
    rw [W8A8ReductionOrder.maximum_fold _ _ (Or.inl rfl) rowMembers] at exactTrace
    have selected := W8A8RowTrace.reduction_words shape width cell bounds input
    change W8A8ReductionCells.words coordinate input (allPoints (shape ++ [width])) cell =
      row width input cell at selected
    rw [selected] at exactTrace
    exact exactTrace
  rw [exactRows] at executed
  exact executed

end MerlinVerify.Rules.W8A8AttentionMaximum
