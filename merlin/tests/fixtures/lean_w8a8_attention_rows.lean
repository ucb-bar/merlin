/- Ordered row geometry, exact maximum execution, reshape preservation and a
cross-row denominator counterexample. The complete captured SSA checker is still open. -/
import MerlinVerify.Rules.W8A8AttentionRowValues

open MerlinVerify Rules W8A8MaskedMaximum Sem.Ops.Linalg

namespace W8A8AttentionRowControls

theorem last_axis_geometry :
    (allPoints ([2, 3] ++ [4])).filter (fun point => decide (point.take 2 = [1, 2])) =
      [[1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 3]] ∧
    W8A8ReductionNamed.outputMap 4 [3] = W8A8RowTrace.outputMap 3 := by
  constructor
  · decide
  · decide

theorem rank_zero_and_empty_reduction :
    (allPoints ([] ++ [2])).filter (fun point => decide (point.take 0 = [])) = [[0], [1]] ∧
    W8A8ReductionCells.words (W8A8IntegerTensor.coordinate [2] (W8A8RowTrace.outputMap 1))
      (fun _ => 0x3f800000) (allPoints [2, 0]) [1] = [] := by decide

theorem arbitrary_ordered_row (shape : List Nat) (width : Nat) (cell : Idx)
    (bounds : Idx.InBounds shape cell) (input : Idx → FBits .f32) :
    W8A8ReductionCells.words (W8A8IntegerTensor.coordinate shape (W8A8RowTrace.outputMap shape.length))
      input (allPoints (shape ++ [width])) cell =
        (List.range width).map (fun column => input (cell ++ [column])) :=
  W8A8RowTrace.reduction_words shape width cell bounds input

def maskedScores (point : Idx) : FBits .f32 :=
  if point = [0, 1] then 0x3f800000 else negativeInfinity

def scoreValues : Idx → Val :=
  (Tensor.canon [2, 2] (.float .f32) (fun point => W8A8OrderedBody.value (maskedScores point))).get

def seedValues : Idx → Val :=
  (Tensor.full [2] (.float .f32) (W8A8OrderedBody.value negativeInfinity)).get

theorem one_finite_row_does_not_fix_another :
    W8A8SoftmaxRow.rowMaximum (W8A8AttentionMaximum.row 2 maskedScores [0]) = 0x3f800000 ∧
    W8A8SoftmaxRow.rowMaximum (W8A8AttentionMaximum.row 2 maskedScores [1]) = negativeInfinity ∧
    ¬ W8A8IEEE32.Finite (W8A8SoftmaxRow.rowMaximum (W8A8AttentionMaximum.row 2 maskedScores [1])) := by decide

theorem complete_maximum_tensor (ctx : Ctx) (call : CallSem) (env : Env) (base : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model base) :
    genericSem ctx [identityMap 2, W8A8RowTrace.outputMap 1] [.parallel, .reduction]
      [.tensor ⟨[2, 2], .float .f32, scoreValues⟩] [.tensor ⟨[2], .float .f32, seedValues⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env (W8A8ReductionBody.region .maximum)) =
      .ok [.tensor (Tensor.canon [2] (.float .f32) fun cell =>
        W8A8OrderedBody.value (W8A8SoftmaxRow.rowMaximum (W8A8AttentionMaximum.row 2 maskedScores cell)))] := by
  have geometry : Check.StructuredOk [identityMap 2, W8A8RowTrace.outputMap 1] 2
      [.tensor [2, 2] (.float .f32), .tensor [2] (.float .f32)] [2, 2] := by
    apply Check.structuredRanges?_spec
    decide
  have initialized : ∀ point, Idx.InBounds [2, 2] point →
      scoreValues point = W8A8OrderedBody.value (maskedScores point) := by
    intro point bounds
    simp only [scoreValues, Tensor.canon, bounds, ↓reduceIte]
  have seeds : ∀ cell, Idx.InBounds [2] cell → seedValues cell = W8A8OrderedBody.value negativeInfinity := by
    intro cell bounds
    simp only [seedValues, Tensor.full, Tensor.canon, bounds, ↓reduceIte]
  have members : ∀ point, Idx.InBounds [2, 2] point → Admissible 179 (maskedScores point) := by
    intro point _
    unfold maskedScores
    split
    · exact Or.inr (by decide)
    · exact Or.inl rfl
  have body : Check.PolicyW8A8ReductionBody.check .maximum (W8A8ReductionBody.region .maximum) = true := by cbv
  exact W8A8AttentionMaximum.generic_sound [2] 2 [.parallel, .reduction] geometry body
    call env base interpretation scoreValues seedValues maskedScores initialized seeds members

theorem actual_row_exponential_association (rank width : Nat) (input : Idx → FBits .f32) (cell : Idx)
    (lengths : cell.length = rank) :
    (List.range width).map (fun column => W8A8AttentionRowValues.exponential rank width input (cell ++ [column])) =
      W8A8SoftmaxRow.exponentials (W8A8AttentionMaximum.row width input cell) :=
  W8A8AttentionRowValues.row_exponentials rank width input cell lengths

theorem reshape_preserves_every_value (tensor : Tensor) (flat : List Nat) (cell : Idx)
    (size : shapeSize flat = shapeSize tensor.shape) (bounds : Idx.InBounds tensor.shape cell) :
    (ReshapeIndex.reshape (ReshapeIndex.reshape tensor flat tensor.elt)
      (tensor.shape ++ [1]) tensor.elt).get (cell ++ [0]) = tensor.get cell :=
  W8A8RowReshape.collapsed_then_unit_get tensor flat cell size bounds

theorem wrong_row_denominator_changes_result :
    W8A8UnitAdd.AtLeastOne (0x3f800000 : FBits .f32) ∧
    W8A8UnitAdd.AtLeastOne (0x40000000 : FBits .f32) ∧
    W8A8IEEE32.div (0x3f800000 : FBits .f32) 0x3f800000 = 0x3f800000 ∧
    W8A8IEEE32.div (0x3f800000 : FBits .f32) 0x40000000 = 0x3f000000 ∧
    W8A8IEEE32.div (0x3f800000 : FBits .f32) 0x3f800000 ≠
      W8A8IEEE32.div (0x3f800000 : FBits .f32) 0x40000000 := by decide

#print axioms last_axis_geometry
#print axioms rank_zero_and_empty_reduction
#print axioms arbitrary_ordered_row
#print axioms one_finite_row_does_not_fix_another
#print axioms complete_maximum_tensor
#print axioms actual_row_exponential_association
#print axioms reshape_preserves_every_value
#print axioms wrong_row_denominator_changes_result

end W8A8AttentionRowControls
