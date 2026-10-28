/- One shared source row determines max/sub/exp/sum/div values. Associations are
derived from coordinates and expressions, not asserted for unrelated SSA tensors. -/
import MerlinVerify.Rules.W8A8AttentionMaximum
import MerlinVerify.Rules.W8A8RowReshape

namespace MerlinVerify.Rules.W8A8AttentionRowValues

open MerlinVerify W8A8MaskedMaximum W8A8AttentionMaximum

def shifted (rank width : Nat) (input : Idx → FBits .f32) (point : Idx) : FBits .f32 :=
  W8A8ArithmeticModel.subtract32 (input point) (W8A8SoftmaxRow.rowMaximum (row width input (point.take rank)))

noncomputable def exponential (rank width : Nat) (input : Idx → FBits .f32) (point : Idx) : FBits .f32 :=
  W8A8RoundedReal.exponential (shifted rank width input point)

noncomputable def probability (rank width : Nat) (input : Idx → FBits .f32) (point : Idx) : FBits .f32 :=
  W8A8IEEE32.div (exponential rank width input point) (W8A8SoftmaxRow.denominator (row width input (point.take rank)))

theorem row_exponentials (rank width : Nat) (input : Idx → FBits .f32) (cell : Idx)
    (lengths : cell.length = rank) :
    (List.range width).map (fun column => exponential rank width input (cell ++ [column])) =
      W8A8SoftmaxRow.exponentials (row width input cell) := by
  have rowCoordinate (column : Nat) : (cell ++ [column]).take rank = cell := by rw [← lengths]; simp
  simp only [W8A8SoftmaxRow.exponentials, row, List.map_map, Function.comp_def,
    exponential, shifted, rowCoordinate]

theorem row_probabilities (rank width : Nat) (input : Idx → FBits .f32) (cell : Idx)
    (lengths : cell.length = rank) :
    (List.range width).map (fun column => probability rank width input (cell ++ [column])) =
      W8A8SoftmaxRow.probabilities (row width input cell) := by
  have rowCoordinate (column : Nat) : (cell ++ [column]).take rank = cell := by rw [← lengths]; simp
  rw [W8A8SoftmaxRow.probabilities, ← row_exponentials rank width input cell lengths, List.map_map]
  simp only [probability, rowCoordinate, Function.comp_def]

/-- The actual last-axis reduction reads precisely the exponential list belonging
to that row, in the same order, so the computed denominator is associated. -/
theorem denominator_trace (shape : List Nat) (width : Nat) (input : Idx → FBits .f32)
    (cell : Idx) (bounds : Idx.InBounds shape cell) :
    W8A8ReductionCells.reduce .sum
      (W8A8ReductionCells.words (W8A8IntegerTensor.coordinate shape (W8A8RowTrace.outputMap shape.length))
        (exponential shape.length width input) (allPoints (shape ++ [width])) cell) (0 : FBits .f32) =
      W8A8SoftmaxRow.denominator (row width input cell) := by
  rw [W8A8RowTrace.reduction_words shape width cell bounds, row_exponentials _ _ _ _ bounds.1]
  exact W8A8ReductionOrder.sum_fold _ _

theorem row_members {cap : Nat} (shape : List Nat) (width : Nat) (input : Idx → FBits .f32)
    (cell : Idx) (bounds : Idx.InBounds shape cell)
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → Admissible cap (input point)) :
    ∀ word ∈ row width input cell, Admissible cap word := by
  intro word member
  obtain ⟨column, selected, rfl⟩ := List.mem_map.mp member
  have columnBounds : Idx.InBounds [width] [column] :=
    (Idx.inBounds_cons_cons ..).mpr ⟨List.mem_range.mp selected, (Idx.inBounds_nil_iff _).mpr rfl⟩
  exact members _ (W8A8RowReshape.append_inBounds bounds columnBounds)

theorem row_witness {cap : Nat} (width : Nat) (input : Idx → FBits .f32) (cell : Idx)
    (unmasked : ∃ column < width, W8A8IEEEBounds.ExponentLE cap (input (cell ++ [column]))) :
    ∃ word ∈ row width input cell, W8A8IEEEBounds.ExponentLE cap word := by
  obtain ⟨column, bounded, finite⟩ := unmasked
  exact ⟨input (cell ++ [column]), List.mem_map.mpr ⟨column, List.mem_range.mpr bounded, rfl⟩, finite⟩

/-- Every point gets the numerical bound for its own row. The finite witness is
required for every actual row; a finite entry in a different row does not suffice. -/
theorem probability_bounded {cap : Nat} (shape : List Nat) (width : Nat) (input : Idx → FBits .f32)
    (safe : cap ≤ 252)
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → Admissible cap (input point))
    (unmasked : ∀ cell, Idx.InBounds shape cell →
      ∃ column < width, W8A8IEEEBounds.ExponentLE cap (input (cell ++ [column])))
    (count : width ≤ 2^22) (point : Idx) (bounds : Idx.InBounds (shape ++ [width]) point) :
    W8A8IEEEBounds.ExponentLE 128 (probability shape.length width input point) ∧
      W8A8IEEE32.Nonnegative (probability shape.length width input point) := by
  let cell := point.take shape.length
  have lengths : (W8A8RowTrace.mask shape.length).length = (shape ++ [width]).length := by
    simp [W8A8RowTrace.mask]
  have cellBounds : Idx.InBounds shape cell := by
    have selected := W8A8ProjectionTrace.select_inBounds lengths bounds
    simpa only [W8A8RowTrace.select_mask, List.take_left] using selected
  have selectedPoint : point ∈ (List.range width).map (fun column => cell ++ [column]) := by
    rw [← W8A8RowTrace.last_axis_points shape width cell cellBounds]
    exact List.mem_filter.mpr ⟨(mem_allPoints _ _).mpr bounds, by simp only [cell, decide_true]⟩
  obtain ⟨column, columnMember, same⟩ := List.mem_map.mp selectedPoint
  have numerical := W8A8SoftmaxRow.probabilities_bounded (row width input cell) safe
    (row_members shape width input cell cellBounds members)
    (row_witness width input cell (unmasked cell cellBounds))
    (by simpa only [row, List.length_map, List.length_range] using count)
  have actualMember : probability shape.length width input point ∈ W8A8SoftmaxRow.probabilities (row width input cell) := by
    rw [← row_probabilities shape.length width input cell cellBounds.1]
    exact List.mem_map.mpr ⟨column, columnMember, congrArg (probability shape.length width input) same⟩
  exact numerical _ actualMember

end MerlinVerify.Rules.W8A8AttentionRowValues
