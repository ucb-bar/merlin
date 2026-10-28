/- Diagonal and row bounds for the explicit causal/group-equality mask formula.
Actual ramp/gather/SSA producer correspondence is a separate checked obligation. -/
import MerlinVerify.Rules.W8A8AttentionMaskBody
import MerlinVerify.Rules.W8A8RowBroadcast
import MerlinVerify.Rules.W8A8MaskedMaximum

namespace MerlinVerify.Rules.W8A8AttentionMaskRows

open MerlinVerify W8A8AttentionMaskBody W8A8MaskedMaximum

/-- Ramps and captured group labels are arbitrary functions. Equal diagonal
indices must use the SAME functions; unrelated gathers cannot be substituted. -/
def mask (rank : Nat) (ramp : Nat → BitVec 64) (groups : Idx → Nat → BitVec 64) (point : Idx) : Bool :=
  !(compare .lessEqual (ramp (point.getD (rank + 1) 0)) (ramp (point.getD rank 0)) &&
    compare .equal (groups (point.take rank) (point.getD rank 0))
      (groups (point.take rank) (point.getD (rank + 1) 0)))

def selected (masked : Idx → Bool) (raw : Idx → FBits .f32) (point : Idx) : FBits .f32 :=
  if masked point then negativeInfinity else raw point

theorem diagonal (cell : Idx) (column : Nat) (ramp : Nat → BitVec 64) (groups : Idx → Nat → BitVec 64) :
    mask cell.length ramp groups (cell ++ [column, column]) = false := by
  simp [mask, W8A8AttentionMaskBody.compare, BitVec.sle]

theorem selected_admissible {cap : Nat} (masked : Idx → Bool) (raw : Idx → FBits .f32) (point : Idx)
    (bounded : W8A8IEEEBounds.ExponentLE cap (raw point)) : Admissible cap (selected masked raw point) := by
  unfold selected
  split
  · exact Or.inl rfl
  · exact Or.inr bounded

/-- Each actual row uses its own diagonal column. The theorem supplies no
finite-row fact until the mask producer is proved to implement this formula. -/
theorem finite_row {cap : Nat} (shape : List Nat) (width : Nat) (ramp : Nat → BitVec 64)
    (groups : Idx → Nat → BitVec 64) (raw : Idx → FBits .f32)
    (bounded : ∀ point, Idx.InBounds (shape ++ [width, width]) point → W8A8IEEEBounds.ExponentLE cap (raw point))
    (cell : Idx) (bounds : Idx.InBounds (shape ++ [width]) cell) :
    ∃ column < width, W8A8IEEEBounds.ExponentLE cap
      (selected (mask shape.length ramp groups) raw (cell ++ [column])) := by
  let leading := cell.take shape.length
  have leadingBounds : Idx.InBounds shape leading := W8A8RowBroadcast.leading_inBounds bounds
  have present : cell ∈ (List.range width).map (fun column => leading ++ [column]) := by
    rw [← W8A8RowTrace.last_axis_points shape width leading leadingBounds]
    exact List.mem_filter.mpr ⟨(mem_allPoints _ _).mpr bounds, by simp only [leading, decide_true]⟩
  obtain ⟨column, columnMember, same⟩ := List.mem_map.mp present
  have columnBounds : Idx.InBounds [width] [column] :=
    (Idx.inBounds_cons_cons ..).mpr ⟨List.mem_range.mp columnMember, (Idx.inBounds_nil_iff _).mpr rfl⟩
  have pointBounds : Idx.InBounds (shape ++ [width, width]) (cell ++ [column]) := by
    simpa only [List.append_assoc, List.cons_append, List.nil_append] using W8A8RowReshape.append_inBounds bounds columnBounds
  have keep : mask shape.length ramp groups (cell ++ [column]) = false := by
    rw [← same, List.append_assoc]
    have length := leadingBounds.1
    simpa only [length, List.cons_append, List.nil_append] using diagonal leading column ramp groups
  refine ⟨column, List.mem_range.mp columnMember, ?_⟩
  simpa only [selected, keep, Bool.false_eq_true, ↓reduceIte] using bounded _ pointBounds

end MerlinVerify.Rules.W8A8AttentionMaskRows
