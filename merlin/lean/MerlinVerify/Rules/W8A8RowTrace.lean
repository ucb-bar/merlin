/- Actual lexicographic last-axis rows. The selected list, not merely its length,
is identified so max/exp/sum can share the same ordered source elements. -/
import MerlinVerify.Rules.W8A8ProjectionTrace
import MerlinVerify.Rules.W8A8ReductionNamed

namespace MerlinVerify.Rules.W8A8RowTrace

open MerlinVerify

def mask (rank : Nat) : List Bool := List.replicate rank true ++ [false]

def outputMap (rank : Nat) : AffineMap :=
  ⟨rank + 1, 0, (List.range rank).map AffineExpr.dim⟩

theorem select_mask (rank : Nat) (point : Idx) :
    W8A8ProjectionTrace.select (mask rank) point = point.take rank := by
  induction rank generalizing point with
  | zero => cases point <;> rfl
  | succ rank induction =>
    cases point with
    | nil => rfl
    | cons head tail =>
      change head :: W8A8ProjectionTrace.select (mask rank) tail = head :: tail.take rank
      rw [induction]

theorem dimensions_mask (rank : Nat) : W8A8ProjectionTrace.dimensions (mask rank) = List.range rank := by
  induction rank with
  | zero => rfl
  | succ rank induction =>
    change [0] ++ (W8A8ProjectionTrace.dimensions (mask rank)).map (· + 1) = List.range (rank + 1)
    rw [induction]
    exact List.range_succ_eq_map.symm

theorem projection_map (rank : Nat) : W8A8ProjectionTrace.outputMap (mask rank) = outputMap rank := by
  unfold W8A8ProjectionTrace.outputMap
  rw [dimensions_mask]
  simp only [mask, List.length_append, List.length_replicate, List.length_singleton, outputMap]

theorem kept_dimensions (rank : Nat) :
    (List.range (rank + 1)).filter (fun d => !([rank].contains d)) = List.range rank := by
  have unchanged : (List.range rank).filter (fun d => !([rank].contains d)) = List.range rank := by
    apply List.filter_eq_self.mpr
    intro index member
    have different : index ≠ rank := Nat.ne_of_lt (List.mem_range.mp member)
    simp [different]
  rw [List.range_succ, List.filter_append, unchanged]
  simp

theorem named_output_map (rank : Nat) : W8A8ReductionNamed.outputMap (rank + 1) [rank] = outputMap rank := by
  rw [W8A8ReductionNamed.outputMap, kept_dimensions]
  rfl

theorem output_coordinate {shape point : List Nat} {width : Nat}
    (bounds : Idx.InBounds (shape ++ [width]) point) :
    W8A8IntegerTensor.coordinate shape (outputMap shape.length) point = point.take shape.length := by
  have lengths : (mask shape.length).length = (shape ++ [width]).length := by simp [mask]
  have selected : W8A8ProjectionTrace.select (mask shape.length) (shape ++ [width]) = shape := by
    rw [select_mask]
    simp
  have actual := W8A8ProjectionTrace.output_coordinate lengths bounds
  simpa only [selected, projection_map, select_mask] using actual

private theorem select_one {α β : Type} [DecidableEq α] (xs : List α) (x : α) (f : α → List β)
    (unique : xs.Nodup) (member : x ∈ xs) :
    xs.flatMap (fun y => if y = x then f y else []) = f x := by
  induction xs with
  | nil => cases member
  | cons y xs induction =>
    have tail := List.nodup_cons.mp unique
    by_cases same : y = x
    · subst y
      have absent : xs.flatMap (fun y => if y = x then f y else []) = [] := by
        rw [List.flatMap_eq_nil_iff]
        intro y present
        have different : y ≠ x := by intro equal; subst y; exact tail.1 present
        simp only [different, ↓reduceIte]
      simp [absent]
    · have present : x ∈ xs := (List.mem_cons.mp member).resolve_left (Ne.symm same)
      simpa [same] using induction tail.2 present

theorem last_axis_points (shape : List Nat) (width : Nat) :
    ∀ row, Idx.InBounds shape row →
      (allPoints (shape ++ [width])).filter (fun point => decide (point.take shape.length = row)) =
        (List.range width).map (fun column => row ++ [column]) := by
  induction shape with
  | nil =>
    intro row bounds
    have empty : row = [] := (Idx.inBounds_nil_iff _).mp bounds
    subst row
    simp [allPoints]
    exact (List.map_eq_flatMap (l := List.range width) (f := fun column => [column])).symm
  | cons extent shape induction =>
    intro row bounds
    cases row with
    | nil => simp [Idx.inBounds_cons_nil] at bounds
    | cons index row =>
      have bounded := (Idx.inBounds_cons_cons ..).mp bounds
      have inner (head : Nat) :
          ((allPoints (shape ++ [width])).map (head :: ·)).filter
              (fun point => decide (point.take (extent :: shape).length = index :: row)) =
            if head = index then
              ((allPoints (shape ++ [width])).filter (fun point => decide (point.take shape.length = row))).map
                (head :: ·) else [] := by
        by_cases same : head = index <;>
          simp [List.filter_map, Function.comp_def, List.take_succ_cons, same]
      rw [List.cons_append, allPoints, List.filter_flatMap]
      simp only [inner]
      rw [select_one (List.range extent) index _ List.nodup_range (List.mem_range.mpr bounded.1),
        induction row bounded.2]
      simp only [List.map_map, Function.comp_def, List.cons_append]

theorem reduction_words (shape : List Nat) (width : Nat) (row : Idx)
    (bounds : Idx.InBounds shape row) (input : Idx → FBits .f32) :
    W8A8ReductionCells.words (W8A8IntegerTensor.coordinate shape (outputMap shape.length))
      input (allPoints (shape ++ [width])) row =
        (List.range width).map (fun column => input (row ++ [column])) := by
  have selected :
      (allPoints (shape ++ [width])).filter (fun point =>
        decide (row = W8A8IntegerTensor.coordinate shape (outputMap shape.length) point)) =
      (allPoints (shape ++ [width])).filter (fun point => decide (point.take shape.length = row)) := by
    apply List.filter_congr
    intro point member
    rw [output_coordinate ((mem_allPoints _ _).mp member)]
    simp only [eq_comm]
  rw [W8A8ReductionCells.words, selected, last_axis_points shape width row bounds, List.map_map]
  rfl

end MerlinVerify.Rules.W8A8RowTrace
