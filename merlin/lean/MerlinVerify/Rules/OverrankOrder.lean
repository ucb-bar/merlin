/- Lexicographic loop trace equals the independently specified increasing-k fold. -/
import MerlinVerify.Rules.OverrankCoordinates
namespace MerlinVerify.Rules.OverrankOrder
open MerlinVerify OverrankIntent OverrankCoordinates

private theorem fold_skip {α β : Type} [DecidableEq β] (rows : List β) (c : β)
    (g : α → α) (z : α) (h : c ∉ rows) :
    rows.foldl (fun a r => if c = r then g a else a) z = z := by
  induction rows generalizing z with
  | nil => rfl
  | cons r rs ih => simp_all

private theorem fold_once {α β : Type} [DecidableEq β] (rows : List β) (c : β)
    (g : α → α) (z : α) (hn : rows.Nodup) (hm : c ∈ rows) :
    rows.foldl (fun a r => if c = r then g a else a) z = g z := by
  induction rows generalizing z with
  | nil => simp at hm
  | cons r rs ih =>
    simp only [List.nodup_cons] at hn
    simp only [List.mem_cons] at hm
    rcases hm with rfl | hm
    · simp only [List.foldl_cons, ite_true]
      exact fold_skip rs c g (g z) hn.1
    · by_cases he : c = r
      · subst c; exact False.elim (hn.1 hm)
      · simp only [List.foldl_cons, if_neg he]
        exact ih z hn.2 hm

theorem fold_congr {α β : Type} (rows : List β) (f g : α → β → α) (z : α)
    (h : ∀ acc row, row ∈ rows → f acc row = g acc row) : rows.foldl f z = rows.foldl g z := by
  induction rows generalizing z with
  | nil => rfl
  | cons r rs ih =>
    simp only [List.foldl_cons]
    rw [h z r (by simp)]
    exact ih _ (fun acc row hr => h acc row (by simp [hr]))

/-- Independent coordinates on a single output's reduction row. -/
theorem coords_append (s : Shape) (row : Idx) (j : Nat)
    (len : row.length = s.batch.length+1) :
    outCoord s (row ++ [j]) = row ∧
    leftCoord s (row ++ [j]) = row.take s.batch.length ++ [j] ∧
    rightCoord s (row ++ [j]) = [j, row[s.batch.length]?.getD 0] := by
  have lt : s.batch.length < row.length := by omega
  have le : s.batch.length ≤ row.length := by omega
  constructor
  · simp [outCoord, ← len]
  constructor
  · simp [leftCoord, ← len, List.take_append, Nat.sub_eq_zero_of_le le,
      List.getElem?_append_right, List.getElem?_append_left, le]
  · simp [rightCoord, ← len, List.getElem?_append_right,
      List.getElem?_append_left lt]

/-- Complete trace, one output cell, no commutativity or reassociation. -/
theorem trace_cell (M : FloatModel) (kind : FKind) (s : Shape)
    (A B : Idx → Option (FBits kind)) (c : Idx) (hc : Idx.InBounds s.out c) :
    (allPoints s.ranges).foldl
      (fun acc p => if c = outCoord s p then
        stepOpt M kind (A (leftCoord s p)) (B (rightCoord s p)) acc else acc)
      (some (zero kind)) = cell M kind s A B c := by
  rw [Shape.ranges, allPoints_append, List.foldl_flatMap]
  have inner (row : Idx) (hr : row ∈ allPoints s.out) (acc : Option (FBits kind)) :
      ((allPoints [s.k]).map (row ++ ·)).foldl
        (fun acc p => if c = outCoord s p then
          stepOpt M kind (A (leftCoord s p)) (B (rightCoord s p)) acc else acc) acc =
      if c = row then
        (List.range s.k).foldl (fun acc j =>
          stepOpt M kind (A (row.take s.batch.length ++ [j]))
            (B [j, row[s.batch.length]?.getD 0]) acc) acc else acc := by
    have len : row.length = s.batch.length+1 := by
      simpa [Shape.out] using ((mem_allPoints _ _).mp hr).1
    have one : allPoints [s.k] = (List.range s.k).map (fun j => [j]) := by
      simp only [allPoints, List.map_cons, List.map_nil]
      exact (List.map_eq_flatMap ..).symm
    rw [one, List.map_map, List.foldl_map]
    by_cases h : c = row
    · subst row
      simp only [Function.comp_def, (fun j => (coords_append s c j len).1),
        (fun j => (coords_append s c j len).2.1), (fun j => (coords_append s c j len).2.2), ite_true]
    · simp only [Function.comp_def, (fun j => (coords_append s row j len).1), if_neg h]
      simp
  have eqfold : (allPoints s.out).foldl
      (fun acc row => (((allPoints [s.k]).map (row ++ ·)).foldl
        (fun acc p => if c = outCoord s p then
          stepOpt M kind (A (leftCoord s p)) (B (rightCoord s p)) acc else acc) acc))
      (some (zero kind)) =
      (allPoints s.out).foldl (fun acc row => if c = row then
        (List.range s.k).foldl (fun acc j =>
          stepOpt M kind (A (c.take s.batch.length ++ [j]))
            (B [j, c[s.batch.length]?.getD 0]) acc) acc else acc) (some (zero kind)) := by
    apply fold_congr
    intro acc row hr
    rw [inner row hr acc]
    split <;> simp_all
  rw [eqfold]
  exact fold_once _ c _ _ (nodup_allPoints _) ((mem_allPoints _ _).mpr hc)
end MerlinVerify.Rules.OverrankOrder
