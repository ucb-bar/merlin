/- A projection contraction has one independent reduction box per output cell, even
when parallel and reduction iterators are interleaved. Count the actual mapped trace
before applying signed-byte accumulation bounds; output size is not a reduction bound. -/
import MerlinVerify.Rules.W8A8IntegerTensor

namespace MerlinVerify.Rules.W8A8ProjectionTrace
open MerlinVerify Sem.Ops.Linalg

def select {α : Type} : List Bool → List α → List α
  | true :: mask, x :: xs => x :: select mask xs
  | false :: mask, _ :: xs => select mask xs
  | _, _ => []

def dimensions : List Bool → List Nat
  | [] => []
  | keep :: mask => (if keep then [0] else []) ++ (dimensions mask).map (· + 1)

def outputMap (mask : List Bool) : AffineMap :=
  ⟨mask.length, 0, (dimensions mask).map AffineExpr.dim⟩

private theorem select_one {α β : Type} [DecidableEq α] (xs : List α) (x : α) (f : α → List β)
    (unique : xs.Nodup) (member : x ∈ xs) :
    xs.flatMap (fun y => if y = x then f y else []) = f x := by
  induction xs with
  | nil => cases member
  | cons y xs ih =>
    have tail := List.nodup_cons.1 unique
    by_cases same : y = x
    · subst y
      have empty : xs.flatMap (fun y => if y = x then f y else []) = [] := by
        rw [List.flatMap_eq_nil_iff]
        intro y found
        have different : y ≠ x := by intro equal; subst y; exact tail.1 found
        simp [different]
      simp [empty]
    · have found : x ∈ xs := (List.mem_cons.1 member).resolve_left (Ne.symm same)
      simpa [same] using ih tail.2 found

private theorem sum_constant {α : Type} (xs : List α) (count : Nat) :
    (xs.map (fun _ => count)).sum = xs.length * count := by
  induction xs <;> simp_all [Nat.succ_mul, Nat.add_comm]

theorem select_inBounds {mask : List Bool} {shape point : List Nat}
    (lengths : mask.length = shape.length) (bounds : Idx.InBounds shape point) :
    Idx.InBounds (select mask shape) (select mask point) := by
  induction mask generalizing shape point with
  | nil => simp [select, Idx.inBounds_nil_iff]
  | cons keep mask ih =>
    cases shape with
    | nil => simp at lengths
    | cons extent shape =>
      cases point with
      | nil => simp [Idx.inBounds_cons_nil] at bounds
      | cons index point =>
        have remaining : mask.length = shape.length := by simpa using lengths
        have bounded := (Idx.inBounds_cons_cons ..).1 bounds
        cases keep
        · exact ih remaining bounded.2
        · exact (Idx.inBounds_cons_cons ..).2 ⟨bounded.1, ih remaining bounded.2⟩

theorem selected_count {mask : List Bool} {shape cell : List Nat}
    (lengths : mask.length = shape.length) (bounds : Idx.InBounds (select mask shape) cell) :
    ((allPoints shape).filter (fun point => decide (select mask point = cell))).length =
      (select (mask.map not) shape).prod := by
  induction mask generalizing shape cell with
  | nil =>
    have empty : shape = [] := List.length_eq_zero_iff.1 lengths.symm
    subst shape
    have emptyCell : cell = [] := (Idx.inBounds_nil_iff _).1 bounds
    subst cell
    simp [select, allPoints]
  | cons keep mask ih =>
    cases shape with
    | nil => simp at lengths
    | cons extent shape =>
      have remaining : mask.length = shape.length := by simpa using lengths
      cases keep with
      | false =>
        have tail := ih remaining bounds
        have inner (x : Nat) :
            ((allPoints shape).map (x :: ·)).filter
              (fun point => decide (select (false :: mask) point = cell)) =
            ((allPoints shape).filter (fun point => decide (select mask point = cell))).map (x :: ·) := by
          rw [List.filter_map]
          rfl
        rw [allPoints, List.filter_flatMap]
        simp only [inner, List.length_flatMap, List.length_map]
        rw [sum_constant, tail]
        simp [select]
      | true =>
        cases cell with
        | nil => simp [select, Idx.inBounds_cons_nil] at bounds
        | cons index cell =>
          have bounded := (Idx.inBounds_cons_cons ..).1 bounds
          have inner (x : Nat) :
              ((allPoints shape).map (x :: ·)).filter
                (fun point => decide (select (true :: mask) point = index :: cell)) =
              if x = index then ((allPoints shape).filter
                (fun point => decide (select mask point = cell))).map (x :: ·) else [] := by
            by_cases same : x = index <;> simp [List.filter_map, Function.comp_def, select, same]
          rw [allPoints, List.filter_flatMap]
          simp only [inner]
          rw [select_one (List.range extent) index _ List.nodup_range (List.mem_range.2 bounded.1)]
          simpa [select] using ih remaining bounded.2

theorem eval_dimensions {mask : List Bool} {point : List Nat} (lengths : mask.length = point.length) :
    ((dimensions mask).map AffineExpr.dim).mapM (evalAffine point) =
      some ((select mask point).map Int.ofNat) := by
  induction mask generalizing point with
  | nil => simp [dimensions, select]
  | cons keep mask ih =>
    cases point with
    | nil => simp at lengths
    | cons index point =>
      have remaining : mask.length = point.length := by simpa using lengths
      have tail := ih remaining
      simp only [List.mapM_map, Function.comp_def, evalAffine] at tail
      cases keep <;>
        simp [dimensions, select, List.mapM_map, List.map_map,
          Function.comp_def, evalAffine, tail]

theorem output_coordinate {mask : List Bool} {shape point : List Nat}
    (lengths : mask.length = shape.length) (bounds : Idx.InBounds shape point) :
    W8A8IntegerTensor.coordinate (select mask shape) (outputMap mask) point = select mask point := by
  have pointLength : mask.length = point.length := lengths.trans bounds.1.symm
  have projected := select_inBounds lengths bounds
  simp only [W8A8IntegerTensor.coordinate, evalMap, outputMap, eval_dimensions pointLength,
    Option.bind_eq_bind, Option.bind_some, toIdx_ofNat projected, Option.getD_some]

theorem mapped_pair_count {mask : List Bool} {shape cell : List Nat}
    (lengths : mask.length = shape.length) (bounds : Idx.InBounds (select mask shape) cell)
    (left right : Idx → BitVec 8) :
    (W8A8IntegerTensor.pairs
      (W8A8IntegerTensor.coordinate (select mask shape) (outputMap mask)) left right
      (allPoints shape) cell).length = (select (mask.map not) shape).prod := by
  have same : (allPoints shape).filter (fun point => decide
      (cell = W8A8IntegerTensor.coordinate (select mask shape) (outputMap mask) point)) =
      (allPoints shape).filter (fun point => decide (select mask point = cell)) := by
    apply List.filter_congr
    intro point member
    rw [output_coordinate lengths ((mem_allPoints _ _).1 member)]
    simp [eq_comm]
  simpa only [W8A8IntegerTensor.pairs, same, List.length_map] using selected_count lengths bounds

theorem every_prefix_exact {mask : List Bool} {shape cell : List Nat}
    (lengths : mask.length = shape.length) (bounds : Idx.InBounds (select mask shape) cell)
    (bound : (select (mask.map not) shape).prod < 131072)
    (left right : Idx → BitVec 8) (count : Nat) :
    let pairs := W8A8IntegerTensor.pairs
      (W8A8IntegerTensor.coordinate (select mask shape) (outputMap mask)) left right (allPoints shape) cell
    (W8A8Accumulation.accumulate (pairs.take count)).toInt = W8A8Accumulation.dot (pairs.take count) := by
  apply W8A8Accumulation.every_prefix_exact
  rw [mapped_pair_count lengths bounds]
  omega

#print axioms select_inBounds
#print axioms selected_count
#print axioms output_coordinate
#print axioms mapped_pair_count
#print axioms every_prefix_exact

end MerlinVerify.Rules.W8A8ProjectionTrace
