/- Standard matmul geometry used by Merlin's W8A8 contraction. Bounds are computed from operand
shapes; the reduction length is K, independently of the number of output cells. -/
import MerlinVerify.Rules.W8A8IntegerTensor

namespace MerlinVerify.Rules.W8A8Matmul

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

def leftMap : AffineMap := { numDims := 3, numSyms := 0, results := [.dim 0, .dim 2] }
def rightMap : AffineMap := { numDims := 3, numSyms := 0, results := [.dim 2, .dim 1] }
def outMap : AffineMap := { numDims := 3, numSyms := 0, results := [.dim 0, .dim 1] }
def maps : List AffineMap := [leftMap, rightMap, outMap]
def iters : List IterKind := [.parallel, .parallel, .reduction]

theorem point_components {m n k : Nat} {point : Idx} (hp : Idx.InBounds [m, n, k] point) :
    ∃ i j t, point = [i, j, t] ∧ i < m ∧ j < n ∧ t < k := by
  cases point with
  | nil => simp [Idx.InBounds] at hp
  | cons i rest =>
    have h := (Idx.inBounds_cons_cons ..).1 hp
    cases rest with
    | nil => simp [Idx.InBounds] at h
    | cons j rest =>
      have h' := (Idx.inBounds_cons_cons ..).1 h.2
      cases rest with
      | nil => simp [Idx.InBounds] at h'
      | cons t rest =>
        have h'' := (Idx.inBounds_cons_cons ..).1 h'.2
        have hr := (Idx.inBounds_nil_iff rest).1 h''.2
        subst rest
        exact ⟨i, j, t, rfl, h.1, h'.1, h''.1⟩

theorem output_coordinate {m n k : Nat} {point : Idx}
    (hp : Idx.InBounds [m, n, k] point) :
    W8A8IntegerTensor.coordinate [m, n] outMap point = point.take 2 := by
  obtain ⟨i, j, t, rfl, hi, hj, _⟩ := point_components hp
  have hb : Idx.InBounds [m, n] [i, j] := by
    simp [Idx.inBounds_cons_cons, Idx.inBounds_nil_iff, hi, hj]
  change ((some ([i, j].map Int.ofNat)) >>= toIdx [m, n]).getD [] = [i, j]
  simp only [Option.bind_eq_bind, Option.bind_some, toIdx_ofNat hb]
  rfl

private theorem select_one (xs : List Nat) (i : Nat) (f : Nat → List Idx)
    (hn : xs.Nodup) (hi : i ∈ xs) :
    xs.flatMap (fun x => if x = i then f x else []) = f i := by
  induction xs with
  | nil => cases hi
  | cons x xs ih =>
    have hn' := List.nodup_cons.1 hn
    by_cases hx : x = i
    · subst x
      have hempty : xs.flatMap (fun x => if x = i then f x else []) = [] := by
        rw [List.flatMap_eq_nil_iff]
        intro x hx
        have hxi : x ≠ i := by intro he; subst x; exact hn'.1 hx
        simp [hxi]
      simp [hempty]
    · have hi' : i ∈ xs := (List.mem_cons.1 hi).resolve_left (Ne.symm hx)
      simpa [hx] using ih hn'.2 hi'

/-- Fixing an output row and column leaves exactly the K reduction points, in their original
order. This prevents an output-size-dependent or caller-supplied accumulation bound. -/
theorem selected_points (m n k i j : Nat) (hi : i < m) (hj : j < n) :
    (allPoints [m, n, k]).filter (fun p => decide ([i, j] = p.take 2)) =
      (List.range k).map (fun t => [i, j, t]) := by
  have inner (x y : Nat) :
      ((List.range k).map (fun t => [x, y, t])).filter
        (fun p => decide ([i, j] = p.take 2)) =
      if x = i ∧ y = j then (List.range k).map (fun t => [x, y, t]) else [] := by
    by_cases hx : x = i <;> by_cases hy : y = j
    all_goals simp [List.filter_map, Function.comp_def, hx, hy, eq_comm (a := i) (b := x),
      eq_comm (a := j) (b := y)]
  have hall : allPoints [m, n, k] = (List.range m).flatMap (fun x => (List.range n).flatMap
      (fun y => (List.range k).map (fun t => [x, y, t]))) := by
    simp [allPoints, List.map_flatMap, ← List.map_eq_flatMap, Function.comp_def]
  rw [hall]
  simp only [List.filter_flatMap, inner]
  have outer (x : Nat) :
      (List.range n).flatMap (fun y =>
        if x = i ∧ y = j then (List.range k).map (fun t => [x, y, t]) else []) =
      if x = i then (List.range k).map (fun t => [x, j, t]) else [] := by
    by_cases hx : x = i
    · simp only [hx, true_and]
      exact select_one (List.range n) j _ (List.nodup_range) (List.mem_range.2 hj)
    · simp [hx]
  simp only [outer]
  exact select_one (List.range m) i _ (List.nodup_range) (List.mem_range.2 hi)

theorem pair_count (m n k : Nat) (left right : Idx → BitVec 8) (cell : Idx)
    (hc : Idx.InBounds [m, n] cell) :
    (W8A8IntegerTensor.pairs (fun p => p.take 2) left right (allPoints [m, n, k]) cell).length = k := by
  cases cell with
  | nil => simp [Idx.InBounds] at hc
  | cons i rest =>
    have h := (Idx.inBounds_cons_cons ..).1 hc
    cases rest with
    | nil => simp [Idx.InBounds] at h
    | cons j rest =>
      have h' := (Idx.inBounds_cons_cons ..).1 h.2
      have hr := (Idx.inBounds_nil_iff rest).1 h'.2
      subst rest
      simp [W8A8IntegerTensor.pairs, selected_points m n k i j h.1 h'.1]

theorem mapped_pair_count (m n k : Nat) (left right : Idx → BitVec 8) (cell : Idx)
    (hc : Idx.InBounds [m, n] cell) :
    (W8A8IntegerTensor.pairs (W8A8IntegerTensor.coordinate [m, n] outMap)
      left right (allPoints [m, n, k]) cell).length = k := by
  have hf : (allPoints [m, n, k]).filter
      (fun p => decide (cell = W8A8IntegerTensor.coordinate [m, n] outMap p)) =
      (allPoints [m, n, k]).filter (fun p => decide (cell = p.take 2)) := by
    apply List.filter_congr
    intro p hp
    rw [output_coordinate ((mem_allPoints _ _).1 hp)]
  simpa only [W8A8IntegerTensor.pairs, hf] using pair_count m n k left right cell hc

/-- Checked K bounds establish the exact signed result of every zero-initialized output cell,
even when captured bytes include -128. No bound is inferred from symmetric-quantization hints. -/
theorem exact_cell (m n k : Nat) (left right : Idx → BitVec 8) (cell : Idx)
    (hc : Idx.InBounds [m, n] cell) (bound : (k : Int) * 16384 < 2147483648) :
    (W8A8TensorTrace.accumulate (W8A8IntegerTensor.coordinate [m, n] outMap)
      (W8A8IntegerTensor.step (some ∘ left) (some ∘ right)) (fun _ => some 0)
      (allPoints [m, n, k]) cell).map BitVec.toInt =
      some (W8A8Accumulation.dot (W8A8IntegerTensor.pairs
        (W8A8IntegerTensor.coordinate [m, n] outMap) left right (allPoints [m, n, k]) cell)) := by
  apply W8A8IntegerTensor.exact_cell
  rwa [mapped_pair_count m n k left right cell hc]

#print axioms output_coordinate
#print axioms selected_points
#print axioms pair_count
#print axioms mapped_pair_count
#print axioms exact_cell

end MerlinVerify.Rules.W8A8Matmul
