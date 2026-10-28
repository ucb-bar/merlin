/- Independent coordinates and reduction order for batched linear. -/
import MerlinVerify.Rules.OverrankIntent
import MerlinVerify.Sem.Facts
namespace MerlinVerify.Rules.OverrankCoordinates
open MerlinVerify OverrankIntent Sem.Ops.Linalg

/-- Enumerating batch, N, K lexicographically visits each output's K values in increasing order. -/
theorem allPoints_append (a b : List Nat) :
    allPoints (a ++ b) = (allPoints a).flatMap (fun p => (allPoints b).map (p ++ ·)) := by
  induction a with
  | nil => simp [allPoints]
  | cons d ds ih => simp [allPoints, ih, List.flatMap_assoc, List.map_flatMap,
      List.map_map, List.flatMap_map, Function.comp_def, List.map_flatMap, List.cons_append]

theorem points_nodup (shape : List Nat) : (allPoints shape).Nodup := nodup_allPoints shape

/-- Projected dim expressions evaluate to the corresponding list slice. -/
theorem eval_lead (p : Idx) (n : Nat) (hn : n ≤ p.length) :
    ((List.range n).map AffineExpr.dim).mapM (evalAffine p) = some ((p.take n).map Int.ofNat) := by
  have hmap : ∀ i ∈ List.range n, evalAffine p (.dim i) = some (Int.ofNat (p[i]?.getD 0)) := by
    intro i hi
    have hi' : i < p.length := by have := List.mem_range.mp hi; omega
    simp [evalAffine, List.getElem?_eq_getElem hi']
  rw [List.mapM_map]
  rw [mapM_eq_some_map _ (fun i => Int.ofNat (p[i]?.getD 0)) _ (by
    intro i hi
    exact hmap i hi)]
  congr 1
  apply List.ext_getElem (by simp; omega)
  intro i hi hj
  simp at hi
  simp [List.getElem?_eq_getElem (by omega : i < p.length), hi]

/-- Projection indexes, independent of checked bounds. -/
def leftCoord (s : Shape) (p : Idx) := p.take s.batch.length ++ [p[s.batch.length+1]?.getD 0]
def rightCoord (s : Shape) (p : Idx) := [p[s.batch.length+1]?.getD 0, p[s.batch.length]?.getD 0]
def outCoord (s : Shape) (p : Idx) := p.take (s.batch.length+1)

/-- The output dimension list is a prefix of the point. -/
theorem eval_out (s : Shape) (p : Idx) (hp : Idx.InBounds s.ranges p) :
    evalMap (maps s)[2]! p = some ((outCoord s p).map Int.ofNat) := by
  have len : s.batch.length + 2 = p.length := by
    simpa [Shape.ranges, Shape.out] using hp.1.symm
  have list : (List.range s.batch.length).map AffineExpr.dim ++ [.dim s.batch.length] =
      (List.range (s.batch.length+1)).map AffineExpr.dim := by
    simp [List.range_succ]
  simpa [maps, evalMap, outCoord, list] using eval_lead p (s.batch.length+1) (by omega)

theorem eval_left (s : Shape) (p : Idx) (hp : Idx.InBounds s.ranges p) :
    evalMap (maps s)[0]! p = some ((leftCoord s p).map Int.ofNat) := by
  have len : s.batch.length + 2 = p.length := by
    simpa [Shape.ranges, Shape.out] using hp.1.symm
  have lead := eval_lead p s.batch.length (by omega)
  simp only [maps, List.getElem!_cons_zero, evalMap, List.mapM_append, lead,
    Option.bind_some, List.mapM_cons, evalAffine, List.mapM_nil, Option.pure_def]
  simp [leftCoord, List.map_append, List.getElem?_eq_getElem (by omega : s.batch.length+1 < p.length)]

theorem eval_right (s : Shape) (p : Idx) (hp : Idx.InBounds s.ranges p) :
    evalMap (maps s)[1]! p = some ((rightCoord s p).map Int.ofNat) := by
  have len : s.batch.length + 2 = p.length := by
    simpa [Shape.ranges, Shape.out] using hp.1.symm
  simp [maps, evalMap, evalAffine, rightCoord,
    List.getElem?_eq_getElem (by omega : s.batch.length+1 < p.length),
    List.getElem?_eq_getElem (by omega : s.batch.length < p.length)]


theorem toIdx_nat_eq : ∀ (shape q i : Idx),
    toIdx shape (q.map Int.ofNat) = some i → i = q
  | [], [], i, h => by simpa [toIdx] using h.symm
  | [], _ :: _, _, h => by simp [toIdx] at h
  | _ :: _, [], _, h => by simp [toIdx] at h
  | d :: ds, q :: qs, i, h => by
    rw [List.map_cons, Check.toIdx_cons] at h
    split at h
    · cases he : toIdx ds (qs.map Int.ofNat) with
      | none => simp [he] at h
      | some is =>
        have ih := toIdx_nat_eq ds qs is he
        simpa [he, ih] using h.symm
    · simp at h

/-- Read indices equal the independent formula, with bounds discharged by the checker. -/
theorem index_of_eval {s ranges : List Nat} {m : AffineMap} {p q : Idx}
    (bound : Check.mapInBounds ranges m s = true) (hp : Idx.InBounds ranges p)
    (he : evalMap m p = some (q.map Int.ofNat)) :
    (evalMap m p >>= toIdx s) = some q := by
  obtain ⟨i, hi, hib⟩ := Check.mapInBounds_sound bound hp
  have hi' := hi
  rw [he] at hi'
  have hq := toIdx_nat_eq s q i hi'
  exact hq ▸ hi

end MerlinVerify.Rules.OverrankCoordinates
