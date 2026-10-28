/- Reassociation composition is checked algebraically for arbitrary runtime source extents. -/
import MerlinVerify.Rules.ReshapeIndex

namespace MerlinVerify.Rules.ReshapeGroups

open MerlinVerify MerlinVerify.Check

def merge (first next : List (List Nat)) : List (List Nat) :=
  next.map fun g => g.flatMap fun i => first.getD i []

theorem size_flatMap (gs : List (List Nat)) (f : Nat → Nat) :
    shapeSize (gs.flatten.map f) = shapeSize (gs.map fun g => shapeSize (g.map f)) := by
  induction gs with
  | nil => rfl
  | cons g gs ih =>
    simp only [List.flatten_cons, List.map_append, shapeSize_append, List.map_cons,
      shapeSize_cons, ih]

theorem valid_get {groups : List (List Nat)} {lo hi : List Nat}
    (h : Sem.Ops.Tensor.reassocValid groups lo hi = true) (i : Nat) (hb : i < hi.length) :
    shapeSize ((groups.getD i []).map fun j => lo.getD j 0) = hi.getD i 0 := by
  simp only [Sem.Ops.Tensor.reassocValid, Bool.and_eq_true, beq_iff_eq] at h
  have hgi : i < groups.length := h.1.1 ▸ hb
  have hm : (groups[i], hi[i]) ∈ groups.zip hi :=
    List.mem_iff_getElem.2 ⟨i, by simp only [List.length_zip]; omega, by simp⟩
  have hv := List.all_eq_true.1 h.2 _ hm
  simpa only [List.getD, List.getElem?_eq_getElem hgi, List.getElem?_eq_getElem hb,
    Option.getD_some, beq_iff_eq] using hv

theorem mem_bound {groups : List (List Nat)} {lo hi : List Nat}
    (h : Sem.Ops.Tensor.reassocValid groups lo hi = true) {g : List Nat} (hg : g ∈ groups)
    {i : Nat} (hi : i ∈ g) : i < lo.length := by
  simp only [Sem.Ops.Tensor.reassocValid, Bool.and_eq_true, beq_iff_eq] at h
  have hm : i ∈ groups.flatten := List.mem_flatten.2 ⟨g, hg, hi⟩
  rw [h.1.2] at hm
  exact List.mem_range.1 hm

theorem size_merge_group {first : List (List Nat)} {lo mid : List Nat}
    (h : Sem.Ops.Tensor.reassocValid first lo mid = true) (g : List Nat)
    (hb : ∀ i ∈ g, i < mid.length) :
    shapeSize ((g.flatMap fun i => first.getD i []).map fun j => lo.getD j 0) =
      shapeSize (g.map fun i => mid.getD i 0) := by
  change shapeSize (((g.map fun i => first.getD i []).flatten).map _) = _
  rw [size_flatMap, List.map_map]
  congr 1
  apply List.map_congr_left
  intro i hi
  exact valid_get h i (hb i hi)

/-- Two collapses compose by concatenating the first groups selected by each second group.
The same algebra, read backwards, composes two expansions. -/
theorem valid_merge {first next : List (List Nat)} {lo mid hi : List Nat}
    (hf : Sem.Ops.Tensor.reassocValid first lo mid = true)
    (hn : Sem.Ops.Tensor.reassocValid next mid hi = true) :
    Sem.Ops.Tensor.reassocValid (merge first next) lo hi = true := by
  have hf' := hf
  have hn' := hn
  simp only [Sem.Ops.Tensor.reassocValid, Bool.and_eq_true, beq_iff_eq] at hf' hn' ⊢
  refine ⟨⟨by simpa [merge] using hn'.1.1, ?_⟩, ?_⟩
  · change (next.flatMap fun g => g.flatMap fun i => first.getD i []) = _
    rw [← List.flatMap_assoc]
    have hflat : next.flatMap (fun g => g) = next.flatten := by
      clear hn hn'
      induction next with
      | nil => rfl
      | cons g gs ih => simp only [List.flatMap_cons, List.flatten_cons, ih]
    rw [hflat]
    rw [hn'.1.2, ← hf'.1.1]
    have hr : (List.range first.length).map (fun i => first.getD i []) = first := by
      apply List.ext_getElem (by simp)
      intro k hk hk'
      simp [List.getD, List.getElem?_eq_getElem hk']
    change ((List.range first.length).map (fun i => first.getD i [])).flatten = _
    rw [hr, hf'.1.2]
  · apply List.all_eq_true.2
    intro pair hp
    obtain ⟨k, hk, he⟩ := List.mem_iff_getElem.1 hp
    have hkm : k < min next.length hi.length := by simpa [merge] using hk
    have hkn : k < next.length := (Nat.lt_min.mp hkm).1
    have hkh : k < hi.length := (Nat.lt_min.mp hkm).2
    have hp' : pair = ((next[k]).flatMap fun i => first.getD i [], hi[k]) := by
      rw [← he]; simp [merge]
    rw [hp']
    simp only [beq_iff_eq]
    rw [size_merge_group hf _ (fun i hi => mem_bound hn (List.getElem_mem hkn) hi)]
    simpa only [List.getD, List.getElem?_eq_getElem hkn, List.getElem?_eq_getElem hkh,
      Option.getD_some] using valid_get hn k hkh

end MerlinVerify.Rules.ReshapeGroups
