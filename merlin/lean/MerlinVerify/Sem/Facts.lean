/-
Facts about the op-semantics building blocks, used by rule proofs.

* `Idx.InBounds` by cases; `mem_allPoints` (the iteration space is exactly the in-bounds indices)
  and `nodup_allPoints` (each point is visited once).
* Identity indexing maps read and write exactly the element at the current point
  (`readAt_identity`, `writeAt_identity`).
-/
import MerlinVerify.Sem.Ops.Linalg
import MerlinVerify.Sem.OutcomeLaws
import Mathlib.Data.List.Nodup

namespace MerlinVerify

open MerlinVerify.Sem.Ops.Linalg

/-! ### In-bounds indices and the iteration space -/

theorem Idx.inBounds_nil_iff (i : Idx) : Idx.InBounds [] i ↔ i = [] := by
  constructor
  · rintro ⟨h, -⟩; exact List.length_eq_zero_iff.1 h
  · rintro rfl; exact ⟨rfl, fun d h₁ => absurd h₁ (Nat.not_lt_zero _)⟩

theorem Idx.inBounds_cons_nil (s : Nat) (ss : List Nat) : ¬ Idx.InBounds (s :: ss) [] := by
  rintro ⟨h, -⟩; simp at h

theorem Idx.inBounds_cons_cons (s : Nat) (ss : List Nat) (x : Nat) (xs : Idx) :
    Idx.InBounds (s :: ss) (x :: xs) ↔ x < s ∧ Idx.InBounds ss xs := by
  constructor
  · rintro ⟨hl, h⟩
    have h0 := h 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    refine ⟨h0, by simpa using hl, fun d h₁ h₂ => ?_⟩
    have hd := h (d + 1) (by simpa using h₁) (by simpa using h₂)
    simpa only [List.getElem_cons_succ] using hd
  · rintro ⟨hx, hl, h⟩
    refine ⟨by simpa using hl, fun d h₁ h₂ => ?_⟩
    cases d with
    | zero => simpa using hx
    | succ d => simpa using h d (by simpa using h₁) (by simpa using h₂)

theorem mem_allPoints : ∀ (s : List Nat) (i : Idx), i ∈ allPoints s ↔ Idx.InBounds s i
  | [], i => by simp [allPoints, Idx.inBounds_nil_iff]
  | d :: ds, [] => by simp [allPoints, Idx.inBounds_cons_nil]
  | d :: ds, x :: xs => by
    simp only [allPoints, List.mem_flatMap, List.mem_range, List.mem_map, List.cons.injEq,
      Idx.inBounds_cons_cons]
    constructor
    · rintro ⟨y, hy, rest, hrest, rfl, rfl⟩
      exact ⟨hy, (mem_allPoints ds rest).1 hrest⟩
    · rintro ⟨hx, hxs⟩
      exact ⟨x, hx, xs, (mem_allPoints ds xs).2 hxs, rfl, rfl⟩

theorem nodup_allPoints : ∀ (s : List Nat), (allPoints s).Nodup
  | [] => by simp [allPoints]
  | d :: ds => by
    simp only [allPoints]
    rw [List.nodup_flatMap]
    refine ⟨fun x _ => (nodup_allPoints ds).map (fun a b h => List.cons_injective h), ?_⟩
    refine List.Pairwise.imp (R := (· ≠ ·)) (fun {a b} hab => ?_) (List.nodup_range)
    rw [Function.onFun, List.disjoint_left]
    intro p hpa hpb
    obtain ⟨r, -, rfl⟩ := List.mem_map.1 hpa
    obtain ⟨r', -, h⟩ := List.mem_map.1 hpb
    exact hab (List.cons.inj h).1.symm

/-! ### Identity indexing maps -/

theorem mapM_eq_some_map {β γ : Type} (g : β → Option γ) (h : β → γ) :
    ∀ (l : List β), (∀ b ∈ l, g b = some (h b)) → l.mapM g = some (l.map h)
  | [], _ => rfl
  | b :: l, hb => by
    rw [List.mapM_cons, hb b (List.mem_cons_self ..),
      mapM_eq_some_map g h l (fun b' h' => hb b' (List.mem_cons_of_mem _ h'))]
    rfl

theorem evalMap_identity (pt : Idx) : evalMap (identityMap pt.length) pt = some (pt.map Int.ofNat) := by
  unfold evalMap identityMap
  rw [List.mapM_map]
  rw [mapM_eq_some_map _ (fun p => Int.ofNat (pt[p]?.getD 0)) _ (fun p hp => by
    have hp' : p < pt.length := List.mem_range.1 hp
    simp [Function.comp, evalAffine, List.getElem?_eq_getElem hp'])]
  congr 1
  apply List.ext_getElem (by simp)
  intro k h₁ h₂
  have hk : k < pt.length := by simpa using h₂
  simp [List.getElem?_eq_getElem hk]

theorem toIdx_ofNat {s : List Nat} {pt : Idx} (h : Idx.InBounds s pt) :
    toIdx s (pt.map Int.ofNat) = some pt := by
  obtain ⟨hl, hb⟩ := h
  unfold toIdx
  rw [ite_eq_right (by simp [hl])]
  rw [mapM_eq_some_map _ (fun p => p.1.toNat) _ (fun p hp => by
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp
    simp only [List.getElem_zip, List.getElem_map]
    have hk1 : k < pt.length := by simpa using (List.lt_length_left_of_zip hk)
    have hk2 : k < s.length := List.lt_length_right_of_zip hk
    have := hb k hk1 hk2
    simp only [Int.ofNat_eq_natCast, Int.toNat_natCast]
    rw [ite_eq_left ⟨by omega, by exact_mod_cast this⟩])]
  congr 1
  apply List.ext_getElem (by simp [hl])
  intro k h₁ h₂
  simp

theorem readAt_identity {t : Tensor} {pt : Idx} (h : Idx.InBounds t.shape pt) :
    readAt (.tensor t) (identityMap t.shape.length) pt = .ok (t.get pt) := by
  have hl : pt.length = t.shape.length := h.1
  simp only [readAt, ← hl, evalMap_identity, Option.bind_eq_bind, Option.bind_some, toIdx_ofNat h]

theorem writeAt_identity {t : Tensor} {pt : Idx} (v : Val) (h : Idx.InBounds t.shape pt) :
    writeAt t (identityMap t.shape.length) pt v = .ok (t.set pt v) := by
  have hl : pt.length = t.shape.length := h.1
  simp only [writeAt, ← hl, evalMap_identity, Option.bind_eq_bind, Option.bind_some, toIdx_ofNat h]

theorem Outcome.mapM_length {α β : Type} (f : α → Outcome β) :
    ∀ (l : List α) (r : List β), Outcome.mapM f l = .ok r → r.length = l.length
  | [], r, h => by simp only [Outcome.mapM] at h; cases h; rfl
  | a :: l, r, h => by
    simp only [Outcome.mapM] at h
    cases hfa : f a with
    | ub => rw [hfa] at h; cases h
    | stuck _ => rw [hfa] at h; cases h
    | ok b =>
      rw [hfa] at h
      simp only [Outcome.bind] at h
      cases hl : Outcome.mapM f l with
      | ub => rw [hl] at h; cases h
      | stuck _ => rw [hl] at h; cases h
      | ok bs =>
        rw [hl] at h
        simp only [Outcome.ok.injEq] at h
        subst h
        simp [Outcome.mapM_length f l bs hl]

/-- `List.mapM` in the `Outcome` monad preserves length. -/
theorem listMapM_length {α β : Type} (f : α → Outcome β) :
    ∀ (l : List α) (r : List β), l.mapM f = .ok r → r.length = l.length
  | [], r, h => by simp only [List.mapM_nil, Outcome.pure_eq] at h; cases h; rfl
  | a :: l, r, h => by
    simp only [List.mapM_cons, Outcome.bindOp_eq] at h
    cases hfa : f a with
    | ub => rw [hfa] at h; cases h
    | stuck _ => rw [hfa] at h; cases h
    | ok b =>
      rw [hfa, Outcome.bind_ok] at h
      cases hl : l.mapM f with
      | ub => rw [hl] at h; cases h
      | stuck _ => rw [hl] at h; cases h
      | ok bs =>
        rw [hl, Outcome.bind_ok, Outcome.pure_eq, Outcome.ok.injEq] at h
        subst h
        simp [listMapM_length f l bs hl]

theorem loopRanges_length {maps : List AffineMap} {shapes : List (List Nat)} {n : Nat}
    {ranges : List Nat} (h : loopRanges maps shapes n = .ok ranges) : ranges.length = n := by
  simpa using listMapM_length _ _ _ h

/-! ### Parallel structured ops with an identity-indexed output -/

/-- `t` with the elements at the points `done` replaced by `F`. -/
def Tensor.updAt (t : Tensor) (F : Idx → Val) (done : List Idx) : Tensor :=
  { t with get := fun i => if i ∈ done then F i else t.get i }

theorem Tensor.updAt_nil (t : Tensor) (F : Idx → Val) : t.updAt F [] = t := by
  cases t; simp [Tensor.updAt]

theorem bodyAt_single_congr {ctx : Ctx} {ins : List RVal} {inMaps : List AffineMap}
    {body : Ctx → RegionSem} {c c' : Tensor} {pt : Idx} (hs : c.shape = c'.shape)
    (hin : Idx.InBounds c.shape pt) (hg : c.get pt = c'.get pt) :
    bodyAt ctx ins inMaps [identityMap c.shape.length] [c] body pt =
      bodyAt ctx ins inMaps [identityMap c'.shape.length] [c'] body pt := by
  have hin' : Idx.InBounds c'.shape pt := hs ▸ hin
  simp only [bodyAt, List.zip_cons_cons, List.zip_nil_right, List.mapM_cons, List.mapM_nil,
    readAt_identity hin, readAt_identity hin', hg, List.length_singleton]

/-- The loop of a structured op whose single output is indexed by the identity map, over a
duplicate-free list of in-bounds points at each of which the body yields `F p` (computed from the
ORIGINAL output element): the output becomes `F` on those points and is unchanged elsewhere. -/
theorem fold_parallel1 (ctx : Ctx) (ins : List RVal) (inMaps : List AffineMap)
    (body : Ctx → RegionSem) (t : Tensor) (F : Idx → Val) :
    ∀ (rest done : List Idx), (done ++ rest).Nodup → (∀ p ∈ rest, Idx.InBounds t.shape p) →
      (∀ p ∈ rest, bodyAt ctx ins inMaps [identityMap t.shape.length] [t] body p = .ok [F p]) →
      rest.foldlM (loopStep ctx ins inMaps [identityMap t.shape.length] body) [t.updAt F done] =
        .ok [t.updAt F (done ++ rest)]
  | [], done, _, _, _ => by simp
  | p :: rest, done, hnd, hin, hF => by
    have hp : p ∉ done := by
      intro hm
      exact (List.nodup_append.1 hnd).2.2 p hm p (List.mem_cons_self ..) rfl
    have hinp : Idx.InBounds t.shape p := hin p (List.mem_cons_self ..)
    have hcur : (t.updAt F done).get p = t.get p := by simp [Tensor.updAt, hp]
    have hbody : bodyAt ctx ins inMaps [identityMap t.shape.length] [t.updAt F done] body p =
        .ok [F p] := by
      have hc : bodyAt ctx ins inMaps [identityMap t.shape.length] [t.updAt F done] body p =
          bodyAt ctx ins inMaps [identityMap t.shape.length] [t] body p :=
        bodyAt_single_congr (c := t.updAt F done) (c' := t) rfl hinp hcur
      rw [hc]
      exact hF p (List.mem_cons_self ..)
    have hw : writeAt (t.updAt F done) (identityMap t.shape.length) p (F p) =
        .ok ((t.updAt F done).set p (F p)) :=
      writeAt_identity (t := t.updAt F done) (F p) hinp
    have hset : (t.updAt F done).set p (F p) = t.updAt F (done ++ [p]) := by
      cases t
      simp only [Tensor.updAt, Tensor.set, Tensor.mk.injEq, true_and]
      funext i
      by_cases hi : i = p
      · subst hi; simp
      · simp [hi]
    have hstep : loopStep ctx ins inMaps [identityMap t.shape.length] body [t.updAt F done] p =
        .ok [t.updAt F (done ++ [p])] := by
      unfold loopStep
      rw [hbody]
      simp only [Outcome.bindOp_eq, Outcome.bind_ok, List.zip_cons_cons, List.zip_nil_right, List.mapM_cons,
        List.mapM_nil, hw, Outcome.pure_eq, hset]
    rw [List.foldlM_cons, hstep]
    simp only [bind, Outcome.bind]
    have := fold_parallel1 ctx ins inMaps body t F rest (done ++ [p])
      (by simpa using hnd) (fun q hq => hin q (List.mem_cons_of_mem _ hq))
      (fun q hq => hF q (List.mem_cons_of_mem _ hq))
    simpa using this

/-- **Parallel structured ops.** A structured op whose single output is indexed by the identity map
over its whole iteration space (`ranges` = the output shape) computes the canonical tensor of the
body's per-point values `F`, provided the body yields `F p` at every point (from the inputs read
through their maps and the original output element). This covers elementwise ops, broadcasts and
transposes, whatever the iterator kinds say. -/
theorem genericSem_parallel1 (ctx : Ctx) (maps : List AffineMap) (iters : List IterKind)
    (ins : List RVal) (t : Tensor) (body : Ctx → RegionSem) (ranges : List Nat) (F : Idx → Val)
    (hlen : maps.length = ins.length + 1)
    (hdims : maps.any (fun m => m.numDims ≠ iters.length || m.numSyms ≠ 0) = false)
    (hrank : (maps.zip ((ins ++ [RVal.tensor t]).map shapeOf)).any
      (fun p => p.1.results.length ≠ p.2.length) = false)
    (hranges : loopRanges maps ((ins ++ [RVal.tensor t]).map shapeOf) iters.length = .ok ranges)
    (hshape : t.shape = ranges)
    (hout : maps.drop ins.length = [identityMap ranges.length])
    (hF : ∀ p, Idx.InBounds ranges p →
      bodyAt ctx ins (maps.take ins.length) [identityMap ranges.length] [t] body p = .ok [F p]) :
    genericSem ctx maps iters ins [.tensor t] body = .ok [.tensor (Tensor.canon ranges t.elt F)] := by
  subst hshape
  have hfold := fold_parallel1 ctx ins (maps.take ins.length) body t F (allPoints t.shape) []
    (by simpa using nodup_allPoints t.shape) (fun p hp => (mem_allPoints _ _).1 hp)
    (fun p hp => hF p ((mem_allPoints _ _).1 hp))
  rw [Tensor.updAt_nil, List.nil_append] at hfold
  unfold genericSem
  have hl : ¬ maps.length ≠ (ins ++ [RVal.tensor t]).length := by simp [hlen]
  rw [ite_eq_right_iff.2 (fun h => absurd h hl)]
  simp only [hdims, Bool.false_eq_true, ↓reduceIte, hrank, List.mapM_cons, List.mapM_nil,
    Outcome.bindOp_eq, Outcome.bind_ok, Outcome.pure_eq, hranges, hout, hfold, List.map_cons,
    List.map_nil, Outcome.ok.injEq, List.cons.injEq, RVal.tensor.injEq, and_true]
  unfold Tensor.canonize
  exact Tensor.canon_ext fun i hi => by
    have hi' : Idx.InBounds t.shape i := hi
    simp [Tensor.updAt, (mem_allPoints _ _).2 hi']

end MerlinVerify
