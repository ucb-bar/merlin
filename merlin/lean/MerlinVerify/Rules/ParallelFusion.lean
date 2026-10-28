/- Exact full-overwrite fusion for identity-indexed parallel tensor loops.
The actual body executions, including captures and poison, remain explicit premises. -/
import MerlinVerify.Sem.Facts

namespace MerlinVerify.Rules.ParallelFusion

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

/-- An identity map finds the actual extent at each dimension of its operand. -/
theorem identity_dimension {shape : List Nat} {d : Nat} (hd : d < shape.length) :
    (((identityMap shape.length).results.zip shape).find?
      (fun p => p.1 = .dim d)).map (·.2) = some shape[d] := by
  have hf : ((identityMap shape.length).results.zip shape).find?
      (fun p => p.1 = .dim d) = some (.dim d, shape[d]) := by
    apply List.find?_eq_some_iff_getElem.mpr
    refine ⟨by simp, d, by simpa [identityMap] using hd, ?_, ?_⟩
    · simp [identityMap, List.getElem_zip, List.getElem_map, List.getElem_range]
    · intro j hj
      have hne : j ≠ d := Nat.ne_of_lt hj
      simp [identityMap, List.getElem_zip, List.getElem_map, List.getElem_range, hne]
  rw [hf]
  rfl

/-- Equal actual operand shapes determine all identity-indexed loop ranges, even for
rank zero or zero extents. There is always at least the output operand. -/
theorem identity_ranges (shape : List Nat) (n : Nat) :
    loopRanges (List.replicate (n + 1) (identityMap shape.length))
      (List.replicate (n + 1) shape) shape.length = .ok shape := by
  have hr (d : Nat) (hd : d < shape.length) :
      loopRange (List.replicate (n + 1) (identityMap shape.length))
        (List.replicate (n + 1) shape) d = some shape[d] := by
    simp only [loopRange, List.zip_replicate']
    rw [List.findSome?_replicate_of_pos (by omega)]
    exact identity_dimension hd
  have hm : ∀ ds : List Nat, (∀ d ∈ ds, d < shape.length) →
      ds.mapM (fun d => Outcome.ofOption s!"loop {d} range not determined by any operand"
        (loopRange (List.replicate (n + 1) (identityMap shape.length))
          (List.replicate (n + 1) shape) d)) =
        .ok (ds.map fun d => shape[d]!) := by
    intro ds
    induction ds with
    | nil => intro _; rfl
    | cons d ds ih =>
      intro h
      have hd := h d (by simp)
      rw [List.mapM_cons, hr d hd, ih (fun j hj => h j (by simp [hj]))]
      simp [Outcome.ofOption, List.getElem?_eq_getElem hd]
  unfold loopRanges
  rw [hm (List.range shape.length) (by simp)]
  congr 1
  apply List.ext_getElem (by simp)
  intro i hi hi'
  simp [List.getElem?_eq_getElem hi']

/-- A total pointwise body on all actual in-bounds points determines the exact canonical
result, independently of the old destination's contents. -/
theorem identity_generic (ctx : Ctx) (shape : List Nat) (ins : List Tensor) (init : Tensor)
    (body : Ctx → RegionSem) (F : Idx → Val)
    (hs : ∀ t ∈ ins ++ [init], t.shape = shape)
    (hF : ∀ pt, Idx.InBounds shape pt →
      bodyAt ctx (ins.map RVal.tensor)
        (List.replicate ins.length (identityMap shape.length))
        [identityMap shape.length] [init] body pt = .ok [F pt]) :
    genericSem ctx (List.replicate (ins.length + 1) (identityMap shape.length))
      (List.replicate shape.length .parallel) (ins.map RVal.tensor) [.tensor init] body =
      .ok [.tensor (Tensor.canon shape init.elt F)] := by
  have hi : init.shape = shape := hs init (by simp)
  have hshapes : (ins.map RVal.tensor ++ [RVal.tensor init]).map shapeOf =
      List.replicate (ins.length + 1) shape := by
    have hu : (ins ++ [init]).map (fun t => t.shape) =
        List.replicate (ins ++ [init]).length shape :=
      List.map_eq_replicate_iff.mpr hs
    simpa [List.map_append, List.map_map, Function.comp_def, shapeOf] using hu
  apply genericSem_parallel1 ctx _ _ _ init body shape F
  · simp
  · simp [identityMap]
  · rw [hshapes, List.zip_replicate']
    simp [identityMap]
  · rw [List.length_replicate, hshapes]
    exact identity_ranges shape ins.length
  · exact hi
  · simp
  · simpa using hF

/-- Full-outcome equality for producer/consumer fusion. The consumer runs on the actual
canonical producer result; the fused destination may be fresh and contain arbitrary poison.
The three body premises must be established by a separate syntax/SSA checker. -/
theorem identity_overwrite_fusion (ctx : Ctx) (shape : List Nat) (ins : List Tensor)
    (pinit cinit finit : Tensor) (producer consumer fused : Ctx → RegionSem)
    (P Q : Idx → Val)
    (hs : ∀ t ∈ ins ++ [pinit, cinit, finit], t.shape = shape)
    (he : finit.elt = cinit.elt)
    (hP : ∀ pt, Idx.InBounds shape pt →
      bodyAt ctx (ins.map RVal.tensor)
        (List.replicate ins.length (identityMap shape.length))
        [identityMap shape.length] [pinit] producer pt = .ok [P pt])
    (hC : ∀ pt, Idx.InBounds shape pt →
      bodyAt ctx [.tensor (Tensor.canon shape pinit.elt P)]
        [identityMap shape.length] [identityMap shape.length] [cinit] consumer pt =
        .ok [Q pt])
    (hF : ∀ pt, Idx.InBounds shape pt →
      bodyAt ctx (ins.map RVal.tensor)
        (List.replicate ins.length (identityMap shape.length))
        [identityMap shape.length] [finit] fused pt = .ok [Q pt]) :
    ((genericSem ctx (List.replicate (ins.length + 1) (identityMap shape.length))
      (List.replicate shape.length .parallel) (ins.map RVal.tensor) [.tensor pinit]
      producer).bind fun produced =>
        genericSem ctx [identityMap shape.length, identityMap shape.length]
          (List.replicate shape.length .parallel) produced [.tensor cinit] consumer) =
      genericSem ctx (List.replicate (ins.length + 1) (identityMap shape.length))
        (List.replicate shape.length .parallel) (ins.map RVal.tensor) [.tensor finit] fused := by
  have hp : ∀ t ∈ ins ++ [pinit], t.shape = shape := by
    intro t ht
    exact hs t (by simp only [List.mem_append, List.mem_cons] at ht ⊢; tauto)
  have hf : ∀ t ∈ ins ++ [finit], t.shape = shape := by
    intro t ht
    exact hs t (by simp only [List.mem_append, List.mem_cons] at ht ⊢; tauto)
  have hc : ∀ t ∈ [Tensor.canon shape pinit.elt P] ++ [cinit], t.shape = shape := by
    intro t ht
    simp only [List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at ht
    rcases ht with ht | ht
    · rw [ht]; rfl
    · rw [ht]; exact hs cinit (by simp)
  rw [identity_generic ctx shape ins pinit producer P hp hP, Outcome.bind_ok]
  have hconsumer := identity_generic ctx shape [Tensor.canon shape pinit.elt P]
    cinit consumer Q hc hC
  have hfused := identity_generic ctx shape ins finit fused Q hf hF
  rw [he] at hfused
  simpa using hconsumer.trans hfused.symm

end MerlinVerify.Rules.ParallelFusion
