/- Runtime consequences of checked structured-loop geometry, shared by safety, definedness
and rewrite proofs. These lemmas do not assume that a loop body terminates or returns clean data. -/
import MerlinVerify.Check.RulesLinalg

namespace MerlinVerify.Check

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

theorem mapM_tensors_total (g : RVal → Outcome Tensor) (h : ∀ t, g (.tensor t) = .ok t) :
    ∀ ts : List Tensor, (ts.map RVal.tensor).mapM g = .ok ts
  | [] => rfl
  | t :: ts => by
    simp only [List.map_cons, List.mapM_cons, Outcome.bindOp_eq, h, mapM_tensors_total g h ts,
      Outcome.bind_ok, Outcome.pure_eq]

/-- Static bounds checks imply that the semantic evaluator accepts the actual runtime shapes. -/
theorem structured_runtime {ib : Nat} {maps : List AffineMap} {n : Nat} {tys : List VTy}
    {ranges : List Nat} {args : List RVal} (hs : StructuredOk maps n tys ranges)
    (ht : HasTys ib args tys) :
    maps.length = args.length ∧
    maps.any (fun m => m.numDims ≠ n || m.numSyms ≠ 0) = false ∧
    (maps.zip (args.map shapeOf)).any (fun p => p.1.results.length ≠ p.2.length) = false ∧
    loopRanges maps (args.map shapeOf) n = .ok ranges := by
  have hshapes := shapes_typed ht
  refine ⟨hs.len.trans (HasTys.length ht).symm, ?_, ?_, ?_⟩
  · rw [List.any_eq_false]
    intro m hm
    simp [(hs.dims m hm).1, (hs.dims m hm).2]
  · rw [hshapes, List.zip_map_right, List.any_map, List.any_eq_false]
    intro p hp
    have hin := List.forall₂_iff_zip.1 hs.inBounds |>.2 hp
    simp only [mapInBounds, Bool.and_eq_true, beq_iff_eq] at hin
    simp [Function.comp, hin.1]
  · rw [hshapes]
    exact mapM_ofOption _ _ _ _ hs.loops

/-- Expose a checked loop's exact execution without strengthening the body's semantics. -/
theorem genericSem_execute {ib : Nat} {maps : List AffineMap} {iters : List IterKind}
    {inTys outTys : List VTy} {ranges : List Nat} {ins : List RVal} {outs : List Tensor}
    (hs : StructuredOk maps iters.length (inTys ++ outTys) ranges)
    (hi : HasTys ib ins inTys) (ho : HasTys ib (outs.map RVal.tensor) outTys)
    (ctx : Ctx) (body : Ctx → RegionSem) :
    genericSem ctx maps iters ins (outs.map RVal.tensor) body =
      ((allPoints ranges).foldlM
        (loopStep ctx ins (maps.take ins.length) (maps.drop ins.length) body) outs).bind
        (fun final => .ok (final.map fun t => RVal.tensor t.canonize)) := by
  obtain ⟨hL, hD, hR, hLR⟩ := structured_runtime hs (List.rel_append hi ho)
  unfold genericSem
  simp only [hL, ne_eq, not_true_eq_false, ↓reduceIte, hD, Bool.false_eq_true, hR,
    Outcome.bindOp_eq]
  rw [mapM_tensors_total _ (by intro _; rfl) outs, Outcome.bind_ok, hLR, Outcome.bind_ok]
  rfl

end MerlinVerify.Check
