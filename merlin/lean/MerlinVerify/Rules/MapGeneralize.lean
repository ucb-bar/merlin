/- Map generalization on an explicit typed runtime domain. No body arithmetic is changed. -/
import MerlinVerify.Check.TypedRewrite
import MerlinVerify.Rules.Generalize

namespace MerlinVerify.Rules.MapGeneralize

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

def generic (s : Op) (inputs rank : Nat) : Op :=
  let keep := fun (p : String × Attr) =>
    !(["indexing_maps", "iterator_types", "operandSegmentSizes"].contains p.1)
  let attrs := [
    ("indexing_maps", Attr.array ((List.replicate (inputs + 1) (identityMap rank)).map Attr.affineMap)),
    ("iterator_types", Attr.array (List.replicate rank (.enum "linalg" "iterator_type" "parallel"))),
    ("operandSegmentSizes", Attr.denseArray (.int 32) [Int.ofNat inputs, 1])] ++ s.attrs.filter keep
  .mk "linalg.generic" s.results s.operands (attrs.mergeSort (fun a b => a.1 ≤ b.1))
    (s.props.filter keep) s.regions s.successors

def legal (s v : Op) (inputs rank : Nat) : Prop :=
  s.name = "linalg.map" ∧ dialectOf s.name = "linalg" ∧ dialectOf v.name = "linalg" ∧
  unmodelledAttr? s.blind = none ∧ unmodelledAttr? v.blind = none ∧
  v.name = "linalg.generic" ∧ v.results.length = s.results.length ∧
  attrDenseArray? v "operandSegmentSizes" = some [Int.ofNat inputs, 1] ∧
  attrAffineMaps? v "indexing_maps" = some (List.replicate (inputs + 1) (identityMap rank)) ∧
  iterKinds? v = some (LinalgNamed.parallel rank)

instance (s v : Op) (inputs rank : Nat) : Decidable (legal s v inputs rank) := by
  unfold legal; infer_instance

theorem meaning {s v : Op} {ni n : Nat} (h : legal s v ni n)
    (ctx : Ctx) (call : CallSem) (ins : List RVal) (init : RVal)
    (hlen : ins.length = ni) (hrank : (shapeOf init).length = n)
    (regions : List (Ctx → RegionSem)) :
    OutLE (opSem ctx call v (ins ++ [init]) regions) (opSem ctx call s (ins ++ [init]) regions) := by
  obtain ⟨hn, hds, hdv, hms, hmv, hvn, hres, hseg, hmaps, hiters⟩ := h
  rw [named_sem_of hms hds (by simp [hn]), linalg_sem_of hmv hdv (.inl hvn)]
  cases regions with
  | nil => exact .inr (.inl ⟨_, rfl⟩)
  | cons body rest =>
    cases rest with
    | cons _ _ => exact .inr (.inl ⟨_, rfl⟩)
    | nil =>
      have hsplit : splitInsOuts v.blind (ins ++ [init]) = some (ins, [init]) := by
        simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hseg ⊢
        rw [hseg]
        simp [← hlen]
      simp only [LinalgNamed.sem, Op.blind_name, hn, List.reverse_append, List.reverse_cons,
        List.reverse_nil, List.nil_append, List.cons_append, List.reverse_reverse,
        List.map_const', List.length_append, List.length_singleton, hlen, hrank,
        LinalgNamed.run, Linalg.sem, hvn, hsplit, attrAffineMaps?_blind, iterKinds?_blind,
        hmaps, hiters, Op.blind_results, List.length_map, hres,
        Outcome.bindOp_eq, Outcome.pure_eq]
      exact Generalize.count_agrees _ _ _ _

def expand (_ib : Nat) (s : Op) (tys : List VTy) : Option Op :=
  match tys.reverse with
  | tout :: insRev =>
    let v := generic s insRev.length tout.shape.length
    if legal s v insRev.length tout.shape.length then some v else none
  | [] => none

theorem expand_sound : ∀ ib s tys v, expand ib s tys = some v →
    v.isTerminator = s.isTerminator ∧ v.operands = s.operands ∧
    v.results = s.results ∧ v.regions = s.regions ∧
    ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args tys →
      OutLE (opSem ctx call v args regions) (opSem ctx call s args regions) := by
  intro ib s tys v he
  unfold expand at he
  split at he
  · next tout insRev hrev =>
    dsimp only at he
    split at he
    · next hc =>
      cases he
      refine ⟨?_, rfl, rfl, rfl, ?_⟩
      · have hn : (generic s insRev.length tout.shape.length).name = "linalg.generic" := rfl
        simp [Op.isTerminator, terminatorNames, hn, hc.1]
      · intro ctx call args regions _ ha
        have ht := forall₂_reverse ha
        rw [hrev] at ht
        cases hv : args.reverse with
        | nil => rw [hv] at ht; cases ht
        | cons init xs =>
          rw [hv] at ht
          obtain _ | ⟨ho, hins⟩ := ht
          have hargs : args = xs.reverse ++ [init] := by
            simpa only [List.reverse_cons] using List.reverse_eq_iff.1 hv
          rw [hargs]
          exact meaning hc ctx call xs.reverse init
            (by simpa only [List.length_reverse] using List.Forall₂.length_eq hins)
            (by rw [shapeOf_typed ho]) regions
    · cases he
  · cases he

def expansion : TypedExpansion := ⟨expand, expand_sound⟩

end MerlinVerify.Rules.MapGeneralize
