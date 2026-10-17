/- Copy generalization retains the actual scalar body, including any supported casts.
The loop rank comes from checked runtime destination types, never an unchecked annotation. -/
import MerlinVerify.Rules.MapGeneralize

namespace MerlinVerify.Rules.CopyGeneralize

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

def generic (s : Op) (rank : Nat) : Op := MapGeneralize.generic s 1 rank

def legal (s v : Op) (rank : Nat) : Prop :=
  s.name = "linalg.copy" ∧ dialectOf s.name = "linalg" ∧ dialectOf v.name = "linalg" ∧
  unmodelledAttr? s.blind = none ∧ unmodelledAttr? v.blind = none ∧
  v.name = "linalg.generic" ∧ v.results.length = s.results.length ∧
  attrDenseArray? s "operandSegmentSizes" = some [1, 1] ∧
  attrDenseArray? v "operandSegmentSizes" = some [1, 1] ∧
  attrAffineMaps? v "indexing_maps" = some [identityMap rank, identityMap rank] ∧
  iterKinds? v = some (LinalgNamed.parallel rank)

instance (s v : Op) (rank : Nat) : Decidable (legal s v rank) := by
  unfold legal; infer_instance

theorem meaning {s v : Op} {rank : Nat} (h : legal s v rank)
    (ctx : Ctx) (call : CallSem) (input : RVal) (init : Tensor)
    (hrank : init.shape.length = rank) (regions : List (Ctx → RegionSem)) :
    OutLE (opSem ctx call v [input, .tensor init] regions)
      (opSem ctx call s [input, .tensor init] regions) := by
  obtain ⟨hn, hds, hdv, hms, hmv, hvn, hres, hss, hvs, hmaps, hiters⟩ := h
  rw [named_sem_of hms hds (by simp [hn]), linalg_sem_of hmv hdv (.inl hvn)]
  have hsplitS : splitInsOuts s.blind [input, .tensor init] = some ([input], [.tensor init]) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hss ⊢
    rw [hss]; rfl
  have hsplitV : splitInsOuts v.blind [input, .tensor init] = some ([input], [.tensor init]) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hvs ⊢
    rw [hvs]; rfl
  cases regions with
  | nil => exact .inr (.inl ⟨_, rfl⟩)
  | cons body rest =>
    cases rest with
    | cons _ _ => exact .inr (.inl ⟨_, rfl⟩)
    | nil =>
      simp only [LinalgNamed.sem, Linalg.sem, Op.blind_name, hn, hvn, hsplitS, hsplitV,
        shapeOf, hrank, LinalgNamed.run, attrAffineMaps?_blind, iterKinds?_blind, hmaps,
        hiters, Op.blind_results, List.length_map, hres, Outcome.bindOp_eq, Outcome.pure_eq]
      exact Generalize.count_agrees _ _ _ _

/-- Successful generic execution also proves the actual named source succeeds. Conditional
refinement alone cannot justify source-definedness because it permits unsupported sources. -/
theorem meaning_success {s v : Op} {rank : Nat} (h : legal s v rank)
    (ctx : Ctx) (call : CallSem) (input : RVal) (init : Tensor)
    (hrank : init.shape.length = rank) (body : Ctx → RegionSem) {rs : List RVal}
    (hok : opSem ctx call v [input, .tensor init] [body] = .ok rs) :
    opSem ctx call s [input, .tensor init] [body] = .ok rs := by
  obtain ⟨hn, hds, hdv, hms, hmv, hvn, hres, hss, hvs, hmaps, hiters⟩ := h
  rw [named_sem_of hms hds (by simp [hn])]
  rw [linalg_sem_of hmv hdv (.inl hvn)] at hok
  have hsplitS : splitInsOuts s.blind [input, .tensor init] = some ([input], [.tensor init]) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hss ⊢
    rw [hss]; rfl
  have hsplitV : splitInsOuts v.blind [input, .tensor init] = some ([input], [.tensor init]) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hvs ⊢
    rw [hvs]; rfl
  simp only [Linalg.sem, Op.blind_name, hvn, hsplitV, attrAffineMaps?_blind, iterKinds?_blind,
    hmaps, hiters, Op.blind_results, List.length_map, hres, Outcome.bindOp_eq, Outcome.pure_eq] at hok
  simp only [LinalgNamed.sem, Op.blind_name, hn, hsplitS, shapeOf, hrank, LinalgNamed.run,
    Op.blind_results, List.length_map]
  exact Generalize.count_success hok

def expand (_ib : Nat) (s : Op) (tys : List VTy) : Option Op :=
  match tys with
  | [_, .tensor shape _] =>
    let v := generic s shape.length
    if legal s v shape.length then some v else none
  | _ => none

theorem expand_sound : ∀ ib s tys v, expand ib s tys = some v →
    v.isTerminator = s.isTerminator ∧ v.operands = s.operands ∧
    v.results = s.results ∧ v.regions = s.regions ∧
    ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args tys →
      OutLE (opSem ctx call v args regions) (opSem ctx call s args regions) := by
  intro ib s tys v he
  unfold expand at he
  split at he
  · next inputTy shape elt =>
    dsimp only at he
    split at he
    · next hc =>
      cases he
      refine ⟨?_, rfl, rfl, rfl, ?_⟩
      · have hn : (generic s shape.length).name = "linalg.generic" := rfl
        simp [Op.isTerminator, terminatorNames, hn, hc.1]
      · intro ctx call args regions _ ha
        obtain _ | ⟨hi, ht⟩ := ha
        obtain _ | ⟨ht, hnil⟩ := ht
        cases hnil
        rename_i input init
        cases init with
        | scalar _ => cases ht
        | tensor t => exact meaning hc ctx call input t (by rw [ht.1]) regions
    · cases he
  · cases he

def expansion : TypedExpansion := ⟨expand, expand_sound⟩

end MerlinVerify.Rules.CopyGeneralize
