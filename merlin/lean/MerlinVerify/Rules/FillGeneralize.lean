/- Fill generalization retains the actual scalar body and obtains the loop rank from checked
runtime tensor types. No initializer contents or floating-point laws are assumed. -/
import MerlinVerify.Check.TypedRewrite
import MerlinVerify.Rules.Generalize

namespace MerlinVerify.Rules.FillGeneralize

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

def maps (rank : Nat) : List AffineMap :=
  [{ numDims := rank, numSyms := 0, results := [] }, identityMap rank]

def generic (s : Op) (rank : Nat) : Op :=
  let keep := fun (p : String × Attr) =>
    !(["indexing_maps", "iterator_types", "operandSegmentSizes"].contains p.1)
  let attrs := [
    ("indexing_maps", Attr.array ((maps rank).map Attr.affineMap)),
    ("iterator_types", Attr.array (List.replicate rank (.enum "linalg" "iterator_type" "parallel"))),
    ("operandSegmentSizes", Attr.denseArray (.int 32) [1, 1])] ++ s.attrs.filter keep
  .mk "linalg.generic" s.results s.operands (attrs.mergeSort (fun a b => a.1 ≤ b.1))
    (s.props.filter keep) s.regions s.successors

/-- Decode the constructed candidate again. The proof consumes only these checked facts. -/
def legal (s v : Op) (rank : Nat) : Prop :=
  s.name = "linalg.fill" ∧ dialectOf s.name = "linalg" ∧ dialectOf v.name = "linalg" ∧
  unmodelledAttr? s.blind = none ∧ unmodelledAttr? v.blind = none ∧
  v.name = "linalg.generic" ∧ v.results.length = s.results.length ∧
  attrDenseArray? s "operandSegmentSizes" = some [1, 1] ∧
  attrDenseArray? v "operandSegmentSizes" = some [1, 1] ∧
  attrAffineMaps? v "indexing_maps" = some (maps rank) ∧
  iterKinds? v = some (List.replicate rank .parallel)

instance (s v : Op) (rank : Nat) : Decidable (legal s v rank) := by
  unfold legal; infer_instance

theorem meaning {s v : Op} {rank : Nat} (h : legal s v rank)
    (ctx : Ctx) (call : CallSem) (value : RVal) (init : Tensor)
    (hrank : init.shape.length = rank) (regions : List (Ctx → RegionSem)) :
    OutLE (opSem ctx call v [value, .tensor init] regions)
      (opSem ctx call s [value, .tensor init] regions) := by
  obtain ⟨hn, hds, hdv, hms, hmv, hvn, hres, hss, hvs, hmaps, hiters⟩ := h
  rw [linalg_sem_of hmv hdv (.inl hvn), linalg_sem_of hms hds (.inr (.inl hn))]
  have hsplitS : splitInsOuts s.blind [value, .tensor init] = some ([value], [.tensor init]) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hss ⊢
    rw [hss]; rfl
  have hsplitV : splitInsOuts v.blind [value, .tensor init] = some ([value], [.tensor init]) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hvs ⊢
    rw [hvs]; rfl
  cases regions with
  | nil => exact .inr (.inl ⟨"malformed linalg.fill", by simp [Linalg.sem, Op.blind_name, hn, hsplitS]⟩)
  | cons body rest =>
    cases rest with
    | cons _ _ => exact .inr (.inl ⟨"malformed linalg.fill", by simp [Linalg.sem, Op.blind_name, hn, hsplitS]⟩)
    | nil =>
      simp only [Linalg.sem, Op.blind_name, hn, hvn, hsplitS, hsplitV,
        attrAffineMaps?_blind, iterKinds?_blind, hmaps, hiters, hrank, maps,
        Op.blind_results, List.length_map, hres, Outcome.bindOp_eq, Outcome.pure_eq]
      exact Generalize.count_agrees _ _ _ _

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
  · next valueTy shape elt =>
    dsimp only at he
    split at he
    · next hc =>
      cases he
      refine ⟨?_, rfl, rfl, rfl, ?_⟩
      · have hn : (generic s shape.length).name = "linalg.generic" := rfl
        simp [Op.isTerminator, terminatorNames, hn, hc.1]
      · intro ctx call args regions _ ha
        obtain _ | ⟨hv, ht⟩ := ha
        obtain _ | ⟨ht, hnil⟩ := ht
        cases hnil
        rename_i value init
        cases init with
        | scalar _ => cases ht
        | tensor t => exact meaning hc ctx call value t (by rw [ht.1]) regions
    · cases he
  · cases he

def expansion : TypedExpansion := ⟨expand, expand_sound⟩

end MerlinVerify.Rules.FillGeneralize
