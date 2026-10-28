/- Typed generalization of transpose, broadcast and variadic reduction. Layouts are derived
from the actual source attributes and checked operand types. Scalar regions remain unchanged. -/
import MerlinVerify.Check.TypedRewrite
import MerlinVerify.Rules.Generalize

namespace MerlinVerify.Rules.LayoutGeneralize

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

structure Layout where
  inputs : Nat
  outputs : Nat
  maps : List AffineMap
  iters : List IterKind

/-- Recompute the named operation's layout. No certificate supplies ranks or dimension lists. -/
def layout (s : Op) (tys : List VTy) : Option Layout :=
  if s.name = "linalg.transpose" then
    match tys, LinalgNamed.natArray? s "permutation" with
    | [_ti, tout], some perm =>
      let n := tout.shape.length
      if ¬(perm.length ≠ n ∨ (List.range n).any (fun p => !(perm.contains p))) then
        some ⟨1, 1, [transposeInMap n perm, identityMap n], LinalgNamed.parallel n⟩
      else none
    | _, _ => none
  else if s.name = "linalg.broadcast" then
    match tys, LinalgNamed.natArray? s "dimensions" with
    | [ti, tout], some dims =>
      let n := tout.shape.length
      let kept := (List.range n).filter (fun d => !(dims.contains d))
      if kept.length = ti.shape.length then
        some ⟨1, 1, [LinalgNamed.mapOf n kept, identityMap n], LinalgNamed.parallel n⟩
      else none
    | _, _ => none
  else if s.name = "linalg.reduce" then
    match reductionSplit s tys, LinalgNamed.natArray? s "dimensions" with
    | (ti :: ins, outs), some dims =>
      let n := ti.shape.length
      let kept := (List.range n).filter (fun d => !(dims.contains d))
      some ⟨ins.length + 1, outs.length,
        List.replicate (ins.length + 1) (identityMap n) ++
          List.replicate outs.length (LinalgNamed.mapOf n kept),
        (List.range n).map fun d => if dims.contains d then .reduction else .parallel⟩
    | _, _ => none
  else none

/-- The generic header is reconstructed; named-only properties are removed. The checker then
reads it back through the same attribute decoders used by the operation semantics. -/
def generic (s : Op) (l : Layout) : Op :=
  let keep := fun (p : String × Attr) =>
    !(["indexing_maps", "iterator_types", "operandSegmentSizes", "permutation", "dimensions"].contains p.1)
  let attrs := [
    ("indexing_maps", Attr.array (l.maps.map Attr.affineMap)),
    ("iterator_types", Attr.array (l.iters.map fun k => .enum "linalg" "iterator_type"
      (match k with | .parallel => "parallel" | .reduction => "reduction"))),
    ("operandSegmentSizes", Attr.denseArray (.int 32) [Int.ofNat l.inputs, Int.ofNat l.outputs])]
      ++ s.attrs.filter keep
  .mk "linalg.generic" s.results s.operands (attrs.mergeSort (fun a b => a.1 ≤ b.1))
    (s.props.filter keep) s.regions s.successors

def legal (s v : Op) (l : Layout) : Prop :=
  (s.name = "linalg.transpose" ∨ s.name = "linalg.broadcast" ∨ s.name = "linalg.reduce") ∧
  dialectOf s.name = "linalg" ∧ dialectOf v.name = "linalg" ∧
  unmodelledAttr? s.blind = none ∧ unmodelledAttr? v.blind = none ∧
  v.name = "linalg.generic" ∧ v.results.length = s.results.length ∧
  attrDenseArray? v "operandSegmentSizes" = some [Int.ofNat l.inputs, Int.ofNat l.outputs] ∧
  attrAffineMaps? v "indexing_maps" = some l.maps ∧ iterKinds? v = some l.iters

instance (s v : Op) (l : Layout) : Decidable (legal s v l) := by
  unfold legal; infer_instance

/-- The computed layout is the actual named source's loop nest, on its checked runtime domain. -/
theorem layout_run {ib : Nat} {s : Op} {tys : List VTy} {l : Layout}
    (hl : layout s tys = some l) {args : List RVal} (ha : HasTys ib args tys)
    (ctx : Ctx) (call : CallSem) (body : Ctx → RegionSem) :
    ∃ ins outs, args = ins ++ outs ∧ ins.length = l.inputs ∧ outs.length = l.outputs ∧
      LinalgNamed.sem ctx call s.blind args [body] =
        LinalgNamed.run ctx s.blind l.maps l.iters ins outs body := by
  unfold layout at hl
  split at hl
  · next hn =>
    split at hl
    · next ti tout perm hp =>
      dsimp only at hl
      split at hl
      · next hc =>
        cases hl
        obtain _ | ⟨hi, _ | ⟨ho, _ | _⟩⟩ := ha
        refine ⟨[_], [_], rfl, rfl, rfl, ?_⟩
        simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hp, shapeOf_typed ho,
          hc, ↓reduceIte, transposeInMap]
      · cases hl
    · cases hl
  · split at hl
    · next hn =>
      split at hl
      · next ti tout dims hd =>
        dsimp only at hl
        split at hl
        · next hc =>
          cases hl
          obtain _ | ⟨hi, _ | ⟨ho, _ | _⟩⟩ := ha
          refine ⟨[_], [_], rfl, rfl, rfl, ?_⟩
          simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hd,
            shapeOf_typed hi, shapeOf_typed ho, hc, ne_eq, not_true_eq_false, ↓reduceIte]
        · cases hl
      · cases hl
    · split at hl
      · next hn =>
        split at hl
        · next ti ts os dims hsplit hd =>
          cases hl
          obtain ⟨ins, outs, hio, hins, houts⟩ := reductionSplit_typed hsplit ha
          obtain _ | ⟨hi, ht⟩ := hins
          rename_i input rest
          refine ⟨input :: rest, outs, ?_, ?_, HasTys.length houts, ?_⟩
          · exact reductionSplit_append s args _ _ hio
          · simp only [List.length_cons, HasTys.length ht]
          · dsimp only [reductionSplit] at hio
            simp only [LinalgNamed.sem, Op.blind_name, hn, natArray?_blind, hd,
              splitInsOuts_blind, hio, shapeOf_typed hi,
              List.map_const', List.length_cons, HasTys.length ht, HasTys.length houts]
        · cases hl
      · cases hl

theorem generic_run {s v : Op} {l : Layout} (h : legal s v l)
    (ctx : Ctx) (call : CallSem) (ins outs : List RVal)
    (hi : ins.length = l.inputs) (ho : outs.length = l.outputs) (body : Ctx → RegionSem) :
    opSem ctx call v (ins ++ outs) [body] =
      (genericSem ctx l.maps l.iters ins outs body).bind fun rs =>
        if rs.length = s.results.length then .ok rs
        else .stuck "linalg.generic: result count ≠ output count (memref form unsupported)" := by
  obtain ⟨_, _, hd, _, hm, hn, hr, hs, hmaps, hiters⟩ := h
  rw [linalg_sem_of hm hd (.inl hn)]
  have hsplit : splitInsOuts v.blind (ins ++ outs) = some (ins, outs) := by
    simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hs ⊢
    rw [hs]
    simp [← hi, ← ho]
  simp only [Linalg.sem, Op.blind_name, hn, hsplit, attrAffineMaps?_blind, iterKinds?_blind,
    hmaps, hiters, Op.blind_results, List.length_map, hr, Outcome.bindOp_eq, Outcome.pure_eq]

theorem meaning {ib : Nat} {s v : Op} {tys : List VTy} {l : Layout}
    (hl : layout s tys = some l) (h : legal s v l)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (ha : HasTys ib args tys)
    (regions : List (Ctx → RegionSem)) :
    OutLE (opSem ctx call v args regions) (opSem ctx call s args regions) := by
  have hs := named_sem_of h.2.2.2.1 h.2.1
    (by rcases h.1 with hn | hn | hn <;> simp [hn]) (ctx := ctx) (call := call) (args := args) (regions := regions)
  cases regions with
  | nil => rw [hs]; exact .inr (.inl ⟨_, rfl⟩)
  | cons body rest =>
    cases rest with
    | cons _ _ => rw [hs]; exact .inr (.inl ⟨_, rfl⟩)
    | nil =>
      obtain ⟨ins, outs, rfl, hi, ho, hr⟩ := layout_run hl ha ctx _ body
      rw [hs, hr, generic_run h ctx call ins outs hi ho body]
      simp only [LinalgNamed.run, Op.blind_results, List.length_map]
      exact Generalize.count_agrees _ _ _ _

/-- Successful target execution transfers to the actual source. This stronger statement is
required for definedness; conditional refinement also permits unsupported source behavior. -/
theorem meaning_success {ib : Nat} {s v : Op} {tys : List VTy} {l : Layout}
    (hl : layout s tys = some l) (h : legal s v l)
    (ctx : Ctx) (call : CallSem) (args : List RVal) (ha : HasTys ib args tys)
    (body : Ctx → RegionSem) {rs : List RVal}
    (hok : opSem ctx call v args [body] = .ok rs) :
    opSem ctx call s args [body] = .ok rs := by
  obtain ⟨ins, outs, rfl, hi, ho, hr⟩ := layout_run hl ha ctx _ body
  rw [generic_run h ctx call ins outs hi ho body] at hok
  rw [named_sem_of h.2.2.2.1 h.2.1
    (by rcases h.1 with hn | hn | hn <;> simp [hn]), hr]
  simp only [LinalgNamed.run, Op.blind_results, List.length_map]
  exact Generalize.count_success hok

def expand (_ib : Nat) (s : Op) (tys : List VTy) : Option Op := do
  let l ← layout s tys
  let v := generic s l
  if legal s v l then some v else none

theorem expand_sound : ∀ ib s tys v, expand ib s tys = some v →
    v.isTerminator = s.isTerminator ∧ v.operands = s.operands ∧
    v.results = s.results ∧ v.regions = s.regions ∧
    ∀ ctx call args regions, ctx.indexBits = ib → HasTys ib args tys →
      OutLE (opSem ctx call v args regions) (opSem ctx call s args regions) := by
  intro ib s tys v he
  unfold expand at he
  cases hl : layout s tys with
  | none => simp [hl] at he
  | some l =>
    simp only [hl, bind, Option.bind] at he
    split at he
    · next hc =>
      cases he
      refine ⟨?_, rfl, rfl, rfl, ?_⟩
      · have hn : (generic s l).name = "linalg.generic" := rfl
        rcases hc.1 with hs | hs | hs <;> simp [Op.isTerminator, terminatorNames, hn, hs]
      · intro ctx call args regions _ ha
        exact meaning hl hc ctx call args ha regions
    · cases he

def expansion : TypedExpansion := ⟨expand, expand_sound⟩

end MerlinVerify.Rules.LayoutGeneralize
