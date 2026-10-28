/- Named contractions expanded to the loop nest specified by their existing semantics.
No arithmetic is reordered or replaced; the original body, including casts, is retained. -/
import MerlinVerify.Rules.OneOp
import MerlinVerify.Check.RulesLinalg

namespace MerlinVerify.Rules.Generalize

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

def maps (s : Op) : List AffineMap :=
  if s.name = "linalg.matmul" then
    LinalgNamed.mapsOr s [LinalgNamed.mapOf 3 [0, 2], LinalgNamed.mapOf 3 [2, 1], LinalgNamed.mapOf 3 [0, 1]]
  else LinalgNamed.mapsOr s [LinalgNamed.mapOf 4 [0, 1, 3], LinalgNamed.mapOf 4 [0, 3, 2], LinalgNamed.mapOf 4 [0, 1, 2]]

def iters (s : Op) : List IterKind :=
  if s.name = "linalg.matmul" then [.parallel, .parallel, .reduction]
  else [.parallel, .parallel, .parallel, .reduction]

/-- Construct a candidate in the authoritative exporter's sorted attribute representation.
The checker below decodes the candidate again, rather than trusting this constructor. -/
def generic (s : Op) : Op :=
  let keep := fun (p : String × Attr) => !(["indexing_maps", "iterator_types", "operandSegmentSizes", "cast"].contains p.1)
  let attrs := [("indexing_maps", Attr.array ((maps s).map Attr.affineMap)),
    ("iterator_types", Attr.array ((iters s).map fun k => .enum "linalg" "iterator_type"
      (match k with | .parallel => "parallel" | .reduction => "reduction"))),
    ("operandSegmentSizes", Attr.denseArray (.int 32) [2, 1])] ++ s.attrs.filter keep
  .mk "linalg.generic" s.results s.operands (attrs.mergeSort (fun a b => a.1 ≤ b.1))
    (s.props.filter keep) s.regions s.successors

/-- Conditions consumed by the semantic proof, all recomputed from the actual operations. -/
def legal (s v : Op) : Prop :=
  (s.name = "linalg.matmul" ∨ s.name = "linalg.batch_matmul") ∧
  dialectOf s.name = "linalg" ∧ dialectOf v.name = "linalg" ∧
  unmodelledAttr? s.blind = none ∧ unmodelledAttr? v.blind = none ∧
  attrDenseArray? s "operandSegmentSizes" = some [2, 1] ∧
  attrDenseArray? v "operandSegmentSizes" = some [2, 1] ∧
  attrAffineMaps? v "indexing_maps" = some (maps s) ∧ iterKinds? v = some (iters s) ∧
  v.name = "linalg.generic" ∧ v.results.length = s.results.length

instance (s v : Op) : Decidable (legal s v) := by unfold legal; infer_instance

theorem count_agrees (x : Outcome (List RVal)) (n : Nat) (sourceReason targetReason : String) :
    OutLE (x.bind fun rs => if rs.length = n then .ok rs else .stuck targetReason)
      (x.bind fun rs => if rs.length = n then .ok rs else .stuck sourceReason) := by
  cases x with
  | ub => exact Or.inl rfl
  | stuck r => exact Or.inr (Or.inl ⟨r, rfl⟩)
  | ok rs =>
    simp only [Outcome.bind_ok]
    split
    · exact Or.inr (Or.inr rfl)
    · exact Or.inr (Or.inl ⟨_, rfl⟩)

theorem meaning {s v : Op} (h : legal s v) (ctx : Ctx) (call : CallSem) (args : List RVal)
    (regions : List (Ctx → RegionSem)) : OutLE (opSem ctx call v args regions) (opSem ctx call s args regions) := by
  obtain ⟨hn, hds, hdv, hms, hmv, hss, hvs, hmaps, hiters, hvn, hres⟩ := h
  rw [linalg_sem_of hmv hdv (.inl hvn)]
  rw [named_sem_of hms hds
    (by rcases hn with h | h <;> simp [h])]
  cases regions with
  | nil => exact Or.inr (Or.inl ⟨_, rfl⟩)
  | cons body rest =>
    cases rest with
    | cons _ _ => exact Or.inr (Or.inl ⟨_, rfl⟩)
    | nil =>
      have hsplit : splitInsOuts v.blind args = splitInsOuts s.blind args := by
        simp only [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hss hvs ⊢
        rw [hss, hvs]
      rcases hn with hn | hn
      all_goals
        simp only [LinalgNamed.sem, Op.blind_name, hn]
        cases hs : splitInsOuts s.blind args with
        | none => exact Or.inr (Or.inl ⟨_, rfl⟩)
        | some io =>
          rcases io with ⟨ins, outs⟩
          cases ins with
          | nil => exact Or.inr (Or.inl ⟨_, rfl⟩)
          | cons a rest =>
            cases rest with
            | nil => exact Or.inr (Or.inl ⟨_, rfl⟩)
            | cons b rest =>
              cases rest with
              | cons _ _ => exact Or.inr (Or.inl ⟨_, rfl⟩)
              | nil =>
                cases outs with
                | nil => exact Or.inr (Or.inl ⟨_, rfl⟩)
                | cons c rest =>
                  cases rest with
                  | cons _ _ => exact Or.inr (Or.inl ⟨_, rfl⟩)
                  | nil =>
                    simp only [Linalg.sem, Op.blind_name, hvn, hsplit, hs, attrAffineMaps?_blind,
                      iterKinds?_blind, hmaps, hiters, LinalgNamed.run, Op.blind_results, List.length_map, hres]
                    have hm : maps s = LinalgNamed.mapsOr s.blind
                        (if s.name = "linalg.matmul" then
                          [LinalgNamed.mapOf 3 [0, 2], LinalgNamed.mapOf 3 [2, 1], LinalgNamed.mapOf 3 [0, 1]]
                        else [LinalgNamed.mapOf 4 [0, 1, 3], LinalgNamed.mapOf 4 [0, 3, 2], LinalgNamed.mapOf 4 [0, 1, 2]]) := by
                      simp only [maps, mapsOr_blind]; split <;> rfl
                    simp only [hn, ↓reduceIte] at hm
                    simp only [hm, iters, hn, ↓reduceIte, Outcome.bindOp_eq, Outcome.pure_eq]
                    exact count_agrees _ _ _ _

theorem count_success {x : Outcome (List RVal)} {n : Nat} {rs : List RVal}
    {sourceReason targetReason : String}
    (h : (x.bind fun ys => if ys.length = n then .ok ys else .stuck targetReason) = .ok rs) :
    (x.bind fun ys => if ys.length = n then .ok ys else .stuck sourceReason) = .ok rs := by
  cases x with
  | ub => cases h
  | stuck _ => cases h
  | ok ys =>
    simp only [Outcome.bind_ok] at h ⊢
    split at h
    · next hc => rw [ite_eq_left hc]; exact h
    · cases h

/-- On the contraction's actual three operands, successful generic execution also establishes
successful source execution. This prevents using an undefined source to claim definedness. -/
theorem meaning_success {s v : Op} (h : legal s v) (ctx : Ctx) (call : CallSem)
    {args : List RVal} (hlen : args.length = 3) (body : Ctx → RegionSem) {rs : List RVal}
    (hok : opSem ctx call v args [body] = .ok rs) : opSem ctx call s args [body] = .ok rs := by
  obtain ⟨hn, hds, hdv, hms, hmv, hss, hvs, hmaps, hiters, hvn, hres⟩ := h
  rw [List.eq_getElem_of_length_eq_three args hlen] at hok ⊢
  rw [linalg_sem_of hmv hdv (.inl hvn)] at hok
  rw [named_sem_of hms hds (by rcases hn with h | h <;> simp [h])]
  have hsplitS : splitInsOuts s.blind args = some (args.take 2, args.drop 2) := by
    simp [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hss ⊢
    rw [hss]
    simp [hlen]
  have hsplitV : splitInsOuts v.blind args = some (args.take 2, args.drop 2) := by
    simp [splitInsOuts, attrDenseArray?, Op.blind_attr?] at hvs ⊢
    rw [hvs]
    simp [hlen]
  rw [List.eq_getElem_of_length_eq_three args hlen] at hsplitS hsplitV
  rcases hn with hn | hn
  all_goals
    simp only [Linalg.sem, Op.blind_name, hvn, hsplitV, attrAffineMaps?_blind, iterKinds?_blind,
      hmaps, hiters, Op.blind_results, List.length_map, hres, Outcome.bindOp_eq, Outcome.pure_eq] at hok
    simp only [LinalgNamed.sem, Op.blind_name, hn, hsplitS, LinalgNamed.run, Op.blind_results,
      List.length_map]
    simp only [maps, iters, hn, ↓reduceIte, mapsOr_blind] at hok ⊢
    exact count_success hok

def expand (s : Op) : Option Op :=
  let v := generic s
  if legal s v ∧ calleeOf v.blind = none then some v else none

theorem expand_sound : ∀ s v, expand s = some v →
    s.isTerminator = false ∧ v.isTerminator = false ∧ v.operands = s.operands ∧
    v.results = s.results ∧ v.regions = s.regions ∧ calleeOf v.blind = none ∧
    ∀ ctx call args regions, OutLE (opSem ctx call v args regions) (opSem ctx call s args regions) := by
  intro s v h
  dsimp only [expand] at h
  split at h
  · next hc =>
    cases h
    refine ⟨?_, rfl, rfl, rfl, rfl, hc.2, meaning hc.1⟩
    rcases hc.1.1 with hn | hn <;> simp [Op.isTerminator, terminatorNames, hn]
  · cases h

def expansion : Expansion := ⟨expand, expand_sound⟩
def rule : Rule := expansion.rule "generalize-contraction"

end MerlinVerify.Rules.Generalize
