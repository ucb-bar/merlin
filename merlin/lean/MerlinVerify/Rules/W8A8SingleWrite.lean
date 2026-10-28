/- Ordered single-write structured execution. The checker must establish that
the actual output-map coordinates are duplicate-free and cover the output.
The scalar body may read the original initialized destination. No reassociation
or replacement of the executed loop is used. -/
import MerlinVerify.Check.DefinedContextLoops

namespace MerlinVerify.Rules.W8A8SingleWrite

open MerlinVerify Sem.Ops.Linalg

def coordinate (shape : List Nat) (map : AffineMap) (point : Idx) : Idx :=
  (evalMap map point >>= toIdx shape).getD []

def State (original current : Tensor) (done : List Idx) (post : Val → Prop) : Prop :=
  current.shape = original.shape ∧ current.elt = original.elt ∧
    ∀ cell, Idx.InBounds original.shape cell →
      (cell ∈ done → post (current.get cell)) ∧
      (cell ∉ done → current.get cell = original.get cell)

theorem state_initial (original : Tensor) (post : Val → Prop) : State original original [] post := by
  refine ⟨rfl, rfl, ?_⟩
  intro cell _bounds
  exact ⟨fun impossible => False.elim (List.not_mem_nil impossible), fun _ => rfl⟩

theorem state_write {original current : Tensor} {done : List Idx} {post : Val → Prop}
    (state : State original current done post) (cell : Idx) {value : Val} (valid : post value) :
    State original (current.set cell value) (done ++ [cell]) post := by
  refine ⟨state.1, state.2.1, ?_⟩
  intro other bounds
  by_cases same : other = cell
  · subst other
    exact ⟨fun _ => by simpa [Tensor.set] using valid, fun absent => False.elim (absent (by simp))⟩
  · simp only [Tensor.set, same, ↓reduceIte]
    constructor
    · intro member
      have previous : other ∈ done := by simpa [same] using member
      exact (state.2.2 other bounds).1 previous
    · intro absent
      exact (state.2.2 other bounds).2 (fun member => absent (List.mem_append_left _ member))

theorem body_original {ctx : Ctx} {ins : List RVal} {maps : List AffineMap} {output : AffineMap}
    {body : Ctx → RegionSem} {original current : Tensor} {point cell : Idx}
    (shape : current.shape = original.shape)
    (mapped : (evalMap output point >>= toIdx original.shape) = some cell)
    (unchanged : current.get cell = original.get cell) :
    bodyAt ctx ins maps [output] [current] body point = bodyAt ctx ins maps [output] [original] body point := by
  simp only [bodyAt, List.zip_cons_cons, List.zip_nil_right, List.mapM_cons, List.mapM_nil,
    readAt, shape, mapped, unchanged, List.length_cons, List.length_nil]

theorem fold_once (ctx : Ctx) (ins : List RVal) (maps : List AffineMap) (output : AffineMap)
    (body : Ctx → RegionSem) (original : Tensor) (post : Val → Prop) :
    ∀ (rest : List Idx) (done : List Idx) (current : Tensor),
      (done ++ rest.map (coordinate original.shape output)).Nodup →
      (∀ point ∈ rest, ∃ cell, (evalMap output point >>= toIdx original.shape) = some cell ∧
        Idx.InBounds original.shape cell) →
      (∀ point ∈ rest, ∃ value, bodyAt ctx ins maps [output] [original] body point = .ok [value] ∧ post value) →
      State original current done post →
      ∃ final, rest.foldlM (loopStep ctx ins maps [output] body) [current] = .ok [final] ∧
        State original final (done ++ rest.map (coordinate original.shape output)) post
  | [], done, current, _unique, _mapped, _total, state =>
    ⟨current, rfl, by simpa using state⟩
  | point :: rest, done, current, unique, mapped, total, state => by
    obtain ⟨cell, mapping, bounds⟩ := mapped point (List.mem_cons_self ..)
    have coord : coordinate original.shape output point = cell := by simp [coordinate, mapping]
    have fresh : cell ∉ done := by
      have disjoint := (List.nodup_append.1 unique).2.2
      intro member
      exact disjoint cell member cell (by simp [coord]) rfl
    have unchanged := (state.2.2 cell bounds).2 fresh
    obtain ⟨value, executed, valid⟩ := total point (List.mem_cons_self ..)
    have actual : bodyAt ctx ins maps [output] [current] body point = .ok [value] :=
      (body_original state.1 mapping unchanged).trans executed
    have written : writeAt current output point value = .ok (current.set cell value) := by
      simp only [writeAt, state.1, mapping]
    have step : loopStep ctx ins maps [output] body [current] point = .ok [current.set cell value] := by
      simp only [loopStep, actual, Outcome.bindOp_eq, Outcome.bind_ok, List.zip_cons_cons,
        List.zip_nil_right, List.mapM_cons, List.mapM_nil, written, Outcome.pure_eq]
    have nextUnique : ((done ++ [cell]) ++ rest.map (coordinate original.shape output)).Nodup := by
      simpa [coord] using unique
    obtain ⟨final, remaining, finalState⟩ := fold_once ctx ins maps output body original post rest
      (done ++ [cell]) (current.set cell value) nextUnique
      (fun point member => mapped point (List.mem_cons_of_mem _ member))
      (fun point member => total point (List.mem_cons_of_mem _ member)) (state_write state cell valid)
    refine ⟨final, ?_, ?_⟩
    · rw [List.foldlM_cons, step]
      simpa only [bind, Outcome.bind] using remaining
    · simpa [coord] using finalState

/-- A source-derived finite loop can change the destination's numerical fact:
each coordinate is read exactly once from the original tensor before being written. -/
theorem execute (ctx : Ctx) (ins : List RVal) (maps : List AffineMap) (output : AffineMap)
    (body : Ctx → RegionSem) (original : Tensor) (points : List Idx) (post : Val → Prop)
    (unique : (points.map (coordinate original.shape output)).Nodup)
    (mapped : ∀ point ∈ points, ∃ cell, (evalMap output point >>= toIdx original.shape) = some cell ∧
      Idx.InBounds original.shape cell)
    (covers : ∀ cell, Idx.InBounds original.shape cell → cell ∈ points.map (coordinate original.shape output))
    (total : ∀ point ∈ points, ∃ value,
      bodyAt ctx ins maps [output] [original] body point = .ok [value] ∧ post value) :
    ∃ final, points.foldlM (loopStep ctx ins maps [output] body) [original] = .ok [final] ∧
      final.shape = original.shape ∧ final.elt = original.elt ∧
      ∀ cell, Idx.InBounds original.shape cell → post (final.get cell) := by
  obtain ⟨final, executed, state⟩ := fold_once ctx ins maps output body original post points [] original
    (by simpa using unique) mapped total (state_initial original post)
  exact ⟨final, executed, state.1, state.2.1, fun cell bounds =>
    (state.2.2 cell bounds).1 (by simpa using covers cell bounds)⟩

#print axioms fold_once
#print axioms execute

end MerlinVerify.Rules.W8A8SingleWrite
