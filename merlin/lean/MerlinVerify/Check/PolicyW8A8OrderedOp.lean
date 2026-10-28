/- Recompute floating contraction forms from the actual operation. Named operations retain
their actual region; malformed explicit maps never silently fall back to default maps. -/
import MerlinVerify.Check.PolicyW8A8OrderedRefined

namespace MerlinVerify.Check.PolicyW8A8OrderedOp

open MerlinVerify Rules Sem.Ops Sem.Ops.Linalg

def matmulMaps : List AffineMap :=
  [LinalgNamed.mapOf 3 [0, 2], LinalgNamed.mapOf 3 [2, 1], LinalgNamed.mapOf 3 [0, 1]]

def batchMaps : List AffineMap :=
  [LinalgNamed.mapOf 4 [0, 1, 3], LinalgNamed.mapOf 4 [0, 3, 2], LinalgNamed.mapOf 4 [0, 1, 2]]

def namedMaps (op : Op) (defaults : List AffineMap) : Option (List AffineMap) :=
  if op.attr? "indexing_maps" = none then some defaults else attrAffineMaps? op "indexing_maps"

theorem namedMaps_sound {op : Op} {defaults maps : List AffineMap}
    (checked : namedMaps op defaults = some maps) : LinalgNamed.mapsOr op defaults = maps := by
  unfold namedMaps at checked
  split at checked
  · next absent =>
    cases checked
    simp [LinalgNamed.mapsOr, attrAffineMaps?, absent]
  · next _ => simp only [LinalgNamed.mapsOr, checked, Option.getD_some]

structure Form where
  maps : List AffineMap
  iterators : List IterKind

def inspect (op : Op) : Option Form :=
  if op.name = "linalg.matmul" then do
    let maps ← namedMaps op matmulMaps
    pure ⟨maps, [.parallel, .parallel, .reduction]⟩
  else if op.name = "linalg.batch_matmul" then do
    let maps ← namedMaps op batchMaps
    pure ⟨maps, [.parallel, .parallel, .parallel, .reduction]⟩
  else if op.name = "linalg.generic" then do
    let maps ← attrAffineMaps? op "indexing_maps"
    let iterators ← iterKinds? op
    pure ⟨maps, iterators⟩
  else none

def Describes (op : Op) (form : Form) : Prop :=
  (op.name = "linalg.matmul" ∧ LinalgNamed.mapsOr op matmulMaps = form.maps ∧
    form.iterators = [.parallel, .parallel, .reduction]) ∨
  (op.name = "linalg.batch_matmul" ∧ LinalgNamed.mapsOr op batchMaps = form.maps ∧
    form.iterators = [.parallel, .parallel, .parallel, .reduction]) ∨
  (op.name = "linalg.generic" ∧ attrAffineMaps? op "indexing_maps" = some form.maps ∧
    iterKinds? op = some form.iterators)

theorem inspect_sound {op : Op} {form : Form} (checked : inspect op = some form) : Describes op form := by
  unfold inspect at checked
  split at checked
  · next named =>
    cases maps : namedMaps op matmulMaps with
    | none => simp [maps] at checked
    | some actual =>
      simp only [maps, bind, Option.bind, pure, Option.pure_def, Option.some.injEq] at checked
      subst form
      exact .inl ⟨named, namedMaps_sound maps, rfl⟩
  · split at checked
    · next named =>
      cases maps : namedMaps op batchMaps with
      | none => simp [maps] at checked
      | some actual =>
        simp only [maps, bind, Option.bind, pure, Option.pure_def, Option.some.injEq] at checked
        subst form
        exact .inr (.inl ⟨named, namedMaps_sound maps, rfl⟩)
    · split at checked
      · next named =>
        cases maps : attrAffineMaps? op "indexing_maps" with
        | none => simp [maps] at checked
        | some actual =>
          cases iters : iterKinds? op with
          | none => simp [maps, iters] at checked
          | some kinds =>
            simp only [maps, iters, bind, Option.bind, pure, Option.pure_def, Option.some.injEq] at checked
            subst form
            exact .inr (.inr ⟨named, maps, iters⟩)
      · cases checked

/-- Lift an already proved complete generic execution through the actual source operation.
Operand segments, the actual region and the result arity are all part of this statement. -/
theorem execute {ctx : Ctx} {call : CallSem} {env : Env} {op : Op} {form : Form}
    {args : List RVal} {left right initial result : RVal} {body : Region}
    (modelled : unmodelledAttr? op.blind = none) (dialect : dialectOf op.name = "linalg")
    (selected : inspect op = some form) (regions : op.regions = [body])
    (splitArgs : splitInsOuts op args = some ([left, right], [initial]))
    (arity : op.results.length = 1)
    (executed : genericSem ctx form.maps form.iterators [left, right] [initial]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env body) = .ok [result]) :
    W8A8SourceSemantics.opSem ctx call op args
      (evalRegions W8A8SourceSemantics.opSem call env op.regions) = .ok [result] := by
  have count : op.blind.results.length = 1 := by simpa only [Op.blind_results, List.length_map] using arity
  rcases inspect_sound selected with ⟨named, maps, iters⟩ | ⟨named, maps, iters⟩ | ⟨named, maps, iters⟩
  all_goals
    have ordinary : W8A8SourceSemantics.isSource op.name = false := by
      simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]
    rw [W8A8SourceSemantics.unchanged ordinary]
  · rw [named_sem_of modelled dialect (by simp [named])]
    simp only [regions, evalRegions, LinalgNamed.sem, Op.blind_name, named, splitInsOuts_blind,
      splitArgs, LinalgNamed.run, LinalgNamed.mapsOr, attrAffineMaps?_blind] at ⊢
    change ((genericSem ctx (LinalgNamed.mapsOr op matmulMaps) _ [left, right] [initial] _).bind _) = _
    rw [maps, ← iters, executed]
    simp [Outcome.bind_ok, arity]
  · rw [named_sem_of modelled dialect (by simp [named])]
    simp only [regions, evalRegions, LinalgNamed.sem, Op.blind_name, named, splitInsOuts_blind,
      splitArgs, LinalgNamed.run, LinalgNamed.mapsOr, attrAffineMaps?_blind] at ⊢
    change ((genericSem ctx (LinalgNamed.mapsOr op batchMaps) _ [left, right] [initial] _).bind _) = _
    rw [maps, ← iters, executed]
    simp [Outcome.bind_ok, arity]
  · rw [linalg_sem_of modelled dialect (.inl named)]
    simp only [regions, evalRegions, Linalg.sem, Op.blind_name, named, splitInsOuts_blind, splitArgs,
      attrAffineMaps?_blind, maps, iterKinds?_blind, iters, Outcome.bindOp_eq, executed,
      Outcome.bind_ok, List.length_cons, List.length_nil, count, ↓reduceIte, Outcome.pure_eq]

#print axioms inspect_sound
#print axioms execute

end MerlinVerify.Check.PolicyW8A8OrderedOp
