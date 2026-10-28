/- Fill is a complete overwrite, with the actual scalar region checked under the same
conditional semantics as its enclosing function. Its yielded numerical facts are preserved. -/
import MerlinVerify.Check.DefinedContextLoops
import MerlinVerify.Check.DefinedGeneric
import MerlinVerify.Check.DefinedPointRegions

namespace MerlinVerify.Check.Definedness.Contextual.Fill

open MerlinVerify Refined MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

def Agreement (sem : OpSem) : Prop :=
  ∀ ctx call op args regions, op.name = "linalg.fill" →
    sem ctx call op args regions = Sem.Ops.opSem ctx call op args regions

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

def pointRule (bodyBook : PointRegions M sem ib allowed) (Γ : Refined.FEnv Info) (op : Op)
    (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.fill" then
    match splitInsOuts op facts, op.regions with
    | some ([input], [⟨⟨.tensor shape elt, _⟩, _⟩]), [body] =>
      if op.results.length = 1 ∧
          structuredRanges? (fillMaps shape.length) shape.length [input.base.ty, .tensor shape elt] = some shape then
        match bodyBook.infer Γ shape body
            ([input, emptyFact M ⟨.tensor shape elt, false⟩].map Refined.elementFact) with
        | some [⟨⟨.scalar yielded, true⟩, info⟩] =>
          if yielded = elt then some [⟨⟨.tensor shape elt, true⟩, info⟩] else none
        | _ => none
      else none
    | _, _ => none
  else none

theorem pointRule_sound (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Agreement sem) (bodyBook : PointRegions M sem ib allowed) :
    ∀ Γ op facts results, pointRule bodyBook Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results := by
  intro Γ op facts results checked ctx call env args width permitted valid inputs
  unfold pointRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named⟩ := header
    split at checked
    · next input shape elt init destInfo body splitFacts regions =>
      split at checked
      · next conditions =>
        obtain ⟨arity, ranges⟩ := conditions
        split at checked
        · next yielded info inferred =>
          split at checked
          · next sameType =>
            subst yielded
            cases checked
            obtain ⟨xs, ys, splitArgs, inputFacts, outputFacts⟩ := splitInsOuts_refined splitFacts inputs
            obtain _ | ⟨inputHead, inputTail⟩ := inputFacts
            cases inputTail
            obtain _ | ⟨outputHead, outputTail⟩ := outputFacts
            cases outputTail
            rename_i value output
            cases output with
            | scalar _ => cases outputHead.1.1
            | tensor tensor =>
              have total : PointBodyTotal M ib allowed call shape (fun ctx => evalRegion sem ctx call env body)
                  (([input] ++ [emptyFact M ⟨.tensor shape elt, false⟩]).map Refined.elementFact)
                  [⟨⟨.scalar elt, true⟩, info⟩] := by
                intro ctx width permitted bounds args holds
                obtain ⟨values, executed, holds, _⟩ :=
                  bodyBook.sound Γ shape body _ _ inferred ctx call env args width permitted bounds valid holds
                exact ⟨values, executed, holds⟩
              have structured : StructuredOk (fillMaps shape.length)
                  (List.replicate shape.length IterKind.parallel).length
                  (([input].map (·.base.ty)) ++ [.tensor shape elt]) shape := by
                simpa only [List.length_replicate, List.map_cons, List.map_nil, List.singleton_append] using
                  structuredRanges?_spec ranges
              obtain ⟨values, executed, holds⟩ := overwriteSem_point_refined elementwise stable width permitted
                structured (by rfl) (.cons inputHead .nil) outputHead.1.1 total
              dsimp only [fillMaps] at executed
              refine ⟨values, ?_, holds⟩
              have count : values.length = op.blind.results.length := by
                rw [HasTys.length holds.typed]
                simp only [List.length_map, List.length_cons, List.length_nil, Op.blind_results, arity]
              rw [agreement ctx call op args (evalRegions sem call env op.regions) named,
                linalg_sem_of modelled dialect (.inr (.inl named))]
              simp only [regions, evalRegions, Linalg.sem, Op.blind_name, named, splitInsOuts_blind, splitArgs,
                outputHead.1.1.1, Outcome.bindOp_eq, executed, Outcome.bind_ok, count, ↓reduceIte, Outcome.pure_eq]
          · cases checked
        · cases checked
      · cases checked
    · cases checked
  · cases checked

def bookPointRegions (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Agreement sem) (bodyBook : PointRegions M sem ib allowed) : Book M sem ib allowed where
  rule := pointRule bodyBook
  sound := pointRule_sound elementwise stable agreement bodyBook

def rule (bodyBook : Regions M sem ib allowed) :
    Refined.FEnv Info → Op → List (Refined.Fact Info) → Option (List (Refined.Fact Info)) :=
  pointRule (.fromRegions bodyBook)

theorem rule_sound (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Agreement sem) (bodyBook : Regions M sem ib allowed) :
    ∀ Γ op facts results, rule bodyBook Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results :=
  pointRule_sound elementwise stable agreement (.fromRegions bodyBook)

def bookRegions (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Agreement sem) (bodyBook : Regions M sem ib allowed) : Book M sem ib allowed where
  rule := rule bodyBook
  sound := rule_sound elementwise stable agreement bodyBook

def book (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Agreement sem) (bodyBook : Book M sem ib allowed) : Book M sem ib allowed :=
  bookRegions elementwise stable agreement (.operations bodyBook)

#print axioms rule_sound
#print axioms pointRule_sound
#print axioms bookPointRegions
#print axioms book
#print axioms bookRegions

end MerlinVerify.Check.Definedness.Contextual.Fill
