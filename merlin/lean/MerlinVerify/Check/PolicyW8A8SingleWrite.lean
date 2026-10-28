/- Initialization-aware single-write refinement for W8A8 numerical analysis.
The entire actual output-coordinate trace is checked. Reductions with repeated
output coordinates refuse this rule and retain their ordinary invariant path. -/
import MerlinVerify.Rules.W8A8SingleWrite
import MerlinVerify.Check.DefinedContextGeneric

namespace MerlinVerify.Check.PolicyW8A8SingleWrite

open MerlinVerify Rules Definedness Refined Sem.Ops Sem.Ops.Linalg

def traceCheck (ranges shape : List Nat) (output : AffineMap) : Bool :=
  if ranges.prod ≤ 65536 ∧ shape.prod ≤ 65536 then
    let points := allPoints ranges
    let cells := points.map (W8A8SingleWrite.coordinate shape output)
    points.all (fun point => decide ((evalMap output point >>= toIdx shape) =
        some (W8A8SingleWrite.coordinate shape output point) ∧
        Idx.InBounds shape (W8A8SingleWrite.coordinate shape output point))) &&
      (cells == allPoints shape)
  else false

theorem traceCheck_sound {ranges shape : List Nat} {output : AffineMap}
    (checked : traceCheck ranges shape output = true) :
    ((allPoints ranges).map (W8A8SingleWrite.coordinate shape output)).Nodup ∧
      (∀ point ∈ allPoints ranges, ∃ cell, (evalMap output point >>= toIdx shape) = some cell ∧
        Idx.InBounds shape cell) ∧
      (∀ cell, Idx.InBounds shape cell → cell ∈ (allPoints ranges).map (W8A8SingleWrite.coordinate shape output)) := by
  unfold traceCheck at checked
  split at checked
  · simp only [Bool.and_eq_true] at checked
    have equal : (allPoints ranges).map (W8A8SingleWrite.coordinate shape output) = allPoints shape := eq_of_beq checked.2
    refine ⟨equal ▸ nodup_allPoints shape, ?_, ?_⟩
    · intro point member
      exact ⟨_, of_decide_eq_true (List.all_eq_true.1 checked.1 point member)⟩
    · intro cell bounds
      rw [equal]
      exact (mem_allPoints _ _).2 bounds
  · cases checked

variable {Info : Type} {M : Model Info} {ib : Nat} {sem : OpSem} {allowed : Contextual.Conditions}

theorem generic_refined (elementwise : Elementwise M) (stable : Contextual.LoopStable allowed)
    {ctx : Ctx} {call : CallSem} (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {ins : List RVal} {inFacts : List (Refined.Fact Info)} {maps : List AffineMap} {iters : List IterKind}
    {tensor : Tensor} {shape ranges : List Nat} {elt : Ty} {initialInfo resultInfo : Info}
    {body : Ctx → RegionSem} {output : AffineMap}
    (structured : StructuredOk maps iters.length (inFacts.map (·.base.ty) ++ [.tensor shape elt]) ranges)
    (outputMap : maps.drop inFacts.length = [output]) (trace : traceCheck ranges shape output = true)
    (inputs : Refined.FactsHold M ib ins inFacts)
    (initial : (⟨⟨.tensor shape elt, true⟩, initialInfo⟩ : Refined.Fact Info).Holds M ib (.tensor tensor))
    (total : Contextual.PointBodyTotal M ib allowed call ranges body
      ((inFacts ++ [(⟨⟨.tensor shape elt, true⟩, initialInfo⟩ : Refined.Fact Info)]).map Refined.elementFact)
      [⟨⟨.scalar elt, true⟩, resultInfo⟩]) :
    ∃ results, genericSem ctx maps iters ins [.tensor tensor] body = .ok results ∧
      Refined.FactsHold M ib results [⟨⟨.tensor shape elt, true⟩, resultInfo⟩] := by
  have allTyped := List.rel_append inputs.typed (.cons initial.1.1 .nil)
  obtain ⟨mapCount, dimensions, ranks, loopRanges⟩ := structured_runtime structured allTyped
  have inputCount : ins.length = inFacts.length := List.Forall₂.length_eq inputs
  have inputMaps : MapsInBounds ranges (maps.take ins.length) (inFacts.map (·.base.ty)) := by
    have found := List.forall₂_take (inFacts.map (·.base.ty)).length structured.inBounds
    rw [List.take_left] at found
    simpa only [List.length_map, ← inputCount] using found
  have outputMaps : MapsInBounds ranges [output] [.tensor shape elt] := by
    have found := List.forall₂_drop (inFacts.map (·.base.ty)).length structured.inBounds
    rw [List.drop_left] at found
    simpa only [List.length_map, outputMap] using found
  have pointResult : ∀ point ∈ allPoints ranges, ∃ value,
      bodyAt ctx ins (maps.take ins.length) [output] [tensor] body point = .ok [value] ∧
      (⟨⟨.scalar elt, true⟩, resultInfo⟩ : Refined.Fact Info).Holds M ib (.scalar value) := by
    intro point member
    have bounds := (mem_allPoints _ _).1 member
    obtain ⟨xs, readInputs, inputFacts⟩ := mapM_read_refined elementwise bounds id
      (fun (value, map) => readAt value map point) (by intro _ _; rfl)
      ins (maps.take ins.length) inFacts (by simpa using inputs) inputMaps
    obtain ⟨ys, readOutputs, outputFacts⟩ := mapM_read_refined elementwise bounds RVal.tensor
      (fun (value, map) => readAt (.tensor value) map point) (by intro _ _; rfl)
      [tensor] [output] [⟨⟨.tensor shape elt, true⟩, initialInfo⟩] (.cons initial .nil) outputMaps
    have arguments : Refined.FactsHold M ib ((xs ++ ys).map RVal.scalar)
        ((inFacts ++ [(⟨⟨.tensor shape elt, true⟩, initialInfo⟩ : Refined.Fact Info)]).map Refined.elementFact) := by
      rw [List.map_append, List.map_append]
      exact List.rel_append inputFacts outputFacts
    obtain ⟨results, executed, holds⟩ := total {ctx with loopIndex := point} width
      (stable ctx call point permitted) bounds ((xs ++ ys).map RVal.scalar) arguments
    obtain _ | ⟨head, tail⟩ := holds
    cases tail
    rename_i result
    cases result with
    | tensor _ => cases head.1.1
    | scalar value =>
      refine ⟨value, ?_, head⟩
      simp only [bodyAt, readInputs, readOutputs, executed, Outcome.bindOp_eq, Outcome.bind_ok,
        List.length_cons, List.length_nil, ne_eq, not_true_eq_false, ↓reduceIte, List.mapM_cons,
        List.mapM_nil, Outcome.pure_eq]
  obtain ⟨unique, mapped, covers⟩ := traceCheck_sound trace
  have shapeEq := initial.1.1.1
  rw [← shapeEq] at unique mapped covers
  obtain ⟨final, executed, finalShape, finalElt, finalValues⟩ := W8A8SingleWrite.execute ctx ins
    (maps.take ins.length) output body tensor (allPoints ranges)
    (fun value => (⟨⟨.scalar elt, true⟩, resultInfo⟩ : Refined.Fact Info).Holds M ib (.scalar value))
    unique mapped covers pointResult
  have finalHolds : (⟨⟨.tensor shape elt, true⟩, resultInfo⟩ : Refined.Fact Info).Holds M ib (.tensor final) := by
    refine ⟨⟨⟨finalShape.trans shapeEq, finalElt.trans initial.1.1.2.1, ?_⟩, ?_⟩, ?_⟩
    · intro cell bounds
      exact (finalValues cell (shapeEq ▸ bounds)).1.1
    · intro _ cell bounds
      exact (finalValues cell (finalShape ▸ bounds)).1.2 rfl
    · apply (elementwise _ _).2
      intro cell bounds
      exact (finalValues cell (finalShape ▸ bounds)).2
  have canonical := canonize_refined elementwise (tensors := [final]) (.cons finalHolds .nil)
  refine ⟨[.tensor final.canonize], ?_, canonical⟩
  simp only [inputCount] at executed
  simp only [genericSem, mapCount, ne_eq, not_true_eq_false, ↓reduceIte, dimensions,
    Bool.false_eq_true, ranks, List.mapM_cons, List.mapM_nil, Outcome.bindOp_eq, Outcome.bind_ok,
    Outcome.pure_eq, loopRanges, inputCount, outputMap, executed, List.map_cons, List.map_nil]

#print axioms traceCheck_sound
#print axioms generic_refined

def rule (bodies : Contextual.PointRegions M sem ib allowed) (Γ : Refined.FEnv Info)
    (op : Op) (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op facts, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨⟨.tensor shape elt, true⟩, initial⟩]), some maps, some iters, [body] => do
      let ranges ← structuredRanges? maps iters.length (ins.map (·.base.ty) ++ [.tensor shape elt])
      let [output] := maps.drop ins.length | none
      if op.results.length = 1 ∧ traceCheck ranges shape output = true then
        match bodies.infer Γ ranges body
            ((ins ++ [(⟨⟨.tensor shape elt, true⟩, initial⟩ : Refined.Fact Info)]).map Refined.elementFact) with
        | some [⟨⟨.scalar yielded, true⟩, info⟩] =>
          if yielded = elt then some [⟨⟨.tensor shape elt, true⟩, info⟩] else none
        | _ => none
      else none
    | _, _, _, _ => none
  else none

theorem rule_sound (elementwise : Elementwise M) (stable : Contextual.LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem)
    (bodies : Contextual.PointRegions M sem ib allowed) : ∀ Γ op facts results,
    rule bodies Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results := by
  intro Γ op facts results checked ctx call env args width permitted valid inputs
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named⟩ := header
    split at checked
    · next ins shape elt initial maps iters body splitFacts mapsAttr iterators regions =>
      cases geometry : structuredRanges? maps iters.length (ins.map (·.base.ty) ++ [.tensor shape elt]) with
      | none => simp [geometry] at checked
      | some ranges =>
        simp only [geometry, bind, Option.bind] at checked
        split at checked
        · next output outputMap =>
          split at checked
          · next conditions =>
            obtain ⟨arity, trace⟩ := conditions
            split at checked
            · next yielded info inferred =>
              split at checked
              · next sameType =>
                subst yielded
                cases checked
                obtain ⟨xs, ys, splitArgs, inputFacts, outputFacts⟩ := splitInsOuts_refined splitFacts inputs
                obtain _ | ⟨head, tail⟩ := outputFacts
                cases tail
                rename_i outputValue
                cases outputValue with
                | scalar _ => cases head.1.1
                | tensor tensor =>
                  have total : Contextual.PointBodyTotal M ib allowed call ranges
                      (fun c => evalRegion sem c call env body)
                      ((ins ++ [(⟨⟨.tensor shape elt, true⟩, initial⟩ : Refined.Fact Info)]).map Refined.elementFact)
                      [⟨⟨.scalar elt, true⟩, info⟩] := by
                    intro c sameWidth permitted bounds arguments holds
                    obtain ⟨values, executed, holds, _⟩ := bodies.sound Γ ranges body _ _ inferred c call env arguments
                      sameWidth permitted bounds valid holds
                    exact ⟨values, executed, holds⟩
                  obtain ⟨values, executed, holds⟩ := generic_refined elementwise stable width permitted
                    (structuredRanges?_spec geometry) outputMap trace inputFacts head total
                  refine ⟨values, ?_, holds⟩
                  have count : values.length = op.blind.results.length := by
                    have length := List.Forall₂.length_eq holds
                    simpa only [List.length_cons, List.length_nil, Op.blind_results, List.length_map, arity] using length
                  rw [agreement ctx call op args (evalRegions sem call env op.regions) named,
                    linalg_sem_of modelled dialect (.inl named)]
                  simp only [regions, evalRegions, Linalg.sem, Op.blind_name, named, splitInsOuts_blind,
                    splitArgs, attrAffineMaps?_blind, mapsAttr, iterKinds?_blind, iterators, Outcome.bindOp_eq,
                    executed, Outcome.bind_ok, count, ↓reduceIte, Outcome.pure_eq]
              · cases checked
            · cases checked
          · cases checked
        · cases checked
    · cases checked
  · cases checked

def book (elementwise : Elementwise M) (stable : Contextual.LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem)
    (bodies : Contextual.PointRegions M sem ib allowed) : Contextual.Book M sem ib allowed where
  rule := rule bodies
  sound := rule_sound elementwise stable agreement bodies

#print axioms rule_sound

end MerlinVerify.Check.PolicyW8A8SingleWrite
