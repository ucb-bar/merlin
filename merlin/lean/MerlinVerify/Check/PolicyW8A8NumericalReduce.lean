/- Named reductions use the frozen generic-form semantics. The checker derives the split,
maps and iterator kinds from the actual operation and facts, and checks the actual body against
a loop invariant. No real-number bound, reassociation or model-specific dimension is assumed. -/
import MerlinVerify.Check.PolicyW8A8NumericalImplication
import MerlinVerify.Check.PolicyW8A8NumericalContext
import MerlinVerify.Rules.W8A8SourceSemantics

namespace MerlinVerify.Check.PolicyW8A8NumericalReduce

open MerlinVerify Rules Definedness Refined Sem.Ops Sem.Ops.Linalg

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

variable {Info : Type} {M : Refined.Model Info} {ib : Nat}

/-- The named semantics falls back to an equal operand split when segment sizes are absent.
Both branches preserve the actual input/output order and all refinements. -/
theorem reductionSplit_refined {op : Op} {args : List RVal} {facts ins outs : List (Refined.Fact Info)}
    (split : reductionSplit op facts = (ins, outs)) (holds : Refined.FactsHold M ib args facts) :
    ∃ xs ys, reductionSplit op args = (xs, ys) ∧
      Refined.FactsHold M ib xs ins ∧ Refined.FactsHold M ib ys outs := by
  cases selected : splitInsOuts op facts with
  | some pair =>
    simp only [reductionSplit, selected, Option.getD_some] at split
    subst pair
    obtain ⟨xs, ys, actual, inputs, outputs⟩ := splitInsOuts_refined selected holds
    exact ⟨xs, ys, by simp [reductionSplit, actual], inputs, outputs⟩
  | none =>
    have actual : splitInsOuts op args = none := by
      have absent : (splitInsOuts op args).isSome = false := by
        rw [splitInsOuts_isSome_length op args facts (List.Forall₂.length_eq holds), selected]
        rfl
      cases decoded : splitInsOuts op args <;> simp_all
    simp only [reductionSplit, selected, Option.getD_none, Prod.mk.injEq] at split
    obtain ⟨rfl, rfl⟩ := split
    refine ⟨args.take (facts.length / 2), args.drop (facts.length / 2), ?_,
      List.forall₂_take _ holds, List.forall₂_drop _ holds⟩
    simp only [reductionSplit, actual, Option.getD_none, List.Forall₂.length_eq holds]

def maps (rank inputCount outputCount : Nat) (dims : List Nat) : List AffineMap :=
  List.replicate inputCount (identityMap rank) ++
    List.replicate outputCount (LinalgNamed.mapOf rank ((List.range rank).filter (fun d => !(dims.contains d))))

def iterators (rank : Nat) (dims : List Nat) : List IterKind :=
  (List.range rank).map fun d => if dims.contains d then .reduction else .parallel

def infer (rules : Contextual.Structured.Invariants M ib)
    (bodies : Contextual.Regions M sourceSem ib allowed)
    (Γ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.reduce" then
    match reductionSplit op facts, LinalgNamed.natArray? op "dimensions", op.regions with
    | (input :: ins, outs), some dims, [body] =>
      let rank := input.base.ty.shape.length
      let indexing := maps rank (ins.length + 1) outs.length dims
      let kinds := iterators rank dims
      if (∀ fact ∈ outs, fact.base.initialized = true) ∧ (∀ fact ∈ outs, fact.base.ty.isTensor = true) ∧
          op.results.length = outs.length ∧
          (structuredRanges? indexing kinds.length (((input :: ins) ++ outs).map (·.base.ty))).isSome then do
        let outputs := outs.map (Contextual.Structured.invariant rules)
        let returned ← bodies.infer Γ body (((input :: ins) ++ outputs).map Refined.elementFact)
        if rules.implies returned (outputs.map Refined.elementFact) then some outputs else none
      else none
    | _, _, _ => none
  else none

theorem infer_sound (elementwise : Refined.Elementwise M)
    (rules : Contextual.Structured.Invariants M ib) (bodies : Contextual.Regions M sourceSem ib allowed) :
    ∀ Γ op facts results, infer rules bodies Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results := by
  intro Γ op facts results checked ctx call env args width permitted valid inputs
  unfold infer at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named⟩ := header
    split at checked
    · next input ins outs dims body splitFacts dimensions regions =>
      dsimp only at checked
      split at checked
      · next conditions =>
        obtain ⟨initialized, tensors, arity, rangesFound⟩ := conditions
        obtain ⟨ranges, rangesFound⟩ := Option.isSome_iff_exists.1 rangesFound
        change ((bodies.infer Γ body
          (((input :: ins) ++ outs.map (Contextual.Structured.invariant rules)).map Refined.elementFact)).bind
          (fun returned => if rules.implies returned
            ((outs.map (Contextual.Structured.invariant rules)).map Refined.elementFact) then
              some (outs.map (Contextual.Structured.invariant rules)) else none)) = some results at checked
        cases inferred : bodies.infer Γ body
            (((input :: ins) ++ outs.map (Contextual.Structured.invariant rules)).map Refined.elementFact) with
        | none =>
          simp only [List.map_append, List.map_cons, List.map_map] at inferred checked
          simp only [inferred, Option.bind] at checked
          cases checked
        | some returned =>
          simp only [List.map_append, List.map_cons, List.map_map] at inferred checked
          simp only [inferred, Option.bind] at checked
          split at checked
          · next subsumed =>
            cases checked
            obtain ⟨xs, ys, splitArgs, inputFacts, outputFacts⟩ := reductionSplit_refined splitFacts inputs
            cases inputFacts with
            | cons head tail =>
              have total : Contextual.BodyTotal M ib allowed call (fun ctx => evalRegion sourceSem ctx call env body)
                  (((input :: ins) ++ outs.map (Contextual.Structured.invariant rules)).map Refined.elementFact)
                  ((outs.map (Contextual.Structured.invariant rules)).map Refined.elementFact) := by
                intro localCtx localWidth localPermitted values holds
                have actualInputs := holds
                simp only [List.map_append, List.map_cons, List.map_map] at actualInputs
                obtain ⟨values, executed, resultFacts, _⟩ :=
                  bodies.sound Γ body _ _ inferred localCtx call env values localWidth localPermitted valid actualInputs
                exact ⟨values, executed, by
                  simpa only [List.map_map] using (rules.implies_sound _ _ subsumed _ resultFacts)⟩
              have structured : StructuredOk (maps input.base.ty.shape.length (ins.length + 1) outs.length dims)
                  (iterators input.base.ty.shape.length dims).length
                  ((input :: ins).map (·.base.ty) ++
                    (outs.map (Contextual.Structured.invariant rules)).map (·.base.ty)) ranges := by
                simpa only [Contextual.Structured.invariant, List.map_map, Function.comp_def,
                  List.map_append, List.map_cons] using structuredRanges?_spec rangesFound
              have outputTensors : ∀ fact ∈ outs.map (Contextual.Structured.invariant rules),
                  fact.base.ty.isTensor = true := by
                intro fact member
                obtain ⟨original, originalMember, rfl⟩ := List.mem_map.1 member
                exact tensors original originalMember
              have outputInitialized : ∀ fact ∈ outs.map (Contextual.Structured.invariant rules),
                  fact.base.initialized = true := by
                intro fact member
                obtain ⟨original, originalMember, rfl⟩ := List.mem_map.1 member
                exact initialized original originalMember
              obtain ⟨values, executed, holds⟩ := Contextual.genericSem_refined elementwise
                PolicyW8A8NumericalContext.loopStable width permitted structured outputTensors outputInitialized
                (List.Forall₂.cons head tail) (Contextual.Structured.invariant_sound rules outputFacts) total
              refine ⟨values, ?_, holds⟩
              have ordinary : W8A8SourceSemantics.isSource op.name = false := by
                simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]
              rw [W8A8SourceSemantics.unchanged ordinary,
                named_sem_of modelled dialect (by simp [named])]
              dsimp only [reductionSplit] at splitArgs
              have count : values.length = op.blind.results.length := by
                rw [HasTys.length holds.typed]
                simp only [List.length_map, Op.blind_results, arity]
              dsimp only [maps, iterators] at executed
              simp only [regions, evalRegions, LinalgNamed.sem, Op.blind_name, named, natArray?_blind,
                dimensions, splitInsOuts_blind, splitArgs, shapeOf_typed head.1.1, List.map_const',
                List.length_cons, List.Forall₂.length_eq tail, List.Forall₂.length_eq outputFacts,
                LinalgNamed.run, executed, Outcome.bind_ok, count, ↓reduceIte]
          · cases checked
      · cases checked
    · cases checked
  · cases checked

def book (elementwise : Refined.Elementwise M) (rules : Contextual.Structured.Invariants M ib)
    (bodies : Contextual.Regions M sourceSem ib allowed) : Contextual.Book M sourceSem ib allowed where
  rule := infer rules bodies
  sound := infer_sound elementwise rules bodies

#print axioms reductionSplit_refined
#print axioms infer_sound
#print axioms book

end MerlinVerify.Check.PolicyW8A8NumericalReduce
