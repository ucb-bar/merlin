/- Conditional generic bodies retain captures and numerical postconditions. Initialized
reductions use a checked invariant; overwrites infer the actual yielded refinement. -/
import MerlinVerify.Check.DefinedContextLoops
import MerlinVerify.Check.DefinedGeneric
import MerlinVerify.Check.DefinedContextFill

namespace MerlinVerify.Check.Definedness.Contextual.Structured

open MerlinVerify Refined MerlinVerify.Sem.Ops MerlinVerify.Sem.Ops.Linalg

structure Invariants {Info : Type} (M : Model Info) (ib : Nat) where
  widen : Info → Info
  widen_sound : ∀ info value, M.holds info value → M.holds (widen info) value
  implies : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool
  implies_sound : ∀ actual required, implies actual required = true →
    ∀ values, Refined.FactsHold M ib values actual → Refined.FactsHold M ib values required

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {allowed : Conditions}

def invariant (rules : Invariants M ib) (fact : Refined.Fact Info) : Refined.Fact Info :=
  ⟨fact.base, rules.widen fact.info⟩

theorem invariant_sound (rules : Invariants M ib) {values : List RVal} {facts : List (Refined.Fact Info)}
    (holds : Refined.FactsHold M ib values facts) :
    Refined.FactsHold M ib values (facts.map (invariant rules)) := by
  induction holds with
  | nil => exact .nil
  | cons head _ ih => exact .cons ⟨head.1, rules.widen_sound _ _ head.2⟩ ih

def initializedPointRule (rules : Invariants M ib) (bodyBook : PointRegions M sem ib allowed)
    (Γ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op facts, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, outs), some maps, some iters, [body] =>
      if (∀ fact ∈ outs, fact.base.initialized = true) ∧ (∀ fact ∈ outs, fact.base.ty.isTensor = true) ∧
          op.results.length = outs.length ∧
          (structuredRanges? maps iters.length ((ins ++ outs).map (·.base.ty))).isSome then do
        let outputs := outs.map (invariant rules)
        let ranges := (structuredRanges? maps iters.length ((ins ++ outs).map (·.base.ty))).getD []
        let results ← bodyBook.infer Γ ranges body ((ins ++ outputs).map Refined.elementFact)
        if rules.implies results (outputs.map Refined.elementFact) then some outputs else none
      else none
    | _, _, _, _ => none
  else none

theorem initializedPointRule_sound (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (rules : Invariants M ib)
    (bodyBook : PointRegions M sem ib allowed) : ∀ Γ op facts results,
    initializedPointRule rules bodyBook Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results := by
  intro Γ op facts results checked ctx call env args width permitted valid inputs
  unfold initializedPointRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named⟩ := header
    split at checked
    · next ins outs maps iters body splitFacts mapsAttr iterators regions =>
      split at checked
      · next conditions =>
        obtain ⟨initialized, tensors, arity, rangesFound⟩ := conditions
        obtain ⟨ranges, rangesFound⟩ := Option.isSome_iff_exists.1 rangesFound
        rw [rangesFound] at checked
        dsimp only [Option.getD] at checked
        cases inferred : bodyBook.infer Γ ranges body
            ((ins ++ outs.map (invariant rules)).map Refined.elementFact) with
        | none =>
          simp only [inferred, bind, Option.bind] at checked
          cases checked
        | some returned =>
          simp only [inferred, bind, Option.bind] at checked
          split at checked
          · next subsumed =>
            cases checked
            obtain ⟨xs, ys, splitArgs, inputFacts, outputFacts⟩ := splitInsOuts_refined splitFacts inputs
            have total : PointBodyTotal M ib allowed call ranges (fun ctx => evalRegion sem ctx call env body)
                ((ins ++ outs.map (invariant rules)).map Refined.elementFact)
                ((outs.map (invariant rules)).map Refined.elementFact) := by
              intro ctx width permitted bounds args holds
              obtain ⟨values, executed, holds, _⟩ :=
                bodyBook.sound Γ ranges body _ _ inferred ctx call env args width permitted bounds valid holds
              exact ⟨values, executed, rules.implies_sound _ _ subsumed _ holds⟩
            have structured : StructuredOk maps iters.length
                (ins.map (·.base.ty) ++ (outs.map (invariant rules)).map (·.base.ty)) ranges := by
              simpa only [invariant, List.map_map, Function.comp_def, List.map_append] using
                structuredRanges?_spec rangesFound
            have outputTensors : ∀ fact ∈ outs.map (invariant rules), fact.base.ty.isTensor = true := by
              intro fact member
              obtain ⟨original, originalMember, rfl⟩ := List.mem_map.1 member
              exact tensors original originalMember
            have outputInitialized : ∀ fact ∈ outs.map (invariant rules), fact.base.initialized = true := by
              intro fact member
              obtain ⟨original, originalMember, rfl⟩ := List.mem_map.1 member
              exact initialized original originalMember
            obtain ⟨values, executed, holds⟩ := genericSem_point_refined elementwise stable width permitted
              structured outputTensors outputInitialized inputFacts (invariant_sound rules outputFacts) total
            refine ⟨values, ?_, holds⟩
            have count : values.length = op.blind.results.length := by
              rw [HasTys.length holds.typed]
              simp only [List.length_map, Op.blind_results, arity]
            rw [agreement ctx call op args (evalRegions sem call env op.regions) named,
              linalg_sem_of modelled dialect (.inl named)]
            simp only [regions, evalRegions, Linalg.sem, Op.blind_name, named, splitInsOuts_blind, splitArgs,
              attrAffineMaps?_blind, mapsAttr, iterKinds?_blind, iterators, Outcome.bindOp_eq, executed,
              Outcome.bind_ok, count, ↓reduceIte, Outcome.pure_eq]
          · cases checked
      · cases checked
    · cases checked
  · cases checked

def overwritePointRule (bodyBook : PointRegions M sem ib allowed) (Γ : Refined.FEnv Info)
    (op : Op) (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.generic" then
    match splitInsOuts op facts, attrAffineMaps? op "indexing_maps", iterKinds? op, op.regions with
    | some (ins, [⟨⟨.tensor shape elt, _⟩, _⟩]), some maps, some iters, [body] =>
      if op.results.length = 1 ∧ maps.drop ins.length = [identityMap shape.length] ∧
          structuredRanges? maps iters.length (ins.map (·.base.ty) ++ [.tensor shape elt]) = some shape then
        match bodyBook.infer Γ shape body
            ((ins ++ [emptyFact M ⟨.tensor shape elt, false⟩]).map Refined.elementFact) with
        | some [⟨⟨.scalar yielded, true⟩, info⟩] =>
          if yielded = elt then some [⟨⟨.tensor shape elt, true⟩, info⟩] else none
        | _ => none
      else none
    | _, _, _, _ => none
  else none

theorem overwritePointRule_sound (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (bodyBook : PointRegions M sem ib allowed) :
    ∀ Γ op facts results, overwritePointRule bodyBook Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results := by
  intro Γ op facts results checked ctx call env args width permitted valid inputs
  unfold overwritePointRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named⟩ := header
    split at checked
    · next ins shape elt init destInfo maps iters body splitFacts mapsAttr iterators regions =>
      split at checked
      · next conditions =>
        obtain ⟨arity, outputMap, ranges⟩ := conditions
        split at checked
        · next yielded info inferred =>
          split at checked
          · next sameType =>
            subst yielded
            cases checked
            obtain ⟨xs, ys, splitArgs, inputFacts, outputFacts⟩ := splitInsOuts_refined splitFacts inputs
            obtain _ | ⟨head, tail⟩ := outputFacts
            cases tail
            rename_i output
            cases output with
            | scalar _ => cases head.1.1
            | tensor tensor =>
              have total : PointBodyTotal M ib allowed call shape (fun ctx => evalRegion sem ctx call env body)
                  ((ins ++ [emptyFact M ⟨.tensor shape elt, false⟩]).map Refined.elementFact)
                  [⟨⟨.scalar elt, true⟩, info⟩] := by
                intro ctx width permitted bounds args holds
                obtain ⟨values, executed, holds, _⟩ :=
                  bodyBook.sound Γ shape body _ _ inferred ctx call env args width permitted bounds valid holds
                exact ⟨values, executed, holds⟩
              obtain ⟨values, executed, holds⟩ := overwriteSem_point_refined elementwise stable width permitted
                (structuredRanges?_spec ranges) outputMap inputFacts head.1.1 total
              refine ⟨values, ?_, holds⟩
              have count : values.length = op.blind.results.length := by
                rw [HasTys.length holds.typed]
                simp only [List.length_map, List.length_cons, List.length_nil, Op.blind_results, arity]
              rw [agreement ctx call op args (evalRegions sem call env op.regions) named,
                linalg_sem_of modelled dialect (.inl named)]
              simp only [regions, evalRegions, Linalg.sem, Op.blind_name, named, splitInsOuts_blind, splitArgs,
                attrAffineMaps?_blind, mapsAttr, iterKinds?_blind, iterators, Outcome.bindOp_eq, executed,
                Outcome.bind_ok, count, ↓reduceIte, Outcome.pure_eq]
          · cases checked
        · cases checked
      · cases checked
    · cases checked
  · cases checked

/-- Try a complete overwrite first so pointwise producers retain their inferred facts.
Reduction initialization requires a separately checked invariant. -/
def bookPointRegions (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (rules : Invariants M ib)
    (bodyBook : PointRegions M sem ib allowed) : Book M sem ib allowed where
  rule Γ op facts := (overwritePointRule bodyBook Γ op facts).orElse fun _ => initializedPointRule rules bodyBook Γ op facts
  sound Γ op facts results checked ctx call env args width permitted valid inputs := by
    cases overwritten : overwritePointRule bodyBook Γ op facts with
    | none =>
      simp only [overwritten, Option.orElse_none] at checked
      exact initializedPointRule_sound elementwise stable agreement rules bodyBook Γ op facts results checked
        ctx call env args width permitted valid inputs
    | some outputs =>
      simp only [overwritten, Option.orElse_some, Option.some.injEq] at checked
      subst results
      exact overwritePointRule_sound elementwise stable agreement bodyBook Γ op facts outputs overwritten
        ctx call env args width permitted valid inputs

def initializedRule (rules : Invariants M ib) (bodyBook : Regions M sem ib allowed) :
    Refined.FEnv Info → Op → List (Refined.Fact Info) → Option (List (Refined.Fact Info)) :=
  initializedPointRule rules (.fromRegions bodyBook)

theorem initializedRule_sound (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (rules : Invariants M ib)
    (bodyBook : Regions M sem ib allowed) : ∀ Γ op facts results,
    initializedRule rules bodyBook Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results :=
  initializedPointRule_sound elementwise stable agreement rules (.fromRegions bodyBook)

def overwriteRule (bodyBook : Regions M sem ib allowed) :
    Refined.FEnv Info → Op → List (Refined.Fact Info) → Option (List (Refined.Fact Info)) :=
  overwritePointRule (.fromRegions bodyBook)

theorem overwriteRule_sound (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (bodyBook : Regions M sem ib allowed) :
    ∀ Γ op facts results, overwriteRule bodyBook Γ op facts = some results → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds M ib Γ env →
    Refined.FactsHold M ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧
      Refined.FactsHold M ib values results :=
  overwritePointRule_sound elementwise stable agreement (.fromRegions bodyBook)

def bookRegions (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (rules : Invariants M ib)
    (bodyBook : Regions M sem ib allowed) : Book M sem ib allowed :=
  bookPointRegions elementwise stable agreement rules (.fromRegions bodyBook)

def book (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (rules : Invariants M ib)
    (bodyBook : Book M sem ib allowed) : Book M sem ib allowed :=
  bookRegions elementwise stable agreement rules (.operations bodyBook)

def nestedBookWith (contracts : Regions M sem ib allowed)
    (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (fillAgreement : Fill.Agreement sem)
    (rules : Invariants M ib)
    (leaf : Book M sem ib allowed) : Nat → Book M sem ib allowed
  | 0 => leaf
  | depth + 1 =>
    let bodyBook := nestedBookWith contracts elementwise stable agreement fillAgreement rules leaf depth
    let regions := contracts.orElse (.operations bodyBook)
    (bookRegions elementwise stable agreement rules regions).orElse
      ((Fill.bookRegions elementwise stable fillAgreement regions).orElse leaf)

def nestedBook (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (fillAgreement : Fill.Agreement sem)
    (rules : Invariants M ib) (leaf : Book M sem ib allowed) : Nat → Book M sem ib allowed :=
  nestedBookWith .empty elementwise stable agreement fillAgreement rules leaf

/-- Region depth is derived from the actual operation; hints cannot cap body inspection. -/
def recursiveBookWith (contracts : Regions M sem ib allowed)
    (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (fillAgreement : Fill.Agreement sem)
    (rules : Invariants M ib)
    (leaf : Book M sem ib allowed) : Book M sem ib allowed where
  rule Γ op facts := (nestedBookWith contracts elementwise stable agreement fillAgreement rules leaf
    (RegionDepth.op op)).rule Γ op facts
  sound Γ op facts results checked ctx call env args width permitted valid inputs :=
    (nestedBookWith contracts elementwise stable agreement fillAgreement rules leaf (RegionDepth.op op)).sound
      Γ op facts results checked ctx call env args width permitted valid inputs

def recursiveBook (elementwise : Elementwise M) (stable : LoopStable allowed)
    (agreement : Definedness.Structured.GenericAgreement sem) (fillAgreement : Fill.Agreement sem)
    (rules : Invariants M ib) (leaf : Book M sem ib allowed) : Book M sem ib allowed :=
  recursiveBookWith .empty elementwise stable agreement fillAgreement rules leaf

#print axioms invariant_sound
#print axioms initializedRule_sound
#print axioms overwriteRule_sound
#print axioms initializedPointRule_sound
#print axioms overwritePointRule_sound
#print axioms bookPointRegions
#print axioms book
#print axioms bookRegions
#print axioms recursiveBookWith
#print axioms recursiveBook

end MerlinVerify.Check.Definedness.Contextual.Structured
