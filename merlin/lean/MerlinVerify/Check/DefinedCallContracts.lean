/- Call rules with typed, initialized and refined result contracts. A contract list is an
untrusted input until its meaning is proved for the actual evaluator's call closure. -/
import MerlinVerify.Check.DefinedContext
import MerlinVerify.Check.Support

namespace MerlinVerify.Check.Definedness.Calls

open MerlinVerify MerlinVerify.Sem.Ops

structure Contract (Info : Type) where
  name : String
  inputs : List (Refined.Fact Info)
  outputs : List (Refined.Fact Info)

variable {Info : Type} (M : Refined.Model Info) (ib : Nat)

def Holds (known : List (Contract Info)) (call : CallSem) : Prop :=
  ∀ contract ∈ known, ∀ args, Refined.FactsHold M ib args contract.inputs →
    ∃ vs, call contract.name args = .ok vs ∧ Refined.FactsHold M ib vs contract.outputs ∧ ResultsDefined vs

def Agreement (sem : OpSem) : Prop :=
  ∀ ctx call op args regions, op.name = "func.call" →
    sem ctx call op args regions = Sem.Ops.opSem ctx call op args regions

def find? (known : List (Contract Info)) (name : String) : Option (Contract Info) :=
  known.find? (fun contract => contract.name == name)

def rule (subsumes : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool)
    (known : List (Contract Info)) (_ : Refined.FEnv Info) (op : Op) (inputs : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) := do
  if unmodelledAttr? op.blind = none ∧ op.name = "func.call" ∧ op.regions = [] ∧ op.successors = [] then
    let callee ← attrSymbol? op "callee"
    let contract ← find? known callee
    if subsumes inputs contract.inputs &&
        op.results.mapM (fun result => vtyOf ib result.ty) == some (contract.outputs.map (·.base.ty)) then
      some contract.outputs else none
  else none

theorem rule_sound {sem : OpSem} (agreement : Agreement sem)
    (subsumes : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool)
    (subsumes_sound : ∀ fs gs, subsumes fs gs = true → ∀ args,
      Refined.FactsHold M ib args fs → Refined.FactsHold M ib args gs)
    (known : List (Contract Info)) : ∀ Γ op fs rs,
    rule ib subsumes known Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Holds M ib known call → Refined.EnvHolds M ib Γ env → Refined.FactsHold M ib args fs →
    ∃ vs, sem ctx call op args (evalRegions sem call env op.regions) = .ok vs ∧ Refined.FactsHold M ib vs rs := by
  intro Γ op fs rs checked ctx call env args _width permitted _valid inputs
  unfold rule at checked
  split at checked
  · next conditions =>
    obtain ⟨modelled, named, regions, _successors⟩ := conditions
    cases callee : attrSymbol? op "callee" with
    | none => simp [callee] at checked
    | some name =>
      simp only [callee, bind, Option.bind] at checked
      cases selected : find? known name with
      | none => simp [selected] at checked
      | some contract =>
        simp only [selected] at checked
        split at checked
        · next accepted =>
          cases checked
          simp only [Bool.and_eq_true, beq_iff_eq] at accepted
          obtain ⟨compatible, results⟩ := accepted
          have member := List.mem_of_find?_eq_some selected
          have same : contract.name = name := eq_of_beq
            (List.find?_some (p := fun c : Contract Info => c.name == name) selected)
          obtain ⟨vs, executed, holds, _initialized⟩ := permitted contract member args
            (subsumes_sound fs contract.inputs compatible args inputs)
          have arity : vs.length = op.results.length := by
            have count := optionMapM_length _ _ _ results
            rw [List.length_map] at count
            exact (List.Forall₂.length_eq holds).trans count
          have dialect : dialectOf op.name = "func" := by rw [named]; cbv
          have target : calleeOf op.blind = some name := by
            unfold attrSymbol? at callee
            split at callee
            · next root encoded =>
              cases callee
              simp only [calleeOf, Op.blind_attr?, encoded]
            · cases callee
          refine ⟨vs, ?_, holds⟩
          rw [agreement ctx call op args _ named, opSem_modelled modelled]
          simp only [dispatch, Op.blind_name]
          rw [dialect]
          simp only [Func.sem, Op.blind_name, named]
          have decoded : attrSymbol? op.blind "callee" = some name := by
            simpa only [attrSymbol?, Op.blind_attr?] using callee
          simp only [decoded, restrictCall, target, ↓reduceIte, ← same, executed, Outcome.bind_ok,
            Op.blind_results, List.length_map, arity]
        · cases checked
  · cases checked

def book {sem : OpSem} (agreement : Agreement sem)
    (subsumes : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool)
    (subsumes_sound : ∀ fs gs, subsumes fs gs = true → ∀ args,
      Refined.FactsHold M ib args fs → Refined.FactsHold M ib args gs)
    (known : List (Contract Info)) : Contextual.Book M sem ib (fun _ call => Holds M ib known call) where
  rule := rule ib subsumes known
  sound := rule_sound M ib agreement subsumes subsumes_sound known

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.Definedness.Calls
