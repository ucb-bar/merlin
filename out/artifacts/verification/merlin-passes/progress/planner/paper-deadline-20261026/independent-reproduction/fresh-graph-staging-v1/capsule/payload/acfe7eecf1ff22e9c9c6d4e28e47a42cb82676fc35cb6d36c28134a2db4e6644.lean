/- UNCOMPILED successor. Complete whole-entry roster with an explicit, proved
single-call context route. Prior EntryRoute and old seals remain unchanged.
-/
import Merlin.GraphLoopsMemorySCFAfterContextChain

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

inductive ContextRoute where
  | kernel
  | unchanged (reachable : List String)
  | caller (wrapper : CallerContext)
  deriving DecidableEq, Repr

structure ContextRow where
  name : String
  route : ContextRoute
  deriving DecidableEq, Repr

def ContextRow.Facts (data : ContextBoundary) (row : ContextRow) : Prop :=
  match row.route with
  | .kernel => row.name = data.kernel.function.name
  | .unchanged reachable => row.name ∈ reachable ∧ ClosedAgreement data.plan.module data.expanded reachable
  | .caller wrapper => row.name = wrapper.function.name ∧ wrapper.Checked data

instance (data : ContextBoundary) (row : ContextRow) : Decidable (row.Facts data) := by
  unfold ContextRow.Facts
  split <;> infer_instance

def ContextRoster (data : ContextBoundary) (sourceHeaders finalHeaders : List EntryHeader) (rows : List ContextRow) : Prop :=
  DefinedHeaders data.source sourceHeaders ∧ DefinedHeaders data.actualFinal finalHeaders ∧ sourceHeaders = finalHeaders ∧
    rows.map ContextRow.name = sourceHeaders.map EntryHeader.name ∧ (rows.map ContextRow.name).Nodup ∧
    ∀ row ∈ rows, row.Facts data

instance (data : ContextBoundary) (sourceHeaders finalHeaders : List EntryHeader) (rows : List ContextRow) :
    Decidable (ContextRoster data sourceHeaders finalHeaders rows) := by unfold ContextRoster; infer_instance

def ContextRow.Domain {Value : Type} (data : ContextBoundary) (row : ContextRow) (inputs : List (Runtime Value)) : Prop :=
  match row.route with
  | .kernel => ExternalValues data.plan.objects.length inputs ∧ ValuesHold data.kernel.function.parameters inputs ∧
      Epoch.CallersValid data.plan.objects.length data.plan.callers (assign emptyEnv data.kernel.function.parameters inputs)
  | .unchanged _ => ExternalValues data.plan.objects.length inputs ∧
      ∃ function, functionAt data.source row.name = some function ∧ ValuesHold function.parameters inputs
  | .caller wrapper => wrapper.Domain data.plan data.kernel inputs

theorem context_every_definition (data : ContextBoundary) (sourceHeaders finalHeaders : List EntryHeader)
    (rows : List ContextRow) (facts : ContextRoster data sourceHeaders finalHeaders rows)
    (function : Function) (defined : function ∈ data.source.functions) : ∃ row ∈ rows, row.name = function.name := by
  have member : function.name ∈ rows.map ContextRow.name := by
    rw [facts.2.2.2.1, facts.1.1]
    exact List.mem_map_of_mem defined
  exact List.mem_map.mp member

theorem checked_context_roster {Value User : Type} (data : ContextBoundary) (boundary : data.Facts)
    (sourceHeaders finalHeaders : List EntryHeader) (rows : List ContextRow)
    (roster : ContextRoster data sourceHeaders finalHeaders rows)
    (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (row : ContextRow) (member : row ∈ rows) (inputs : List (Runtime Value)) (domain : row.Domain data inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.source row.name inputs heap user observed ↔
    Epoch.graphOutcomes data.plan.objects.length (data.meaning scalar literal) data.actualFinal row.name inputs heap user observed := by
  have rowFacts := roster.2.2.2.2.2 row member
  cases route : row.route with
  | kernel =>
    have name : row.name = data.kernel.function.name := by simpa only [ContextRow.Facts, route] using rowFacts
    have valid : ExternalValues data.plan.objects.length inputs ∧ ValuesHold data.kernel.function.parameters inputs ∧
        Epoch.CallersValid data.plan.objects.length data.plan.callers (assign emptyEnv data.kernel.function.parameters inputs) := by
      simpa only [ContextRow.Domain, route] using domain
    rw [name]
    exact data.kernel_boundary boundary scalar literal inputs valid.2.1 valid.2.2 heap user observed
  | caller wrapper =>
    have valid : row.name = wrapper.function.name ∧ wrapper.Checked data := by
      simpa only [ContextRow.Facts, route] using rowFacts
    have inputsValid : wrapper.Domain data.plan data.kernel inputs := by simpa only [ContextRow.Domain, route] using domain
    rw [valid.1]
    exact wrapper.checked_boundary data boundary valid.2 scalar literal inputs inputsValid heap user observed
  | unchanged reachable =>
    have valid : row.name ∈ reachable ∧ ClosedAgreement data.plan.module data.expanded reachable := by
      simpa only [ContextRow.Facts, route] using rowFacts
    rcases boundary with ⟨sourceValid, _, _, normalized, _, _, _, _, _, materialized, _⟩
    exact unchanged_entry_outcomes data.source data.bits sourceValid data.plan normalized data.kernel data.expansions
      reachable valid.2 data.actualFinal data.swaps data.sourceFrame data.finalFrame materialized
      (data.meaning scalar literal) row.name valid.1 inputs heap user observed

end Merlin.GraphLoopsMemorySCF.After
