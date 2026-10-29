/- UNCOMPILED successor. Complete defined-entry roster, explicit interfaces and
per-entry domains. Declaration/visibility metadata is an independently decoded
input bound to both raw modules; it is not native linkage/access-control semantics.
-/
import Merlin.GraphLoopsMemorySCFAfterEntriesMeaning

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

structure EntryHeader where
  name : String
  visibility : String
  declaration : Bool
  parameters : List Kind
  returned : List Kind
  deriving DecidableEq, Repr

def HeaderFits (module : Module) (header : EntryHeader) : Prop :=
  match functionAt module header.name with
  | none => header.declaration = true
  | some function => header.declaration = false ∧
      function.parameters.map Binding.kind = header.parameters ∧ function.returned = header.returned

instance (module : Module) (header : EntryHeader) : Decidable (HeaderFits module header) := by
  unfold HeaderFits
  split <;> infer_instance

def DefinedHeaders (module : Module) (headers : List EntryHeader) : Prop :=
  headers.map EntryHeader.name = module.functions.map Function.name ∧
    (headers.map EntryHeader.name).Nodup ∧
    (∀ header ∈ headers, header.declaration = false ∧ HeaderFits module header)

instance (module : Module) (headers : List EntryHeader) : Decidable (DefinedHeaders module headers) := by
  unfold DefinedHeaders
  infer_instance

inductive EntryRoute where
  | kernel
  | unchanged (reachable : List String)
  deriving DecidableEq, Repr

structure EntryRow where
  name : String
  route : EntryRoute
  deriving DecidableEq, Repr

def EntryRow.Facts (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) (row : EntryRow) : Prop :=
  match row.route with
  | .kernel => row.name = kernel.function.name
  | .unchanged reachable => row.name ∈ reachable ∧ ClosedAgreement plan.module (expandedModule plan kernel expansions) reachable

instance (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) (row : EntryRow) :
    Decidable (row.Facts plan kernel expansions) := by
  unfold EntryRow.Facts
  split <;> infer_instance

def EntryRosterFacts (source actualFinal : Module) (sourceHeaders finalHeaders : List EntryHeader)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) (rows : List EntryRow) : Prop :=
  DefinedHeaders source sourceHeaders ∧ DefinedHeaders actualFinal finalHeaders ∧ sourceHeaders = finalHeaders ∧
    rows.map EntryRow.name = sourceHeaders.map EntryHeader.name ∧
    (rows.map EntryRow.name).Nodup ∧ ∀ row ∈ rows, row.Facts plan kernel expansions

instance (source actualFinal : Module) (sourceHeaders finalHeaders : List EntryHeader)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) (rows : List EntryRow) :
    Decidable (EntryRosterFacts source actualFinal sourceHeaders finalHeaders plan kernel expansions rows) := by
  unfold EntryRosterFacts
  infer_instance

def ExternalInput {Value : Type} (bound : Nat) : Runtime Value → Prop
  | .view _ descriptor _ => ¬ descriptor.allocation < bound
  | _ => True

def EntryDomain {Value : Type} (source : Module) (plan : Epoch.PlanContext) (kernel : Epoch.Kernel)
    (row : EntryRow) (inputs : List (Runtime Value)) : Prop :=
  (∀ input ∈ inputs, ExternalInput plan.objects.length input) ∧
  match row.route with
  | .kernel => ValuesHold kernel.function.parameters inputs ∧
      Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs)
  | .unchanged _ => ∃ function, functionAt source row.name = some function ∧ ValuesHold function.parameters inputs

theorem roster_has_every_definition (source actualFinal : Module) (sourceHeaders finalHeaders : List EntryHeader)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (expansions : List Expansion) (rows : List EntryRow)
    (facts : EntryRosterFacts source actualFinal sourceHeaders finalHeaders plan kernel expansions rows)
    (function : Function) (member : function ∈ source.functions) : ∃ row ∈ rows, row.name = function.name := by
  have nameMember : function.name ∈ rows.map EntryRow.name := by
    rw [facts.2.2.2.1, facts.1.1]
    exact List.mem_map_of_mem member
  exact List.mem_map.mp nameMember

theorem typed_input_arity {Value : Type} (bindings : List Binding) (inputs : List (Runtime Value))
    (typed : ValuesHold bindings inputs) : inputs.length = bindings.length := by
  induction typed with
  | nil => rfl
  | cons holds typed induction => exact congrArg Nat.succ induction

theorem wrong_arity_no_success {Value World : Type} (meaning : Interpretation Value World) (module : Module)
    (name : String) (function : Function) (found : functionAt module name = some function)
    (inputs : List (Runtime Value)) (wrong : inputs.length ≠ function.parameters.length)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    ¬ complete meaning module false name inputs before outputs after := by
  rintro ⟨depth, run⟩
  cases depth with
  | zero => exact run
  | succ depth =>
    simp only [GraphLoopsShape.invoke, found, Function.invoke] at run
    exact wrong (typed_input_arity function.parameters inputs run.1)

end Merlin.GraphLoopsMemorySCF.After
