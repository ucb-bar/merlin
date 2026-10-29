/- UNCOMPILED. Transport of an ordinary retained action using only its actual
operand list and typed scalar/index results. All partial effects and the whole
world stay inside the same action relation. This is not a whole-program premise.
-/
import Merlin.GraphLoopsMemorySCFAfterInvariant

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem action_scoped_forward {Value World : Type} (objects : GraphLoopsReuse.Objects)
    (context : Context) (meaning : GraphLoopsShape.Interpretation Value World) (key : String) (operands results : List Binding)
    (logical actual logicalNext : GraphLoopsShape.Env Value)
    (related : RegistersRelated objects context logical actual)
    (arguments : readArguments false [] actual operands = readArguments false [] logical operands)
    (fresh : Fresh context results) (scalarResults : NoViewBindings results)
    (before after : World)
    (run : (Epoch.Atom.action key operands results).step meaning logical before logicalNext after) :
    ∃ actualNext, (Epoch.Atom.action key operands results).step meaning actual before actualNext after ∧
      RegistersRelated objects (extendContext context results) logicalNext actualNext := by
  obtain ⟨values, action, typed, rfl⟩ := run
  refine ⟨GraphLoopsShape.assign actual results values, ⟨values, ?_, typed, rfl⟩, ?_⟩
  · rw [arguments]; exact action
  · have transported := related.assign results values fresh typed
    rw [scalar_values_fixed objects results values typed scalarResults] at transported
    exact transported

theorem action_scoped_backward {Value World : Type} (objects : GraphLoopsReuse.Objects)
    (context : Context) (meaning : GraphLoopsShape.Interpretation Value World) (key : String) (operands results : List Binding)
    (logical actual actualNext : GraphLoopsShape.Env Value)
    (related : RegistersRelated objects context logical actual)
    (arguments : readArguments false [] actual operands = readArguments false [] logical operands)
    (fresh : Fresh context results) (scalarResults : NoViewBindings results)
    (before after : World)
    (run : (Epoch.Atom.action key operands results).step meaning actual before actualNext after) :
    ∃ logicalNext, (Epoch.Atom.action key operands results).step meaning logical before logicalNext after ∧
      RegistersRelated objects (extendContext context results) logicalNext actualNext := by
  obtain ⟨values, action, typed, rfl⟩ := run
  refine ⟨GraphLoopsShape.assign logical results values, ⟨values, ?_, typed, rfl⟩, ?_⟩
  · rw [arguments] at action; exact action
  · have transported := related.assign results values fresh typed
    rw [scalar_values_fixed objects results values typed scalarResults] at transported
    exact transported

/-- Candidate reuse/segmentation flags do not alter a retained instruction.
This is proved by the two frozen constructor normal forms, not by assuming a
source execution on the candidate heap.
-/
theorem retained_modes {Value User : Type} (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (bound : Nat) (specifications : Epoch.Specifications) (mapping : Aliases) (scalar : Epoch.BinaryMeaning Value)
    (key : String) (spec : Epoch.Spec) (selected : Epoch.specAt specifications key = some spec)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation bound specifications mapping scalar) true true
      (.retained key) before after ↔
    GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation bound specifications mapping scalar) false false
      (.retained key) before after := by
  rw [candidate_retained_iff bound specifications scalar mapping key spec selected,
    Epoch.retained_step_iff bound specifications scalar mapping key spec selected]

/-- Only a single retained action is abstracted by `localLaw`. Binary actions and
the concrete Touch observer discharge this with the existing scalar/observer
theorems. This lemma alone is NOT checked-plan soundness.
-/
theorem retained_scoped_backward {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (clock : Nat) (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (key operation : String) (spec : Epoch.Spec) (selected : Epoch.specAt plan.specifications key = some spec)
    (operands results : List Binding) (fresh : Fresh context results) (scalarResults : NoViewBindings results)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual next : GraphLoopsShape.Env Value) (related : RegistersRelated plan.objects context before.observed.registers actual)
    (arguments : readArguments false [] actual operands = readArguments false [] before.observed.registers operands)
    (world : MemoryWorld Value (Epoch.AllocationPublic User))
    (localLaw : ∀ logicalNext final,
      (Epoch.Atom.action operation operands results).step meaning before.observed.registers
        (Epoch.clockWorld plan.objects.length clock before) logicalNext final ↔
      ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) false false
        (.retained key) before after ∧ logicalNext = after.observed.registers ∧
        final = Epoch.clockWorld plan.objects.length clock after)
    (run : (Epoch.Atom.action operation operands results).step meaning actual
      (Epoch.clockWorld plan.objects.length clock before) next world) :
    ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      (.retained key) before after ∧ world = Epoch.clockWorld plan.objects.length clock after ∧
      RegistersRelated plan.objects (extendContext context results) after.observed.registers next := by
  obtain ⟨logicalNext, logicalRun, nextRelated⟩ := action_scoped_backward plan.objects context meaning operation
    operands results before.observed.registers actual next related arguments fresh scalarResults _ world run
  obtain ⟨after, oldRun, rfl, final⟩ := (localLaw logicalNext world).mp logicalRun
  exact ⟨after, (retained_modes plan.objects plan.configs plan.objects.length plan.specifications plan.mapping
    scalar key spec selected before after).mpr oldRun, final, nextRelated⟩

theorem retained_scoped_forward {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (clock : Nat) (scalar : Epoch.BinaryMeaning Value)
    (meaning : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (key operation : String) (spec : Epoch.Spec) (selected : Epoch.specAt plan.specifications key = some spec)
    (operands results : List Binding) (fresh : Fresh context results) (scalarResults : NoViewBindings results)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated plan.objects context before.observed.registers actual)
    (arguments : readArguments false [] actual operands = readArguments false [] before.observed.registers operands)
    (localLaw : ∀ logicalNext final,
      (Epoch.Atom.action operation operands results).step meaning before.observed.registers
        (Epoch.clockWorld plan.objects.length clock before) logicalNext final ↔
      ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) false false
        (.retained key) before after ∧ logicalNext = after.observed.registers ∧
        final = Epoch.clockWorld plan.objects.length clock after)
    (run : GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      (.retained key) before after) :
    ∃ next, (Epoch.Atom.action operation operands results).step meaning actual
      (Epoch.clockWorld plan.objects.length clock before) next (Epoch.clockWorld plan.objects.length clock after) ∧
      RegistersRelated plan.objects (extendContext context results) after.observed.registers next := by
  have oldRun := (retained_modes plan.objects plan.configs plan.objects.length plan.specifications plan.mapping
    scalar key spec selected before after).mp run
  exact action_scoped_forward plan.objects context meaning operation operands results before.observed.registers actual
    after.observed.registers related arguments fresh scalarResults _ _ ((localLaw _ _).mpr ⟨after, oldRun, rfl, rfl⟩)

end Merlin.GraphLoopsMemorySCF.After
