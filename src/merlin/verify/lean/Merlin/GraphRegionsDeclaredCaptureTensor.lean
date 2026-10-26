/- Pointwise tensor lifting for the restricted captured-source interpretation.
Tensor lifting and surrounding-context transport for the restricted captured
source slice. Native calls, transfers, allocator/alias/completion contracts and
the frozen Core.Sem bridge are deliberately not replaced by this source model. -/
import Merlin.GraphRegionsDeclaredCapturePrefixBridge

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegionsDeclared

def sourceTensor {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (program : Program) (names : List Nat) (references : List Reference)
    (leading : List Prefix.Instruction) (initialCells : Prefix.Cells α)
    (arguments : List (Tensor elements α)) : Tensor elements α :=
  fun index =>
    let runtime : Environment α := fun slot => (arguments[slot]?.map (· index)).getD default
    sourceRun meaning (references.map (·.slot)) program.inputs.length program.instructions
      (prefixCache (references.map (·.slot)) names (Prefix.run meaning leading initialCells) runtime) program.returned

theorem sourceTensor_agrees_closed {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (arguments : List (Tensor elements α)) (initialCells : Prefix.Cells α)
    {actualPrefix : List Op} {leading : List Prefix.Instruction} {available : List Nat}
    {program : Program} {names : List Nat} {references : List Reference}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (matched : checkReferences actualPrefix program names references = true)
    (eligible : checkEligible (references.map (·.slot)) program.inputs.length program.instructions = true)
    (valid : program.Valid) :
    sourceTensor meaning program names references leading initialCells arguments = tensorEvaluate program meaning arguments := by
  funext index
  exact executed_prefix_closure_result meaning _ initialCells matchedPrefix validPrefix matched eligible valid

def sourceStep {α : Type} [Inhabited α] {elements : Nat} (meaning : Primitive → List α → α)
    (site : GenericSite) (references : List Reference) (leading : List Prefix.Instruction) (initialCells : Prefix.Cells α)
    (state : StateEffects.State (Tensor elements α)) : StateEffects.State (Tensor elements α) :=
  bind state site.stage.output
    (sourceTensor meaning site.stage.program site.names references leading initialCells (site.stage.inputs.map state))

/-- Local source-step agreement is DERIVED from exact literal definitions and
explicit prefix execution. It is not an arbitrary Action agreement hypothesis. -/
theorem sourceStep_agrees_closed {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α)
    {actualPrefix : List Op} {actual : Op} {leading : List Prefix.Instruction} {available : List Nat}
    {site : GenericSite} {references : List Reference}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (binding : RegionBinding actualPrefix actual site references)
    (eligible : checkEligible (references.map (·.slot)) site.stage.program.inputs.length
      site.stage.program.instructions = true) :
    sourceStep (elements := elements) meaning site references leading initialCells = site.stage.step meaning := by
  funext state
  unfold sourceStep ScalarStage.step
  rw [sourceTensor_agrees_closed meaning _ initialCells matchedPrefix validPrefix binding.2.2 eligible binding.2.1.1]

/-- Any unchanged deterministic context can include failure/terminal-world data
in its result type. This transports the derived equality without asserting the
semantics of every unchanged model computation or any native fault suppression. -/
theorem surrounding_context {α Result : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α)
    {actualPrefix : List Op} {actual : Op} {leading : List Prefix.Instruction} {available : List Nat}
    {site : GenericSite} {references : List Reference}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (binding : RegionBinding actualPrefix actual site references)
    (eligible : checkEligible (references.map (·.slot)) site.stage.program.inputs.length
      site.stage.program.instructions = true)
    (before : StateEffects.State (Tensor elements α) → StateEffects.State (Tensor elements α))
    (after : StateEffects.State (Tensor elements α) → Result) (initial : StateEffects.State (Tensor elements α)) :
    after (sourceStep meaning site references leading initialCells (before initial)) =
      after (site.stage.step meaning (before initial)) := by
  rw [sourceStep_agrees_closed meaning initialCells matchedPrefix validPrefix binding eligible]

end Merlin.GraphRegionsDeclared.Capture
