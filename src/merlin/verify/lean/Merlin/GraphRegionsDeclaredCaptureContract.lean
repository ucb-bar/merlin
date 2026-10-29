/- Restricted captured-source initialization and closure contract.
Bind complete parent/region syntax, explicit source literal initialization and
independently submitted closed declarations. This is an additive region model;
whole parent execution/failure correspondence and actual-pass qualification are
separate pending obligations. No native implementation is defined here. -/
import Merlin.GraphRegionsDeclaredCaptureInitialize

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegionsDeclared

/-- Original-region execution in canonical SSA coordinates. Captured slots read
the already interpreted outer literals; local constants/operations execute in
the original region order. The closed function has a different evaluator. -/
inductive SourceExecution {α : Type} (meaning : Primitive → List α → α) :
    List Op → Op → Environment α → α → Prop where
  | region (leading : List Op) (actual : Op) (site : GenericSite) (references : List Reference)
      (matched : RegionBinding leading actual site references)
      (eligible : checkEligible (references.map (·.slot)) site.stage.program.inputs.length
        site.stage.program.instructions = true) (arguments : Environment α) :
    SourceExecution meaning leading actual arguments
      (sourceRun meaning (references.map (·.slot)) site.stage.program.inputs.length site.stage.program.instructions
        (cacheLiterals meaning (references.map (·.slot)) site.stage.program.inputs.length
          site.stage.program.instructions arguments) site.stage.program.returned)

theorem region_agrees_closed {α : Type} {leading : List Op} {actual : Op}
    {site : GenericSite} {references : List Reference}
    (binding : RegionBinding leading actual site references)
    (eligible : checkEligible (references.map (·.slot)) site.stage.program.inputs.length
      site.stage.program.instructions = true)
    (meaning : Primitive → List α → α) (arguments : Environment α) :
    SourceExecution meaning leading actual arguments (site.stage.program.evaluate meaning arguments) := by
  have same := initialized_closure_result meaning (references.map (·.slot)) site.stage.program
    binding.2.1.1 eligible arguments
  rw [← same]
  exact .region leading actual site references binding eligible arguments

/-- Conditional acceptance-to-region execution, with complete retained parent IR
and actual dominating definitions. This does not claim full parent execution or
reclassify an unqualified capture/native receipt as approved verification. -/
theorem checked_capture_contract {α : Type}
    {actual retained closedSource declaration : Module} {site : GenericSite} {references : List Reference}
    {sourceName declarationName : String} {sourceVisibility declarationVisibility : Option String}
    (parent : ProgramBinding actual retained site references)
    (eligible : checkEligible (references.map (·.slot)) site.stage.program.inputs.length
      site.stage.program.instructions = true)
    (closed : checkScalar closedSource declaration sourceName declarationName sourceVisibility declarationVisibility
      site.stage.program = true)
    (meaning : Primitive → List α → α) (arguments : Environment α) :
    actual = retained ∧
    ∃ leading operation,
      RegionBinding leading operation site references ∧
      SourceExecution meaning leading operation arguments (site.stage.program.evaluate meaning arguments) ∧
      ScalarExecution meaning closedSource arguments (site.stage.program.evaluate meaning arguments) ∧
      ScalarExecution meaning declaration arguments (site.stage.program.evaluate meaning arguments) := by
  obtain ⟨sameParent, body⟩ := parent
  refine ⟨sameParent, ?_⟩
  split at body
  · obtain ⟨operation, _, region⟩ := body
    obtain ⟨sourceExecution, declaredExecution⟩ := checkScalar_sound closed meaning arguments
    exact ⟨_, operation, region, region_agrees_closed region eligible meaning arguments,
      sourceExecution, declaredExecution⟩
  · contradiction

end Merlin.GraphRegionsDeclared.Capture
