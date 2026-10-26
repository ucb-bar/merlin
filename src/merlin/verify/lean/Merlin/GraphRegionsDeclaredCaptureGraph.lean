/- Per-source-site execution is constructed
from actual checked prefix/region syntax and common primitive interpretations.
The equality supplied to existing GraphRegions roots is a derived conclusion.
This file grants no actual-pass, native or frozen Core.Sem qualification. -/
import Merlin.GraphRegionsDeclaredCaptureParent

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

structure Site where
  generic : GenericSite
  references : List Reference

def SitesBound (actualPrefix : List Op) (sites : List Site) : Prop :=
  ∀ site ∈ sites,
    RegionBinding actualPrefix (originalGeneric site.generic site.references) site.generic site.references ∧
    checkEligible (site.references.map (·.slot)) site.generic.stage.program.inputs.length
      site.generic.stage.program.instructions = true

def executeSource {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (leading : List Prefix.Instruction) (initialCells : Prefix.Cells α) :
    List Site → StateEffects.State (Tensor elements α) → StateEffects.State (Tensor elements α)
  | [], initial => initial
  | site :: rest, initial => executeSource meaning leading initialCells rest
      (sourceStep meaning site.generic site.references leading initialCells initial)

/-- Every admitted source site's local meaning is constructed. No arbitrary
total Action is assumed to denote that site, and no whole-region equality is a
premise. Distinct selected sites remain separate in the ordered execution. -/
theorem executeSource_eq_stages {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α)
    {actualPrefix : List Op} {leading : List Prefix.Instruction} {available : List Nat}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (sites : List Site) (bound : SitesBound actualPrefix sites) (initial : StateEffects.State (Tensor elements α)) :
    executeSource meaning leading initialCells sites initial =
      executeStages meaning (sites.map (fun site => site.generic.stage)) initial := by
  induction sites generalizing initial with
  | nil => rfl
  | cons site rest ih =>
    have current := bound site (by simp)
    have tail : SitesBound actualPrefix rest := fun next member => bound next (by simp [member])
    change executeSource meaning leading initialCells rest
      (sourceStep meaning site.generic site.references leading initialCells initial) =
      executeStages meaning (rest.map (fun site => site.generic.stage)) (site.generic.stage.step meaning initial)
    rw [sourceStep_agrees_closed meaning initialCells matchedPrefix validPrefix current.1 current.2]
    exact ih tail _

/-- Direct bridge to the existing checked GraphRegions.TypedAction execution.
The registry lookup is constructive, and actual source/capture checks supply the
site meanings. Primitive, native and unchanged-context contracts stay explicit. -/
theorem executeSource_eq_graph {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    {actualPrefix : List Op} {leading : List Prefix.Instruction} {available : List Nat}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (sites : List Site) (bound : SitesBound actualPrefix sites)
    (unique : ((sites.map (fun site => site.generic.stage)).map ScalarStage.node).Nodup)
    (initial : StateEffects.State (Tensor elements α)) :
    executeSource meaning leading initialCells sites initial =
      executeNodes (registry meaning fallback (sites.map (fun site => site.generic.stage)))
        (sites.map (fun site => site.generic.stage.node)) initial := by
  rw [executeSource_eq_stages meaning initialCells matchedPrefix validPrefix sites bound]
  have result := executeStages_nodes meaning (registry meaning fallback (sites.map (fun site => site.generic.stage)))
    (sites.map (fun site => site.generic.stage)) (registry_steps meaning fallback _ unique) initial
  simpa only [List.map_map, Function.comp_def] using result

/-- Surrounding deterministic contexts may return failures/world observations.
Their executions are unchanged functions; source-site agreement is derived above.
This does not establish native transfer/no-fault or asynchronous equivalence. -/
theorem graph_context_correspondence {α Result : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    {actualPrefix : List Op} {leading : List Prefix.Instruction} {available : List Nat}
    (matchedPrefix : Prefix.Matches actualPrefix leading) (validPrefix : Prefix.Valid available leading)
    (sites : List Site) (bound : SitesBound actualPrefix sites)
    (unique : ((sites.map (fun site => site.generic.stage)).map ScalarStage.node).Nodup)
    (before : StateEffects.State (Tensor elements α) → StateEffects.State (Tensor elements α))
    (after : StateEffects.State (Tensor elements α) → Result) (initial : StateEffects.State (Tensor elements α)) :
    after (executeSource meaning leading initialCells sites (before initial)) =
      after (executeNodes (registry meaning fallback (sites.map (fun site => site.generic.stage)))
        (sites.map (fun site => site.generic.stage.node)) (before initial)) := by
  rw [executeSource_eq_graph meaning initialCells fallback matchedPrefix validPrefix sites bound unique]

end Merlin.GraphRegionsDeclared.Capture
