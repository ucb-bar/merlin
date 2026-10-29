/- Bind the multi-site
source graph bridge to an entire independent parent AST, including every source
operation, all outputs and actual SSA identities. The supported parent is one
uniform static integer-tensor function with a literal/static-empty prefix and
an ordered set of parallel generics. Extra operations/functions refuse. -/
import Merlin.GraphRegionsDeclaredCaptureGraph

namespace Merlin.GraphRegionsDeclared.Capture
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

structure GraphParent where
  name : String
  visibility : Option String
  inputCount : Nat
  shape : List Nat
  width : Nat
  leading : List Prefix.Instruction
  sites : List Site
  observations : List Nat

def GraphParent.tensorType (parent : GraphParent) : Ty :=
  .tensor (parent.shape.map some) (.int parent.width) none

def GraphParent.graph (parent : GraphParent) : Graph :=
  ⟨parent.sites.map (fun site => site.generic.stage.node), List.range parent.inputCount, parent.observations, []⟩

def Site.privateNames (site : Site) : List Nat :=
  site.generic.names.filter (fun name => !(site.references.map (·.sourceId)).contains name) ++ [site.generic.unused]

def GraphParent.globalIds (parent : GraphParent) : List Nat :=
  List.range parent.inputCount ++ parent.leading.map Prefix.Instruction.result ++
    parent.sites.map (fun site => site.generic.stage.output)

def GraphParent.module (parent : GraphParent) : Module :=
  let arguments := (List.range parent.inputCount).map fun id => (⟨id, parent.tensorType⟩ : Value)
  let attrs := [("function_type", .type (.function (List.replicate parent.inputCount parent.tensorType)
      (List.replicate parent.observations.length parent.tensorType))), ("sym_name", .str parent.name)] ++
    parent.visibility.toList.map (fun value => ("sym_visibility", .str value))
  let body := parent.leading.map Prefix.Instruction.emit ++
    parent.sites.map (fun site => originalGeneric site.generic site.references) ++
      [.mk "func.return" [] parent.observations [] [] [] []]
  ⟨[.mk "func.func" [] [] attrs [] [.mk [.mk arguments body]] []], [], []⟩

def GraphParent.Valid (parent : GraphParent) : Prop :=
  0 < parent.width ∧ parent.width ≤ 256 ∧ parent.width % 8 = 0 ∧
  (∀ dimension ∈ parent.shape, 0 < dimension) ∧
  Prefix.Valid (List.range parent.inputCount) parent.leading ∧ parent.graph.Valid ∧
  SitesBound (parent.leading.map Prefix.Instruction.emit) parent.sites ∧
  (∀ (index : Nat) (site : Site), parent.sites[index]? = some site →
    site.generic.stage.ordinal = parent.leading.length + index ∧ site.generic.shape = parent.shape ∧
    site.generic.stage.program.output = parent.width ∧
    site.generic.stage.inputTypes = List.replicate site.generic.stage.inputs.length (typeTag parent.tensorType) ∧
    site.generic.stage.outputType = typeTag parent.tensorType ∧
    site.generic.stage.output ∉ parent.leading.map Prefix.Instruction.result ∧
    (∃ (ordinal : Nat), parent.leading[ordinal]? =
      some (Prefix.Instruction.empty site.generic.initializer parent.shape parent.width))) ∧
  (parent.sites.flatMap Site.privateNames).Nodup ∧
  (∀ id ∈ parent.sites.flatMap Site.privateNames, id ∉ parent.globalIds)

/-- Acceptance consumes both complete actual parent buffers. Equality against
the emitted grammar is a kernel-checked structural relation, not a hash law.
All source sites are covered because the actual body must equal the entire list. -/
def GraphParent.Accepts (parent : GraphParent) (actual retained : Module) : Prop :=
  actual = parent.module ∧ retained = actual ∧ parent.Valid

/-- Candidate interpretation of the complete recognized parent. Prefix literals
are executed independently of runtime tensor inputs; uninitialized empty results
are excluded from all source reads by checked scalar-region syntax. -/
inductive GraphParentExecution {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) : Module → StateEffects.State (Tensor elements α) →
      List (Tensor elements α) → Prop where
  | evaluated (parent : GraphParent) (valid : parent.Valid) (initialCells : Prefix.Cells α)
      (initial : StateEffects.State (Tensor elements α))
      (extent : elements = parent.shape.foldl (· * ·) 1) :
    GraphParentExecution meaning parent.module initial
      (parent.observations.map (executeSource meaning parent.leading initialCells parent.sites initial))

/-- Actual complete-parent correspondence to existing GraphRegions execution.
The interpretation is additive; actual compiler-buffer association, native
fidelity and frozen-semantics identification are separate obligations. -/
theorem actual_parent_graph_correspondence {α : Type} [Inhabited α] {elements : Nat}
    {parent : GraphParent} {actual retained : Module}
    (accepted : parent.Accepts actual retained)
    (meaning : Primitive → List α → α) (initialCells : Prefix.Cells α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (unique : ((parent.sites.map (fun site => site.generic.stage)).map ScalarStage.node).Nodup)
    (initial : StateEffects.State (Tensor elements α)) (extent : elements = parent.shape.foldl (· * ·) 1) :
    actual = retained ∧
    GraphParentExecution meaning actual initial
      (parent.graph.observations.map (executeNodes
        (registry meaning fallback (parent.sites.map (fun site => site.generic.stage))) parent.graph.nodes initial)) := by
  obtain ⟨sameModule, retained, valid⟩ := accepted
  refine ⟨retained.symm, ?_⟩
  rw [sameModule]
  have corresponding := executeSource_eq_graph meaning initialCells fallback
    (show Prefix.Matches (parent.leading.map Prefix.Instruction.emit) parent.leading from rfl)
    valid.2.2.2.2.1 parent.sites valid.2.2.2.2.2.2.1 unique initial
  have observed := congrArg (fun final => parent.observations.map final) corresponding
  change GraphParentExecution meaning parent.module initial
    (parent.observations.map (executeNodes _ (parent.sites.map (fun site => site.generic.stage.node)) initial))
  rw [← observed]
  exact .evaluated parent valid initialCells initial extent

end Merlin.GraphRegionsDeclared.Capture
