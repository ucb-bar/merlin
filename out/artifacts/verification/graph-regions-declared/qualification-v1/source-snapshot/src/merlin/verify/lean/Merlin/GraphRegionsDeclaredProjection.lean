/- Recompute graph roles, SSA uses and observers from the complete authoritative
module AST. Unknown surrounding node meanings remain explicit total/pure Actions;
no claim is made that those operations have frozen operational interpretations. -/
import Merlin.GraphRegionsDeclaredGeneric

namespace Merlin.GraphRegionsDeclared
open MerlinVerify Merlin.GraphRegions

def typeTag : Ty → String
  | .int width => "i" ++ toString width
  | .tensor shape (.int width) none =>
    "tensor[" ++ String.intercalate "," (shape.map (fun n => n.map toString |>.getD "?")) ++ "]i" ++ toString width
  | _ => "unsupported-context-type"

def typedIds (values : List Value) : List (Nat × String) := values.map fun value => (value.id, typeTag value.ty)

structure GenericSite where
  stage : ScalarStage
  shape : List Nat
  names : List Nat
  initializer : Nat
  unused : Nat
  deriving DecidableEq, Repr

def findSite (sites : List GenericSite) (ordinal : Nat) : Option GenericSite :=
  sites.find? (fun site => site.stage.ordinal == ordinal)

def projectedNode (types : List (Nat × String)) (ordinal : Nat) (op : Op) : Node :=
  ⟨ordinal, op.name, op.operands, op.results.map (·.id),
    op.operands.map (fun slot => (types.lookup slot).getD "missing-type"), op.results.map (fun value => typeTag value.ty)⟩

def projectOps (sites : List GenericSite) : List (Nat × String) → Nat → List Op → List Node
  | _, _, [] => []
  | types, ordinal, op :: rest =>
    (match findSite sites ordinal with
      | some site => site.stage.node
      | none => projectedNode types ordinal op) ::
      projectOps sites (types ++ typedIds op.results) (ordinal + 1) rest

def SitesMatch (sites : List GenericSite) : List (Nat × String) → Nat → List Op → Prop
  | _, _, [] => True
  | types, ordinal, op :: rest =>
    (match findSite sites ordinal with
      | some site => checkGeneric op site.stage site.shape site.names site.initializer site.unused ∧
          site.stage.inputTypes = site.stage.inputs.map (fun slot => (types.lookup slot).getD "missing-type") ∧
          site.stage.outputType = typeTag (.tensor (site.shape.map some) (.int site.stage.program.output) none)
      | none => True) ∧ SitesMatch sites (types ++ typedIds op.results) (ordinal + 1) rest

def projectModule (actual : Module) (sites : List GenericSite) : Option Graph :=
  match actual.ops with
  | [.mk "func.func" [] [] _ [] [.mk [.mk arguments operations]] []] =>
    match operations.getLast? with
    | some (.mk "func.return" [] observations [] [] [] []) =>
      some ⟨projectOps sites (typedIds arguments) 0 operations.dropLast, arguments.map (·.id), observations, []⟩
    | _ => none
  | _ => none

/-- Complete AST equality binds all retained functions/attributes/regions and
unchanged observers. SitesMatch additionally checks each selected scalar body;
projectModule recomputes graph SSA/observer fields rather than trusting a hint. -/
def SourceBinding (before after : Module) (sites : List GenericSite) (graph : Graph) : Prop :=
  before = after ∧ projectModule before sites = some graph ∧
  match before.ops with
  | [.mk "func.func" [] [] _ [] [.mk [.mk arguments operations]] []] =>
    SitesMatch sites (typedIds arguments) 0 operations.dropLast ∧
    (∀ site ∈ sites, ∃ op, operations[site.stage.ordinal]? = some op ∧
      checkGeneric op site.stage site.shape site.names site.initializer site.unused)
  | _ => False

theorem source_binding_retained {before after : Module} {sites : List GenericSite} {graph : Graph}
    (accepted : SourceBinding before after sites graph) :
    before = after ∧ projectModule before sites = some graph := ⟨accepted.1, accepted.2.1⟩

end Merlin.GraphRegionsDeclared
