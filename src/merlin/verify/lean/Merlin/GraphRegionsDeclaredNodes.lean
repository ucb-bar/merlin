/- Construct source node meanings from the admitted ordered scalar declarations.
Tensor elements are initialized mathematical values; a total primitive meaning
is explicit. This module claims neither frozen Core.Sem nor machine fidelity. -/
import Merlin.GraphRegionsDeclaredEvaluation

namespace Merlin.GraphRegionsDeclared
open Merlin.GraphRegions

structure ScalarStage where
  ordinal : Nat
  role : String
  inputs : List Nat
  output : Nat
  inputTypes : List String
  outputType : String
  program : Program
  deriving DecidableEq, Repr

def ScalarStage.node (stage : ScalarStage) : Node :=
  ⟨stage.ordinal, stage.role, stage.inputs, [stage.output], stage.inputTypes, [stage.outputType]⟩

abbrev Tensor (elements : Nat) (α : Type) := Fin elements → α

def tensorEvaluate {α : Type} [Inhabited α] {elements : Nat} (program : Program)
    (meaning : Primitive → List α → α) (arguments : List (Tensor elements α)) : Tensor elements α :=
  fun index => program.evaluate meaning (fun slot => (arguments[slot]?.map (· index)).getD default)

def ScalarStage.step {α : Type} [Inhabited α] {elements : Nat} (stage : ScalarStage)
    (meaning : Primitive → List α → α) (initial : StateEffects.State (Tensor elements α)) :
    StateEffects.State (Tensor elements α) :=
  bind initial stage.output (tensorEvaluate stage.program meaning (stage.inputs.map initial))

/-- Read locality is derived from the actual ordered operand list. The result is
computed from the checked scalar program, rather than an arbitrary total Action. -/
def ScalarStage.action {α : Type} [Inhabited α] {elements : Nat} (stage : ScalarStage)
    (meaning : Primitive → List α → α) : StateEffects.Action stage.node.footprint (Tensor elements α) where
  step := stage.step meaning
  frame := by
    intro initial slot outside
    simp only [ScalarStage.node, Node.footprint, List.mem_singleton] at outside
    simp [ScalarStage.step, bind, outside]
  locality := by
    intro first second agree slot output
    have same : stage.inputs.map first = stage.inputs.map second := List.map_congr_left agree
    simp only [ScalarStage.node, Node.footprint, List.mem_singleton] at output
    subst slot
    simp only [ScalarStage.step, bind, ↓reduceIte, same]

/-- Width soundness remains separately available through Program.typed and its
explicit primitive typing contract. This adapter uses the inhabited initialized
tensor domain; it does not assert global machine initialization. -/
def ScalarStage.typedAction {α : Type} [Inhabited α] {elements : Nat} (stage : ScalarStage)
    (meaning : Primitive → List α → α) : TypedAction stage.node (Tensor elements α) (fun _ _ => True) where
  action := stage.action meaning
  preservesTypes := by intros; trivial

def executeStages {α : Type} [Inhabited α] {elements : Nat}
    (meaning : Primitive → List α → α) : List ScalarStage →
    StateEffects.State (Tensor elements α) → StateEffects.State (Tensor elements α)
  | [], initial => initial
  | stage :: rest, initial => executeStages meaning rest (stage.step meaning initial)

def registry {α : Type} [Inhabited α] {elements : Nat} (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True)) :
    List ScalarStage → ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True)
  | [], node => fallback node
  | stage :: rest, node =>
    if equal : stage.node = node then equal ▸ stage.typedAction primitive
    else registry primitive fallback rest node

theorem registry_steps {α : Type} [Inhabited α] {elements : Nat}
    (primitive : Primitive → List α → α)
    (fallback : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (stages : List ScalarStage) (unique : (stages.map ScalarStage.node).Nodup) :
    ∀ stage ∈ stages, (registry primitive fallback stages stage.node).action.step = stage.step primitive := by
  induction stages with
  | nil => intro stage member; cases member
  | cons first rest ih =>
    intro stage member
    rcases List.mem_cons.mp member with same | member
    · subst stage
      simp [registry, ScalarStage.typedAction, ScalarStage.action]
    · have different : first.node ≠ stage.node := by
        intro equal
        exact (List.nodup_cons.mp unique).1 (List.mem_map.mpr ⟨stage, member, equal.symm⟩)
      simp only [registry, different, ↓reduceDIte]
      exact ih (List.nodup_cons.mp unique).2 stage member

/-- The only matching premise here is a registry lookup for each actual node.
The scalar meaning itself has already been constructed above. -/
theorem executeStages_nodes {α : Type} [Inhabited α] {elements : Nat}
    (primitive : Primitive → List α → α)
    (meaning : ∀ node, TypedAction node (Tensor elements α) (fun _ _ => True))
    (stages : List ScalarStage)
    (lookup : ∀ stage ∈ stages, (meaning stage.node).action.step = stage.step primitive)
    (initial : StateEffects.State (Tensor elements α)) :
    executeStages primitive stages initial = executeNodes meaning (stages.map ScalarStage.node) initial := by
  induction stages generalizing initial with
  | nil => rfl
  | cons stage rest ih =>
    change executeStages primitive rest (stage.step primitive initial) =
      executeNodes meaning (rest.map ScalarStage.node) ((meaning stage.node).action.step initial)
    rw [lookup stage (by simp)]
    exact ih (fun next member => lookup next (by simp [member])) _

end Merlin.GraphRegionsDeclared
