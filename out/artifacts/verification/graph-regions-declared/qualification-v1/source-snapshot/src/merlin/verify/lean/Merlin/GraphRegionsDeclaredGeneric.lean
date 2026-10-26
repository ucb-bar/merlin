/- Structural correspondence from an actual parallel generic region to the
matched closed scalar function. Global SSA and the unread output initializer
remain distinct. This is an additive pointwise interpretation, not Core.Sem. -/
import Merlin.GraphRegionsDeclaredNodes

namespace Merlin.GraphRegionsDeclared
open MerlinVerify

def renameSlot (names : List Nat) (slot : Nat) : Nat := names[slot]?.getD 0

def renamedOperations (program : Program) (names : List Nat) : Nat → List Instruction → List Op
  | _, [] => [.mk "linalg.yield" [] [renameSlot names program.returned] [] [] [] []]
  | next, instruction :: rest =>
    .mk instruction.primitive.name [⟨renameSlot names next, .int instruction.primitive.width⟩]
      (instruction.operands.map (renameSlot names)) instruction.primitive.attrs [] [] [] ::
      renamedOperations program names (next + 1) rest

def identityMap (rank : Nat) : Attr :=
  .affineMap ⟨rank, 0, (List.range rank).map AffineExpr.dim⟩

def genericOperation (stage : ScalarStage) (shape names : List Nat) (initializer unused : Nat) : Op :=
  let type := Ty.tensor (shape.map some) (.int stage.program.output) none
  .mk "linalg.generic" [⟨stage.output, type⟩] (stage.inputs ++ [initializer])
    [("indexing_maps", .array (List.replicate (stage.inputs.length + 1) (identityMap shape.length))),
     ("iterator_types", .array (List.replicate shape.length (.enum "linalg" "iterator_type" "parallel"))),
     ("operandSegmentSizes", .denseArray (.int 32) [Int.ofNat stage.inputs.length, 1])]
    [] [.mk [.mk
      (((List.range stage.program.inputs.length).zip stage.program.inputs).map
        (fun (slot, width) => ⟨renameSlot names slot, .int width⟩) ++ [⟨unused, .int stage.program.output⟩])
      (renamedOperations stage.program names stage.program.inputs.length stage.program.instructions)]] []

def genericWellFormed (stage : ScalarStage) (shape names : List Nat) (unused : Nat) : Prop :=
  stage.program.Valid ∧ stage.inputs.length = stage.program.inputs.length ∧
  stage.program.inputs = List.replicate stage.inputs.length stage.program.output ∧
  names.length = stage.program.inputs.length + stage.program.instructions.length ∧
  names.Nodup ∧ unused ∉ names ∧ (∀ dimension ∈ shape, 0 < dimension)

instance (stage : ScalarStage) (shape names : List Nat) (unused : Nat) :
    Decidable (genericWellFormed stage shape names unused) := by unfold genericWellFormed; infer_instance

def checkGeneric (actual : Op) (stage : ScalarStage) (shape names : List Nat) (initializer unused : Nat) : Prop :=
  actual = genericOperation stage shape names initializer unused ∧ genericWellFormed stage shape names unused

/-- Full generic operation, properties, maps, ordered SSA names and unread output
argument are checked. No scalar expression hash or initializer value is used. -/
theorem checkGeneric_sound {actual : Op} {stage : ScalarStage} {shape names : List Nat} {initializer unused : Nat}
    (accepted : checkGeneric actual stage shape names initializer unused) :
    actual = genericOperation stage shape names initializer unused ∧ stage.program.Valid ∧ unused ∉ names :=
  ⟨accepted.1, accepted.2.1, accepted.2.2.2.2.2.2.1⟩

/-- The admitted exact generic is interpreted pointwise using the complete
checked canonical SSA function. This supplies the source/declaration relation
for this structural model. The historical frozen operational bridge remains
UNQUALIFIED and is not assumed or imported. -/
inductive GenericExecution {α : Type} [Inhabited α] {elements : Nat}
    (primitive : Primitive → List α → α) : Op → StateEffects.State (Tensor elements α) →
      StateEffects.State (Tensor elements α) → Prop where
  | pointwise (stage : ScalarStage) (shape names : List Nat) (initializer unused : Nat)
      (valid : genericWellFormed stage shape names unused) (initial : StateEffects.State (Tensor elements α)) :
    GenericExecution primitive (genericOperation stage shape names initializer unused) initial (stage.step primitive initial)

theorem generic_execution {α : Type} [Inhabited α] {elements : Nat}
    {actual : Op} {stage : ScalarStage} {shape names : List Nat} {initializer unused : Nat}
    (accepted : checkGeneric actual stage shape names initializer unused)
    (primitive : Primitive → List α → α) (initial : StateEffects.State (Tensor elements α)) :
    GenericExecution primitive actual initial (stage.step primitive initial) := by
  rw [accepted.1]
  exact .pointwise stage shape names initializer unused accepted.2 initial

end Merlin.GraphRegionsDeclared
