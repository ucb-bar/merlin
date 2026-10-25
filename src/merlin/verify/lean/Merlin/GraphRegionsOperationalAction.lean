/- Constructed source meanings for the existing GraphRegions.Action model.
Outside the initialized source domain, integer element extraction maps malformed
or poison values to zero. Agreement with opSem is proved only on the named domain.
This is not a native implementation or a change to the frozen source semantics. -/
import Merlin.GraphRegionsSourceStage

namespace Merlin.GraphRegionsSource
open MerlinVerify

structure Entry where
  stage : Stage
  key : String
  deriving DecidableEq, Repr

def Entry.node (entry : Entry) : GraphRegions.Node := entry.stage.node entry.key

def defaultTensor (type : TensorType) : Tensor :=
  Tensor.full type.shape (.int type.width) (.some (.bits type.width 0))

theorem default_initialized (type : TensorType) : type.initialized (defaultTensor type) := by
  refine ⟨rfl, rfl, ?_⟩
  intro point bounds
  refine ⟨0, ?_⟩
  simp [defaultTensor, Tensor.full, Tensor.canon, bounds]

def selectEntry (entries : List Entry) (types : Nat → TensorType) (node : GraphRegions.Node) : Option Entry :=
  entries.find? fun entry => decide (entry.node = node ∧ types entry.stage.result = entry.stage.output)

theorem selectEntry_sound {entries : List Entry} {types : Nat → TensorType} {node : GraphRegions.Node}
    {entry : Entry} (selected : selectEntry entries types node = some entry) :
    entry.node = node ∧ types entry.stage.result = entry.stage.output := by
  have predicate := List.find?_some selected
  exact of_decide_eq_true predicate

def compute (entry : Entry) (state : Nat → Tensor) : Tensor :=
  entry.stage.body.tensor entry.stage.shape (entry.stage.inputs.map state)

def sourceStep (entries : List Entry) (types : Nat → TensorType) (node : GraphRegions.Node)
    (state : Nat → Tensor) : Nat → Tensor := fun slot =>
  if slot ∈ node.results then
    match selectEntry entries types node with
    | some entry => if slot = entry.stage.result then compute entry state else defaultTensor (types slot)
    | none => defaultTensor (types slot)
  else state slot

/-- Concrete typed total action, constructed from the checked source body.
Unknown descriptors have an explicit dummy interpretation and cannot satisfy the
separate source/program acceptance conditions. -/
def sourceMeaning (entries : List Entry) (types : Nat → TensorType) (node : GraphRegions.Node) :
    GraphRegions.TypedAction node Tensor (fun slot tensor => (types slot).initialized tensor) where
  action := {
    step := sourceStep entries types node
    frame := by
      intro state slot outside
      change slot ∉ node.results at outside
      simp [sourceStep, outside]
    locality := by
      intro first second inputs slot written
      change slot ∈ node.results at written
      simp only [sourceStep, written, ↓reduceIte]
      cases selected : selectEntry entries types node with
      | none => rfl
      | some entry =>
        have matchNode := (selectEntry_sound selected).1
        have same : entry.stage.inputs.map first = entry.stage.inputs.map second := by
          apply List.map_congr_left
          intro id member
          apply inputs id
          change id ∈ node.operands
          simpa only [← matchNode, Entry.node, Stage.node] using member
        simp only [compute, same] }
  preservesTypes := by
    intro state typed slot
    by_cases written : slot ∈ node.results
    · simp only [sourceStep, written, ↓reduceIte]
      cases selected : selectEntry entries types node with
      | none => exact default_initialized _
      | some entry =>
        by_cases output : slot = entry.stage.result
        · subst slot
          simp only [↓reduceIte, (selectEntry_sound selected).2]
          exact tensor_result_initialized _ _ _
        · simp only [output, ↓reduceIte]
          exact default_initialized _
    · simpa only [sourceStep, written, ↓reduceIte] using typed slot

theorem sourceStep_selected {entries : List Entry} {types : Nat → TensorType} {node : GraphRegions.Node}
    {entry : Entry} (selected : selectEntry entries types node = some entry) (state : Nat → Tensor) :
    sourceStep entries types node state = put state entry.stage.result (compute entry state) := by
  have matchNode := (selectEntry_sound selected).1
  funext slot
  simp only [sourceStep, selected]
  by_cases output : slot = entry.stage.result <;>
    simp [← matchNode, Entry.node, Stage.node, put, output]

/-- Source execution agrees with the constructed action at the complete result.
The caller supplies only entry input/initializer facts, not source success. -/
theorem source_stage_agrees {entries : List Entry} {types : Nat → TensorType} {node : GraphRegions.Node}
    {entry : Entry} (selected : selectEntry entries types node = some entry) (valid : entry.stage.Valid)
    (state : Nat → Tensor)
    (inputs : InputsInitialized entry.stage.body.inputs entry.stage.shape (entry.stage.inputs.map state))
    (init : Tensor) (shape : init.shape = entry.stage.shape) (element : init.elt = .int entry.stage.body.width)
    (ctx : Ctx) (call : CallSem) (environment : Env)
    (captures : ScalarAgree (constantTypes entry.stage.body.captures) environment
      (constantState entry.stage.body.captures)) :
    Sem.Ops.opSem ctx call entry.stage.operation
      ((entry.stage.inputs.map state).map RVal.tensor ++ [.tensor init])
      (evalRegions Sem.Ops.opSem call environment entry.stage.operation.regions) =
      .ok [.tensor ((sourceMeaning entries types node).action.step state entry.stage.result)] := by
  rw [stage_execution entry.stage valid _ inputs init shape element ctx call environment captures]
  rw [show (sourceMeaning entries types node).action.step = sourceStep entries types node from rfl,
    sourceStep_selected selected]
  simp [put, compute]

end Merlin.GraphRegionsSource
