/- Actual generic operation construction/recognition and frozen opSem agreement.
No source operation is represented by an assumed total-action summary. -/
import Merlin.GraphRegionsTensorGeneric
import Merlin.GraphRegionsFusion

namespace Merlin.GraphRegionsSource
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open MerlinVerify.Sem.Ops.Linalg

structure TensorType where
  shape : List Nat
  width : Nat
  deriving DecidableEq, Repr

def TensorType.ty (type : TensorType) : Ty := .tensor (type.shape.map some) (.int type.width) none

def TensorType.Valid (type : TensorType) : Prop :=
  0 < type.width ∧ type.width % 8 = 0 ∧ ∀ dimension ∈ type.shape, 0 < dimension

def TensorType.initialized (type : TensorType) : Tensor → Prop := TensorInitialized type.shape type.width

def TensorType.render (type : TensorType) : String :=
  "tensor<" ++ String.join (type.shape.map fun dimension => toString dimension ++ "x") ++ "i" ++
    toString type.width ++ ">"

structure Stage where
  ordinal : Nat
  inputs : List Nat
  destination : Nat
  result : Nat
  shape : List Nat
  body : Body
  deriving DecidableEq, Repr

def Stage.output (stage : Stage) : TensorType := ⟨stage.shape, stage.body.width⟩
def Stage.inputTypes (stage : Stage) : List TensorType :=
  stage.body.inputs.map fun (_, width) => ⟨stage.shape, width⟩

def Stage.node (stage : Stage) (key : String) : GraphRegions.Node :=
  ⟨stage.ordinal, key, stage.inputs, [stage.result], stage.inputTypes.map TensorType.render,
    [stage.output.render]⟩

def Stage.attrs (stage : Stage) : List (String × Attr) :=
  [("indexing_maps", .array (List.replicate (stage.inputs.length + 1)
      (.affineMap (identityMap stage.shape.length)))),
   ("iterator_types", .array (List.replicate stage.shape.length (.enum "linalg" "iterator_type" "parallel"))),
   ("operandSegmentSizes", .denseArray (.int 32) [Int.ofNat stage.inputs.length, 1])]

def Stage.operation (stage : Stage) : Op :=
  .mk "linalg.generic" [⟨stage.result, stage.output.ty⟩] (stage.inputs ++ [stage.destination])
    stage.attrs [] [stage.body.region] []

def Stage.Valid (stage : Stage) : Prop :=
  stage.output.Valid ∧ (∀ type ∈ stage.inputTypes, type.Valid) ∧
    stage.inputs.length = stage.body.inputs.length ∧ stage.body.Valid

theorem Stage.modelled (stage : Stage) : unmodelledAttr? stage.operation.blind = none := by rfl

theorem Stage.nonterminator (stage : Stage) : stage.operation.isTerminator = false := by rfl

theorem decode_identity_maps (count rank : Nat) :
    (List.replicate count (Attr.affineMap (identityMap rank))).mapM
      (fun | Attr.affineMap map => some map | _ => none) =
      some (List.replicate count (identityMap rank)) := by
  induction count with
  | zero => rfl
  | succ count ih => simp [List.replicate_succ, ih]

theorem decode_parallel_iterators (count : Nat) :
    (List.replicate count (Attr.enum "linalg" "iterator_type" "parallel")).mapM
      (fun | .enum "linalg" "iterator_type" "parallel" => some IterKind.parallel
           | .enum "linalg" "iterator_type" "reduction" => some IterKind.reduction
           | _ => none) = some (List.replicate count .parallel) := by
  induction count with
  | zero => rfl
  | succ count ih => simp [List.replicate_succ, ih]

/-- The full source header, scalar closure and tensor overwrite execute using the
existing opSem. The destination's elements remain arbitrary, including poison. -/
theorem stage_execution (stage : Stage) (valid : stage.Valid) (inputs : List Tensor)
    (initialized : InputsInitialized stage.body.inputs stage.shape inputs)
    (init : Tensor) (initShape : init.shape = stage.shape) (initElement : init.elt = .int stage.body.width)
    (ctx : Ctx) (call : CallSem) (environment : Env)
    (captures : ScalarAgree (constantTypes stage.body.captures) environment (constantState stage.body.captures)) :
    opSem ctx call stage.operation (inputs.map RVal.tensor ++ [.tensor init])
      (evalRegions opSem call environment stage.operation.regions) =
      .ok [.tensor (stage.body.tensor stage.shape inputs)] := by
  have count : inputs.length = stage.inputs.length := initialized.length_eq.symm.trans valid.2.2.1.symm
  have maps : attrAffineMaps? stage.operation.blind "indexing_maps" =
      some (List.replicate (stage.inputs.length + 1) (identityMap stage.shape.length)) := by
    change (List.replicate (stage.inputs.length + 1) (Attr.affineMap (identityMap stage.shape.length))).mapM
      (fun | Attr.affineMap map => some map | _ => none) = _
    exact decode_identity_maps (stage.inputs.length + 1) stage.shape.length
  have iterators : iterKinds? stage.operation.blind =
      some (List.replicate stage.shape.length .parallel) := by
    change (List.replicate stage.shape.length (Attr.enum "linalg" "iterator_type" "parallel")).mapM
      (fun | .enum "linalg" "iterator_type" "parallel" => some IterKind.parallel
           | .enum "linalg" "iterator_type" "reduction" => some IterKind.reduction
           | _ => none) = _
    exact decode_parallel_iterators stage.shape.length
  have split : splitInsOuts stage.operation.blind (inputs.map RVal.tensor ++ [.tensor init]) =
      some (inputs.map RVal.tensor, [.tensor init]) := by
    simp [splitInsOuts, attrDenseArray?, Stage.operation, Stage.attrs, Op.blind,
      Op.attr?, Op.props, Op.attrs, List.lookup, ← count]
  have execution := generic_execution valid.2.2.2 initialized init initShape initElement ctx call environment captures
  have dialect : dialectOf stage.operation.blind.name = "linalg" := by
    change dialectOf "linalg.generic" = "linalg"
    cbv
  rw [opSem_modelled stage.modelled]
  simp only [dispatch, dialect]
  change Linalg.sem ctx call stage.operation.blind (inputs.map RVal.tensor ++ [.tensor init])
    (evalRegions opSem call environment [stage.body.region]) = _
  have name : stage.operation.blind.name = "linalg.generic" := rfl
  have results : stage.operation.blind.results.length = 1 := rfl
  simp only [Linalg.sem, name, results, evalRegions]
  rw [split, maps, iterators]
  simp only [← count, execution, Outcome.bindOp_eq, Outcome.pure_eq, Outcome.bind_ok]
  rfl

end Merlin.GraphRegionsSource
