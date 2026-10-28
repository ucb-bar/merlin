/- Canonical six-operation mask application: invert kept positions, construct
-Inf, then select scores. This does not specify or assume the mask producer. -/
import MerlinVerify.Rules.W8A8AttentionExecution
import MerlinVerify.Rules.W8A8AttentionMaskTensor

namespace MerlinVerify.Rules.W8A8AttentionMaskApply

open MerlinVerify Check Sem.Ops Sem.Ops.Linalg W8A8AttentionTemplate

def boolTy (shape : List Nat) : Ty := .tensor (shape.map some) (.int 1) none

def booleanEmpty (shape : List Nat) : Op :=
  .mk "tensor.empty" [⟨2, boolTy shape⟩] [] [] [] [] []

def inversion (shape : List Nat) : Op :=
  .mk "linalg.generic" [⟨3, boolTy shape⟩] [1, 2]
    [("indexing_maps", .array [.affineMap (identityMap shape.length), .affineMap (identityMap shape.length)]),
     ("iterator_types", .array (List.replicate shape.length (.enum "linalg" "iterator_type" "parallel"))),
     ("operandSegmentSizes", .denseArray (.int 32) [1, 1])] []
    [freshRegion 1000 W8A8AttentionMaskBody.inversionRegion] []

def selectMaps (shape : List Nat) (maskMap : AffineMap) : List AffineMap :=
  [maskMap, W8A8AttentionMaskTensor.scalarMap shape.length, identityMap shape.length, identityMap shape.length]

def selection (shape : List Nat) (maskMap : AffineMap) : Op :=
  generic shape [3, 5, 0] 6 7 (selectMaps shape maskMap) (freshRegion 1100 W8A8AttentionMaskBody.selectRegion)

def operations (maskShape shape : List Nat) (maskMap : AffineMap) : List Op :=
  [booleanEmpty maskShape, inversion maskShape, constant 4 0xff800000, splat [] 4 5,
    empty shape 6, selection shape maskMap]

theorem static_bool (shape : List Nat) : (boolTy shape).staticTensor? = some (shape, .int 1) := by
  have dimensions : (shape.map some).mapM id = some shape := by
    induction shape with
    | nil => rfl
    | cons head tail induction => simp [induction]
  simp [boolTy, Ty.staticTensor?, dimensions]

private theorem dialect_empty : dialectOf "tensor.empty" = "tensor" := by cbv
private theorem dialect_generic : dialectOf "linalg.generic" = "linalg" := by cbv

theorem empty_execute (ctx : Ctx) (call : CallSem) (shape : List Nat) :
    W8A8SourceSemantics.opSem ctx call (booleanEmpty shape) [] [] =
      .ok [.tensor (Tensor.full shape (.int 1) .poison)] := by
  simp [W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
    Sem.Ops.opSem, unmodelledAttr?, interpretedAttrs, dispatch, booleanEmpty, Op.blind,
    Op.name, Op.results, Op.props, Op.attrs, Op.successors, Tensor.sem,
    dialect_empty, Tensor.resultTensorTy, static_bool]

theorem inversion_body : PolicyW8A8AttentionMaskBody.check .inversion
    (freshRegion 1000 W8A8AttentionMaskBody.inversionRegion) = true := by cbv

theorem selection_body : PolicyW8A8AttentionMaskBody.check .select
    (freshRegion 1100 W8A8AttentionMaskBody.selectRegion) = true := by cbv

theorem inversion_execute (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat)
    (geometry : StructuredOk [identityMap shape.length, identityMap shape.length] shape.length
      [.tensor shape (.int 1), .tensor shape (.int 1)] shape)
    (values : Idx → Val) (input : Idx → Bool)
    (initialized : ∀ point, Idx.InBounds shape point → values point = W8A8AttentionMaskBody.boolean (input point)) :
    W8A8SourceSemantics.opSem ctx call (inversion shape)
      [.tensor ⟨shape, .int 1, values⟩, .tensor (Tensor.full shape (.int 1) .poison)]
      (evalRegions W8A8SourceSemantics.opSem call env (inversion shape).regions) =
      .ok [.tensor (Tensor.canon shape (.int 1) fun point => W8A8AttentionMaskBody.boolean (!(input point)))] := by
  have executed := W8A8AttentionMaskTensor.inversion_sound inversion_body ctx call env shape
    (List.replicate shape.length .parallel) (by simpa only [List.length_replicate] using geometry)
    values (Tensor.full shape (.int 1) .poison).get input initialized
  have ordinary : W8A8SourceSemantics.isSource (inversion shape).name = false := by cbv
  have modelled : unmodelledAttr? (inversion shape).blind = none := by
    simp [inversion, Op.blind, unmodelledAttr?, interpretedAttrs, Op.name, Op.props, Op.attrs]
  rw [W8A8SourceSemantics.unchanged ordinary, linalg_sem_of modelled dialect_generic (Or.inl rfl)]
  have split : splitInsOuts (inversion shape)
      [RVal.tensor ⟨shape, .int 1, values⟩, .tensor (Tensor.full shape (.int 1) .poison)] =
      some ([.tensor ⟨shape, .int 1, values⟩], [.tensor (Tensor.full shape (.int 1) .poison)]) := by rfl
  have selectedMaps : attrAffineMaps? (inversion shape) "indexing_maps" =
      some [identityMap shape.length, identityMap shape.length] := by rfl
  have selectedIters : iterKinds? (inversion shape) = some (List.replicate shape.length .parallel) := by
    have decoded (rank : Nat) :
        (List.replicate rank (Attr.enum "linalg" "iterator_type" "parallel")).mapM (fun
          | .enum "linalg" "iterator_type" "parallel" => some IterKind.parallel
          | .enum "linalg" "iterator_type" "reduction" => some IterKind.reduction
          | _ => none) = some (List.replicate rank IterKind.parallel) := by
      induction rank with
      | zero => rfl
      | succ rank induction => simp [List.replicate_succ, induction]
    simp only [inversion, iterKinds?, Op.attr?, Op.props, Op.attrs, List.lookup]
    exact decoded shape.length
  have regions : (inversion shape).regions = [freshRegion 1000 W8A8AttentionMaskBody.inversionRegion] := rfl
  simp only [regions, evalRegions, Linalg.sem, Op.blind_name,
    show (inversion shape).name = "linalg.generic" from rfl, splitInsOuts_blind, split,
    attrAffineMaps?_blind, selectedMaps, iterKinds?_blind, selectedIters]
  change genericSem ctx _ _ _ [.tensor (Tensor.full shape (.int 1) .poison)] _ = _ at executed
  rw [executed]
  rfl

structure Geometry (maskShape shape : List Nat) (maskMap : AffineMap) : Prop where
  inversion : StructuredOk [identityMap maskShape.length, identityMap maskShape.length] maskShape.length
    [.tensor maskShape (.int 1), .tensor maskShape (.int 1)] maskShape
  selection : StructuredOk (selectMaps shape maskMap) shape.length
    [.tensor maskShape (.int 1), .tensor [] (.float .f32),
      .tensor shape (.float .f32), .tensor shape (.float .f32)] shape

def geometryCheck (maskShape shape : List Nat) (maskMap : AffineMap) : Bool :=
  decide (
    structuredRanges? [identityMap maskShape.length, identityMap maskShape.length] maskShape.length
      [.tensor maskShape (.int 1), .tensor maskShape (.int 1)] = some maskShape ∧
    structuredRanges? (selectMaps shape maskMap) shape.length
      [.tensor maskShape (.int 1), .tensor [] (.float .f32),
        .tensor shape (.float .f32), .tensor shape (.float .f32)] = some shape)

theorem geometryCheck_sound {maskShape shape : List Nat} {maskMap : AffineMap}
    (checked : geometryCheck maskShape shape maskMap = true) : Geometry maskShape shape maskMap := by
  have parts := of_decide_eq_true checked
  exact ⟨structuredRanges?_spec parts.1, structuredRanges?_spec parts.2⟩

def inverted (maskShape : List Nat) (keep : Idx → Bool) : Tensor :=
  Tensor.canon maskShape (.int 1) (fun point => W8A8AttentionMaskBody.boolean (!(keep point)))

def result (maskShape shape : List Nat) (maskMap : AffineMap) (keep : Idx → Bool) (scores : Idx → FBits .f32) : Tensor :=
  Tensor.canon shape (.float .f32) fun point => W8A8OrderedBody.value
    (if !(keep (W8A8TensorOperands.coordinate maskShape maskMap point)) then
      W8A8MaskedMaximum.negativeInfinity else scores point)

theorem selection_execute (ctx : Ctx) (call : CallSem) (env : Env)
    (maskShape shape : List Nat) (maskMap : AffineMap) (geometry : Geometry maskShape shape maskMap)
    (keep : Idx → Bool) (values : Idx → Val) (scores : Idx → FBits .f32)
    (initialized : ∀ point, Idx.InBounds shape point → values point = W8A8OrderedBody.value (scores point)) :
    W8A8SourceSemantics.opSem ctx call (selection shape maskMap)
      [.tensor (inverted maskShape keep),
        .tensor (Tensor.full [] (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)),
        .tensor ⟨shape, .float .f32, values⟩, .tensor (Tensor.full shape (.float .f32) .poison)]
      (evalRegions W8A8SourceSemantics.opSem call env (selection shape maskMap).regions) =
      .ok [.tensor (result maskShape shape maskMap keep scores)] := by
  have masks : ∀ point, Idx.InBounds maskShape point →
      (inverted maskShape keep).get point = W8A8AttentionMaskBody.boolean (!(keep point)) := by
    intro point bounds
    simp only [inverted, Tensor.canon, bounds, ↓reduceIte]
  have executed := W8A8AttentionMaskTensor.select_sound selection_body ctx call env maskShape shape maskMap
    (List.replicate shape.length .parallel)
    (by simpa only [List.length_replicate, selectMaps] using geometry.selection)
    (inverted maskShape keep).get values (Tensor.full shape (.float .f32) .poison).get
    (fun point => !(keep point)) scores masks initialized
  exact W8A8AttentionTemplateOps.generic_execute ctx call env shape [3, 5, 0] 6 7 (selectMaps shape maskMap)
    (freshRegion 1100 W8A8AttentionMaskBody.selectRegion)
    [.tensor (inverted maskShape keep),
      .tensor (Tensor.full [] (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)),
      .tensor ⟨shape, .float .f32, values⟩] (Tensor.full shape (.float .f32) .poison)
    (result maskShape shape maskMap keep scores) rfl executed

theorem template_execute (ctx : Ctx) (call : CallSem) (env : Env)
    (maskShape shape : List Nat) (maskMap : AffineMap) (geometry : Geometry maskShape shape maskMap)
    (keepValues scoreValues : Idx → Val) (keep : Idx → Bool) (scores : Idx → FBits .f32)
    (maskSupplied : env.get? 1 = some (.tensor ⟨maskShape, .int 1, keepValues⟩))
    (scoresSupplied : env.get? 0 = some (.tensor ⟨shape, .float .f32, scoreValues⟩))
    (maskInitialized : ∀ point, Idx.InBounds maskShape point → keepValues point = W8A8AttentionMaskBody.boolean (keep point))
    (scoresInitialized : ∀ point, Idx.InBounds shape point → scoreValues point = W8A8OrderedBody.value (scores point)) :
    evalOps W8A8SourceSemantics.opSem ctx call env
      (operations maskShape shape maskMap ++ [terminator 7]) =
      .ok [.tensor (result maskShape shape maskMap keep scores)] := by
  change List.lookup 1 env = some (.tensor ⟨maskShape, .int 1, keepValues⟩) at maskSupplied
  change List.lookup 0 env = some (.tensor ⟨shape, .float .f32, scoreValues⟩) at scoresSupplied
  let e2 : Env := (2, .tensor (Tensor.full maskShape (.int 1) .poison)) :: env
  let e3 : Env := (3, .tensor (inverted maskShape keep)) :: e2
  let e4 : Env := (4, .scalar (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)) :: e3
  let e5 : Env := (5, .tensor (Tensor.full [] (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity))) :: e4
  let e6 : Env := (6, .tensor (Tensor.full shape (.float .f32) .poison)) :: e5
  let e7 : Env := (7, .tensor (result maskShape shape maskMap keep scores)) :: e6
  simp only [operations, List.cons_append, List.nil_append]
  rw [W8A8AttentionExecution.head_execute (env := env) (executed := empty_execute ctx call maskShape)
    (output := rfl) (header := by cbv)
    (reads := by simp [booleanEmpty, Op.operands, Env.getAll, Outcome.mapM, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e2 _ = _
  rw [W8A8AttentionExecution.head_execute (env := e2)
    (executed := inversion_execute ctx call e2 maskShape geometry.inversion keepValues keep maskInitialized)
    (output := rfl) (header := by cbv)
    (reads := by simp [e2, inversion, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
      Outcome.ofOption, maskSupplied])]
  change evalOps W8A8SourceSemantics.opSem ctx call e3 _ = _
  rw [W8A8AttentionExecution.head_execute (env := e3)
    (executed := W8A8AttentionTemplateOps.constant_execute ctx call 4 0xff800000)
    (output := rfl) (header := by cbv)
    (reads := by simp [constant, Op.operands, Env.getAll, Outcome.mapM, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e4 _ = _
  rw [W8A8AttentionExecution.head_execute (env := e4)
    (executed := W8A8AttentionTemplateOps.splat_execute ctx call [] 4 5 (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity))
    (output := rfl) (header := by cbv)
    (reads := by simp [e4, e3, e2, splat, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e5 _ = _
  rw [W8A8AttentionExecution.head_execute (env := e5)
    (executed := W8A8AttentionTemplateOps.empty_execute ctx call shape 6)
    (output := rfl) (header := by cbv)
    (reads := by simp [empty, Op.operands, Env.getAll, Outcome.mapM, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e6 _ = _
  rw [W8A8AttentionExecution.head_execute (env := e6)
    (executed := selection_execute ctx call e6 maskShape shape maskMap geometry keep scoreValues scores scoresInitialized)
    (output := rfl) (header := by cbv)
    (reads := by simp [e6, e5, e4, e3, e2, selection, generic, Op.operands, Env.getAll, Outcome.mapM,
      Env.get?, List.lookup, Outcome.ofOption, scoresSupplied])]
  change evalOps W8A8SourceSemantics.opSem ctx call e7 [terminator 7] = _
  simp [e7, terminator, evalOps, Op.isTerminator, terminatorNames, Op.name,
    Env.getAll, Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption]

end MerlinVerify.Rules.W8A8AttentionMaskApply
