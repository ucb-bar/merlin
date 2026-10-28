/- The five numerical stages of the source softmax template, connected to actual
dispatch, actual scalar bodies, and matching row-major reshape/broadcast reads. -/
import MerlinVerify.Rules.W8A8AttentionTemplateOps
import MerlinVerify.Rules.W8A8AttentionPointwiseRows

namespace MerlinVerify.Rules.W8A8AttentionTemplateStages

open MerlinVerify Check Sem.Ops Sem.Ops.Linalg W8A8AttentionTemplate

def reductionIters (shape : List Nat) : List IterKind :=
  W8A8ReductionNamed.iterators (shape.length + 1) [shape.length]

def reductionMaps (shape : List Nat) : List AffineMap :=
  [identityMap (shape.length + 1), W8A8RowTrace.outputMap shape.length]

def binaryMaps (shape : List Nat) (width : Nat) : List AffineMap :=
  [identityMap (shape ++ [width]).length, broadcastMap shape.length, identityMap (shape ++ [width]).length]

def unaryMaps (shape : List Nat) (width : Nat) : List AffineMap :=
  [identityMap (shape ++ [width]).length, identityMap (shape ++ [width]).length]

/-- Static execution obligations, recomputed from the shape rather than supplied
as runtime numerical assumptions. -/
structure Geometry (shape : List Nat) (width : Nat) : Prop where
  reduction : StructuredOk (reductionMaps shape) (reductionIters shape).length
    [.tensor (shape ++ [width]) (.float .f32), .tensor shape (.float .f32)] (shape ++ [width])
  binary : StructuredOk (binaryMaps shape width) (shape ++ [width]).length
    [.tensor (shape ++ [width]) (.float .f32), .tensor (shape ++ [1]) (.float .f32),
      .tensor (shape ++ [width]) (.float .f32)] (shape ++ [width])
  unary : StructuredOk (unaryMaps shape width) (shape ++ [width]).length
    [.tensor (shape ++ [width]) (.float .f32), .tensor (shape ++ [width]) (.float .f32)] (shape ++ [width])
  collapse : Tensor.reassocValid [List.range shape.length] shape [shapeSize shape] = true
  expand : Tensor.reassocValid [List.range (shape ++ [1]).length] (shape ++ [1]) [shapeSize shape] = true

def geometryCheck (shape : List Nat) (width : Nat) : Bool :=
  decide (
    structuredRanges? (reductionMaps shape) (reductionIters shape).length
      [.tensor (shape ++ [width]) (.float .f32), .tensor shape (.float .f32)] = some (shape ++ [width]) ∧
    structuredRanges? (binaryMaps shape width) (shape ++ [width]).length
      [.tensor (shape ++ [width]) (.float .f32), .tensor (shape ++ [1]) (.float .f32),
        .tensor (shape ++ [width]) (.float .f32)] = some (shape ++ [width]) ∧
    structuredRanges? (unaryMaps shape width) (shape ++ [width]).length
      [.tensor (shape ++ [width]) (.float .f32), .tensor (shape ++ [width]) (.float .f32)] = some (shape ++ [width]) ∧
    Tensor.reassocValid [List.range shape.length] shape [shapeSize shape] = true ∧
    Tensor.reassocValid [List.range (shape ++ [1]).length] (shape ++ [1]) [shapeSize shape] = true)

theorem geometryCheck_sound {shape : List Nat} {width : Nat} (checked : geometryCheck shape width = true) :
    Geometry shape width := by
  obtain ⟨reduction, binary, unary, collapse, expand⟩ := of_decide_eq_true checked
  exact ⟨structuredRanges?_spec reduction, structuredRanges?_spec binary, structuredRanges?_spec unary,
    collapse, expand⟩

def maximumWords (width : Nat) (scores : Idx → FBits .f32) (cell : Idx) : FBits .f32 :=
  W8A8SoftmaxRow.rowMaximum (W8A8AttentionMaximum.row width scores cell)

noncomputable def denominatorWords (width : Nat) (scores : Idx → FBits .f32) (cell : Idx) : FBits .f32 :=
  W8A8SoftmaxRow.denominator (W8A8AttentionMaximum.row width scores cell)

theorem maximum_body : PolicyW8A8ReductionBody.check .maximum
    (freshRegion 100 (W8A8ReductionBody.region .maximum)) = true := by cbv
theorem sum_body : PolicyW8A8ReductionBody.check .sum
    (freshRegion 200 (W8A8ReductionBody.region .sum)) = true := by cbv
theorem subtract_body : PolicyW8A8AttentionMapBody.checkBinary .subtract
    (freshRegion 300 (W8A8AttentionMapBody.binaryRegion .subtract)) = true := by cbv
theorem exponential_body : PolicyW8A8AttentionMapBody.checkExponential
    (freshRegion 400 W8A8AttentionMapBody.exponentialRegion) = true := by cbv
theorem divide_body : PolicyW8A8AttentionMapBody.checkBinary .divide
    (freshRegion 500 (W8A8AttentionMapBody.binaryRegion .divide)) = true := by cbv

theorem reduce_execute (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat) (width : Nat)
    (input seed result : Nat) (kind : W8A8ReductionBody.Kind) (values initial : Idx → Val) (output : Tensor)
    (executed : genericSem ctx (reductionMaps shape) (reductionIters shape)
      [.tensor ⟨shape ++ [width], .float .f32, values⟩] [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env
        (freshRegion (if kind == .maximum then 100 else 200) (W8A8ReductionBody.region kind))) =
      .ok [.tensor output]) :
    W8A8SourceSemantics.opSem ctx call (reduce shape input seed result kind)
      [.tensor ⟨shape ++ [width], .float .f32, values⟩, .tensor ⟨shape, .float .f32, initial⟩]
      (evalRegions W8A8SourceSemantics.opSem call env (reduce shape input seed result kind).regions) =
      .ok [.tensor output] := by
  have ordinary : W8A8SourceSemantics.isSource (reduce shape input seed result kind).name = false := by cbv
  have dialect : dialectOf (reduce shape input seed result kind).name = "linalg" := by cbv
  have modelled : unmodelledAttr? (reduce shape input seed result kind).blind = none := by
    simp [reduce, Op.blind, unmodelledAttr?, interpretedAttrs, Op.name, Op.props, Op.attrs]
  rw [W8A8SourceSemantics.unchanged ordinary, named_sem_of modelled dialect (by cbv)]
  have dimensions : LinalgNamed.natArray? (reduce shape input seed result kind) "dimensions" =
      some [shape.length] := by
    simp [reduce, LinalgNamed.natArray?, attrDenseArray?, Op.attr?, Op.props, Op.attrs]
  have split : splitInsOuts (reduce shape input seed result kind)
      [RVal.tensor ⟨shape ++ [width], .float .f32, values⟩, RVal.tensor ⟨shape, .float .f32, initial⟩] = none := by
    simp [reduce, splitInsOuts, attrDenseArray?, Op.attr?, Op.props, Op.attrs, List.lookup]
  have named : (reduce shape input seed result kind).name = "linalg.reduce" := rfl
  have regions : (reduce shape input seed result kind).regions =
      [freshRegion (if kind == .maximum then 100 else 200) (W8A8ReductionBody.region kind)] := rfl
  simp only [
      regions, evalRegions, LinalgNamed.sem, Op.blind_name, named, natArray?_blind, dimensions,
      splitInsOuts_blind, split, Option.getD_none, List.length_cons, List.length_nil,
      Nat.reduceAdd, Nat.reduceDiv, List.take_succ_cons, List.take_zero, List.drop_succ_cons,
      List.drop_zero, shapeOf, List.length_append, List.map_cons, List.map_nil, List.cons_append,
      List.nil_append, W8A8RowTrace.kept_dimensions, LinalgNamed.run]
  change (genericSem ctx (reductionMaps shape) (reductionIters shape)
    [.tensor ⟨shape ++ [width], .float .f32, values⟩] [.tensor ⟨shape, .float .f32, initial⟩]
    (fun c => evalRegion W8A8SourceSemantics.opSem c call env
      (freshRegion (if kind == .maximum then 100 else 200) (W8A8ReductionBody.region kind)))).bind
      (fun results => if results.length = 1 then .ok results else .stuck "linalg.reduce: result count") = _
  rw [executed]
  rfl

end MerlinVerify.Rules.W8A8AttentionTemplateStages
