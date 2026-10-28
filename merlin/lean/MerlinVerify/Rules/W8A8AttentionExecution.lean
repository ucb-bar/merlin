/- Initialized execution of the complete source softmax template. Admissible
all-masked rows remain allowed here and produce initialized NaN words. -/
import MerlinVerify.Rules.W8A8AttentionTemplateNumerics

namespace MerlinVerify.Rules.W8A8AttentionExecution

open MerlinVerify Check W8A8AttentionTemplate W8A8AttentionTemplateStages
  W8A8AttentionTemplateNumerics W8A8AttentionPointwiseRows W8A8AttentionRowValues

theorem head_execute {ctx : Ctx} {call : CallSem} {env : Env} {op : Op} {rest : List Op}
    {result : Value} {value : RVal} {args : List RVal}
    (executed : W8A8SourceSemantics.opSem ctx call op args
      (evalRegions W8A8SourceSemantics.opSem call env op.regions) = .ok [value])
    (output : op.results = [result]) (header : op.isTerminator = false)
    (reads : env.getAll op.operands = .ok args) :
    evalOps W8A8SourceSemantics.opSem ctx call env (op :: rest) =
      evalOps W8A8SourceSemantics.opSem ctx call ((result.id, value) :: env) rest := by
  rw [evalOps_cons_nonterm _ _ _ _ _ _ header, reads, Outcome.bind_ok, executed, Outcome.bind_ok]
  simp only [output, Env.bindAll]
  rfl

theorem template_execute {cap : Nat} (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (geometry : Geometry shape width)
    (values : Idx → Val) (scores : Idx → FBits .f32)
    (supplied : env.get? 0 = some (.tensor ⟨shape ++ [width], .float .f32, values⟩))
    (initialized : ∀ point, Idx.InBounds (shape ++ [width]) point → values point = W8A8OrderedBody.value (scores point))
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → W8A8MaskedMaximum.Admissible cap (scores point)) :
    evalOps W8A8SourceSemantics.opSem ctx call env (operations shape width ++ [terminator 16]) =
      .ok [.tensor (rowTensor (shape ++ [width]) (probability shape.length width scores))] := by
  change List.lookup 0 env = some (.tensor ⟨shape ++ [width], .float .f32, values⟩) at supplied
  let maximum := rowTensor shape (maximumWords width scores)
  let flatMaximum := ReshapeIndex.reshape maximum [shapeSize shape] (.float .f32)
  let denominator := rowTensor shape (denominatorWords width scores)
  let flatDenominator := ReshapeIndex.reshape denominator [shapeSize shape] (.float .f32)
  let e1 : Env := (1, .scalar (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)) :: env
  let e2 : Env := (2, .tensor (Tensor.full shape (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity))) :: e1
  let e3 : Env := (3, .tensor maximum) :: e2
  let e4 : Env := (4, .tensor flatMaximum) :: e3
  let e5 : Env := (5, .tensor (expanded shape (maximumWords width scores))) :: e4
  let e6 : Env := (6, .tensor (Tensor.full (shape ++ [width]) (.float .f32) .poison)) :: e5
  let e7 : Env := (7, .tensor (rowTensor (shape ++ [width]) (shifted shape.length width scores))) :: e6
  let e8 : Env := (8, .tensor (Tensor.full (shape ++ [width]) (.float .f32) .poison)) :: e7
  let e9 : Env := (9, .tensor (rowTensor (shape ++ [width]) (exponential shape.length width scores))) :: e8
  let e10 : Env := (10, .scalar (W8A8OrderedBody.value (0 : FBits .f32))) :: e9
  let e11 : Env := (11, .tensor (Tensor.full shape (.float .f32) (W8A8OrderedBody.value (0 : FBits .f32)))) :: e10
  let e12 : Env := (12, .tensor denominator) :: e11
  let e13 : Env := (13, .tensor flatDenominator) :: e12
  let e14 : Env := (14, .tensor (expanded shape (denominatorWords width scores))) :: e13
  let e15 : Env := (15, .tensor (Tensor.full (shape ++ [width]) (.float .f32) .poison)) :: e14
  let e16 : Env := (16, .tensor (rowTensor (shape ++ [width]) (probability shape.length width scores))) :: e15
  have subtractStage := subtract_execute ctx call e6 fallback interpretation shape width geometry values scores initialized
  have exponentialStage := exponential_execute ctx call e8 fallback interpretation shape width geometry scores
  have divideStage := divide_execute ctx call e15 fallback interpretation shape width geometry scores
  dsimp only [binaryMaps] at subtractStage divideStage
  dsimp only [unaryMaps] at exponentialStage
  simp only [operations, List.cons_append, List.nil_append]
  rw [head_execute (env := env) (executed := W8A8AttentionTemplateOps.constant_execute ctx call 1 0xff800000)
    (output := rfl) (header := by cbv) (reads := by simp [constant, Op.operands, Env.getAll, Outcome.mapM, Env.get?, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e1 _ = _
  rw [head_execute (env := e1) (executed := W8A8AttentionTemplateOps.splat_execute ctx call shape 1 2 (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity))
    (output := rfl) (header := by cbv) (reads := by simp [e1, splat, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e2 _ = _
  rw [head_execute (env := e2) (executed := maximum_execute ctx call e2 fallback interpretation shape width geometry values scores initialized members)
    (output := rfl) (header := by cbv) (reads := by simp [
      e2, e1, reduce, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
      Outcome.ofOption, supplied])]
  change evalOps W8A8SourceSemantics.opSem ctx call e3 _ = _
  rw [head_execute (env := e3) (executed := W8A8AttentionTemplateOps.collapse_execute ctx call shape 3 4 maximum rfl rfl geometry.collapse)
    (output := rfl) (header := by cbv) (reads := by simp [
      e3, e2, e1, collapse, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
      Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e4 _ = _
  rw [head_execute (env := e4) (executed := W8A8AttentionTemplateOps.expand_execute ctx call (shape ++ [1]) 4 5 flatMaximum rfl geometry.expand)
    (output := rfl) (header := by cbv) (reads := by simp [
      e4, e3, e2, e1, expand, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
      Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e5 _ = _
  rw [head_execute (env := e5) (executed := W8A8AttentionTemplateOps.empty_execute ctx call (shape ++ [width]) 6)
    (output := rfl) (header := by cbv) (reads := by simp [
      e5, e4, e3, e2, e1, empty, Op.operands, Env.getAll, Outcome.mapM, Env.get?, List.lookup,
      Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e6 _ = _
  rw [head_execute (env := e6) (executed := subtractStage)
    (output := rfl) (header := by cbv) (reads := by simp [
      e6, e5, e4, e3, e2, e1, generic, Op.operands, Env.getAll, Outcome.mapM, Env.get?,
      List.lookup, Outcome.ofOption, supplied])]
  change evalOps W8A8SourceSemantics.opSem ctx call e7 _ = _
  rw [head_execute (env := e7) (executed := W8A8AttentionTemplateOps.empty_execute ctx call (shape ++ [width]) 8)
    (output := rfl) (header := by cbv) (reads := by simp [
      e7, e6, e5, e4, e3, e2, e1, empty, Op.operands, Env.getAll, Outcome.mapM, Env.get?,
      List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e8 _ = _
  rw [head_execute (env := e8) (executed := exponentialStage)
    (output := rfl) (header := by cbv) (reads := by simp [
      e8, e7, e6, e5, e4, e3, e2, e1, generic, Op.operands, Env.getAll, Outcome.mapM, Env.get?,
      List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e9 _ = _
  rw [head_execute (env := e9) (executed := W8A8AttentionTemplateOps.constant_execute ctx call 10 0)
    (output := rfl) (header := by cbv) (reads := by simp [
      e9, e8, e7, e6, e5, e4, e3, e2, e1, constant, Op.operands, Env.getAll, Outcome.mapM,
      Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e10 _ = _
  rw [head_execute (env := e10) (executed := W8A8AttentionTemplateOps.splat_execute ctx call shape 10 11 (W8A8OrderedBody.value (0 : FBits .f32)))
    (output := rfl) (header := by cbv) (reads := by simp [
      e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, splat, Op.operands, Env.getAll, Outcome.mapM,
      Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e11 _ = _
  rw [head_execute (env := e11) (executed := sum_execute ctx call e11 fallback interpretation shape width geometry scores)
    (output := rfl) (header := by cbv) (reads := by simp [
      e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, reduce, Op.operands, Env.getAll, Outcome.mapM,
      Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e12 _ = _
  rw [head_execute (env := e12) (executed := W8A8AttentionTemplateOps.collapse_execute ctx call shape 12 13 denominator rfl rfl geometry.collapse)
    (output := rfl) (header := by cbv) (reads := by simp [
      e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, collapse, Op.operands, Env.getAll,
      Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e13 _ = _
  rw [head_execute (env := e13) (executed := W8A8AttentionTemplateOps.expand_execute ctx call (shape ++ [1]) 13 14 flatDenominator rfl geometry.expand)
    (output := rfl) (header := by cbv) (reads := by simp [
      e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, expand, Op.operands, Env.getAll,
      Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e14 _ = _
  rw [head_execute (env := e14) (executed := W8A8AttentionTemplateOps.empty_execute ctx call (shape ++ [width]) 15)
    (output := rfl) (header := by cbv) (reads := by simp [
      e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, empty, Op.operands, Env.getAll,
      Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e15 _ = _
  rw [head_execute (env := e15) (executed := divideStage)
    (output := rfl) (header := by cbv) (reads := by simp [
      e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, generic, Op.operands,
      Env.getAll, Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption])]
  change evalOps W8A8SourceSemantics.opSem ctx call e16 [terminator 16] = _
  simp [
      e16, terminator, evalOps, Op.isTerminator, terminatorNames, Op.name, Env.getAll,
      Outcome.mapM, Env.get?, List.lookup, Outcome.ofOption]

end MerlinVerify.Rules.W8A8AttentionExecution
