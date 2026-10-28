/- Actual source-operation execution for the canonical five softmax stages.
This file composes row values and tensor semantics; no native float law is assumed. -/
import MerlinVerify.Rules.W8A8AttentionTemplateStages

namespace MerlinVerify.Rules.W8A8AttentionTemplateNumerics

open MerlinVerify Sem.Ops.Linalg W8A8AttentionTemplate W8A8AttentionTemplateStages
  W8A8AttentionPointwiseRows W8A8AttentionRowValues

theorem maximum_execute {cap : Nat} (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (geometry : Geometry shape width)
    (values : Idx → Val) (scores : Idx → FBits .f32)
    (initialized : ∀ point, Idx.InBounds (shape ++ [width]) point → values point = W8A8OrderedBody.value (scores point))
    (members : ∀ point, Idx.InBounds (shape ++ [width]) point → W8A8MaskedMaximum.Admissible cap (scores point)) :
    W8A8SourceSemantics.opSem ctx call (reduce shape 0 2 3 .maximum)
      [.tensor ⟨shape ++ [width], .float .f32, values⟩,
        .tensor (Tensor.full shape (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity))]
      (evalRegions W8A8SourceSemantics.opSem call env (reduce shape 0 2 3 .maximum).regions) =
      .ok [.tensor (rowTensor shape (maximumWords width scores))] := by
  have seeds : ∀ cell, Idx.InBounds shape cell →
      (Tensor.full shape (.float .f32) (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity)).get cell =
        W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity := by
    intro cell bounds
    simp only [Tensor.full, Tensor.canon, bounds, ↓reduceIte]
  exact reduce_execute ctx call env shape width 0 2 3 .maximum values _ _
    (W8A8AttentionMaximum.generic_sound shape width (reductionIters shape) geometry.reduction maximum_body
      call env fallback interpretation values _ scores initialized seeds members)

theorem sum_execute (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (geometry : Geometry shape width) (scores : Idx → FBits .f32) :
    W8A8SourceSemantics.opSem ctx call (reduce shape 9 11 12 .sum)
      [.tensor (rowTensor (shape ++ [width]) (exponential shape.length width scores)),
        .tensor (Tensor.full shape (.float .f32) (W8A8OrderedBody.value (0 : FBits .f32)))]
      (evalRegions W8A8SourceSemantics.opSem call env (reduce shape 9 11 12 .sum).regions) =
      .ok [.tensor (rowTensor shape (denominatorWords width scores))] := by
  have initialized : ∀ point, Idx.InBounds (shape ++ [width]) point →
      (rowTensor (shape ++ [width]) (exponential shape.length width scores)).get point =
        W8A8OrderedBody.value (exponential shape.length width scores point) := by
    intro point bounds
    simp only [rowTensor, Tensor.canon, bounds, ↓reduceIte]
  have seeds : ∀ cell, Idx.InBounds shape cell →
      (Tensor.full shape (.float .f32) (W8A8OrderedBody.value (0 : FBits .f32))).get cell =
        W8A8OrderedBody.value (0 : FBits .f32) := by
    intro cell bounds
    simp only [Tensor.full, Tensor.canon, bounds, ↓reduceIte]
  exact reduce_execute ctx call env shape width 9 11 12 .sum _ _ _
    (W8A8AttentionSum.generic_sound shape width (reductionIters shape) geometry.reduction sum_body
      call env fallback interpretation _ _ scores initialized seeds)

theorem subtract_execute (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (geometry : Geometry shape width)
    (values : Idx → Val) (scores : Idx → FBits .f32)
    (initialized : ∀ point, Idx.InBounds (shape ++ [width]) point → values point = W8A8OrderedBody.value (scores point)) :
    W8A8SourceSemantics.opSem ctx call
      (generic (shape ++ [width]) [0, 5] 6 7 (binaryMaps shape width)
        (freshRegion 300 (W8A8AttentionMapBody.binaryRegion .subtract)))
      [.tensor ⟨shape ++ [width], .float .f32, values⟩, .tensor (expanded shape (maximumWords width scores)),
        .tensor (Tensor.full (shape ++ [width]) (.float .f32) .poison)]
      (evalRegions W8A8SourceSemantics.opSem call env
        [freshRegion 300 (W8A8AttentionMapBody.binaryRegion .subtract)]) =
      .ok [.tensor (rowTensor (shape ++ [width]) (shifted shape.length width scores))] := by
  have checkedGeometry := geometry.binary
  have executed := W8A8AttentionPointwiseRows.binary_sound subtract_body ctx call env fallback interpretation
    shape width (List.replicate (shape ++ [width]).length .parallel)
    (by simpa only [List.length_replicate, binaryMaps, unaryMaps] using checkedGeometry)
    values (Tensor.full (shape ++ [width]) (.float .f32) .poison).get scores (maximumWords width scores) initialized
  exact W8A8AttentionTemplateOps.generic_execute ctx call env (shape ++ [width]) [0, 5] 6 7
    (binaryMaps shape width) (freshRegion 300 (W8A8AttentionMapBody.binaryRegion .subtract))
    [.tensor ⟨shape ++ [width], .float .f32, values⟩, .tensor (expanded shape (maximumWords width scores))]
    (Tensor.full (shape ++ [width]) (.float .f32) .poison)
    (rowTensor (shape ++ [width]) (shifted shape.length width scores)) rfl executed

theorem exponential_execute (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (geometry : Geometry shape width) (scores : Idx → FBits .f32) :
    W8A8SourceSemantics.opSem ctx call
      (generic (shape ++ [width]) [7] 8 9 (unaryMaps shape width) (freshRegion 400 W8A8AttentionMapBody.exponentialRegion))
      [.tensor (rowTensor (shape ++ [width]) (shifted shape.length width scores)),
        .tensor (Tensor.full (shape ++ [width]) (.float .f32) .poison)]
      (evalRegions W8A8SourceSemantics.opSem call env [freshRegion 400 W8A8AttentionMapBody.exponentialRegion]) =
      .ok [.tensor (rowTensor (shape ++ [width]) (exponential shape.length width scores))] := by
  have initialized : ∀ point, Idx.InBounds (shape ++ [width]) point →
      (rowTensor (shape ++ [width]) (shifted shape.length width scores)).get point =
        W8A8OrderedBody.value (shifted shape.length width scores point) := by
    intro point bounds
    simp only [rowTensor, Tensor.canon, bounds, ↓reduceIte]
  have executed := W8A8AttentionMapTensor.exponential_sound exponential_body ctx call env fallback interpretation
    (shape ++ [width]) (shape ++ [width]) (identityMap (shape ++ [width]).length)
    (List.replicate (shape ++ [width]).length .parallel)
    (by simpa only [List.length_replicate, binaryMaps, unaryMaps] using geometry.unary)
    (rowTensor (shape ++ [width]) (shifted shape.length width scores)).get
    (Tensor.full (shape ++ [width]) (.float .f32) .poison).get (shifted shape.length width scores) initialized
  have results :
      Tensor.canon (shape ++ [width]) (.float .f32) (fun point => W8A8OrderedBody.value
        (W8A8RoundedReal.exponential (shifted shape.length width scores
          (W8A8TensorOperands.coordinate (shape ++ [width]) (identityMap (shape ++ [width]).length) point)))) =
      rowTensor (shape ++ [width]) (exponential shape.length width scores) := by
    apply Tensor.canon_ext
    intro point bounds
    rw [W8A8TensorOperands.coordinate_identity bounds]
    rfl
  rw [results] at executed
  exact W8A8AttentionTemplateOps.generic_execute ctx call env (shape ++ [width]) [7] 8 9
    (unaryMaps shape width) (freshRegion 400 W8A8AttentionMapBody.exponentialRegion)
    [.tensor (rowTensor (shape ++ [width]) (shifted shape.length width scores))]
    (Tensor.full (shape ++ [width]) (.float .f32) .poison)
    (rowTensor (shape ++ [width]) (exponential shape.length width scores)) rfl executed

theorem divide_execute (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback)
    (shape : List Nat) (width : Nat) (geometry : Geometry shape width) (scores : Idx → FBits .f32) :
    W8A8SourceSemantics.opSem ctx call
      (generic (shape ++ [width]) [9, 14] 15 16 (binaryMaps shape width)
        (freshRegion 500 (W8A8AttentionMapBody.binaryRegion .divide)))
      [.tensor (rowTensor (shape ++ [width]) (exponential shape.length width scores)),
        .tensor (expanded shape (denominatorWords width scores)),
        .tensor (Tensor.full (shape ++ [width]) (.float .f32) .poison)]
      (evalRegions W8A8SourceSemantics.opSem call env
        [freshRegion 500 (W8A8AttentionMapBody.binaryRegion .divide)]) =
      .ok [.tensor (rowTensor (shape ++ [width]) (probability shape.length width scores))] := by
  have initialized : ∀ point, Idx.InBounds (shape ++ [width]) point →
      (rowTensor (shape ++ [width]) (exponential shape.length width scores)).get point =
        W8A8OrderedBody.value (exponential shape.length width scores point) := by
    intro point bounds
    simp only [rowTensor, Tensor.canon, bounds, ↓reduceIte]
  have executed := W8A8AttentionPointwiseRows.binary_sound divide_body ctx call env fallback interpretation
    shape width (List.replicate (shape ++ [width]).length .parallel)
    (by simpa only [List.length_replicate, binaryMaps, unaryMaps] using geometry.binary)
    (rowTensor (shape ++ [width]) (exponential shape.length width scores)).get
    (Tensor.full (shape ++ [width]) (.float .f32) .poison).get (exponential shape.length width scores)
    (denominatorWords width scores) initialized
  exact W8A8AttentionTemplateOps.generic_execute ctx call env (shape ++ [width]) [9, 14] 15 16
    (binaryMaps shape width) (freshRegion 500 (W8A8AttentionMapBody.binaryRegion .divide))
    [.tensor (rowTensor (shape ++ [width]) (exponential shape.length width scores)),
      .tensor (expanded shape (denominatorWords width scores))]
    (Tensor.full (shape ++ [width]) (.float .f32) .poison)
    (rowTensor (shape ++ [width]) (probability shape.length width scores)) rfl executed

end MerlinVerify.Rules.W8A8AttentionTemplateNumerics
