/- Full initialized source softmax execution, static geometry and exceptional rows.
Captured diagonal-mask admission and source quantization remain separate obligations. -/
import MerlinVerify.Check.PolicyW8A8AttentionExecution

open MerlinVerify Rules W8A8AttentionTemplateStages W8A8AttentionPointwiseRows

namespace W8A8AttentionExecutionControls

theorem shapes_are_checked : geometryCheck [2] 3 = true ∧ geometryCheck [1, 32, 8] 8 = true ∧
    geometryCheck [2] 0 = true := by decide

theorem arbitrary_nonempty_tensor (ctx : Ctx) (call : CallSem) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback) (scores : Idx → FBits .f32)
    (members : ∀ point, Idx.InBounds [2, 3] point → W8A8MaskedMaximum.Admissible 179 (scores point)) :
    evalOps W8A8SourceSemantics.opSem ctx call
      [(0, .tensor (rowTensor [2, 3] scores))]
      (W8A8AttentionTemplate.operations [2] 3 ++ [W8A8AttentionTemplate.terminator 16]) =
      .ok [.tensor (rowTensor [2, 3] (W8A8AttentionRowValues.probability 1 3 scores))] := by
  exact W8A8AttentionExecution.template_execute ctx call _ fallback interpretation [2] 3
    (geometryCheck_sound shapes_are_checked.1) (rowTensor [2, 3] scores).get scores rfl
    (fun point bounds => by
      change Idx.InBounds [2, 3] point at bounds
      simp only [rowTensor, Tensor.canon, bounds, ↓reduceIte]) members

theorem exact_row_broadcast (tensor : Tensor) (point : Idx) (width : Nat)
    (bounds : Idx.InBounds (tensor.shape ++ [width]) point) :
    (ReshapeIndex.reshape (ReshapeIndex.reshape tensor [shapeSize tensor.shape] tensor.elt)
      (tensor.shape ++ [1]) tensor.elt).get
      (W8A8TensorOperands.coordinate (tensor.shape ++ [1])
        (W8A8AttentionTemplate.broadcastMap tensor.shape.length) point) =
      tensor.get (point.take tensor.shape.length) :=
  W8A8RowBroadcast.reshaped_read tensor _ point width
    (by simp only [shapeSize, List.foldl_cons, List.foldl_nil, Nat.one_mul]) bounds

theorem masked_row_still_nan :
    W8A8AttentionRowValues.probability 1 1 (fun _ => W8A8MaskedMaximum.negativeInfinity) [0, 0] = 0x7fc00000 := by
  have maximum : W8A8ByteFloat.maximum W8A8MaskedMaximum.negativeInfinity W8A8MaskedMaximum.negativeInfinity =
      W8A8MaskedMaximum.negativeInfinity := by decide
  have scalar := W8A8RoundedReal.exponential_nan W8A8AttentionDifference.negative_infinity_self_nan
  have exponential : W8A8AttentionRowValues.exponential 1 1
      (fun _ => W8A8MaskedMaximum.negativeInfinity) [0, 0] = 0x7fc00000 := by
    simp only [W8A8AttentionRowValues.exponential, W8A8AttentionRowValues.shifted,
      W8A8AttentionMaximum.row, List.range_succ, List.range_zero, List.nil_append,
      List.map_cons, List.map_nil, W8A8SoftmaxRow.rowMaximum, List.foldl_cons, List.foldl_nil, maximum, scalar]
  have denominator : W8A8SoftmaxRow.denominator [W8A8MaskedMaximum.negativeInfinity] = 0x7fc00000 := by
    simp only [W8A8SoftmaxRow.denominator, W8A8SoftmaxRow.exponentials, W8A8SoftmaxRow.rowMaximum,
      List.foldl_cons, List.foldl_nil, List.map_cons, List.map_nil, maximum, scalar]
    decide
  simp only [W8A8AttentionRowValues.probability, exponential, W8A8AttentionMaximum.row,
    List.range_succ, List.range_zero, List.nil_append, List.map_cons, List.map_nil, denominator]
  decide

theorem all_masked_program_nan (ctx : Ctx) (call : CallSem) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model fallback) :
    ∃ tensor, evalOps W8A8SourceSemantics.opSem ctx call
      [(0, .tensor (rowTensor [1, 1] (fun _ => W8A8MaskedMaximum.negativeInfinity)))]
      (W8A8AttentionTemplate.operations [1] 1 ++ [W8A8AttentionTemplate.terminator 16]) = .ok [.tensor tensor] ∧
      tensor.get [0, 0] = W8A8OrderedBody.value (0x7fc00000 : FBits .f32) := by
  refine ⟨rowTensor [1, 1] (W8A8AttentionRowValues.probability 1 1 (fun _ => W8A8MaskedMaximum.negativeInfinity)), ?_, ?_⟩
  · exact W8A8AttentionExecution.template_execute (cap := 179) ctx call _ fallback interpretation [1] 1
      (geometryCheck_sound (by decide)) (rowTensor [1, 1] (fun _ => W8A8MaskedMaximum.negativeInfinity)).get
      (fun _ => W8A8MaskedMaximum.negativeInfinity) rfl
      (fun point bounds => by
        change Idx.InBounds [1, 1] point at bounds
        simp only [rowTensor, Tensor.canon, bounds, ↓reduceIte])
      (fun _ _ => Or.inl rfl)
  · simp only [rowTensor, Tensor.canon, show Idx.InBounds [1, 1] [0, 0] by decide, ↓reduceIte,
      masked_row_still_nan]

#print axioms shapes_are_checked
#print axioms arbitrary_nonempty_tensor
#print axioms exact_row_broadcast
#print axioms masked_row_still_nan
#print axioms all_masked_program_nan
#print axioms Check.PolicyW8A8AttentionExecution.check_sound
#print axioms Check.PolicyW8A8AttentionExecution.checked_exec
#print axioms Check.PolicyW8A8AttentionExecution.output_bounded

end W8A8AttentionExecutionControls
