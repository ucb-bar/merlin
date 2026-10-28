/- Whole-segment SSA correspondence and reviewed parent-annotation normalization. -/
import MerlinVerify.Check.PolicyW8A8AttentionTemplate

open MerlinVerify Rules

namespace W8A8AttentionTemplateControls

def annotated : Op := .mk "tensor.empty" [⟨1, W8A8AttentionTemplate.tensorTy [2, 3]⟩] []
  [("prov.origin", .str "capture"), ("linalg.memoized_indexing_maps", .unit)] [] [] []

theorem erases_only_reviewed_metadata :
    (W8A8AttentionAnnotations.clean annotated).attrs = [] ∧
    (W8A8AttentionAnnotations.clean
      (.mk "tensor.empty" [] [] [("llvm.emit_c_interface", .unit), ("roundingmode", .str "bad")] [] [] [])).attrs =
      [("llvm.emit_c_interface", .unit), ("roundingmode", .str "bad")] := by cbv

theorem metadata_does_not_change_execution (ctx : Ctx) (call : CallSem) (args : List RVal)
    (regions : List (Ctx → RegionSem)) :
    W8A8SourceSemantics.opSem ctx call (W8A8AttentionAnnotations.clean annotated) args regions =
      W8A8SourceSemantics.opSem ctx call annotated args regions :=
  W8A8AttentionAnnotations.source_clean ctx call annotated args regions (by exact Or.inr (Or.inr (Or.inl rfl)))

theorem strict_body_keys_remain :
    W8A8AttentionAnnotations.semantic [("fastmath", .str "fast"), ("prov.origin", .str "x"),
      ("indexing_maps", .array []), ("value", .float 0 (.float .f32))] =
      [("fastmath", .str "fast"), ("indexing_maps", .array []), ("value", .float 0 (.float .f32))] := by cbv

#print axioms erases_only_reviewed_metadata
#print axioms metadata_does_not_change_execution
#print axioms strict_body_keys_remain
#print axioms W8A8AttentionAnnotations.eval_clean
#print axioms Check.PolicyW8A8AttentionTemplate.check_sound

end W8A8AttentionTemplateControls
