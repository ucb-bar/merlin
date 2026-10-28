/- Complete mask application, including the actual leading-head broadcast map.
The kept-position producer is intentionally an arbitrary initialized boolean tensor. -/
import MerlinVerify.Check.PolicyW8A8AttentionMaskApply

open MerlinVerify Rules W8A8AttentionMaskApply

namespace W8A8AttentionMaskApplyControls

def maskMap : AffineMap := ⟨4, 0, [.dim 0, .const 0, .dim 2, .dim 3]⟩

theorem actual_broadcast_geometry : geometryCheck [1, 1, 2, 2] [1, 3, 2, 2] maskMap = true := by decide

def keptTensor (keep : Idx → Bool) : Tensor :=
  Tensor.canon [1, 1, 2, 2] (.int 1) (fun point => W8A8AttentionMaskBody.boolean (keep point))

def scoreTensor (scores : Idx → FBits .f32) : Tensor :=
  Tensor.canon [1, 3, 2, 2] (.float .f32) (fun point => W8A8OrderedBody.value (scores point))

theorem complete_application (ctx : Ctx) (call : CallSem) (keep : Idx → Bool) (scores : Idx → FBits .f32) :
    evalOps W8A8SourceSemantics.opSem ctx call [(0, .tensor (scoreTensor scores)), (1, .tensor (keptTensor keep))]
      (operations [1, 1, 2, 2] [1, 3, 2, 2] maskMap ++ [W8A8AttentionTemplate.terminator 7]) =
      .ok [.tensor (result [1, 1, 2, 2] [1, 3, 2, 2] maskMap keep scores)] := by
  apply template_execute ctx call _ [1, 1, 2, 2] [1, 3, 2, 2] maskMap
    (geometryCheck_sound actual_broadcast_geometry) (keptTensor keep).get (scoreTensor scores).get keep scores rfl rfl
  · intro point bounds
    simp only [keptTensor, Tensor.canon, bounds, ↓reduceIte]
  · intro point bounds
    simp only [scoreTensor, Tensor.canon, bounds, ↓reduceIte]

theorem wrong_head_map_is_refused :
    geometryCheck [1, 1, 2, 2] [1, 3, 2, 2] ⟨4, 0, [.dim 0, .dim 1, .dim 2, .dim 3]⟩ = false := by decide

#print axioms actual_broadcast_geometry
#print axioms complete_application
#print axioms wrong_head_map_is_refused
#print axioms Check.PolicyW8A8AttentionMaskApply.check_sound

end W8A8AttentionMaskApplyControls
