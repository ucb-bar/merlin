/- Exact i64 ramp execution and the explicit index-width boundary. -/
import MerlinVerify.Rules.W8A8AttentionRampTensor

open MerlinVerify Rules

namespace W8A8AttentionRampControls

theorem canonical_body : Check.PolicyW8A8AttentionRamp.check W8A8AttentionRamp.region = true := by cbv

theorem arbitrary_scalar (ctx : Ctx) (call : CallSem) (env : Env) (column : Nat) (old : Val)
    (width : ctx.indexBits = 64) (position : ctx.loopIndex[0]? = some column) :
    evalRegion W8A8SourceSemantics.opSem ctx call env W8A8AttentionRamp.region [.scalar old] =
      .ok [.scalar (W8A8AttentionMaskBody.integer (BitVec.ofNat 64 column))] :=
  W8A8AttentionRamp.region_sound ctx call env column old width position

theorem complete_nonempty_ramp (ctx : Ctx) (call : CallSem) (env : Env) (width : ctx.indexBits = 64) :
    Sem.Ops.Linalg.genericSem ctx [Sem.Ops.Linalg.identityMap 1] [.parallel] []
      [.tensor (Tensor.full [8] (.int 64) .poison)]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env W8A8AttentionRamp.region) =
      .ok [.tensor (W8A8AttentionRampTensor.ramp 8)] := by
  exact W8A8AttentionRampTensor.generic_sound canonical_body ctx call env 8
    (Tensor.full [8] (.int 64) .poison).get width (Check.structuredRanges?_spec (by decide))

theorem fixed_width_wrap_and_signed_extension :
    BitVec.ofNat 64 (2^64) = (0 : BitVec 64) ∧
    ((BitVec.ofNat 32 (2^31)).signExtend 64).toInt = -(2^31 : Int) ∧
    (BitVec.ofNat 64 (2^31)).toInt = (2^31 : Int) := by decide

#print axioms canonical_body
#print axioms arbitrary_scalar
#print axioms complete_nonempty_ramp
#print axioms fixed_width_wrap_and_signed_extension
#print axioms Check.PolicyW8A8AttentionRamp.check_sound
#print axioms W8A8AttentionRampTensor.generic_sound
#print axioms W8A8AttentionRampTensor.matching_words

end W8A8AttentionRampControls
