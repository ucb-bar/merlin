/- Actual SSA/body recognition for the softmax subtract/exp/div sequence. Each
accepted body retains its precise model and unused-destination execution theorem. -/
import MerlinVerify.Rules.W8A8AttentionMapBody

namespace MerlinVerify.Check.PolicyW8A8AttentionMapBody

open MerlinVerify Rules W8A8AttentionMapBody

def BinaryCorrect (kind : BinaryKind) (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel),
    ctx.M = W8A8RoundedReal.model fallback → ∀ (left right : FBits .f32) (destination : Val),
      evalRegion W8A8SourceSemantics.opSem ctx call env actual
        [.scalar (W8A8OrderedBody.value left), .scalar (W8A8OrderedBody.value right), .scalar destination] =
        .ok [.scalar (W8A8OrderedBody.value (binary kind left right))]

def ExponentialCorrect (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel),
    ctx.M = W8A8RoundedReal.model fallback → ∀ (input : FBits .f32) (destination : Val),
      evalRegion W8A8SourceSemantics.opSem ctx call env actual
        [.scalar (W8A8OrderedBody.value input), .scalar destination] =
        .ok [.scalar (W8A8OrderedBody.value (W8A8RoundedReal.exponential input))]

def checkBinary (kind : BinaryKind) (actual : Region) : Bool :=
  alphaRegion [] (binaryRegion kind) actual

def checkExponential (actual : Region) : Bool :=
  alphaRegion [] exponentialRegion actual

theorem checkBinary_sound {kind : BinaryKind} {actual : Region} (checked : checkBinary kind actual = true) :
    BinaryCorrect kind actual := by
  intro ctx call env fallback interpretation left right destination
  have same := ParametricAlpha.alphaRegion_sound W8A8SourceAlpha.laws call call []
    (binaryRegion kind) actual ctx env env checked
    (by intro x y missing; simp [Rho.find, List.lookup] at missing) (fun _ _ => rfl)
  rw [← same, binary_region_sound, interpretation, binary_model]

theorem checkExponential_sound {actual : Region} (checked : checkExponential actual = true) :
    ExponentialCorrect actual := by
  intro ctx call env fallback interpretation input destination
  have same := ParametricAlpha.alphaRegion_sound W8A8SourceAlpha.laws call call []
    exponentialRegion actual ctx env env checked
    (by intro x y missing; simp [Rho.find, List.lookup] at missing) (fun _ _ => rfl)
  rw [← same, exponential_region_sound, interpretation, W8A8RoundedReal.model_exponential]

end MerlinVerify.Check.PolicyW8A8AttentionMapBody
