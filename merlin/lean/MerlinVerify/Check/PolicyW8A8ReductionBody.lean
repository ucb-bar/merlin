/- Recompute the scalar max/sum body from actual SSA and attributes. This proves
execution of the accepted region; row geometry and initialization remain separate. -/
import MerlinVerify.Rules.W8A8ReductionBody

namespace MerlinVerify.Check.PolicyW8A8ReductionBody

open MerlinVerify Rules W8A8ReductionBody

def result (kind : Kind) (input : FBits .f32) (accumulator : Option (FBits .f32)) : Val :=
  W8A8OrderedBody.accumulatorValue (accumulator.map fun current => combine kind input current)

def Correct (kind : Kind) (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel),
    ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback) →
    ∀ (input : FBits .f32) (accumulator : Option (FBits .f32)),
      evalRegion W8A8SourceSemantics.opSem ctx call env actual
        [.scalar (W8A8OrderedBody.value input), .scalar (W8A8OrderedBody.accumulatorValue accumulator)] =
        .ok [.scalar (result kind input accumulator)]

def check (kind : Kind) (actual : Region) : Bool :=
  alphaRegion [] (region kind) actual

theorem check_sound {kind : Kind} {actual : Region} (checked : check kind actual = true) :
    Correct kind actual := by
  intro ctx call env fallback interpretation input accumulator
  have same := ParametricAlpha.alphaRegion_sound W8A8SourceAlpha.laws call call []
    (region kind) actual ctx env env checked
    (by intro x y missing; simp [Rho.find, List.lookup] at missing) (fun _ _ => rfl)
  rw [← same, region_optional_sound]
  cases accumulator <;>
    simp only [rawResult, result, Option.map_none, Option.map_some,
      W8A8OrderedBody.accumulatorValue, interpretation, model_combine]

end MerlinVerify.Check.PolicyW8A8ReductionBody
