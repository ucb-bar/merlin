/- Actual floating MAC-region recognition under the already constructed W8A8 arithmetic
family. Numerical bounds and tensor-coordinate counts are separate checked obligations. -/
import MerlinVerify.Rules.W8A8OrderedBody

namespace MerlinVerify.Check.PolicyW8A8OrderedBody

open MerlinVerify Rules

def result (a b : FBits .f32) (accumulator : Option (FBits .f32)) : Val :=
  W8A8OrderedBody.accumulatorValue (accumulator.map fun z => W8A8OrderedDot.step z (a, b))

def Correct (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (fallback : FloatModel),
    ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback) →
    ∀ (a b : FBits .f32) (accumulator : Option (FBits .f32)),
      evalRegion W8A8SourceSemantics.opSem ctx call env actual
        [.scalar (W8A8OrderedBody.value a), .scalar (W8A8OrderedBody.value b),
          .scalar (W8A8OrderedBody.accumulatorValue accumulator)] = .ok [.scalar (result a b accumulator)]

def check (actual : Region) : Bool :=
  alphaRegion [] (W8A8OrderedBody.region false) actual ||
    alphaRegion [] (W8A8OrderedBody.region true) actual

theorem check_sound {actual : Region} (checked : check actual = true) : Correct actual := by
  have selected : ∃ first, alphaRegion [] (W8A8OrderedBody.region first) actual = true := by
    have alternatives : alphaRegion [] (W8A8OrderedBody.region false) actual = true ∨
        alphaRegion [] (W8A8OrderedBody.region true) actual = true := by
      simpa only [check, Bool.or_eq_true] using checked
    cases alternatives with
    | inl accepted => exact ⟨false, accepted⟩
    | inr accepted => exact ⟨true, accepted⟩
  obtain ⟨first, accepted⟩ := selected
  intro ctx call env fallback interpretation a b accumulator
  have same := ParametricAlpha.alphaRegion_sound W8A8SourceAlpha.laws call call []
    (W8A8OrderedBody.region first) actual ctx env env accepted
    (by intro x y missing; simp [Rho.find, List.lookup] at missing) (fun _ _ => rfl)
  rw [← same, W8A8OrderedBody.region_optional_sound]
  cases accumulator <;>
    simp only [W8A8OrderedBody.rawResult, result, Option.map_none, Option.map_some,
      W8A8OrderedBody.accumulatorValue, interpretation, W8A8OrderedBody.rawStep_explicit]

#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8OrderedBody
