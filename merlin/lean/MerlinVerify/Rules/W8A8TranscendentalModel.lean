/- Exact unary-f32 source keys with unchanged W8A8 arithmetic and numerical context.
Noncomputable mathematical candidate; no native fidelity or independent approval is claimed. -/
import MerlinVerify.Rules.W8A8Transcendentals
import MerlinVerify.Rules.W8A8UnarySign
import MerlinVerify.Check.PolicyW8A8IEEEContext

namespace MerlinVerify.Rules.W8A8RoundedReal

open MerlinVerify Rules

noncomputable section

def transcendentalFallback (base : FloatModel) : FloatModel :=
  { base with op := fun key kind arguments =>
    match kind, arguments with
    | .f32, [bits] =>
      if key = Spec.Dequant.defaultKey "math.sin" then sine bits
      else if key = Spec.Dequant.defaultKey "math.cos" then cosine bits
      else if key = Spec.Dequant.defaultKey "math.exp" then exponential bits
      else base.op key .f32 [bits]
    | kind, arguments => base.op key kind arguments }

def model (base : FloatModel) : FloatModel :=
  W8A8ByteModel.extend (W8A8IEEEPrimitives.model (transcendentalFallback (W8A8UnarySign.model base)))

theorem model_sine (base : FloatModel) (bits : FBits .f32) :
    (model base).op (Spec.Dequant.defaultKey "math.sin") .f32 [bits] = sine bits := by
  simp [model, W8A8ByteModel.extend, W8A8IEEEPrimitives.model, transcendentalFallback,
    Spec.Dequant.defaultKey, W8A8IEEEPrimitives.key]

theorem model_cosine (base : FloatModel) (bits : FBits .f32) :
    (model base).op (Spec.Dequant.defaultKey "math.cos") .f32 [bits] = cosine bits := by
  simp [model, W8A8ByteModel.extend, W8A8IEEEPrimitives.model, transcendentalFallback,
    Spec.Dequant.defaultKey, W8A8IEEEPrimitives.key]

theorem model_exponential (base : FloatModel) (bits : FBits .f32) :
    (model base).op (Spec.Dequant.defaultKey "math.exp") .f32 [bits] = exponential bits := by
  simp [model, W8A8ByteModel.extend, W8A8IEEEPrimitives.model, transcendentalFallback,
    Spec.Dequant.defaultKey, W8A8IEEEPrimitives.key]

theorem all_original_numerical_laws (base : FloatModel) :
    W8A8FiniteLaws.Laws (W8A8IEEEPrimitives.model (transcendentalFallback (W8A8UnarySign.model base))) ∧
    W8A8DynamicLaws.Laws (W8A8IEEEPrimitives.model (transcendentalFallback (W8A8UnarySign.model base))) ∧
    W8A8ArithmeticModel.Agreement (W8A8IEEEPrimitives.model (transcendentalFallback (W8A8UnarySign.model base))) :=
  ⟨W8A8IEEELaws.finite_laws _, W8A8IEEELaws.dynamic_laws _, W8A8IEEEPrimitives.agreement _⟩

def context (base : FloatModel) (externals : ExtEnv) (indexBits : Nat) : Ctx :=
  Check.PolicyW8A8IEEEContext.context (transcendentalFallback (W8A8UnarySign.model base)) externals indexBits

theorem context_implies_original (base : FloatModel) (externals : ExtEnv) (indexBits : Nat) :
    Check.PolicyW8A8NumericalContext.ContextOk (context base externals indexBits) :=
  Check.PolicyW8A8IEEEContext.context_ok _ _ _

theorem model_negation (base : FloatModel) (bits : FBits .f32) :
    (model base).op (Spec.Dequant.defaultKey "arith.negf") .f32 [bits] = W8A8UnarySign.negate bits := by
  simp [model, W8A8ByteModel.extend, W8A8IEEEPrimitives.model, transcendentalFallback,
    W8A8UnarySign.model, Spec.Dequant.defaultKey, W8A8IEEEPrimitives.key]

end
end MerlinVerify.Rules.W8A8RoundedReal
