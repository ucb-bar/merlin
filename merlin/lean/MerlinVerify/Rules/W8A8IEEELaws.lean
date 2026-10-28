/- All existing source and dynamic primitive laws for one explicit arithmetic construction.
The unconstrained fallback is arbitrary; no field of the existing law records is assumed. -/
import MerlinVerify.Rules.W8A8IEEEReciprocal64

namespace MerlinVerify.Rules.W8A8IEEELaws

open MerlinVerify

theorem finite_laws (fallback : FloatModel) : W8A8FiniteLaws.Laws (W8A8IEEEPrimitives.model fallback) where
  widen _ normal := W8A8IEEEWiden.widen_wide normal
  reciprocal _ wide := W8A8IEEEReciprocal64.reciprocal_narrowable wide
  narrow _ bound := W8A8IEEENarrow.narrow_finite bound
  integer _ allowed value := W8A8IEEEInteger.signed_finite allowed value
  multiply _ _ left right := W8A8IEEE32.mul_finite_notNaN left right
  round _ nonNaN := W8A8IEEEPrimitives.roundeven_nonNaN nonNaN
  add _ _ nonNaN finite := W8A8IEEE32.add_nonNaN_finite nonNaN finite

theorem dynamic_laws (fallback : FloatModel) : W8A8DynamicLaws.Laws (W8A8IEEEPrimitives.model fallback) where
  source := finite_laws fallback
  absolute _ finite := W8A8IEEEPrimitives.absolute_finite finite
  scaleDivide _ finite := W8A8IEEEDivision32.scale_divide_finite finite
  reciprocal32 _ normal := W8A8IEEEDivision32.reciprocal_finite normal

#print axioms finite_laws
#print axioms dynamic_laws

end MerlinVerify.Rules.W8A8IEEELaws
