/- Kernel-checked examples and negative controls. Conditional source-quantizer theorems do
not construct a model satisfying the numerical context. -/
import MerlinVerify.Check.PolicyW8A8NumericalInhabited

open MerlinVerify MerlinVerify.Rules MerlinVerify.Check Definedness

namespace W8A8NumericalExamples

theorem rsqrt_positive_zero : W8A8IEEE32.rsqrt 0 = 0x7f800000#32 := by rfl

theorem finite_overflow_permitted :
    W8A8IEEE32.mul 0x7f7fffff#32 0x40000000#32 = 0x7f800000#32 := by rfl

theorem subnormal_inverse_refused : W8A8ArithmeticModel.inverseCheck 0x00000001#32 = false := by decide

theorem maximum_scale_inverse_nonzero : W8A8ArithmeticModel.inverseCheck 0x7f7fffff#32 = true := by decide

theorem zero_gain_outside_candidate :
    PolicyW8A8NumericalFacts.bitsCheck .nonzeroFinite 0x00000000#32 = false ∧
    PolicyW8A8NumericalFacts.bitsCheck .nonzeroFinite 0x80000000#32 = false := by decide

theorem nan_is_not_quantizer_input :
    PolicyW8A8NumericalFacts.bitsCheck .nonNaN 0x7fc00000#32 = false := by decide

theorem positive_infinity_source_conversion (base : FloatModel)
    (agreement : W8A8ArithmeticModel.Agreement base) (laws : W8A8FiniteLaws.Laws base) :
    W8A8SourceQuantize.scalar 64 (W8A8ByteModel.extend base) (some 0x7f800000#32)
      0x3f800000#32 0 ≠ .poison :=
  W8A8ArithmeticModel.scalar_defined agreement laws (Or.inr rfl) (by decide) (by decide)

theorem negative_infinity_source_conversion (base : FloatModel)
    (agreement : W8A8ArithmeticModel.Agreement base) (laws : W8A8FiniteLaws.Laws base) :
    W8A8SourceQuantize.scalar 64 (W8A8ByteModel.extend base) (some 0xff800000#32)
      0x3f800000#32 0 ≠ .poison :=
  W8A8ArithmeticModel.scalar_defined agreement laws (Or.inr rfl) (by decide) (by decide)

theorem candidate_inputs_exist : ∃ args,
    PolicyW8A8NumericalDefined.Domain 64
      [.tensor [2, 2048] (.float .f32), .tensor [2048] (.float .f32), .tensor [2] (.int 64)]
      [none, none, some ⟨0, 17⟩] args :=
  PolicyW8A8NumericalInhabited.check_sound (by decide)

theorem empty_integer_interval_refused : PolicyW8A8NumericalInhabited.check 64
    [.tensor [2] (.int 64)] [some ⟨3, 3⟩] = false := by decide

#print axioms rsqrt_positive_zero
#print axioms finite_overflow_permitted
#print axioms positive_infinity_source_conversion
#print axioms negative_infinity_source_conversion
#print axioms candidate_inputs_exist

end W8A8NumericalExamples
