/- Kernel examples for one joint law/agreement witness, including guarded endpoints. -/
import MerlinVerify.Check.PolicyW8A8IEEEPrefix

open MerlinVerify MerlinVerify.Rules MerlinVerify.Check

abbrev primitive := W8A8IEEEPrimitives.model W8A8IEEEPrimitives.unsupported
abbrev key := Spec.Dequant.defaultKey

theorem source_laws : W8A8FiniteLaws.Laws primitive := W8A8IEEELaws.finite_laws _
theorem dynamic_laws : W8A8DynamicLaws.Laws primitive := W8A8IEEELaws.dynamic_laws _
theorem arithmetic_agreement : W8A8ArithmeticModel.Agreement primitive := W8A8IEEEPrimitives.agreement _
theorem joint_context : PolicyW8A8NumericalContext.ContextOk PolicyW8A8IEEEContext.candidate :=
  PolicyW8A8IEEEContext.candidate_ok

theorem boundary_controls :
    primitive.convert (key "arith.extf") .f32 .f64 0x00800000 = 0x3810000000000000 ∧
    primitive.op (key "arith.divf") .f64 [0x3ff0000000000000, 0x3810000000000000] = 0x47d0000000000000 ∧
    primitive.convert (key "arith.truncf") .f64 .f32 0x47d0000000000000 = 0x7e800000 ∧
    primitive.op (key "arith.divf") .f32 [0x3f800000, 0x00800000] = 0x7e800000 ∧
    W8A8FiniteLaws.scaleCheck 0x00000001 = false ∧
    primitive.op (key "arith.divf") .f32 [0x3f800000, 0x00000001] = 0x7f800000 := by decide

theorem signed_integer_controls :
    primitive.ofInt (key "arith.sitofp") 32 .f32 0x80000000 = 0xcf000000 ∧
    primitive.ofInt (key "arith.sitofp") 32 .f32 0x7fffffff = 0x4f000000 ∧
    primitive.ofInt (key "arith.sitofp") 64 .f32 0x8000000000000000 = 0xdf000000 ∧
    primitive.ofInt (key "arith.sitofp") 64 .f32 0x7fffffffffffffff = 0x5f000000 := by decide

theorem rounding_and_overflow_controls :
    primitive.op (key "math.roundeven") .f32 [0x3f000000] = 0 ∧
    primitive.op (key "math.roundeven") .f32 [0xbf000000] = 0x80000000 ∧
    primitive.op (key "math.roundeven") .f32 [0x3fc00000] = 0x40000000 ∧
    primitive.op (key "math.roundeven") .f32 [0xbfc00000] = 0xc0000000 ∧
    primitive.op (key "math.absf") .f32 [0x80000001] = 1 ∧
    primitive.op (key "arith.mulf") .f32 [0x7f7fffff, 0x40000000] = 0x7f800000 := by decide

theorem supported_joint_domain :
    ∃ ctx args, PolicyW8A8IEEEQualified.conditions 64 ctx ∧
      PolicyW8A8RotaryDomain.Domain 64 [.tensor [2, 3] (.float .f32)] [none] [some 250] args := by
  exact PolicyW8A8IEEEQualified.domain_inhabited (by decide)

#print axioms source_laws
#print axioms dynamic_laws
#print axioms arithmetic_agreement
#print axioms joint_context
#print axioms supported_joint_domain
#print axioms boundary_controls
#print axioms signed_integer_controls
#print axioms rounding_and_overflow_controls
