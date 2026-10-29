import MerlinVerify.Rules.W8A8DynamicLaws

namespace MerlinVerify.Rules.W8A8FiniteProductsCountermodel

open MerlinVerify W8A8FiniteLaws

/-- An abstract countermodel, deliberately not an IEEE implementation. Every multiply
returns infinity; the current primitive laws require non-NaN but not finite products. -/
def model : FloatModel where
  op operation kind _ := match kind with
    | .f32 => if operation = Spec.Dequant.defaultKey "arith.mulf" then 0x7f800000#32 else 0#32
    | .f64 => 0#64
    | .f16 => 0#16
    | .bf16 => 0#16
  cmp _ _ _ _ := false
  convert _ _ destination _ := match destination with
    | .f64 => 0x3810000000000000#64
    | .f32 => 0#32
    | .f16 => 0#16
    | .bf16 => 0#16
  toInt _ _ _ _ := none
  ofInt _ _ kind _ := 0#kind.width

theorem source_laws : W8A8FiniteLaws.Laws model where
  widen := by intros; change WideScale 0x3810000000000000#64; unfold WideScale; decide
  reciprocal := by intros; change Narrowable 0#64; unfold Narrowable; decide
  narrow := by intros; change Finite 0#32; decide
  integer := by intros; change Finite 0#32; decide
  multiply := by intros; change W8A8ByteFloat.isNaN 0x7f800000#32 = false; decide
  round := by intros; change W8A8ByteFloat.isNaN 0#32 = false; decide
  add := by intros; change W8A8ByteFloat.isNaN 0#32 = false; decide

theorem dynamic_laws : W8A8DynamicLaws.Laws model where
  source := source_laws
  absolute := by intros; change Finite 0#32; decide
  scaleDivide := by intros; change Finite 0#32; decide
  reciprocal32 := by intros; change Finite 0#32; decide

/-- Even bounds containing only ordinary finite operands cannot supply a missing
concrete multiplication law. The operands here are both positive one. -/
theorem finite_product_not_entailed :
    ¬ (∀ M : FloatModel, W8A8DynamicLaws.Laws M →
      ∀ x y : FBits .f32, Finite x → Finite y →
        Finite (M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y])) := by
  intro proposed
  have impossible := proposed model dynamic_laws 0x3f800000#32 0x3f800000#32
    (by decide) (by decide)
  change W8A8ByteFloat.exponent 0x7f800000#32 < 255 at impossible
  exact (by decide : ¬ W8A8ByteFloat.exponent 0x7f800000#32 < 255) impossible

#print axioms source_laws
#print axioms dynamic_laws
#print axioms finite_product_not_entailed

end MerlinVerify.Rules.W8A8FiniteProductsCountermodel
