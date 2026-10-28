/- Checked numerical postconditions for source per-tensor dequantization. The actual scale
and zero-point bits determine a finite byte table. No claim about a lowering pass is made. -/
import MerlinVerify.Rules.W8A8ArithmeticModel
import MerlinVerify.Rules.W8A8DequantWidth

namespace MerlinVerify.Rules.W8A8DequantNumerics

open MerlinVerify

def scalar {width : Nat} (q : BitVec 8) (zero : BitVec width) (scale : FBits .f32) : FBits .f32 :=
  W8A8IEEE32.mul
    (W8A8ArithmeticModel.subtract32 (W8A8ArithmeticModel.signedInteger32 q)
      (W8A8ArithmeticModel.signedInteger32 zero)) scale

theorem scalar_agrees {base : FloatModel} (agreement : W8A8ArithmeticModel.Agreement base)
    {width : Nat} (q : BitVec 8) (zero : BitVec width) (scale : FBits .f32)
    (differenceNonNaN : W8A8IEEE32.NonNaN (W8A8ArithmeticModel.subtract32
      (W8A8ArithmeticModel.signedInteger32 q) (W8A8ArithmeticModel.signedInteger32 zero)))
    (resultNonNaN : W8A8IEEE32.NonNaN (scalar q zero scale)) :
    W8A8DequantWidth.perTensor width (W8A8ByteModel.extend base) q zero scale = scalar q zero scale := by
  have subtraction (x y : FBits .f32) :
      (W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "arith.subf") .f32 [x, y] =
        base.op (Spec.Dequant.defaultKey "arith.subf") .f32 [x, y] := by
    simp [W8A8ByteModel.extend, Spec.Dequant.defaultKey]
  simp only [W8A8DequantWidth.perTensor, W8A8ByteModel.multiply_eq]
  change base.op (Spec.Dequant.defaultKey "arith.mulf") .f32
    [(W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "arith.subf") .f32
      [base.ofInt (Spec.Dequant.defaultKey "arith.sitofp") 8 .f32 q,
        base.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 zero], scale] = _
  rw [subtraction, agreement.signedInteger32, agreement.signedInteger32,
    agreement.subtract32 _ _ differenceNonNaN, agreement.multiply32 _ _ resultNonNaN]
  rfl

def finiteCheck {width : Nat} (zero : BitVec width) (scale : FBits .f32) : Bool :=
  (List.range 256).all fun n =>
    decide (W8A8IEEE32.NonNaN (W8A8ArithmeticModel.subtract32
      (W8A8ArithmeticModel.signedInteger32 (BitVec.ofNat 8 n)) (W8A8ArithmeticModel.signedInteger32 zero))) &&
    decide (W8A8IEEE32.Finite (scalar (BitVec.ofNat 8 n) zero scale))

theorem finiteCheck_sound {base : FloatModel} (agreement : W8A8ArithmeticModel.Agreement base)
    {width : Nat} {zero : BitVec width} {scale : FBits .f32} (checked : finiteCheck zero scale = true)
    (q : BitVec 8) :
    W8A8IEEE32.Finite (W8A8DequantWidth.perTensor width (W8A8ByteModel.extend base) q zero scale) := by
  have member : q.toNat ∈ List.range 256 := List.mem_range.2 q.isLt
  have fact := List.all_eq_true.1 checked q.toNat member
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq, Bool.and_eq_true, decide_eq_true_eq] at fact
  rw [scalar_agrees agreement q zero scale fact.1 (W8A8IEEE32.finite_notNaN fact.2)]
  exact fact.2

#print axioms scalar_agrees
#print axioms finiteCheck_sound

end MerlinVerify.Rules.W8A8DequantNumerics
