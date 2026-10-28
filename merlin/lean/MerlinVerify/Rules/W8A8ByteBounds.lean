/- Exact signed-byte tables derive dequantized weight bounds from the actual
scale and zero-point bits, preserving conversion/subtraction/multiplication order. -/
import MerlinVerify.Rules.W8A8DequantNumerics
import MerlinVerify.Rules.W8A8IEEEBounds

namespace MerlinVerify.Rules.W8A8ByteBounds

open MerlinVerify

def cap {width : Nat} (zero : BitVec width) (scale : FBits .f32) : Nat :=
  ((List.range 256).map fun n => W8A8ByteFloat.exponent
    (W8A8DequantNumerics.scalar (BitVec.ofNat 8 n) zero scale)).foldl max 0

def check {width : Nat} (zero : BitVec width) (scale : FBits .f32) (upper : Nat) : Bool :=
  (List.range 256).all fun n =>
    decide (W8A8IEEE32.NonNaN (W8A8ArithmeticModel.subtract32
      (W8A8ArithmeticModel.signedInteger32 (BitVec.ofNat 8 n)) (W8A8ArithmeticModel.signedInteger32 zero))) &&
    decide (W8A8IEEEBounds.ExponentLE upper (W8A8DequantNumerics.scalar (BitVec.ofNat 8 n) zero scale))

theorem check_sound {base : FloatModel} (agreement : W8A8ArithmeticModel.Agreement base)
    {width upper : Nat} {zero : BitVec width} {scale : FBits .f32} (checked : check zero scale upper = true)
    (q : BitVec 8) : W8A8IEEEBounds.ExponentLE upper
      (W8A8DequantWidth.perTensor width (W8A8ByteModel.extend base) q zero scale) := by
  have member : q.toNat ∈ List.range 256 := List.mem_range.2 q.isLt
  have fact := List.all_eq_true.1 checked q.toNat member
  simp only [BitVec.ofNat_toNat, BitVec.setWidth_eq, Bool.and_eq_true, decide_eq_true_eq] at fact
  rw [W8A8DequantNumerics.scalar_agrees agreement q zero scale fact.1 (W8A8IEEE32.finite_notNaN fact.2.1)]
  exact fact.2

#print axioms check_sound

end MerlinVerify.Rules.W8A8ByteBounds
