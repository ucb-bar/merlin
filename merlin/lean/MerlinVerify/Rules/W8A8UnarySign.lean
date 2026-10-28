/- Explicit binary32 negation flips only the sign bit. The source key, format and arity
are exact; all other fallback fields are preserved. This is a logical primitive definition. -/
import MerlinVerify.Rules.W8A8IEEEBounds

namespace MerlinVerify.Rules.W8A8UnarySign

open MerlinVerify

def negate (bits : FBits .f32) : FBits .f32 := bits ^^^ 0x80000000

theorem magnitude (bits : FBits .f32) :
    W8A8ByteFloat.magnitudeWord (negate bits) = W8A8ByteFloat.magnitudeWord bits := by
  unfold W8A8ByteFloat.magnitudeWord negate
  rw [BitVec.toNat_xor, Nat.xor_mod_two_pow]
  rw [show (0x80000000 : FBits .f32).toNat % 2^31 = 0 by decide, Nat.xor_zero]

theorem exponent (bits : FBits .f32) :
    W8A8ByteFloat.exponent (negate bits) = W8A8ByteFloat.exponent bits := by
  simp only [W8A8ByteFloat.exponent, magnitude]

theorem bounded {bits : FBits .f32} {cap : Nat} (finite : W8A8IEEEBounds.ExponentLE cap bits) :
    W8A8IEEEBounds.ExponentLE cap (negate bits) := by
  change W8A8ByteFloat.exponent (negate bits) < 255 ∧ W8A8ByteFloat.exponent (negate bits) ≤ cap
  rw [exponent]
  exact finite

def model (fallback : FloatModel) : FloatModel :=
  { fallback with op := fun key kind arguments =>
    match kind, arguments with
    | .f32, [bits] => if key = Spec.Dequant.defaultKey "arith.negf" then negate bits
        else fallback.op key .f32 [bits]
    | kind, arguments => fallback.op key kind arguments }

end MerlinVerify.Rules.W8A8UnarySign
