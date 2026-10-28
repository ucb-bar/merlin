/- Conservative bounds for one actual binary32 addition. Both operands may have either
sign; rounding and finite-output safety follow from the existing aligned integer-unit proof. -/
import MerlinVerify.Rules.W8A8OrderedSum

namespace MerlinVerify.Rules.W8A8PairBounds

open MerlinVerify W8A8OrderedMagnitude

def addCap (left right : Nat) : Nat := max left right + 2

theorem input_units_bound {x y : FBits .f32} {left right : Nat}
    (leftBound : W8A8IEEEBounds.ExponentLE left x)
    (rightBound : W8A8IEEEBounds.ExponentLE right y) :
    units x + units y < 2^(max left right + 24) := by
  let cap := max left right
  have leftUnits : units x < 2^(cap + 23) :=
    Nat.lt_of_lt_of_le (units_lt_of_exponent leftBound)
      (Nat.pow_le_pow_right (by decide) (Nat.add_le_add_right (Nat.le_max_left _ _) 23))
  have rightUnits : units y < 2^(cap + 23) :=
    Nat.lt_of_lt_of_le (units_lt_of_exponent rightBound)
      (Nat.pow_le_pow_right (by decide) (Nat.add_le_add_right (Nat.le_max_right _ _) 23))
  have nextPower : 2^(cap + 24) = 2 * 2^(cap + 23) := by
    rw [show cap + 24 = (cap + 23) + 1 by omega, Nat.pow_succ, Nat.mul_comm]
  rw [nextPower]
  omega

theorem result_bounded {x y result : FBits .f32} {left right : Nat}
    (leftBound : W8A8IEEEBounds.ExponentLE left x)
    (rightBound : W8A8IEEEBounds.ExponentLE right y) (finite : W8A8IEEE32.Finite result)
    (rounded : units result ≤ units x + units y + 2^(max left right)) :
    W8A8IEEEBounds.ExponentLE (addCap left right) result := by
  let cap := max left right
  have sumBound : units x + units y < 2^(cap + 24) := input_units_bound leftBound rightBound
  change units result ≤ units x + units y + 2^cap at rounded
  have smallRounding : 2^cap ≤ 2^(cap + 24) := Nat.pow_le_pow_right (by decide) (by omega)
  have lastPower : 2^(cap + 25) = 2 * 2^(cap + 24) := by
    rw [show cap + 25 = (cap + 24) + 1 by omega, Nat.pow_succ, Nat.mul_comm]
  have total : units result < 2^(cap + 25) := by rw [lastPower]; omega
  have exponent := exponent_le_of_units total
  refine ⟨finite, ?_⟩
  change W8A8ByteFloat.exponent result ≤ max left right + 2
  dsimp only [cap] at exponent
  omega

theorem add_bounded {x y : FBits .f32} {left right : Nat}
    (leftBound : W8A8IEEEBounds.ExponentLE left x)
    (rightBound : W8A8IEEEBounds.ExponentLE right y) (safe : max left right ≤ 252) :
    W8A8IEEEBounds.ExponentLE (addCap left right) (W8A8IEEE32.add x y) := by
  obtain ⟨finite, rounded⟩ := W8A8OrderedMagnitude.add_bounded leftBound.1 rightBound.1 safe
    (input_units_bound leftBound rightBound)
  exact result_bounded leftBound rightBound finite rounded

end MerlinVerify.Rules.W8A8PairBounds
