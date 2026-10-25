/- Integer-unit magnitude bounds for the actual ordered binary32 computation.
This additive family preserves the earlier qualified checker and its assumptions. -/
import MerlinVerify.Rules.W8A8IEEEBounds

namespace MerlinVerify.Rules.W8A8OrderedMagnitude

open MerlinVerify Float.Model.UnpackedFloat W8A8IEEE32
open Float.Model (UnpackedFloat)

def units (x : FBits .f32) : Nat :=
  let exponent := W8A8ByteFloat.exponent x
  let fraction := W8A8ByteFloat.fraction x
  if exponent = 0 then fraction else (2^23 + fraction) * 2^(exponent - 1)

theorem unpack_units {x : FBits .f32} {s : Sign} {m : Nat} {e : Int} {positive : 0 < m}
    (decoded : W8A8IEEE32.unpack x = .finite s m e positive) :
    -149 ≤ e ∧ units x = m * 2^(e + 149).toNat := by
  unfold W8A8IEEE32.unpack UnpackedFloat.unpack at decoded
  dsimp only at decoded
  split at decoded
  · split at decoded <;> cases decoded
  · split at decoded
    · next exponentZero =>
      split at decoded
      · cases decoded
      · cases decoded
        simp [units, exponent_eq_unpack, fraction_eq_unpack, exponentZero,
          format, Float.Model.Format.binary32, Float.Model.Format.exponentBias]
    · next exponentNonzero =>
      have nonzero : (unpackExponent (spec := format) x).toNat ≠ 0 := by
        intro equal
        apply exponentNonzero
        exact BitVec.eq_of_toNat_eq equal
      cases decoded
      constructor
      · change -149 ≤ ((unpackExponent (spec := format) x).toNat : Int) - (127 + 23)
        omega
      · have shift : (((unpackExponent (spec := format) x).toNat : Int) - (127 + 23) + 149).toNat =
            (unpackExponent (spec := format) x).toNat - 1 := by omega
        simp only [units, exponent_eq_unpack, fraction_eq_unpack, ite_eq_right nonzero,
          append_toNat, BitVec.toNat_ofNat]
        change (8388608 + (unpackMantissa (spec := format) x).toNat) *
          2^((unpackExponent (spec := format) x).toNat - 1) =
          (8388608 + (unpackMantissa (spec := format) x).toNat) *
          2^(((unpackExponent (spec := format) x).toNat : Int) - (127 + 23) + 149).toNat
        rw [shift]

theorem pack_units_le {s : Sign} {m : Nat} {e : Int} {positive : 0 < m}
    (lower : -149 ≤ e) (upper : e ≤ 104) :
    units (W8A8IEEE32.pack (.finite s m e positive)) ≤ m * 2^(e + 149).toNat := by
  change units (UnpackedFloat.pack format (.finite s m e positive)) ≤ _
  simp only [UnpackedFloat.pack]
  split
  · next overflow => change 256 ≤ (e + 127 + 23).toNat + 1 at overflow; omega
  · split
    · next normal =>
      have log : m.log2 = 23 := by change m.log2 + 1 = 24 at normal; omega
      obtain ⟨low, high⟩ := (Nat.log2_eq_iff (Nat.ne_of_gt positive)).1 log
      have exponentLow : 0 < (e + 127 + 23).toNat := by omega
      have exponentHigh : (e + 127 + 23).toNat < 256 := by omega
      have shifted : (e + 127 + 23).toNat - 1 = (e + 149).toNat := by omega
      have mantissa : 2^23 + m % 2^23 = m := by omega
      simp only [units, exponent_packComponents, fraction_packComponents, BitVec.toNat_ofNat]
      change (if (e + 127 + 23).toNat % 256 = 0 then m % 8388608
        else (8388608 + m % 8388608) * 2^((e + 127 + 23).toNat % 256 - 1)) ≤ _
      rw [Nat.mod_eq_of_lt exponentHigh, ite_eq_right (Nat.ne_of_gt exponentLow), shifted, mantissa]
      exact Nat.le_refl _
    · simp only [units, exponent_packComponents, fraction_packComponents, BitVec.toNat_ofNat]
      change m % 8388608 ≤ m * 2^(e + 149).toNat
      have power := Nat.two_pow_pos (e + 149).toNat
      have remainder := Nat.mod_le m 8388608
      exact Nat.le_trans remainder (Nat.le_mul_of_pos_right m power)

theorem zero_units (s : Sign) :
    Finite (W8A8IEEE32.pack (.zero s)) ∧ units (W8A8IEEE32.pack (.zero s)) = 0 := by
  cases s <;> decide

theorem units_lt_of_exponent {x : FBits .f32} {upper : Nat}
    (bounded : W8A8IEEEBounds.ExponentLE upper x) : units x < 2^(upper + 23) := by
  have fraction : W8A8ByteFloat.fraction x < 2^23 := by
    rw [fraction_eq_unpack]
    exact (unpackMantissa (spec := format) x).isLt
  unfold units
  dsimp only
  split
  · exact Nat.lt_of_lt_of_le fraction (Nat.pow_le_pow_right (by decide) (by omega))
  · next nonzero =>
    have positive : 0 < W8A8ByteFloat.exponent x := Nat.pos_of_ne_zero nonzero
    have mantissa : 2^23 + W8A8ByteFloat.fraction x < 2^24 := by omega
    have bound := Nat.mul_lt_mul_of_pos_right mantissa (Nat.two_pow_pos (W8A8ByteFloat.exponent x - 1))
    have exponent : 24 + (W8A8ByteFloat.exponent x - 1) = W8A8ByteFloat.exponent x + 23 := by omega
    rw [← Nat.pow_add, exponent] at bound
    exact Nat.lt_of_lt_of_le bound (Nat.pow_le_pow_right (by decide) (Nat.add_le_add_right bounded.2 23))

theorem exponent_le_of_units {x : FBits .f32} {power : Nat} (bounded : units x < 2^power) :
    W8A8ByteFloat.exponent x ≤ power - 23 := by
  by_cases zero : W8A8ByteFloat.exponent x = 0
  · rw [zero]
    exact Nat.zero_le _
  · have positive : 0 < W8A8ByteFloat.exponent x := Nat.pos_of_ne_zero zero
    have leading : 2^(W8A8ByteFloat.exponent x + 22) ≤ units x := by
      unfold units
      rw [ite_eq_right zero]
      have exponent : W8A8ByteFloat.exponent x + 22 = 23 + (W8A8ByteFloat.exponent x - 1) := by omega
      rw [exponent, Nat.pow_add]
      exact Nat.mul_le_mul_right _ (Nat.le_add_right _ _)
    have strict := Nat.lt_of_le_of_lt leading bounded
    have exponents : W8A8ByteFloat.exponent x + 22 < power :=
      (Nat.pow_lt_pow_iff_right (by decide)).1 strict
    omega

#print axioms unpack_units
#print axioms pack_units_le
#print axioms zero_units
#print axioms units_lt_of_exponent
#print axioms exponent_le_of_units

end MerlinVerify.Rules.W8A8OrderedMagnitude
