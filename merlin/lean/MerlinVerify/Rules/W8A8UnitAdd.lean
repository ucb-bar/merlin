/- Actual NN binary32 addition preserves a representable1threshold. Ordered folds use
an actual unit member; a nonzero-only divisor fact is never silently strengthened. -/
import MerlinVerify.Rules.W8A8UnitRounding
import MerlinVerify.Rules.W8A8PositiveAdd

namespace MerlinVerify.Rules.W8A8UnitAdd

open MerlinVerify W8A8OrderedMagnitude
open Float.Model (UnpackedFloat)
open Float.Model.UnpackedFloat

def AtLeastOne (bits : FBits .f32) : Prop :=
  W8A8IEEE32.Nonnegative bits ∧ 127 ≤ W8A8ByteFloat.exponent bits

instance (bits : FBits .f32) : Decidable (AtLeastOne bits) :=
  inferInstanceAs (Decidable (W8A8IEEE32.Nonnegative bits ∧ 127 ≤ W8A8ByteFloat.exponent bits))

theorem unpack_unit_fields {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive)
    (lower : 127 ≤ W8A8ByteFloat.exponent bits) : 2^23 ≤ mantissa ∧ -23 ≤ exponent := by
  unfold W8A8IEEE32.unpack UnpackedFloat.unpack at decoded
  dsimp only at decoded
  split at decoded
  · split at decoded <;> cases decoded
  · split at decoded
    · next zero =>
      have exponentZero : W8A8ByteFloat.exponent bits = 0 :=
        (W8A8IEEE32.exponent_eq_unpack bits).trans (congrArg BitVec.toNat zero)
      omega
    · cases decoded
      have exponentLower : 127 ≤ (unpackExponent (spec := W8A8IEEE32.format) bits).toNat := by
        rw [← W8A8IEEE32.exponent_eq_unpack]
        exact lower
      rw [W8A8IEEE32.append_toNat]
      simp only [BitVec.toNat_ofNat]
      change 8388608 ≤ 1 * 8388608 + _ ∧
        -23 ≤ ((unpackExponent (spec := W8A8IEEE32.format) bits).toNat : Int) - (127 + 23)
      omega

theorem units_lower {bits : FBits .f32} (lower : 127 ≤ W8A8ByteFloat.exponent bits) :
    2^149 ≤ units bits := by
  by_contra below
  have exponent := exponent_le_of_units (power := 149) (x := bits) (show units bits < 2^149 by omega)
  change W8A8ByteFloat.exponent bits ≤ 126 at exponent
  omega

theorem add_unit_left {left right : FBits .f32}
    (leftUnit : AtLeastOne left) (rightNonnegative : W8A8IEEE32.Nonnegative right) :
    AtLeastOne (W8A8IEEE32.add left right) := by
  refine ⟨W8A8IEEE32.add_nonnegative leftUnit.1 rightNonnegative, ?_⟩
  have leftSign := W8A8IEEE32.unpack_positiveSign leftUnit.1
  have rightSign := W8A8IEEE32.unpack_positiveSign rightNonnegative
  have leftNotNaN := W8A8IEEE32.unpack_notNaN leftUnit.1.1
  have rightNotNaN := W8A8IEEE32.unpack_notNaN rightNonnegative.1
  have leftNonzero : W8A8IEEE32.Nonzero left := by
    intro zero
    have exponentZero : W8A8ByteFloat.exponent left = 0 := by simp [W8A8ByteFloat.exponent, zero]
    have lower := leftUnit.2
    omega
  have leftNotZero := W8A8IEEE32.unpack_notZero leftNonzero
  cases leftDecoded : W8A8IEEE32.unpack left with
  | notANumber => exact (leftNotNaN leftDecoded).elim
  | zero sign => exact (leftNotZero sign leftDecoded).elim
  | infinity sign =>
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity rightInfinitySign =>
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      cases sign <;> cases rightInfinitySign <;> decide
    | zero _ =>
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      cases sign <;> decide
    | finite _ _ _ _ =>
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      cases sign <;> decide
  | finite sign leftMantissa leftExponent leftPositive =>
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity rightInfinitySign =>
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      cases rightInfinitySign <;> decide
    | zero _ =>
      obtain ⟨mantissaLower, exponentLower⟩ := unpack_unit_fields leftDecoded leftUnit.2
      have mantissaUpper := (W8A8PositiveAdd.unpack_fields leftDecoded).1
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      exact W8A8UnitRounding.packed_unit_lower sign leftPositive mantissaLower mantissaUpper exponentLower
    | finite rightFiniteSign rightMantissa rightExponent rightPositive =>
      have signPositive : sign = .positive := by simpa only [leftDecoded, W8A8IEEE32.PositiveSign] using leftSign
      have rightSignPositive : rightFiniteSign = .positive := by
        simpa only [rightDecoded, W8A8IEEE32.PositiveSign] using rightSign
      subst sign
      subst rightFiniteSign
      obtain ⟨leftLower, leftUnits⟩ := unpack_units leftDecoded
      obtain ⟨rightLower, rightUnits⟩ := unpack_units rightDecoded
      let common := min leftExponent rightExponent
      let sum := (leftMantissa <<< (leftExponent - common).toNat) +
        (rightMantissa <<< (rightExponent - common).toNat)
      have commonLower : -149 ≤ common := by dsimp only [common]; omega
      have leftAligned : common ≤ leftExponent := by dsimp only [common]; omega
      have rightAligned : common ≤ rightExponent := by dsimp only [common]; omega
      have sumUnits : sum * 2^(common + 149).toNat = units left + units right := by
        dsimp only [sum]
        rw [Nat.add_mul, aligned_units _ commonLower leftAligned, aligned_units _ commonLower rightAligned,
          ← leftUnits, ← rightUnits]
      have lowerUnits : 2^149 ≤ sum * 2^(common + 149).toNat := by
        rw [sumUnits]
        exact Nat.le_trans (units_lower leftUnit.2) (Nat.le_add_right _ _)
      have leading := W8A8UnitRounding.leading_of_units commonLower lowerUnits
      have shiftedPositive : 0 < leftMantissa <<< (leftExponent - common).toNat := by
        rw [Nat.shiftLeft_eq]
        exact Nat.mul_pos leftPositive (Nat.two_pow_pos _)
      have sumPositive : 0 < sum := Nat.lt_of_lt_of_le shiftedPositive (Nat.le_add_right _ _)
      have intPositive : (0:Int) < (sum : Int) := by exact_mod_cast sumPositive
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add, Sign.apply]
      change 127 ≤ W8A8ByteFloat.exponent
        (W8A8IEEE32.pack (UnpackedFloat.normalize W8A8IEEE32.format (sum : Int) common .positive))
      rw [UnpackedFloat.normalize, Int.compare_eq_gt.mpr intPositive]
      simp only [Int.toNat_natCast]
      exact W8A8UnitRounding.rounded_unit_lower .positive sumPositive leading

theorem add_unit_right {left right : FBits .f32}
    (leftNonnegative : W8A8IEEE32.Nonnegative left) (rightUnit : AtLeastOne right) :
    AtLeastOne (W8A8IEEE32.add left right) := by
  rw [W8A8OrderedMagnitude.add_comm]
  exact add_unit_left rightUnit leftNonnegative

theorem fold_unit_initial (words : List (FBits .f32))
    (members : ∀ word ∈ words, W8A8IEEE32.Nonnegative word) :
    ∀ initial, AtLeastOne initial → AtLeastOne (words.foldl W8A8IEEE32.add initial) := by
  induction words with
  | nil => intro initial unit; exact unit
  | cons head tail induction =>
    intro initial unit
    exact induction (fun word member => members word (by simp [member])) _
      (add_unit_left unit (members head (by simp)))

theorem fold_unit_member (words : List (FBits .f32)) :
    ∀ initial, W8A8IEEE32.Nonnegative initial → (∀ word ∈ words, W8A8IEEE32.Nonnegative word) →
    (∃ word ∈ words, AtLeastOne word) → AtLeastOne (words.foldl W8A8IEEE32.add initial) := by
  induction words with
  | nil => intro _ _ _ witness; simp at witness
  | cons head tail induction =>
    intro initial nonnegative members witness
    obtain ⟨word, member, unit⟩ := witness
    rcases List.mem_cons.mp member with rfl | later
    · exact fold_unit_initial tail (fun word member => members word (by simp [member])) _
        (add_unit_right nonnegative unit)
    · exact induction _ (W8A8IEEE32.add_nonnegative nonnegative (members head (by simp)))
        (fun word member => members word (by simp [member])) ⟨word, later, unit⟩

end MerlinVerify.Rules.W8A8UnitAdd
