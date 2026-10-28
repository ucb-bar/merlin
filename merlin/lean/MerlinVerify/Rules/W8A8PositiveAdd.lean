/- Actual nonnegative binary32 addition preserves an existing positive member. Ordered
folds containing a positive member therefore cannot become zero. Infinity remains allowed
by this lemma; separate counted bounds establish a finite denominator. -/
import MerlinVerify.Rules.W8A8PositiveRounding
import MerlinVerify.Rules.W8A8OrderedSum

namespace MerlinVerify.Rules.W8A8PositiveAdd

open MerlinVerify W8A8OrderedMagnitude
open Float.Model (UnpackedFloat)
open Float.Model.UnpackedFloat

theorem unpack_finite {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive) :
    W8A8IEEE32.Finite bits := by
  unfold W8A8IEEE32.unpack UnpackedFloat.unpack at decoded
  dsimp only at decoded
  split at decoded
  · split at decoded <;> cases decoded
  · next notMaximum =>
    have small : W8A8ByteFloat.exponent bits < 256 := by
      rw [W8A8IEEE32.exponent_eq_unpack]
      exact (unpackExponent (spec := W8A8IEEE32.format) bits).isLt
    have different : W8A8ByteFloat.exponent bits ≠ 255 := by
      intro maximum
      apply notMaximum
      apply BitVec.eq_of_toNat_eq
      exact (W8A8IEEE32.exponent_eq_unpack bits).symm.trans maximum
    change W8A8ByteFloat.exponent bits < 255
    omega

theorem unpack_fields {bits : FBits .f32} {sign : Sign} {mantissa : Nat} {exponent : Int}
    {positive : 0 < mantissa} (decoded : W8A8IEEE32.unpack bits = .finite sign mantissa exponent positive) :
    mantissa < 2^24 ∧ -149 ≤ exponent := by
  have finite := unpack_finite decoded
  have cap : W8A8IEEEBounds.ExponentLE 254 bits := ⟨finite, by change W8A8ByteFloat.exponent bits < 255 at finite; omega⟩
  exact ⟨(W8A8IEEEBounds.unpack_finite_bounds cap decoded).1, (unpack_units decoded).1⟩

theorem add_positive_left {left right : FBits .f32}
    (leftPositive : W8A8IEEE32.Positive left) (rightNonnegative : W8A8IEEE32.Nonnegative right) :
    W8A8IEEE32.Positive (W8A8IEEE32.add left right) := by
  refine ⟨W8A8IEEE32.add_nonnegative leftPositive.1 rightNonnegative, ?_⟩
  have leftSign := W8A8IEEE32.unpack_positiveSign leftPositive.1
  have rightSign := W8A8IEEE32.unpack_positiveSign rightNonnegative
  have leftNotNaN := W8A8IEEE32.unpack_notNaN leftPositive.1.1
  have rightNotNaN := W8A8IEEE32.unpack_notNaN rightNonnegative.1
  have leftNotZero := W8A8IEEE32.unpack_notZero leftPositive.2
  cases leftDecoded : W8A8IEEE32.unpack left with
  | notANumber => exact (leftNotNaN leftDecoded).elim
  | zero sign => exact (leftNotZero sign leftDecoded).elim
  | infinity sign =>
    have signPositive : sign = .positive := by simpa only [leftDecoded, W8A8IEEE32.PositiveSign] using leftSign
    subst sign
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity rightInfinitySign =>
      have positive : rightInfinitySign = .positive := by
        simpa only [rightDecoded, W8A8IEEE32.PositiveSign] using rightSign
      subst rightInfinitySign
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      decide
    | zero _ => simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]; decide
    | finite _ _ _ _ => simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]; decide
  | finite sign leftMantissa leftExponent leftNonzero =>
    have signPositive : sign = .positive := by simpa only [leftDecoded, W8A8IEEE32.PositiveSign] using leftSign
    subst sign
    cases rightDecoded : W8A8IEEE32.unpack right with
    | notANumber => exact (rightNotNaN rightDecoded).elim
    | infinity sign =>
      have positive : sign = .positive := by simpa only [rightDecoded, W8A8IEEE32.PositiveSign] using rightSign
      subst sign
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      decide
    | zero _ =>
      obtain ⟨small, lower⟩ := unpack_fields leftDecoded
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add]
      exact W8A8PositiveRounding.packed_nonzero .positive leftNonzero small lower
    | finite sign rightMantissa rightExponent rightNonzero =>
      have signPositive : sign = .positive := by simpa only [rightDecoded, W8A8IEEE32.PositiveSign] using rightSign
      subst sign
      let common := min leftExponent rightExponent
      let sum := (leftMantissa <<< (leftExponent - common).toNat) +
        (rightMantissa <<< (rightExponent - common).toNat)
      have leftLower := (unpack_fields leftDecoded).2
      have rightLower := (unpack_fields rightDecoded).2
      have commonLower : -149 ≤ common := by dsimp only [common]; omega
      have leftShiftPositive : 0 < leftMantissa <<< (leftExponent - common).toNat := by
        rw [Nat.shiftLeft_eq]
        exact Nat.mul_pos leftNonzero (Nat.two_pow_pos _)
      have sumPositive : 0 < sum := Nat.lt_of_lt_of_le leftShiftPositive (Nat.le_add_right _ _)
      have intPositive : (0:Int) < (sum : Int) := by exact_mod_cast sumPositive
      simp only [W8A8IEEE32.add, leftDecoded, rightDecoded, UnpackedFloat.add, Sign.apply]
      change W8A8IEEE32.Nonzero
        (W8A8IEEE32.pack (UnpackedFloat.normalize W8A8IEEE32.format (sum : Int) common .positive))
      rw [UnpackedFloat.normalize, Int.compare_eq_gt.mpr intPositive]
      simp only [Int.toNat_natCast]
      exact W8A8PositiveRounding.round_nonzero .positive sumPositive commonLower

theorem add_positive_right {left right : FBits .f32}
    (leftNonnegative : W8A8IEEE32.Nonnegative left) (rightPositive : W8A8IEEE32.Positive right) :
    W8A8IEEE32.Positive (W8A8IEEE32.add left right) := by
  rw [W8A8OrderedMagnitude.add_comm]
  exact add_positive_left rightPositive leftNonnegative

theorem fold_positive_initial (words : List (FBits .f32))
    (members : ∀ word ∈ words, W8A8IEEE32.Nonnegative word) :
    ∀ initial, W8A8IEEE32.Positive initial → W8A8IEEE32.Positive (words.foldl W8A8IEEE32.add initial) := by
  induction words with
  | nil => intro initial positive; exact positive
  | cons head tail induction =>
    intro initial positive
    exact induction (fun word member => members word (by simp [member])) _
      (add_positive_left positive (members head (by simp)))

theorem fold_positive_member (words : List (FBits .f32)) :
    ∀ initial, W8A8IEEE32.Nonnegative initial → (∀ word ∈ words, W8A8IEEE32.Nonnegative word) →
    (∃ word ∈ words, W8A8IEEE32.Positive word) → W8A8IEEE32.Positive (words.foldl W8A8IEEE32.add initial) := by
  induction words with
  | nil => intro _ _ _ witness; simp at witness
  | cons head tail induction =>
    intro initial nonnegative members witness
    obtain ⟨word, member, positive⟩ := witness
    rcases List.mem_cons.mp member with rfl | later
    · exact fold_positive_initial tail (fun word member => members word (by simp [member])) _
        (add_positive_right nonnegative positive)
    · exact induction _ (W8A8IEEE32.add_nonnegative nonnegative (members head (by simp)))
        (fun word member => members word (by simp [member])) ⟨word, later, positive⟩

end MerlinVerify.Rules.W8A8PositiveAdd
