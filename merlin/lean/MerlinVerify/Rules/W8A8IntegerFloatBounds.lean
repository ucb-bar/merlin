/- Small integer intervals are converted using their exact signed width and the
executable rounded f32 conversion. Range tables are checked, not native oracles. -/
import MerlinVerify.Rules.W8A8IEEEBounds
import MerlinVerify.Rules.W8A8ArithmeticModel
import MerlinVerify.Check.IndexRanges

namespace MerlinVerify.Rules.W8A8IntegerFloatBounds

open MerlinVerify Check Check.Definedness

def cap (width : Nat) (range : IndexRanges.Interval) : Nat :=
  ((List.range range.upper).map fun n => W8A8ByteFloat.exponent
    (W8A8ArithmeticModel.signedInteger32 (BitVec.ofNat width n))).foldl max 0

def check (width : Nat) (range : IndexRanges.Interval) (upper : Nat) : Bool :=
  range.upper ≤ 4096 && (List.range range.upper).all fun n =>
    if range.lower ≤ n then decide (W8A8IEEEBounds.ExponentLE upper
      (W8A8ArithmeticModel.signedInteger32 (BitVec.ofNat width n))) else true

theorem check_sound {width upper : Nat} {range : IndexRanges.Interval}
    (checked : check width range upper = true) (bits : BitVec width)
    (bounds : (range.lower : Int) ≤ bits.toInt ∧ bits.toInt < (range.upper : Int)) :
    W8A8IEEEBounds.ExponentLE upper (W8A8ArithmeticModel.signedInteger32 bits) := by
  have nonnegative : 0 ≤ bits.toInt := by omega
  have member : bits.toInt.toNat ∈ List.range range.upper := List.mem_range.2 (by omega)
  simp only [check, Bool.and_eq_true] at checked
  have actual := List.all_eq_true.1 checked.2 bits.toInt.toNat member
  have lower : range.lower ≤ bits.toInt.toNat := by omega
  rw [ite_eq_left lower] at actual
  have encoded : BitVec.ofNat width bits.toInt.toNat = bits := by
    rw [← BitVec.ofInt_natCast, Int.toNat_of_nonneg nonnegative]
    exact BitVec.ofInt_toInt
  rw [encoded] at actual
  exact of_decide_eq_true actual

#print axioms check_sound

end MerlinVerify.Rules.W8A8IntegerFloatBounds
