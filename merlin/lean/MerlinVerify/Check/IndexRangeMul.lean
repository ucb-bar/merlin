/- Nonnegative multiplication bounds use actual scalar widths. An interval is retained
only when every possible product fits the signed result range; modular singletons remain
available separately. No flag or sampled product establishes this condition. -/
import MerlinVerify.Check.IndexArithmetic

namespace MerlinVerify.Check.Definedness.IndexRangeMul

open MerlinVerify Sem.Ops IndexRanges

def evidence (width : Nat) (left right : Option Interval) : Option Interval :=
  match left, right with
  | some a, some b =>
    if 0 < width ∧ a.lower < a.upper ∧ b.lower < b.upper ∧
        (a.upper - 1) * (b.upper - 1) < 2 ^ (width - 1) then
      some ⟨a.lower * b.lower, (a.upper - 1) * (b.upper - 1) + 1⟩
    else none
  | _, _ => none

theorem evidence_sound {width : Nat} (left right : Option Interval) (x y : BitVec width)
    (hx : model.holds left (.scalar (.some (.bits width x))))
    (hy : model.holds right (.scalar (.some (.bits width y)))) :
    model.holds (evidence width left right) (.scalar (.some (.bits width (x * y)))) := by
  cases left with
  | none => trivial
  | some a =>
    cases right with
    | none => trivial
    | some b =>
      unfold evidence
      dsimp only
      split
      · next checked =>
        obtain ⟨_, _, _, fits⟩ := checked
        change a.Holds (.scalar (.some (.bits width x))) at hx
        change b.Holds (.scalar (.some (.bits width y))) at hy
        obtain ⟨u, decodedU, lowerU, upperU⟩ := hx
        obtain ⟨v, decodedV, lowerV, upperV⟩ := hy
        have xInt : x.toInt = u := by simpa [value?, Sem.Ops.Tensor.indexVal?] using decodedU
        have yInt : y.toInt = v := by simpa [value?, Sem.Ops.Tensor.indexVal?] using decodedV
        have uNonnegative : 0 ≤ u := by omega
        have vNonnegative : 0 ≤ v := by omega
        have uNat : (u.toNat : Int) = u := Int.natCast_toNat_eq_self.mpr uNonnegative
        have vNat : (v.toNat : Int) = v := Int.natCast_toNat_eq_self.mpr vNonnegative
        have lowU : a.lower ≤ u.toNat := by omega
        have lowV : b.lower ≤ v.toNat := by omega
        have highU : u.toNat ≤ a.upper - 1 := by omega
        have highV : v.toNat ≤ b.upper - 1 := by omega
        have low := Nat.mul_le_mul lowU lowV
        have high := Nat.mul_le_mul highU highV
        have product : x.toInt * y.toInt = ((u.toNat * v.toNat : Nat) : Int) := by
          simp only [Int.natCast_mul, uNat, vNat, xInt, yInt]
        have lower : 0 ≤ x.toInt * y.toInt := by rw [product]; omega
        have upper : x.toInt * y.toInt < (2 : Int) ^ (width - 1) := by
          have bounded : u.toNat * v.toNat < 2 ^ (width - 1) := Nat.lt_of_le_of_lt high fits
          rw [product]
          exact_mod_cast bounded
        have safe : ¬ BitVec.smulOverflow x y := by
          have power : (0 : Int) ≤ 2 ^ (width - 1) := Int.pow_nonneg (by decide)
          simp only [BitVec.smulOverflow, Bool.or_eq_true, decide_eq_true_eq, not_or, Int.not_le, Int.not_lt]
          constructor <;> omega
        have actual := BitVec.toInt_mul_of_not_smulOverflow safe
        change ∃ z, value? (.scalar (.some (.bits width (x * y)))) = some z ∧ _ ∧ _
        refine ⟨x.toInt * y.toInt, ?_, ?_, ?_⟩
        · simpa [value?, Sem.Ops.Tensor.indexVal?] using congrArg some actual
        · dsimp only
          rw [product]
          exact_mod_cast low
        · dsimp only
          rw [product]
          exact_mod_cast Nat.lt_succ_of_le high
      · trivial

#print axioms evidence_sound

end MerlinVerify.Check.Definedness.IndexRangeMul
