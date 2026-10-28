/- Nonnegative interval addition is valid only when the entire exclusive result range fits
below the actual signed index limit. This transfer does not assume that index values are bounded. -/
import MerlinVerify.Check.IndexRanges

namespace MerlinVerify.Check.Definedness.IndexRangeAdd

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

def evidence (ib : Nat) (left right : Option Interval) : Option Interval :=
  match left, right with
  | some a, some b =>
    if 0 < ib ∧ a.lower < a.upper ∧ b.lower < b.upper ∧
        a.upper + b.upper - 1 ≤ 2 ^ (ib - 1) then
      some ⟨a.lower + b.lower, a.upper + b.upper - 1⟩
    else none
  | _, _ => none

theorem evidence_sound {ib : Nat} (left right : Option Interval) (x y : BitVec ib)
    (hx : model.holds left (.scalar (.some (.bits ib x))))
    (hy : model.holds right (.scalar (.some (.bits ib y)))) :
    model.holds (evidence ib left right) (.scalar (.some (.bits ib (x + y)))) := by
  cases left with
  | none => trivial
  | some a =>
    cases right with
    | none => trivial
    | some b =>
      unfold evidence
      dsimp only
      split
      · next hc =>
        obtain ⟨_, ha', hb', hlimit⟩ := hc
        change a.Holds (.scalar (.some (.bits ib x))) at hx
        change b.Holds (.scalar (.some (.bits ib y))) at hy
        obtain ⟨u, hu, hul, huh⟩ := hx
        obtain ⟨v, hv, hvl, hvh⟩ := hy
        have hux : x.toInt = u := by simpa [value?, Sem.Ops.Tensor.indexVal?] using hu
        have hvy : y.toInt = v := by simpa [value?, Sem.Ops.Tensor.indexVal?] using hv
        have hsize : 1 ≤ a.upper + b.upper := by omega
        have hcast : ((a.upper + b.upper - 1 : Nat) : Int) = (a.upper : Int) + b.upper - 1 := by omega
        have hhi : x.toInt + y.toInt < (2 : Int) ^ (ib - 1) := by
          have hlim : ((a.upper + b.upper - 1 : Nat) : Int) ≤ (2 : Int) ^ (ib - 1) := by
            exact_mod_cast hlimit
          omega
        have hlo : 0 ≤ x.toInt + y.toInt := by omega
        have hsafe : ¬ BitVec.saddOverflow x y := by
          have hpow : 0 ≤ (2 : Int) ^ (ib - 1) := Int.pow_nonneg (by omega)
          simp only [BitVec.saddOverflow, Bool.or_eq_true, decide_eq_true_eq, not_or, Int.not_le, Int.not_lt]
          constructor <;> omega
        have heq := BitVec.toInt_add_of_not_saddOverflow hsafe
        change ∃ z, value? (.scalar (.some (.bits ib (x + y)))) = some z ∧ _ ∧ _
        refine ⟨x.toInt + y.toInt, ?_, ?_, ?_⟩
        · simpa [value?, Sem.Ops.Tensor.indexVal?] using congrArg some heq
        · dsimp; omega
        · dsimp; omega
      · trivial

end MerlinVerify.Check.Definedness.IndexRangeAdd
