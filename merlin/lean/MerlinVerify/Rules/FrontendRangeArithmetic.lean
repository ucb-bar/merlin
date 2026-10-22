/- Bounds transfers use actual modular integer semantics. A range is retained only
when its entire signed result range is representable. No floating laws are used. -/
import MerlinVerify.Rules.FrontendRangeFrame
import MerlinVerify.Rules.FrontendDefinedIndex

namespace MerlinVerify.Rules.FrontendRangeArithmetic
open MerlinVerify MerlinVerify.Check Sem.Ops
open Definedness.IndexRanges (Interval)
open FrontendTensorRanges

namespace Mul

def evidence (ib : Nat) (left right : Option Interval) : Option Interval :=
  match left, right with
  | some a, some b =>
    if 0 < ib ∧ a.lower < a.upper ∧ b.lower < b.upper ∧
        (a.upper - 1) * (b.upper - 1) + 1 ≤ 2 ^ (ib - 1) then
      some ⟨a.lower * b.lower, (a.upper - 1) * (b.upper - 1) + 1⟩
    else none
  | _, _ => none

theorem sound {ib : Nat} (left right : Option Interval) (x y : BitVec ib)
    (hx : Definedness.IndexRanges.model.holds left (.scalar (.some (.bits ib x))))
    (hy : Definedness.IndexRanges.model.holds right (.scalar (.some (.bits ib y)))) :
    Definedness.IndexRanges.model.holds (evidence ib left right) (.scalar (.some (.bits ib (x * y)))) := by
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
        obtain ⟨_, ha, hb, hlimit⟩ := hc
        obtain ⟨u, hu, hul, huh⟩ := hx
        obtain ⟨v, hv, hvl, hvh⟩ := hy
        have hux : x.toInt = u := by simpa [Definedness.IndexRanges.value?, Tensor.indexVal?] using hu
        have hvy : y.toInt = v := by simpa [Definedness.IndexRanges.value?, Tensor.indexVal?] using hv
        have hun : 0 ≤ u := by omega
        have hvn : 0 ≤ v := by omega
        have hulo : a.lower ≤ u.toNat := by omega
        have hvlo : b.lower ≤ v.toNat := by omega
        have huhi : u.toNat ≤ a.upper - 1 := by omega
        have hvhi : v.toNat ≤ b.upper - 1 := by omega
        have hlo := Nat.mul_le_mul hulo hvlo
        have hhi := Nat.mul_le_mul huhi hvhi
        have hpos : 0 ≤ u * v := Int.mul_nonneg hun hvn
        have hmul : u * v < (2 : Int) ^ (ib - 1) := by
          have h : u.toNat * v.toNat < 2 ^ (ib - 1) := by omega
          have hc' : ((u.toNat * v.toNat : Nat) : Int) < (2 : Int) ^ (ib - 1) := by exact_mod_cast h
          simpa [Int.toNat_of_nonneg hun, Int.toNat_of_nonneg hvn] using hc'
        have hsafe : ¬ BitVec.smulOverflow x y := by
          have hp : 0 ≤ (2 : Int) ^ (ib - 1) := Int.pow_nonneg (by omega)
          simp only [BitVec.smulOverflow, Bool.or_eq_true, decide_eq_true_eq, not_or, Int.not_le, Int.not_lt]
          rw [hux, hvy]
          constructor <;> omega
        have heq := BitVec.toInt_mul_of_not_smulOverflow hsafe
        change ∃ z, Definedness.IndexRanges.value? (.scalar (.some (.bits ib (x * y)))) = some z ∧ _ ∧ _
        refine ⟨u * v, ?_, ?_, ?_⟩
        · simpa [Definedness.IndexRanges.value?, Tensor.indexVal?, hux, hvy] using congrArg some heq
        · have h : ((a.lower * b.lower : Nat) : Int) ≤ u * v := by
            simpa [Int.toNat_of_nonneg hun, Int.toNat_of_nonneg hvn] using (Int.ofNat_le.mpr hlo)
          simpa using h
        · have h : u * v < (((a.upper - 1) * (b.upper - 1) + 1 : Nat) : Int) := by
            have h := Int.ofNat_le.mpr hhi
            simp only [Int.natCast_mul, Int.toNat_of_nonneg hun, Int.toNat_of_nonneg hvn] at h
            omega
          exact h
      · trivial
end Mul

def evidence (ib : Nat) (kind : Definedness.IndexArithmetic.Kind) (left right : Fact) : Option Interval :=
  if kind = .mul then Mul.evidence ib left.info right.info
  else Definedness.IndexArithmetic.evidence ib kind left right

theorem evidence_sound {ib : Nat} (kind : Definedness.IndexArithmetic.Kind) (left right : Fact) (x y : BitVec ib)
    (hx : Definedness.Refined.Fact.Holds model ib (.scalar (.some (.bits ib x))) left)
    (hy : Definedness.Refined.Fact.Holds model ib (.scalar (.some (.bits ib y))) right) :
    model.holds (evidence ib kind left right) (.scalar (.some (.bits ib (kind.bits x y)))) := by
  have hx' : Definedness.Refined.Fact.Holds Definedness.IndexRanges.model ib (.scalar (.some (.bits ib x))) left := by
    exact ⟨hx.1, by cases hi : left.info <;> simpa [model, Holds, Definedness.IndexRanges.model, hi] using hx.2⟩
  have hy' : Definedness.Refined.Fact.Holds Definedness.IndexRanges.model ib (.scalar (.some (.bits ib y))) right := by
    exact ⟨hy.1, by cases hi : right.info <;> simpa [model, Holds, Definedness.IndexRanges.model, hi] using hy.2⟩
  have h : Definedness.IndexRanges.model.holds (evidence ib kind left right)
      (.scalar (.some (.bits ib (kind.bits x y)))) := by
    unfold evidence
    split
    · next hk => subst kind; exact Mul.sound _ _ x y hx'.2 hy'.2
    · exact Definedness.IndexArithmetic.evidence_sound kind left right x y hx' hy'
  exact (from_scalar (ib := ib) (f := ⟨⟨.scalar (.int ib), true⟩, evidence ib kind left right⟩)
    ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩, h⟩).2

/-- Fixed-width initialized integer arithmetic, with the same flag checks as the old index rule. -/
def rule (ib : Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  match fs, op.results, Definedness.IndexArithmetic.kind? op.name with
  | [left, right], [r], some kind =>
    if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧
        left.base = ⟨.scalar (.int ib), true⟩ ∧ right.base = ⟨.scalar (.int ib), true⟩ ∧
        r.ty = .int ib ∧ op.regions = [] ∧ overflowFlags op = [] ∧ Arith.fastmathKey op = "0" then
      some [⟨⟨.scalar (.int ib), true⟩, evidence ib kind left right⟩]
    else none
  | _, _, _ => none

def book (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := rule ib
  sound Γ op fs rs h ctx call env args _ _ ha := by
    unfold rule at h
    split at h
    · next _ _ _ left right r kind hr hk =>
      split at h
      · next hc =>
        obtain ⟨hm, hd, hl, hright, hty, hregs, hflags, _⟩ := hc
        cases h
        obtain _ | ⟨hx, hrest⟩ := ha
        obtain _ | ⟨hy, hnil⟩ := hrest
        cases hnil
        have hxl := hx.1
        have hyr := hy.1
        rw [hl] at hxl
        rw [hright] at hyr
        obtain ⟨x, rfl⟩ := Definedness.initialized_int hxl
        obtain ⟨y, rfl⟩ := Definedness.initialized_int hyr
        refine ⟨[.scalar (.some (.bits ib (kind.bits x y)))], ?_,
          .cons ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
            evidence_sound kind left right x y hx hy⟩ .nil⟩
        have hn := Definedness.IndexArithmetic.kind?_name hk
        have hresult : Arith.resultElt op.blind = some (.int ib) := by
          simp [Arith.resultElt, Op.blind_results, hr, hty]
        have hf : overflowFlags op.blind = [] := by simpa [overflowFlags, attrEnum?] using hflags
        rw [hregs, evalRegions, opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        cases kind <;> simp only [Definedness.IndexArithmetic.Kind.name] at hn
        all_goals
          simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult]
          rw [show [.scalar (.some (.bits ib x)), .scalar (.some (.bits ib y))] =
            ([.some (.bits ib x), .some (.bits ib y)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
          simp [Arith.intBin, Arith.arithWithFlags, hf, Arith.bv, Definedness.IndexArithmetic.Kind.bits]
      · cases h
    · cases h

#print axioms book
end MerlinVerify.Rules.FrontendRangeArithmetic
