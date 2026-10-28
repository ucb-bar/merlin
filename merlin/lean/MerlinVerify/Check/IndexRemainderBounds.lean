/- Unsigned remainder is total for an actual positive divisor. Nonnegative divisor intervals
below the signed index limit provide nonnegative bounds for subsequent tensor reads. -/
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.DefinedControl

namespace MerlinVerify.Check.Definedness.IndexRemainderBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

def divisor? (ib : Nat) : Option Interval → Option Interval
  | some r => if 0 < ib ∧ 0 < r.lower ∧ r.upper ≤ 2 ^ (ib - 1) then some r else none
  | none => none

theorem divisor_sound {ib : Nat} {info : Option Interval} {r : Interval} {y : BitVec ib}
    (h : divisor? ib info = some r) (hy : model.holds info (.scalar (.some (.bits ib y)))) :
    0 < ib ∧ 0 < y.toNat ∧ y.toNat < r.upper ∧ r.upper ≤ 2 ^ (ib - 1) := by
  cases info with
  | none => cases h
  | some interval =>
    by_cases hc : 0 < ib ∧ 0 < interval.lower ∧ interval.upper ≤ 2 ^ (ib - 1)
    · simp [divisor?, hc] at h
      subst r
      obtain ⟨z,hz,hlo,hhi⟩ := hy
      have heq : y.toInt = z := by simpa [value?, Sem.Ops.Tensor.indexVal?] using hz
      have hn : 0 ≤ y.toInt := by omega
      have hnat := BitVec.toInt_eq_toNat_of_lt (BitVec.toInt_pos_iff.mp hn)
      have hypos : (0 : Int) < y.toNat := by omega
      have hyupper : (y.toNat : Int) < interval.upper := by omega
      exact ⟨hc.1, by exact_mod_cast hypos, by exact_mod_cast hyupper, hc.2.2⟩
    · simp [divisor?, hc] at h

/-- Subtracting one from the exclusive divisor bound recovers the exact bound for a
constant divisor: [d,d+1) yields remainder [0,d). No source numerator bound is needed. -/
theorem remainder_bound {ib : Nat} {info : Option Interval} {r : Interval} (x y : BitVec ib)
    (h : divisor? ib info = some r) (hy : model.holds info (.scalar (.some (.bits ib y)))) :
    y ≠ 0 ∧ (Interval.mk 0 (r.upper - 1)).Holds (.scalar (.some (.bits ib (x % y)))) := by
  obtain ⟨hi,hpositive,hupper,hlimit⟩ := divisor_sound h hy
  have hnz : y ≠ 0 := by intro hz; simp [hz] at hpositive
  have hrem : (x % y).toNat < y.toNat := by
    rw [BitVec.toNat_umod]
    exact Nat.mod_lt _ hpositive
  have hbound : (x % y).toNat < r.upper - 1 := by omega
  have hpow : 2 ^ ib = 2 * 2 ^ (ib - 1) := by
    have hw : ib = (ib - 1) + 1 := by omega
    conv_lhs => rw [hw]
    rw [Nat.pow_succ, Nat.mul_comm]
  have heq := BitVec.toInt_eq_toNat_of_lt (x := x % y) (by omega : 2 * (x % y).toNat < 2 ^ ib)
  refine ⟨hnz,(x % y).toInt,rfl,?_,?_⟩
  · dsimp; omega
  · dsimp
    rw [heq]
    exact_mod_cast hbound

def rule (ib : Nat) (_ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) :
    Option (List IndexRanges.Fact) :=
  match fs, op.results with
  | [left,right], [result] =>
    if op.name = "arith.remui" ∧ dialectOf op.name = "arith" ∧
        unmodelledAttr? op.blind = none ∧ op.regions = [] ∧ op.successors = [] ∧
        left.base = ⟨.scalar .index,true⟩ ∧ right.base = ⟨.scalar .index,true⟩ ∧ result.ty = .index ∧
        overflowFlags op = [] ∧ Arith.fastmathKey op = "0" ∧ hasUnit op "nonNeg" = false ∧
        hasUnit op "isExact" = false ∧ hasUnit op "disjoint" = false then
      match divisor? ib right.info with
      | some r => some [⟨⟨.scalar .index,true⟩,some ⟨0,r.upper - 1⟩⟩]
      | none => none
    else none
  | _, _ => none

theorem rule_sound (ib : Nat) : ∀ Γ op fs rs,
    rule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args _ _ ha
  unfold rule at h
  split at h
  · next _ _ _ left right result hr =>
    split at h
    · next hc =>
      obtain ⟨hn,hd,hm,hregs,_,hl,hright,hty,_⟩ := hc
      cases hb : divisor? ib right.info with
      | none => simp [hb] at h
      | some r =>
        simp only [hb, Option.some.injEq] at h
        cases h
        obtain _ | ⟨hx,htail⟩ := ha
        obtain _ | ⟨hy,hnil⟩ := htail
        cases hnil
        have hxl := hx.1
        have hyr := hy.1
        rw [hl] at hxl
        rw [hright] at hyr
        obtain ⟨x,rfl⟩ := initialized_integer (ib := ib) (w := ib) (ty := .index) rfl hxl
        obtain ⟨y,rfl⟩ := initialized_integer (ib := ib) (w := ib) (ty := .index) rfl hyr
        obtain ⟨hnz,hbound⟩ := remainder_bound x y hb hy.2
        refine ⟨[.scalar (.some (.bits ib (x % y)))], ?_,
          .cons ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
            hbound⟩ .nil⟩
        have hresult : Arith.resultElt op.blind = some .index := by
          simp [Arith.resultElt, Op.blind_results, hr, hty]
        rw [hregs, evalRegions, opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult]
        rw [show [.scalar (.some (.bits ib x)), .scalar (.some (.bits ib y))] =
          ([.some (.bits ib x),.some (.bits ib y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp only [Arith.intDiv]
        rw [ite_eq_right hnz]
        simp [Arith.bv]
    · cases h
  · cases h

def book (ib : Nat) : Refined.Book model opSem ib where
  rule := rule ib
  sound := rule_sound ib

end MerlinVerify.Check.Definedness.IndexRemainderBounds
