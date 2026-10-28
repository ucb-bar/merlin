/- An unsigned minimum is bounded by either operand. Only an actual nonnegative bound below
this claim's signed index limit can justify subsequent tensor indexing. -/
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.DefinedControl

namespace MerlinVerify.Check.Definedness.IndexClampBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

def bits {ib : Nat} (x y : BitVec ib) : BitVec ib := if x.ult y then x else y

theorem bits_toNat {ib : Nat} (x y : BitVec ib) : (bits x y).toNat = min x.toNat y.toNat := by
  by_cases h : x.toNat < y.toNat
  · simp [bits, BitVec.ult_eq_decide, h, Nat.min_eq_left (Nat.le_of_lt h)]
  · simp [bits, BitVec.ult_eq_decide, h, Nat.min_eq_right (by omega : y.toNat ≤ x.toNat)]

def upper? (ib : Nat) : Option Interval → Option Nat
  | some r => if 0 < ib ∧ r.upper ≤ 2 ^ (ib - 1) then some r.upper else none
  | none => none

theorem upper_sound {ib upper : Nat} {info : Option Interval} {x : BitVec ib}
    (h : upper? ib info = some upper) (hx : model.holds info (.scalar (.some (.bits ib x)))) :
    0 < ib ∧ x.toNat < upper ∧ upper ≤ 2 ^ (ib - 1) := by
  cases info with
  | none => cases h
  | some r =>
    by_cases hc : 0 < ib ∧ r.upper ≤ 2 ^ (ib - 1)
    · simp [upper?, hc] at h
      subst upper
      obtain ⟨z,hz,hlo,hhi⟩ := hx
      have heq : x.toInt = z := by simpa [value?, Sem.Ops.Tensor.indexVal?] using hz
      have hn : 0 ≤ x.toInt := by omega
      have hnat := BitVec.toInt_eq_toNat_of_lt (BitVec.toInt_pos_iff.mp hn)
      refine ⟨hc.1, ?_, hc.2⟩
      have : (x.toNat : Int) < r.upper := by omega
      exact_mod_cast this
    · simp [upper?, hc] at h

def evidence (ib : Nat) (left right : Option Interval) : Option Interval :=
  match upper? ib left, upper? ib right with
  | some a, some b => some ⟨0,min a b⟩
  | some a, none => some ⟨0,a⟩
  | none, some b => some ⟨0,b⟩
  | none, none => none

theorem bounded_bits {ib upper : Nat} (x y : BitVec ib) (hi : 0 < ib)
    (hu : (bits x y).toNat < upper) (hlimit : upper ≤ 2 ^ (ib - 1)) :
    (Interval.mk 0 upper).Holds (.scalar (.some (.bits ib (bits x y)))) := by
  have hpow : 2 ^ ib = 2 * 2 ^ (ib - 1) := by
    have hw : ib = (ib - 1) + 1 := by omega
    conv_lhs => rw [hw]
    rw [Nat.pow_succ, Nat.mul_comm]
  have heq := BitVec.toInt_eq_toNat_of_lt (x := bits x y) (by omega : 2 * (bits x y).toNat < 2 ^ ib)
  refine ⟨(bits x y).toInt, rfl, ?_, ?_⟩
  · dsimp; omega
  · dsimp
    rw [heq]
    exact_mod_cast hu

theorem evidence_sound {ib : Nat} (left right : Option Interval) (x y : BitVec ib)
    (hx : model.holds left (.scalar (.some (.bits ib x))))
    (hy : model.holds right (.scalar (.some (.bits ib y)))) :
    model.holds (evidence ib left right) (.scalar (.some (.bits ib (bits x y)))) := by
  unfold evidence
  cases hl : upper? ib left with
  | none =>
    cases hr : upper? ib right with
    | none => trivial
    | some upper =>
      obtain ⟨hi,hyu,hlimit⟩ := upper_sound hr hy
      apply bounded_bits x y hi _ hlimit
      rw [bits_toNat]
      omega
  | some upper =>
    obtain ⟨hi,hxu,hlimit⟩ := upper_sound hl hx
    cases hr : upper? ib right with
    | none =>
      apply bounded_bits x y hi _ hlimit
      rw [bits_toNat]
      omega
    | some other =>
      obtain ⟨_,hyu,otherLimit⟩ := upper_sound hr hy
      apply bounded_bits x y hi
      · rw [bits_toNat]; omega
      · omega

def rule (ib : Nat) (_ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) : Option (List IndexRanges.Fact) :=
  match fs, op.results with
  | [left,right], [r] =>
    if op.name = "arith.minui" ∧ dialectOf op.name = "arith" ∧
        unmodelledAttr? op.blind = none ∧ op.regions = [] ∧ op.successors = [] ∧
        left.base = ⟨.scalar .index,true⟩ ∧ right.base = ⟨.scalar .index,true⟩ ∧ r.ty = .index ∧
        overflowFlags op = [] ∧ Arith.fastmathKey op = "0" ∧ hasUnit op "nonNeg" = false ∧
        hasUnit op "isExact" = false ∧ hasUnit op "disjoint" = false then
      some [⟨⟨.scalar .index,true⟩, evidence ib left.info right.info⟩]
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
  · next _ _ _ left right r hr =>
    split at h
    · next hc =>
      obtain ⟨hn,hd,hm,hregs,_,hl,hright,hty,_⟩ := hc
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
      refine ⟨[.scalar (.some (.bits ib (bits x y)))], ?_,
        .cons ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
          evidence_sound left.info right.info x y hx.2 hy.2⟩ .nil⟩
      have hresult : Arith.resultElt op.blind = some .index := by
        simp [Arith.resultElt, Op.blind_results, hr, hty]
      rw [hregs, evalRegions, opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult]
      rw [show [.scalar (.some (.bits ib x)), .scalar (.some (.bits ib y))] =
        ([.some (.bits ib x),.some (.bits ib y)] : List Val).map RVal.scalar from rfl,
        liftEager_scalars]
      simp [Arith.intBin, Arith.bv, bits]
    · cases h
  · cases h

def book (ib : Nat) : Refined.Book model opSem ib where
  rule := rule ib
  sound := rule_sound ib

end MerlinVerify.Check.Definedness.IndexClampBounds
