/- Unflagged index arithmetic is total on initialized values. Singleton propagation computes
actual modular bits; addition also propagates intervals only when their entire range cannot wrap. -/
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.IndexRangeAdd
import MerlinVerify.Check.DefinedControl

namespace MerlinVerify.Check.Definedness.IndexArithmetic

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

inductive Kind where
  | add | sub | mul
  deriving DecidableEq

def Kind.name : Kind → String
  | .add => "arith.addi" | .sub => "arith.subi" | .mul => "arith.muli"

def Kind.bits (kind : Kind) {ib : Nat} (x y : BitVec ib) : BitVec ib :=
  match kind with
  | .add => x + y | .sub => x - y | .mul => x * y

def kind? (name : String) : Option Kind :=
  if name = "arith.addi" then some .add else
  if name = "arith.subi" then some .sub else
  if name = "arith.muli" then some .mul else none

def singletonEvidence (ib : Nat) (kind : Kind) (left right : IndexRanges.Fact) : Option Interval :=
  match left.info, right.info with
  | some a, some b =>
    if a.upper = a.lower + 1 ∧ b.upper = b.lower + 1 then
      let result := (kind.bits (BitVec.ofInt ib a.lower) (BitVec.ofInt ib b.lower)).toInt
      if 0 ≤ result then some ⟨result.toNat, result.toNat + 1⟩ else none
    else none
  | _, _ => none

theorem singleton_bits {ib : Nat} {f : IndexRanges.Fact} {r : Interval} {x : BitVec ib}
    (hf : Refined.Fact.Holds model ib (.scalar (.some (.bits ib x))) f)
    (hi : f.info = some r) (hs : r.upper = r.lower + 1) :
    x = BitVec.ofInt ib r.lower := by
  have hr : r.Holds (.scalar (.some (.bits ib x))) := by simpa [model, hi] using hf.2
  obtain ⟨z, hz, hlo, hhi⟩ := hr
  have hz' : x.toInt = z := by simpa [value?, Sem.Ops.Tensor.indexVal?] using hz
  have hx : x.toInt = (r.lower : Int) := by omega
  rw [← hx, BitVec.ofInt_toInt]

theorem singletonEvidence_sound {ib : Nat} (kind : Kind) (left right : IndexRanges.Fact) (x y : BitVec ib)
    (hx : Refined.Fact.Holds model ib (.scalar (.some (.bits ib x))) left)
    (hy : Refined.Fact.Holds model ib (.scalar (.some (.bits ib y))) right) :
    model.holds (singletonEvidence ib kind left right) (.scalar (.some (.bits ib (kind.bits x y)))) := by
  unfold singletonEvidence
  split
  · next a b ha hb =>
    split
    · next hs =>
      obtain ⟨hsa, hsb⟩ := hs
      have hxa := singleton_bits hx ha hsa
      have hyb := singleton_bits hy hb hsb
      subst x
      subst y
      dsimp only
      split
      · next hn =>
        simp only [BitVec.ofInt_natCast] at hn
        change ∃ z, value? (.scalar (.some (.bits ib _))) = some z ∧ _ ∧ _
        refine ⟨_, rfl, ?_, ?_⟩ <;> dsimp <;> omega
      · trivial
    · trivial
  · trivial

/-- Exact singleton computation retains modular-wrap precision; a non-singleton addition
uses the independently proved no-wrap range transfer. Other uncertain ranges become unknown. -/
def evidence (ib : Nat) (kind : Kind) (left right : IndexRanges.Fact) : Option Interval :=
  (singletonEvidence ib kind left right).orElse fun _ =>
    if kind = .add then IndexRangeAdd.evidence ib left.info right.info else none

theorem evidence_sound {ib : Nat} (kind : Kind) (left right : IndexRanges.Fact) (x y : BitVec ib)
    (hx : Refined.Fact.Holds model ib (.scalar (.some (.bits ib x))) left)
    (hy : Refined.Fact.Holds model ib (.scalar (.some (.bits ib y))) right) :
    model.holds (evidence ib kind left right) (.scalar (.some (.bits ib (kind.bits x y)))) := by
  unfold evidence
  cases h : singletonEvidence ib kind left right with
  | some r =>
    simpa [h] using singletonEvidence_sound kind left right x y hx hy
  | none =>
    simp only [Option.orElse_none]
    split
    · next hk =>
      subst kind
      exact IndexRangeAdd.evidence_sound left.info right.info x y hx.2 hy.2
    · trivial

theorem kind?_name {name : String} {kind : Kind} (h : kind? name = some kind) : name = kind.name := by
  unfold kind? at h
  split at h
  · cases h; assumption
  · split at h
    · cases h; assumption
    · split at h
      · cases h; assumption
      · cases h

/-- A runtime index remains initialized even when its bounds are unknown or wrap negative.
Proved singletons or a no-wrap addition range can contribute new interval evidence. -/
def rule (ib : Nat) (_ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) :
    Option (List IndexRanges.Fact) :=
  match fs, op.results, kind? op.name with
  | [left, right], [r], some kind =>
    if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧
        left.base = ⟨.scalar .index, true⟩ ∧ right.base = ⟨.scalar .index, true⟩ ∧
        r.ty = .index ∧ op.regions = [] ∧ overflowFlags op = [] then
      some [⟨⟨.scalar .index, true⟩, evidence ib kind left right⟩]
    else none
  | _, _, _ => none

theorem rule_sound (ib : Nat) : ∀ Γ op fs rs,
    rule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args _ _ ha
  unfold rule at h
  split at h
  · next _ _ _ left right r kind hr hk =>
    split at h
    · next hc =>
      obtain ⟨hm, hd, hl, hright, hty, hregs, hflags⟩ := hc
      cases h
      obtain _ | ⟨hx, hrest⟩ := ha
      obtain _ | ⟨hy, hnil⟩ := hrest
      cases hnil
      have hxl := hx.1
      have hyr := hy.1
      rw [hl] at hxl
      rw [hright] at hyr
      obtain ⟨x, rfl⟩ := initialized_integer (ib := ib) (w := ib) (ty := .index) rfl hxl
      obtain ⟨y, rfl⟩ := initialized_integer (ib := ib) (w := ib) (ty := .index) rfl hyr
      refine ⟨[.scalar (.some (.bits ib (kind.bits x y)))], ?_,
        .cons ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
          evidence_sound kind left right x y hx hy⟩ .nil⟩
      have hn := kind?_name hk
      have hresult : Arith.resultElt op.blind = some .index := by
        simp [Arith.resultElt, Op.blind_results, hr, hty]
      have hf : overflowFlags op.blind = [] := by simpa [overflowFlags, attrEnum?] using hflags
      rw [hregs, evalRegions, opSem_modelled hm]
      simp only [dispatch, Op.blind_name, hd]
      cases kind <;> simp only [Kind.name] at hn
      all_goals
        simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hresult]
        rw [show [.scalar (.some (.bits ib x)), .scalar (.some (.bits ib y))] =
          ([.some (.bits ib x), .some (.bits ib y)] : List Val).map RVal.scalar from rfl,
          liftEager_scalars]
        simp [Arith.intBin, Arith.arithWithFlags, hf, Arith.bv, Kind.bits]
    · cases h
  · cases h

def book (ib : Nat) : Refined.Book model opSem ib where
  rule := rule ib
  sound := rule_sound ib

end MerlinVerify.Check.Definedness.IndexArithmetic
