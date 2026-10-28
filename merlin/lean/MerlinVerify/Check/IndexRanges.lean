/- Nonnegative index intervals are value properties. Constants are decoded at the claim's actual
index width, including signed interpretation after wraparound; annotations never supply bounds. -/
import MerlinVerify.Check.DefinedRefinements
import MerlinVerify.Check.DefinedRules

namespace MerlinVerify.Check.Definedness.IndexRanges

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

structure Interval where
  lower : Nat
  upper : Nat
  deriving DecidableEq

def value? : RVal → Option Int
  | .scalar v => Sem.Ops.Tensor.indexVal? v
  | _ => none

def Interval.Holds (r : Interval) (v : RVal) : Prop :=
  ∃ z, value? v = some z ∧ (r.lower : Int) ≤ z ∧ z < (r.upper : Int)

def model : Refined.Model (Option Interval) where
  holds info v := match info with
    | none => True
    | some r => r.Holds v
  empty := none
  empty_sound _ := trivial

abbrev Fact := Refined.Fact (Option Interval)
abbrev FEnv := Refined.FEnv (Option Interval)

def constantRule (ib : Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.constant" ∧
      op.regions = [] ∧ fs.isEmpty = true then
    match op.results, op.attr? "value" with
    | [r], some (.int n .index) =>
      let actual := (BitVec.ofInt ib n).toInt
      if r.ty = .index ∧ 0 ≤ actual then
        some [⟨⟨.scalar .index, true⟩, some ⟨actual.toNat, actual.toNat + 1⟩⟩]
      else none
    | _, _ => none
  else none

theorem constantRule_sound (ib : Nat) : ∀ Γ op fs rs,
    constantRule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args hib _ ha
  unfold constantRule at h
  split at h
  · next hc =>
    obtain ⟨hm, hd, hn, hregs, hfs⟩ := hc
    have hnil : fs = [] := List.isEmpty_iff.1 hfs
    subst fs
    cases ha
    split at h
    · next r n hr hv =>
      dsimp only at h
      split at h
      · next hc =>
        obtain ⟨_, hnonneg⟩ := hc
        cases h
        refine ⟨[.scalar (.some (.bits ib (BitVec.ofInt ib n)))], ?_, .cons ⟨?_, ?_⟩ .nil⟩
        · rw [hregs, evalRegions, opSem_modelled hm, ← hib]
          simp only [dispatch, Op.blind_name, hd]
          simp [Arith.sem, hn, hr, hv, scalarOfAttr]
        · exact ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩
        · change ∃ z, value? (.scalar (.some (.bits ib (BitVec.ofInt ib n)))) = some z ∧ _ ∧ _
          refine ⟨(BitVec.ofInt ib n).toInt, rfl, ?_, ?_⟩ <;> dsimp <;> omega
      · cases h
    · cases h
  · cases h

def constantBook (ib : Nat) : Refined.Book model opSem ib where
  rule := constantRule ib
  sound := constantRule_sound ib

end MerlinVerify.Check.Definedness.IndexRanges
