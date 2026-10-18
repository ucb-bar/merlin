/- Checked singleton evidence through integer/index casts. Unknown bounds remain unknown; casts
compute the actual signed/unsigned extension or truncation before decoding the result. -/
import MerlinVerify.Check.IndexRanges
import MerlinVerify.Check.DefinedControl

namespace MerlinVerify.Check.Definedness.IndexCastBounds

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

inductive Kind where
  | signed | unsigned
  deriving DecidableEq

def Kind.name : Kind → String
  | .signed => "arith.index_cast" | .unsigned => "arith.index_castui"

def Kind.bits (kind : Kind) {w : Nat} (target : Nat) (x : BitVec w) : BitVec target :=
  match kind with
  | .signed => if target ≤ w then x.setWidth target else x.signExtend target
  | .unsigned => x.setWidth target

def kind? (name : String) : Option Kind :=
  if name = "arith.index_cast" then some .signed else
  if name = "arith.index_castui" then some .unsigned else none

def width? (ib : Nat) : Ty → Option Nat
  | .index => some ib | .int w => some w | _ => none

def castPair (source target : Ty) : Bool :=
  match source, target with
  | .index, .int _ | .int _, .index => true
  | _, _ => false

def evidence (kind : Kind) (source target : Nat) (fact : IndexRanges.Fact) : Option Interval :=
  match fact.info with
  | some r =>
    if r.upper = r.lower + 1 then
      let result := (kind.bits target (BitVec.ofInt source r.lower)).toInt
      if 0 ≤ result then some ⟨result.toNat, result.toNat + 1⟩ else none
    else none
  | none => none

theorem singleton_bits {ib w : Nat} {f : IndexRanges.Fact} {r : Interval} {x : BitVec w}
    (hf : Refined.Fact.Holds model ib (.scalar (.some (.bits w x))) f)
    (hi : f.info = some r) (hs : r.upper = r.lower + 1) :
    x = BitVec.ofInt w r.lower := by
  have hr : r.Holds (.scalar (.some (.bits w x))) := by simpa [model, hi] using hf.2
  obtain ⟨z, hz, hlo, hhi⟩ := hr
  have hz' : x.toInt = z := by simpa [value?, Sem.Ops.Tensor.indexVal?] using hz
  have hx : x.toInt = (r.lower : Int) := by omega
  rw [← hx, BitVec.ofInt_toInt]

theorem evidence_sound {ib w : Nat} (kind : Kind) (target : Nat) (fact : IndexRanges.Fact) (x : BitVec w)
    (hx : Refined.Fact.Holds model ib (.scalar (.some (.bits w x))) fact) :
    model.holds (evidence kind w target fact) (.scalar (.some (.bits target (kind.bits target x)))) := by
  unfold evidence
  split
  · next r hi =>
    split
    · next hs =>
      have heq := singleton_bits hx hi hs
      subst x
      dsimp only
      split
      · next hn =>
        simp only [BitVec.ofInt_natCast] at hn
        change ∃ z, value? (.scalar (.some (.bits target _))) = some z ∧ _ ∧ _
        refine ⟨_, rfl, ?_, ?_⟩ <;> dsimp <;> omega
      · trivial
    · trivial
  · trivial

theorem kind?_name {name : String} {kind : Kind} (h : kind? name = some kind) : name = kind.name := by
  unfold kind? at h
  split at h
  · cases h; assumption
  · split at h
    · cases h; assumption
    · cases h

/-- Zero extension from a narrower integer has a nonnegative range for every actual input
bitvector. This is useful even when the input is an arbitrary initialized function argument. -/
def resultEvidence (kind : Kind) (source target : Nat) (allowUnsigned : Bool)
    (fact : IndexRanges.Fact) : Option Interval :=
  if allowUnsigned = true ∧ kind = .unsigned ∧ source < target then some ⟨0, 2 ^ source⟩
  else evidence kind source target fact

theorem resultEvidence_sound {ib w : Nat} (kind : Kind) (target : Nat) (allowUnsigned : Bool)
    (fact : IndexRanges.Fact) (x : BitVec w)
    (hx : Refined.Fact.Holds model ib (.scalar (.some (.bits w x))) fact) :
    model.holds (resultEvidence kind w target allowUnsigned fact)
      (.scalar (.some (.bits target (kind.bits target x)))) := by
  unfold resultEvidence
  split
  · next h =>
    obtain ⟨_, hk, hw⟩ := h
    subst kind
    have hz : (x.setWidth target).toInt = (x.toNat : Int) := by
      rw [← BitVec.setWidth'_eq (Nat.le_of_lt hw), BitVec.toInt_setWidth'_of_lt hw]
    change ∃ z, value? (.scalar (.some (.bits target (x.setWidth target)))) = some z ∧ _ ∧ _
    refine ⟨x.toNat, ?_, ?_, ?_⟩
    · simpa [value?, Sem.Ops.Tensor.indexVal?] using congrArg some hz
    · simp
    · exact_mod_cast x.isLt
  · exact evidence_sound kind target fact x hx

def unsignedIntoIndex : Ty → Ty → Bool
  | .int _, .index => true
  | _, _ => false

def rule (ib : Nat) (_ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) :
    Option (List IndexRanges.Fact) :=
  match fs, op.results, kind? op.name with
  | [fact], [r], some kind =>
    match fact.base.ty, width? ib r.ty with
    | .scalar source, some target =>
      match width? ib source with
      | some input =>
        if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧
            fact.base.initialized = true ∧ castPair source r.ty = true ∧
            op.regions = [] ∧ hasUnit op "nonNeg" = false then
          some [⟨⟨.scalar r.ty, true⟩, resultEvidence kind input target (unsignedIntoIndex source r.ty) fact⟩]
        else none
      | none => none
    | _, _ => none
  | _, _, _ => none

theorem width?_intWidth {ib w : Nat} {ty : Ty} (h : width? ib ty = some w) :
    intWidth? ib ty = some w := by
  cases ty <;> simp_all [width?, intWidth?]

theorem width?_bitWidth {ib w : Nat} {ty : Ty} (h : width? ib ty = some w) :
    ty.bitWidth? ib = some w := by
  cases ty <;> simp_all [width?, Ty.bitWidth?]

theorem width?_typed {ib w : Nat} {ty : Ty} (h : width? ib ty = some w) (x : BitVec w) :
    RVal.HasTy ib (.scalar (.some (.bits w x))) (.scalar ty) := by
  cases ty <;> simp_all [width?, RVal.HasTy, Val.HasTy, Scalar.conforms]

theorem rule_sound (ib : Nat) : ∀ Γ op fs rs,
    rule ib Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args hib _ ha
  unfold rule at h
  split at h
  · next fact r kind hr hk =>
    split at h
    · next source target hsource htarget =>
      split at h
      · next input hinput =>
        split at h
        · next hc =>
          obtain ⟨hm, hd, hi, _, hregs, hnneg⟩ := hc
          cases h
          obtain _ | ⟨hx, hnil⟩ := ha
          cases hnil
          have hbase : fact.base = ⟨.scalar source, true⟩ := by
            calc fact.base = ⟨fact.base.ty, fact.base.initialized⟩ := rfl
                 _ = _ := by rw [hsource, hi]
          have hxbase := hx.1
          rw [hbase] at hxbase
          obtain ⟨x, rfl⟩ := initialized_integer (width?_intWidth hinput) hxbase
          refine ⟨[.scalar (.some (.bits target (kind.bits target x)))], ?_,
            .cons ⟨⟨width?_typed htarget _, by simp [RVal.Defined]⟩,
              resultEvidence_sound kind target _ fact x hx⟩ .nil⟩
          have hn := kind?_name hk
          have hresult : Arith.resultElt op.blind = some r.ty := by
            cases ht : r.ty <;> simp_all [width?, Arith.resultElt, Op.blind_results]
          have hw : Arith.widthOf ctx.indexBits op.blind = .ok target := by
            simp [Arith.widthOf, hresult, hib, width?_bitWidth htarget, Outcome.ofOption]
          have hn' : hasUnit op.blind "nonNeg" = false := by simpa [hasUnit] using hnneg
          rw [hregs, evalRegions, opSem_modelled hm]
          simp only [dispatch, Op.blind_name, hd]
          cases kind <;> simp only [Kind.name] at hn
          all_goals
            simp only [Arith.sem, Op.blind_name, hn, hw, Outcome.bindOp_eq, Outcome.bind_ok, Arith.unOp, Arith.pureN, hresult, hn']
            rw [show [.scalar (.some (.bits input x))] =
              ([.some (.bits input x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
            simp [Arith.intUn, Arith.bv, Kind.bits]
        · cases h
      · cases h
    · cases h
  · cases h

def book (ib : Nat) : Refined.Book model opSem ib where
  rule := rule ib
  sound := rule_sound ib

end MerlinVerify.Check.Definedness.IndexCastBounds
