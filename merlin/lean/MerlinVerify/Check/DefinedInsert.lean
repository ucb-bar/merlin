/- Tensor writes are total only at actual proved in-bounds indices. An initialized payload
preserves an initialized destination; rank zero additionally proves a complete overwrite. -/
import MerlinVerify.Check.IndexBounds
import MerlinVerify.Rules.InsertExtractKernel

namespace MerlinVerify.Check.Definedness.Insert

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops
open IndexRanges

/-- A rank-zero tensor has exactly its empty coordinate. No positive-rank dirty destination
is promoted to initialized by this rule. -/
theorem updated_fact {ib : Nat} {t : Tensor} {shape : List Nat} {element : Ty} {initialized : Bool}
    {value : Val} {coordinate : Idx}
    (ht : Fact.Holds ib (.tensor t) ⟨.tensor shape element, initialized⟩)
    (hv : Fact.Holds ib (.scalar value) ⟨.scalar element, true⟩)
    (hc : Idx.InBounds shape coordinate) (hfull : initialized = true ∨ shape = []) :
    Fact.Holds ib (.tensor (t.set coordinate value).canonize) ⟨.tensor shape element, true⟩ := by
  obtain ⟨hshape, helt, hget⟩ := ht.1
  refine ⟨⟨hshape, helt, ?_⟩, ?_⟩
  · intro i hi
    simp only [Tensor.canonize, Tensor.canon, Tensor.set, hshape, hi, ↓reduceIte]
    split
    · exact hv.1
    · exact hget i hi
  · intro _ i hi
    change Idx.InBounds (t.set coordinate value).canonize.shape i at hi
    change Idx.InBounds t.shape i at hi
    have his : Idx.InBounds shape i := hshape ▸ hi
    simp only [Tensor.canonize, Tensor.canon, Tensor.set, hi, ↓reduceIte]
    split
    · exact hv.2 rfl
    · next hne =>
      rcases hfull with hinit | hzero
      · exact ht.2 hinit i hi
      · have hiz : i = [] := List.length_eq_zero_iff.1 (by simpa [hzero] using his.1)
        have hcz : coordinate = [] := List.length_eq_zero_iff.1 (by simpa [hzero] using hc.1)
        exact (hne (hiz.trans hcz.symm)).elim

def rule (_ : IndexRanges.FEnv) (op : Op) (fs : List IndexRanges.Fact) :
    Option (List IndexRanges.Fact) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "tensor" ∧ op.name = "tensor.insert" ∧
      op.regions = [] then
    match fs with
    | ⟨⟨.scalar inputElement, true⟩, _⟩ :: ⟨⟨.tensor shape element, initialized⟩, _⟩ :: indices =>
      if inputElement = element ∧ Sem.Ops.Tensor.resultTensorTy op = some (shape, element) ∧
          Bounds.within indices shape = true ∧ (initialized = true ∨ shape = []) then
        some [Refined.emptyFact model ⟨.tensor shape element, true⟩]
      else none
    | _ => none
  else none

theorem rule_sound (ib : Nat) : ∀ Γ op fs rs,
    rule Γ op fs = some rs → ∀ ctx call env args,
    ctx.indexBits = ib → Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args fs →
    ∃ vs, opSem ctx call op args (evalRegions opSem call env op.regions) = .ok vs ∧
      Refined.FactsHold model ib vs rs := by
  intro Γ op fs rs h ctx call env args _ _ ha
  unfold rule at h
  split at h
  · next checks =>
    obtain ⟨hm, _, hn, hregs⟩ := checks
    split at h
    · next inputElement info shape element initialized destInfo indices =>
      split at h
      · next checks =>
        obtain ⟨rfl, _, hbound, hfull⟩ := checks
        cases h
        obtain _ | ⟨hx, hrest⟩ := ha
        obtain _ | ⟨ht, hi⟩ := hrest
        rename_i scalar tensor indexArgs
        cases scalar with
        | tensor _ => cases hx.1.1
        | scalar value =>
          cases tensor with
          | scalar _ => cases ht.1.1
          | tensor t =>
            obtain ⟨coords, hc, hlen, hb⟩ := Bounds.within_sound hi hbound
            have hs := ht.1.1.1
            have hpoint : Idx.InBounds shape (coords.map Int.toNat) := coords_inBounds _ _ hlen hb
            have hdecode : Rules.InsertExtract.coordinates indexArgs = some coords := hc
            refine ⟨[.tensor (t.set (coords.map Int.toNat) value).canonize], ?_,
              .cons ⟨updated_fact ht.1 hx.1 hpoint hfull, trivial⟩ .nil⟩
            rw [hregs, evalRegions]
            exact Rules.InsertExtract.insert_sem hn hm hdecode (hs.symm ▸ hlen) (hs.symm ▸ hb) ctx call value
      · cases h
    · cases h
  · cases h

def book (ib : Nat) : Refined.Book model opSem ib where
  rule := rule
  sound := rule_sound ib

end MerlinVerify.Check.Definedness.Insert
