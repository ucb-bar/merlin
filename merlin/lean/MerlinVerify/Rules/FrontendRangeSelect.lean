/- Interval joins for initialized selection, with no assumption on the condition value. -/
import MerlinVerify.Rules.FrontendRangeGeneric

namespace MerlinVerify.Rules.FrontendRangeSelect
open MerlinVerify MerlinVerify.Check Sem.Ops
open FrontendTensorRanges
open Definedness.IndexRanges (Interval)

def join (a b : Option Interval) : Option Interval :=
  match a, b with
  | some a, some b => some ⟨min a.lower b.lower, max a.upper b.upper⟩
  | _, _ => none

theorem interval_mono {a b : Interval} {v : RVal} (hlo : b.lower ≤ a.lower) (hhi : a.upper ≤ b.upper)
    (h : a.Holds v) : b.Holds v := by
  obtain ⟨z, hz, hl, hh⟩ := h
  refine ⟨z, hz, ?_, ?_⟩ <;> omega

theorem holds_mono {a b : Interval} {v : RVal} (hlo : b.lower ≤ a.lower) (hhi : a.upper ≤ b.upper)
    (h : Holds a v) : Holds b v := by
  cases v with
  | scalar _ => exact interval_mono hlo hhi h
  | tensor t => exact fun p hp => interval_mono hlo hhi (h p hp)

theorem join_left (a b : Option Interval) (v : RVal) (h : model.holds a v) :
    model.holds (join a b) v := by
  cases a with
  | none => trivial
  | some a =>
    cases b with
    | none => trivial
    | some b => exact holds_mono (Nat.min_le_left _ _) (Nat.le_max_left _ _) h

theorem join_right (a b : Option Interval) (v : RVal) (h : model.holds b v) :
    model.holds (join a b) v := by
  cases a with
  | none => trivial
  | some a =>
    cases b with
    | none => trivial
    | some b => exact holds_mono (Nat.min_le_right _ _) (Nat.le_max_right _ _) h

theorem canonize_holds (info : Option Interval) (v : RVal) (h : model.holds info v) :
    model.holds info v.canonize := by
  cases info with
  | none => trivial
  | some r =>
    cases v with
    | scalar _ => exact h
    | tensor t =>
      intro p hp
      change Idx.InBounds t.shape p at hp
      simpa only [RVal.canonize, Tensor.canonize, Tensor.canon, hp, ↓reduceIte] using h p hp

/-- A branch interval is established for both possible initialized conditions. -/
def rule (ib : Nat) (_ : FEnv) (op : Op) (fs : List Fact) : Option (List Fact) :=
  if unmodelledAttr? op.blind = none ∧ op.regions = [] then
    match fs, op.results with
    | [⟨⟨.scalar (.int 1), true⟩, _⟩, ⟨⟨t, true⟩, a⟩, ⟨⟨t', true⟩, b⟩], [r] =>
      if dialectOf op.name = "arith" ∧ op.name = "arith.select" ∧ t' = t ∧ vtyOf ib r.ty = some t then
        some [⟨⟨t, true⟩, join a b⟩]
      else none
    | _, _ => none
  else none

def book (ib : Nat) : Definedness.Refined.Book model opSem ib where
  rule := rule ib
  sound Γ op fs rs h ctx call env args _ _ ha := by
    unfold rule at h
    split at h
    · next hc =>
      obtain ⟨hm, hregs⟩ := hc
      split at h
      · next _ _ _ condInfo t a t' b r hres =>
        split at h
        · next hc =>
          obtain ⟨hd, hn, hty, _⟩ := hc
          subst t'
          cases h
          obtain _ | ⟨hc, hrest⟩ := ha
          obtain _ | ⟨ht, hrest⟩ := hrest
          obtain _ | ⟨hf, hnil⟩ := hrest
          cases hnil
          obtain ⟨c, rfl⟩ := Definedness.initialized_int hc.1
          rw [hregs, evalRegions, opSem_modelled hm]
          simp only [dispatch, Op.blind_name, hd]
          simp only [Arith.sem, Op.blind_name, hn]
          split
          · exact ⟨_, rfl, .cons ⟨⟨hasTy_canonize ht.1.1, fun _ => Definedness.defined_canonize (ht.1.2 rfl)⟩,
              canonize_holds _ _ (join_left _ _ _ ht.2)⟩ .nil⟩
          · exact ⟨_, rfl, .cons ⟨⟨hasTy_canonize hf.1.1, fun _ => Definedness.defined_canonize (hf.1.2 rfl)⟩,
              canonize_holds _ _ (join_right _ _ _ hf.2)⟩ .nil⟩
        · cases h
      · cases h
    · cases h

#print axioms book
end MerlinVerify.Rules.FrontendRangeSelect
