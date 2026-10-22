/- Bounds for linalg.index come from the enclosing checked iteration space. The encoding
must fit the actual signed index width; loop coordinates cannot be assumed at function scope. -/
import MerlinVerify.Check.BoundedFloatFacts
import MerlinVerify.Check.DefinedPointRegions
import MerlinVerify.Check.DefinedGeneric

namespace MerlinVerify.Check.BoundedFloatFacts.PointIndex

open MerlinVerify Definedness Sem.Ops

def Agreement (sem : OpSem) : Prop :=
  ∀ ctx call op args regions, op.name = "linalg.index" → sem ctx call op args regions = opSem ctx call op args regions

def rule (ib : Nat) (ranges : List Nat) (_ : Refined.FEnv Info) (op : Op)
    (facts : List (Refined.Fact Info)) : Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "linalg" ∧ op.name = "linalg.index" ∧
      op.regions = [] ∧ facts = [] then
    match attrInt? op "dim", op.results with
    | some dim, [result] => do
      let upper ← ranges[dim.toNat]?
      if result.ty = .index ∧ 0 ≤ dim ∧ 0 < ib ∧ upper ≤ 2 ^ (ib - 1) then
        some [⟨⟨.scalar .index, true⟩, ⟨FloatFacts.model.empty, some ⟨0, upper⟩⟩⟩]
      else none
    | _, _ => none
  else none

theorem rule_sound {sem : OpSem} {allowed : Contextual.Conditions} (agreement : Agreement sem)
    (ib : Nat) (ranges : List Nat) : ∀ Γ op facts outputs, rule ib ranges Γ op facts = some outputs →
    ∀ ctx call env args, ctx.indexBits = ib → Contextual.PointConditions allowed ranges ctx call →
    Refined.EnvHolds model ib Γ env → Refined.FactsHold model ib args facts → ∃ values,
      sem ctx call op args (evalRegions sem call env op.regions) = .ok values ∧ Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args width permitted _ inputs
  unfold rule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, rfl⟩ := header
    cases inputs
    split at checked
    · next dim result dimension results =>
      cases found : ranges[dim.toNat]? with
      | none => simp [found] at checked
      | some upper =>
        simp only [found, bind, Option.bind] at checked
        split at checked
        · next conditions =>
          obtain ⟨_, nonnegative, positive, fits⟩ := conditions
          cases checked
          obtain ⟨inRange, upperEq⟩ := List.getElem?_eq_some_iff.1 found
          have inPoint : dim.toNat < ctx.loopIndex.length := by have := permitted.2.1; omega
          let point := ctx.loopIndex[dim.toNat]
          have pointFound : ctx.loopIndex[dim.toNat]? = some point := List.getElem?_eq_getElem inPoint
          have small : point < upper := by simpa only [upperEq] using permitted.2.2 dim.toNat inPoint inRange
          have fitsInt : (point : Int) < (2 : Int) ^ (ib - 1) := by exact_mod_cast Nat.lt_of_lt_of_le small fits
          have encoded : (BitVec.ofNat ib point).toInt = (point : Int) := by
            change (BitVec.ofInt ib (point : Int)).toInt = _
            apply BitVec.toInt_ofInt_eq_self positive _ fitsInt
            have power : (0 : Int) ≤ 2 ^ (ib - 1) := Int.pow_nonneg (by decide)
            omega
          refine ⟨[.scalar (.some (.bits ib (BitVec.ofNat ib point)))], ?_,
            .cons ⟨⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
              FloatFacts.model.empty_sound _, ?_⟩ .nil⟩
          · rw [agreement ctx call op [] (evalRegions sem call env op.regions) named,
              regions, evalRegions, linalg_sem_of modelled dialect (.inr (.inr named))]
            simp only [Linalg.sem, Op.blind_name, named, attrInt?_blind, dimension, pointFound,
              nonnegative, ↓reduceIte]
            rw [width]
          · refine ⟨point, ?_, ?_, ?_⟩
            · simp [IndexRanges.value?, Sem.Ops.Tensor.indexVal?, encoded]
            · change (0 : Int) ≤ (point : Int); omega
            · change (point : Int) < (upper : Int); exact_mod_cast small
        · cases checked
    · cases checked
  · cases checked

def book {sem : OpSem} {allowed : Contextual.Conditions} (agreement : Agreement sem)
    (ib : Nat) (ranges : List Nat) : Contextual.Book model sem ib (Contextual.PointConditions allowed ranges) where
  rule := rule ib ranges
  sound := rule_sound agreement ib ranges

#print axioms rule_sound
#print axioms book

end MerlinVerify.Check.BoundedFloatFacts.PointIndex
