/- Initialization transfer for the observed rank-zero scalar-carrier constructor. The actual
input fact supplies scalar typing and initialization; result syntax alone supplies neither. -/
import MerlinVerify.Check.DefinedRules
import MerlinVerify.Rules.ScalarCarrierKernel

namespace MerlinVerify.Check.Definedness.Carrier

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def rule : Rule := fun s fs =>
  if s.name = "tensor.from_elements" then
    match fs, Sem.Ops.Tensor.resultTensorTy s with
    | [⟨.scalar e, true⟩], some ([], element) =>
      if e = element then some [⟨.tensor [] element, true⟩] else none
    | _, _ => none
  else none

theorem rule_sound (ib : Nat) : RuleSound ib rule := by
  intro s fs rs hm h ctx call args _ ha
  unfold rule at h
  split at h
  · next hn =>
    split at h
    · next e element ht =>
      split at h
      · next he =>
        subst e
        cases h
        obtain _ | ⟨hv, hnil⟩ := ha
        cases hnil
        rename_i value
        cases value with
        | tensor _ => cases hv.1
        | scalar value =>
          refine ⟨[.tensor (Tensor.full [] element value)],
            Rules.ScalarCarrier.from_elements_sem hn hm ht ctx call value,
            .cons ⟨hasTy_full hv.1, ?_⟩ .nil⟩
          intro _ i hi
          change Idx.InBounds [] i at hi
          simpa only [RVal.Defined, Tensor.full, Tensor.canon, hi, ↓reduceIte] using hv.2 rfl
      · cases h
    · cases h
  · cases h

def book (ib : Nat) : Book opSem ib where
  rule _ s fs :=
    if unmodelledAttr? s.blind = none ∧ s.regions = [] then rule s fs else none
  sound Γ s fs rs h ctx call env args hi _ ha := by
    split at h
    · next hc =>
      simpa only [hc.2, evalRegions] using rule_sound ib s fs rs hc.1 h ctx call args hi ha
    · cases h

end MerlinVerify.Check.Definedness.Carrier
