/- Rank-one construction initializes exactly the elements supplied by actual initialized scalar
facts. Zero inputs produce a defined empty tensor, without authorizing any extraction from it. -/
import MerlinVerify.Check.DefinedRules
import MerlinVerify.Rules.FromElementsKernel

namespace MerlinVerify.Check.Definedness.FromElements

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

theorem tensor_fact {ib : Nat} {element : Ty} {values : List Val}
    (ht : ValsTyped ib values (List.replicate values.length element))
    (hd : ∀ v ∈ values, v ≠ .poison) :
    Fact.Holds ib (.tensor (Rules.FromElements.tensor element values))
      ⟨.tensor [values.length] element, true⟩ := by
  constructor
  · refine ⟨rfl, rfl, fun i hi => ?_⟩
    simp only [Rules.FromElements.tensor, Tensor.canon, hi, ↓reduceIte]
    rcases toArray_getD_mem values (linearIndex [values.length] i) .poison with hp | hp
    · rw [hp]; exact hasTy_poison _ _
    · exact ValsTyped.of_all ht (fun _ h => List.eq_of_mem_replicate h) _ hp
  · intro _ i hi
    change Idx.InBounds [values.length] i at hi
    have hk : linearIndex [values.length] i < values.length := by
      simpa [shapeSize] using linearIndex_lt [values.length] i hi
    simpa [Rules.FromElements.tensor, Tensor.canon, hi, List.getElem?_toArray,
      List.getElem?_eq_getElem hk] using hd values[linearIndex [values.length] i] (List.getElem_mem hk)

def rule : Rule := fun s fs =>
  if s.name = "tensor.from_elements" then
    match Sem.Ops.Tensor.resultTensorTy s with
    | some ([size], element) =>
      if fs = List.replicate size (⟨.scalar element, true⟩ : Fact) then
        some [⟨.tensor [size] element, true⟩]
      else none
    | _ => none
  else none

theorem rule_sound (ib : Nat) : RuleSound ib rule := by
  intro s fs rs hm h ctx call args _ ha
  unfold rule at h
  split at h
  · next hn =>
    split at h
    · next size element ht =>
      split at h
      · next hfs =>
        subst fs
        cases h
        have htyped := ha.typed
        have hdefined := ha.defined (by intro f hf; have := List.eq_of_mem_replicate hf; subst f; rfl)
        have hrep : (List.replicate size (⟨.scalar element, true⟩ : Fact)).map (·.ty) =
            (List.replicate size element).map (liftTy none) := by simp [liftTy]
        rw [hrep] at htyped
        obtain ⟨values, rfl, hvalues⟩ := hasTys_lift_scalar htyped
        have hlen : values.length = size := by rw [List.Forall₂.length_eq hvalues, List.length_replicate]
        refine ⟨[.tensor (Rules.FromElements.tensor element values)],
          Rules.FromElements.constructor_sem hn hm ht values hlen ctx call, .cons ?_ .nil⟩
        rw [← hlen] at hvalues ⊢
        apply tensor_fact hvalues
        intro value hv
        exact hdefined (.scalar value) (List.mem_map_of_mem hv)
      · cases h
    · cases h
  · cases h

def book (ib : Nat) : Book opSem ib where
  rule _ s fs := if unmodelledAttr? s.blind = none ∧ s.regions = [] then rule s fs else none
  sound Γ s fs rs h ctx call env args hi _ ha := by
    split at h
    · next hc =>
      simpa only [hc.2, evalRegions] using rule_sound ib s fs rs hc.1 h ctx call args hi ha
    · cases h

end MerlinVerify.Check.Definedness.FromElements
