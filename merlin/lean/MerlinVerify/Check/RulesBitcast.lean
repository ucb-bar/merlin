/- Additive scalar bitcast support for exactly the raw-bit domains of the existing semantics. -/
import MerlinVerify.Rules.ScalarBitcast

namespace MerlinVerify.Check.Bitcast

open MerlinVerify MerlinVerify.Sem.Ops MerlinVerify.Rules.ScalarBitcast

def rule : OpRule := fun _ s argTys resTys =>
  match argTys, resTys, s.results with
  | [.scalar src], [.scalar dst], [r] =>
    if s.name = "arith.bitcast" ∧ Compatible src dst ∧ Plain s ∧ r.ty = dst ∧
        s.regions = [] ∧ s.successors = [] then some [] else none
  | _, _, _ => none

theorem rule_sound (ib : Nat) : RuleSound ib rule := by
  intro d s argTys resTys sigs hm h
  unfold rule at h
  split at h
  · next src dst r hr =>
    split at h
    · next hc =>
      obtain ⟨hn, hc, _, hty, _, _⟩ := hc
      cases h
      intro ctx call args regions hib _ ha _
      obtain _ | ⟨hv, hnil⟩ := ha
      cases hnil
      rename_i value
      cases value with
      | tensor _ => cases hv
      | scalar value =>
        have hresult := resultElt_eq hr hty hc
        have hvalue : Val.HasTy ctx.indexBits src value := by simpa only [RVal.HasTy, hib] using hv
        obtain ⟨target, hs, ht, _⟩ := meaning hn hm hresult hc ctx call regions hvalue
        rw [hs]
        exact Safe.of_ok (.cons (by simpa only [RVal.HasTy, hib] using ht) .nil)
    · cases h
  · cases h

end MerlinVerify.Check.Bitcast
