/- Scalar raw-bit reinterpretations preserve initialization. The proof shares the exact runtime
kernel with the additive progress adapter and assumes no numerical float identities. -/
import MerlinVerify.Check.DefinedRules
import MerlinVerify.Rules.ScalarBitcast

namespace MerlinVerify.Check.DefinedBitcast

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Rules.ScalarBitcast

def rule : Definedness.Rule := fun s fs =>
  match fs, s.results with
  | [⟨.scalar src, true⟩], [r] =>
    if s.name = "arith.bitcast" ∧ Compatible src r.ty ∧ Plain s then
      some [⟨.scalar r.ty, true⟩]
    else none
  | _, _ => none

theorem rule_sound (ib : Nat) : Definedness.RuleSound ib rule := by
  intro s fs rs hm h ctx call args hib ha
  unfold rule at h
  split at h
  · next src r hr =>
    split at h
    · next hc =>
      obtain ⟨hn, hc, _⟩ := hc
      cases h
      obtain _ | ⟨hv, hnil⟩ := ha
      cases hnil
      rename_i value
      cases value with
      | tensor _ => cases hv.1
      | scalar value =>
        have hresult := resultElt_eq hr rfl hc
        have hvalue : Val.HasTy ctx.indexBits src value := by simpa only [RVal.HasTy, hib] using hv.1
        obtain ⟨target, hs, ht, hi⟩ := meaning hn hm hresult hc ctx call [] hvalue
        exact ⟨[.scalar target], hs, .cons ⟨by simpa only [RVal.HasTy, hib] using ht,
          fun _ => hi (hv.2 rfl)⟩ .nil⟩
    · cases h
  · cases h

def book (ib : Nat) : Definedness.Book opSem ib where
  rule _ s fs :=
    if unmodelledAttr? s.blind = none ∧ s.regions = [] then rule s fs else none
  sound Γ s fs rs h ctx call env args hi _ ha := by
    split at h
    · next hc =>
      simpa only [hc.2, evalRegions] using rule_sound ib s fs rs hc.1 h ctx call args hi ha
    · cases h

end MerlinVerify.Check.DefinedBitcast
