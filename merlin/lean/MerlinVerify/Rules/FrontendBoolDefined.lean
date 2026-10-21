/- Additive f32 comparison totality for frontend execution qualification.
This proves initialization for any comparison interpretation, without a numerical float law. -/
import MerlinVerify.Check.DefinedLinalg
import MerlinVerify.Rules.FrontendBoolSimulation

namespace MerlinVerify.Rules.FrontendBoolDefined
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops

/-- Only the actual no-fast-math f32 comparison vocabulary used by the repair is admitted. -/
def compareRule : Definedness.Rule := fun op fs =>
  if FrontendBoolSimulation.CmpFacts op ∧ op.regions = [] ∧
      fs = [⟨.scalar (.float .f32), true⟩, ⟨.scalar (.float .f32), true⟩] then
    some [⟨.scalar (.int 1), true⟩]
  else none

theorem compareRule_sound (ib : Nat) : RuleSound ib compareRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold compareRule at h
  split at h
  · next hc =>
    obtain ⟨⟨_, hd, hn, hr, hk⟩, _, rfl⟩ := hc
    cases h
    obtain _ | ⟨hx, hy⟩ := ha
    obtain _ | ⟨hy, hnil⟩ := hy
    cases hnil
    obtain ⟨x, rfl⟩ := initialized_float hx
    obtain ⟨y, rfl⟩ := initialized_float hy
    rw [opSem_modelled hm]
    simp only [dispatch, Op.blind_name, hd]
    simp only [Arith.sem, Op.blind_name, hn, Arith.binOp, Arith.pureN, hr]
    rw [show [.scalar (.some (.flt .f32 x)), .scalar (.some (.flt .f32 y))] =
      ([.some (.flt .f32 x), .some (.flt .f32 y)] : List Val).map RVal.scalar from rfl,
      liftEager_scalars]
    exact ⟨_, rfl, .cons ⟨by simp [Arith.boolVal, RVal.HasTy, Val.HasTy, Scalar.conforms],
      by simp [Arith.boolVal, RVal.Defined]⟩ .nil⟩
  · cases h

def comparisonBook (ib : Nat) : Book opSem ib where
  rule _ op fs := if unmodelledAttr? op.blind = none then compareRule op fs else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hm =>
      have hregs : op.regions = [] := by
        unfold compareRule at h
        split at h
        · next hc => exact hc.2.1
        · cases h
      simpa only [hregs, evalRegions] using compareRule_sound ib op fs rs hm h ctx call args hib ha
    · cases h

/-- The existing body-totality interface lifts the new scalar rule through actual generics.
Every fallback retains its existing proof and refuses unsupported operations. -/
def nestedBook (ib : Nat) : Nat → Book opSem ib
  | 0 => (comparisonBook ib).orElse (Definedness.book ib)
  | depth + 1 => (genericBodyBook ib (nestedBook ib depth)).orElse (nestedBook ib 0)

def book (ib : Nat) : Book opSem ib where
  rule Γ op fs := (nestedBook ib (RegionDepth.op op)).rule Γ op fs
  sound Γ op fs rs h ctx call env args hib hΓ ha :=
    (nestedBook ib (RegionDepth.op op)).sound Γ op fs rs h ctx call env args hib hΓ ha

#print axioms compareRule_sound
end MerlinVerify.Rules.FrontendBoolDefined
