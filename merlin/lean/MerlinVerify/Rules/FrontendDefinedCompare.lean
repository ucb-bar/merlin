/- Additive f32 comparison totality for frontend execution qualification.
This proves initialization for any comparison interpretation, without a numerical float law. -/
import MerlinVerify.Check.DefinedLinalg
import MerlinVerify.Rules.FrontendBoolSimulation

namespace MerlinVerify.Rules.FrontendDefinedCompare
open MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness MerlinVerify.Sem.Ops

/-- This proves initialization, not numerical identities. Only f32, known predicates and no
fast-math are accepted; FloatModel remains parametric. -/
def metadata (op : Op) : Prop :=
  dialectOf op.name = "arith" ∧ op.name = "arith.cmpf" ∧ Arith.resultElt op.blind = some (.int 1) ∧
  (match attrInt? op "predicate" with | some p => 0 ≤ p ∧ p < 16 | none => False) ∧
  op.attr? "fastmath" = some (.enum "arith" "fastmath" "none")
instance (op : Op) : Decidable (metadata op) := by
  unfold metadata
  cases attrInt? op "predicate" <;> infer_instance

def compareRule : Definedness.Rule := fun op fs =>
  if metadata op ∧ op.regions = [] ∧
      fs = [⟨.scalar (.float .f32), true⟩, ⟨.scalar (.float .f32), true⟩] then
    some [⟨.scalar (.int 1), true⟩]
  else none

theorem compareRule_sound (ib : Nat) : RuleSound ib compareRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold compareRule at h
  split at h
  · next hc =>
    obtain ⟨⟨hd, hn, hr, _⟩, _, rfl⟩ := hc
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


#print axioms compareRule_sound
end MerlinVerify.Rules.FrontendDefinedCompare
