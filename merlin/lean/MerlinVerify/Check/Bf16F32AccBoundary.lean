/- A positive pass-local MACC boundary combines two independent checks on the same
actual modules: complete equality to the candidate reference (including errors),
and strict initialized-operand qualification of every reconstructed source site.
This is a conditional theorem about the existing draft contract, never policy
approval or a claim that half accumulation equals f32 accumulation. -/
import MerlinVerify.Check.Bf16F32AccSites

namespace MerlinVerify.Check.Bf16F32AccBoundary

open MerlinVerify Rules.Bf16F32AccReference

/-- Empty/identity boundaries cannot count as positive MACC coverage. -/
def check (ib : Nat) (before after : Module) : Bool :=
  Bf16F32AccSites.check ib before after &&
    match Bf16F32AccSites.collect before after with
    | none => false
    | some pairs => !pairs.isEmpty

/-- The policy reference comes only from the bound source, local obligations only
from the bound pair, and complete output equality uses explicit SSA/call laws.
The independent review of that numerical policy is an additional requirement. -/
theorem check_sound {sem : OpSem} (laws : ParametricAlpha.Laws sem)
    {ib : Nat} {before after : Module} (accepted : check ib before after = true) :
    ∃ hints reference pairs,
      expand hints before = some reference ∧
      Bf16F32AccSites.collect before after = some pairs ∧
      pairs ≠ [] ∧
      (∀ ctx fn args, denote sem ctx after fn args = denote sem ctx reference fn args) ∧
      ∀ pair ∈ pairs, Bf16F32Acc.Qualified ib pair.before pair.after := by
  simp only [check, Bool.and_eq_true] at accepted
  obtain ⟨exactMatch, pairs, collected, qualified⟩ :=
    Bf16F32AccSites.check_sound accepted.1
  obtain ⟨hints, reference, expanded, same⟩ := Bf16F32AccExact.check_sound laws exactMatch
  refine ⟨hints, reference, pairs, expanded, collected, ?_, same, qualified⟩
  intro empty
  simpa [collected, empty] using accepted.2

#print axioms check_sound

end MerlinVerify.Check.Bf16F32AccBoundary
