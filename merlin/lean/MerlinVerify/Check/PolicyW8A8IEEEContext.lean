/- An explicit witness for the previously hypothetical joint numerical context. Only the
constrained arithmetic slice is qualified; unsupported fallback operations remain explicit.
This does not claim native fidelity, policy admission or whole-model source-definedness. -/
import MerlinVerify.Rules.W8A8IEEELaws
import MerlinVerify.Check.PolicyW8A8NumericalContext

namespace MerlinVerify.Check.PolicyW8A8IEEEContext

open MerlinVerify Rules

def context (fallback : FloatModel) (externals : ExtEnv) (indexBits : Nat) : Ctx where
  M := W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)
  ext := externals
  indexBits := indexBits

theorem context_ok (fallback : FloatModel) (externals : ExtEnv) (indexBits : Nat) :
    PolicyW8A8NumericalContext.ContextOk (context fallback externals indexBits) :=
  ⟨W8A8IEEEPrimitives.model fallback, W8A8IEEELaws.dynamic_laws fallback,
    W8A8IEEEPrimitives.agreement fallback, rfl⟩

/-- The fallback is deliberately unqualified for other operations. All ten constrained laws
and nine arithmetic agreement fields follow from the actual overrides, not these sentinels. -/
def candidate : Ctx := context W8A8IEEEPrimitives.unsupported
  (fun name _ => .stuck s!"uninterpreted W8A8 external {name}") 64

theorem candidate_ok : PolicyW8A8NumericalContext.ContextOk candidate :=
  context_ok _ _ _

theorem inhabited : ∃ ctx, PolicyW8A8NumericalContext.ContextOk ctx := ⟨candidate, candidate_ok⟩

#print axioms context_ok
#print axioms inhabited

end MerlinVerify.Check.PolicyW8A8IEEEContext
