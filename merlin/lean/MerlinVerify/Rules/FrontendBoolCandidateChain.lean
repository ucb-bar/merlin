/- Repair-policy composition for the inactive runtime interpretation. -/
import MerlinVerify.Rules.FrontendBoolSimulationWith

namespace MerlinVerify.Rules.FrontendBoolCandidateChain
open MerlinVerify FrontendMaskRuntime

/-- A suffix proved under this same interpretation composes with the accepted f32 repair.
The suffix equality is an explicit premise: frozen-model receipts cannot discharge it. The
conclusion matches the original program's declared A2 repair policy, not its raw fptosi meaning. -/
theorem compose {src repaired final : Module}
    (hrepair : FrontendBoolSimulation.check src repaired = true)
    (limit : Nat) (base : FloatModel) (ctx : Ctx) (name : String) (args : List RVal)
    (hsuffix : denote (sem limit) {ctx with M := FrontendF32Comparison.model base} final name args =
      denote (sem limit) {ctx with M := FrontendF32Comparison.model base} repaired name args) :
    denote (sem limit) {ctx with M := FrontendF32Comparison.model base} final name args =
      FrontendBoolSimulationWith.policyDenote (sem limit) src
        {ctx with M := FrontendF32Comparison.model base} name args := by
  exact hsuffix.trans (FrontendBoolSimulationWith.check_sound_candidate hrepair limit base ctx name args).symm

#print axioms compose
end MerlinVerify.Rules.FrontendBoolCandidateChain
