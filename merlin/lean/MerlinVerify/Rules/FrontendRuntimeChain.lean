/- Checked suffix composition under exactly the same candidate interpretation. -/
import MerlinVerify.Rules.FrontendRuntimeVerification

namespace MerlinVerify.Rules.FrontendRuntimeChain
open MerlinVerify MerlinVerify.Check

/-- An intentionally narrow suffix checker: complete AST alpha correspondence. It never imports
frozen-model receipts or treats a numerical policy as ordinary execution equality. -/
def boundary (src tgt : Module) : Bool :=
  alphaOps [] src.ops tgt.ops && (outsideChanges src tgt).unverified.isEmpty &&
    FrontendBoolSimulation.functions [] src.funcs tgt.funcs

theorem boundary_sound {src tgt : Module} (h : boundary src tgt = true)
    (limit : Nat) (base : FloatModel) (ctx : Ctx) (name : String) (args : List RVal) :
    denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base} src name args =
      denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base} tgt name args := by
  simp only [boundary, Bool.and_eq_true] at h
  have he := FrontendBoolSimulationWith.evalFunc_sound _
    (FrontendBoolSimulationWith.candidateLaws limit) h.2
    {ctx with M := FrontendF32Comparison.model base} (FrontendF32Comparison.model_correct base)
  have hl := FrontendBoolSimulation.functions_length src.funcs tgt.funcs h.2
  rw [FrontendBoolSimulationWith.policy_nil] at he
  simp only [denote]
  rw [he, hl]

/-- The tail is explicit, so neither the order nor an intermediate boundary can be omitted. -/
def suffix (first : Module) : List Module → Bool
  | [] => true
  | next :: rest => boundary first next && suffix next rest

def final (first : Module) : List Module → Module
  | [] => first
  | next :: rest => final next rest

theorem suffix_sound (first : Module) : ∀ tail, suffix first tail = true →
    ∀ (limit : Nat) (base : FloatModel) (ctx : Ctx) (name : String) (args : List RVal),
    denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
        (final first tail) name args =
      denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base} first name args
  | [], _, _, _, _, _, _ => rfl
  | next :: rest, h, limit, base, ctx, name, args => by
    simp only [suffix, Bool.and_eq_true] at h
    exact (suffix_sound next rest h.2 limit base ctx name args).trans
      (boundary_sound h.1 limit base ctx name args).symm

/-- A checked suffix discharges the equality premise in the existing repair-policy composition
lemma. The final claim still matches A2's source-only repair policy, not raw source semantics. -/
theorem compose {src repaired : Module} {tail : List Module} {entry : String} {stop limit : Nat}
    (hrepair : FrontendRuntimeVerification.check src repaired entry stop limit = true)
    (hsuffix : suffix repaired tail = true) (base : FloatModel) (ctx : Ctx)
    (name : String) (args : List RVal) :
    denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
        (final repaired tail) name args =
      FrontendBoolSimulationWith.policyDenote (FrontendMaskRuntime.sem limit) src
        {ctx with M := FrontendF32Comparison.model base} name args :=
  FrontendBoolCandidateChain.compose (FrontendRuntimeVerification.check_sound hrepair).1 limit base ctx name args
    (suffix_sound repaired tail hsuffix limit base ctx name args)

#print axioms boundary_sound
#print axioms suffix_sound
#print axioms compose
end MerlinVerify.Rules.FrontendRuntimeChain
