/- Actual-boundary repair policy and initialized checkpoint under one inactive interpretation. -/
import MerlinVerify.Rules.FrontendRuntimeCheckpoint
import MerlinVerify.Rules.FrontendBoolCandidateChain

namespace MerlinVerify.Rules.FrontendRuntimeVerification
open MerlinVerify MerlinVerify.Check FrontendRuntimeCheckpoint

/-- Hints locate the checkpoint; every operation, root and type is recovered from both modules.
The full repair check also accounts for all changes outside this checkpoint. -/
def checkpoint (src tgt : Module) (name : String) (stop limit : Nat) : Bool :=
  match FrontendBoolProgram.defsOps 64 src.ops, project? src name stop, project? tgt name stop with
  | some defs, some s, some t =>
    match t.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) with
    | none => false
    | some a => t.check limit a &&
      FrontendBoolSimulation.region (FrontendBoolProgram.sourceTypes defs) [] s.body t.body
  | _, _, _ => false

def check (src tgt : Module) (name : String) (stop limit : Nat) : Bool :=
  FrontendBoolSimulation.check src tgt && checkpoint src tgt name stop limit

/-- Acceptance proves both complete-program equality to the declared repair policy and successful
execution of the actual initial segment. It does not claim that the rest of the model is defined,
or that repaired execution preserves the potentially poison-producing raw source semantics. -/
theorem check_sound {src tgt : Module} {name : String} {stop limit : Nat}
    (h : check src tgt name stop limit = true) :
    FrontendBoolSimulation.check src tgt = true ∧
    ∃ defs, ∃ (s : Scope src name stop) (t : Scope tgt name stop), ∃ a,
      FrontendBoolProgram.defsOps 64 src.ops = some defs ∧
      project? src name stop = some s ∧ project? tgt name stop = some t ∧
      t.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) = some a ∧
      ∀ (base : FloatModel) (ctx : Ctx) (call : CallSem) (args : List RVal),
        ctx.indexBits = 64 → 1024 ≤ limit → HasTys 64 args a → ResultsDefined args →
        ∃ results,
          evalRegion (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
            call [] t.body args = .ok results ∧
          evalRegion (FrontendBoolSimulationWith.policySem (FrontendMaskRuntime.sem limit)
              (FrontendBoolProgram.sourceTypes defs)) {ctx with M := FrontendF32Comparison.model base}
            call [] s.body args = .ok results ∧
          HasTys 64 results FrontendMaskTyped.resultTypes ∧ ResultsDefined results := by
  unfold check checkpoint at h
  simp only [Bool.and_eq_true] at h
  refine ⟨h.1, ?_⟩
  rcases h with ⟨_, h⟩
  split at h
  · next defs s t hdefs hs ht =>
    split at h
    · cases h
    · next a ha =>
      simp only [Bool.and_eq_true] at h
      refine ⟨defs, s, t, a, hdefs, hs, ht, ha, ?_⟩
      intro base ctx call args hib hl hty hd
      obtain ⟨results, hv, hr, hi⟩ := scope_sound h.1 base ctx call args hty hd hib hl
      have he := FrontendBoolSimulationWith.region_sound _
        (FrontendBoolSimulationWith.candidateLaws limit) _ call [] s.body t.body
        {ctx with M := FrontendF32Comparison.model base} [] [] h.2
        (FrontendF32Comparison.model_correct base) (by intro x y hxy; simp [Rho.find] at hxy)
      exact ⟨results, hv, (congrFun he args).trans hv, hr, hi⟩
  · cases h

/-- Complete-program policy matching is unconditional on successful execution. Poison, UB and
unsupported outcomes remain observable in the same candidate evaluator. -/
theorem full_policy {src tgt : Module} {name : String} {stop limit : Nat}
    (h : check src tgt name stop limit = true) (base : FloatModel) (ctx : Ctx)
    (entry : String) (args : List RVal) :
    FrontendBoolSimulationWith.policyDenote (FrontendMaskRuntime.sem limit) src
        {ctx with M := FrontendF32Comparison.model base} entry args =
      denote (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base} tgt entry args :=
  FrontendBoolSimulationWith.check_sound_candidate (check_sound h).1 limit base ctx entry args

#print axioms check_sound
#print axioms full_policy
end MerlinVerify.Rules.FrontendRuntimeVerification
