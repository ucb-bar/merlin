/- Before/after program execution against A2's source-only repair policy,
including the actual position embedding and attention-mask continuation. -/
import MerlinVerify.Rules.FrontendPositionExecution

namespace MerlinVerify.Rules.FrontendPositionVerification
open MerlinVerify MerlinVerify.Check FrontendPositionScope
open FrontendTensorRanges

/-- The source policy is applied to the actual original region. The target's
initialized execution is proved independently from its actual SSA producers. -/
def checkpoint (src tgt : Module) (name : String) (stop count : Nat) : Bool :=
  match FrontendBoolProgram.defsOps 64 src.ops, project? src name stop count, project? tgt name stop count with
  | some defs, some s, some t => t.check &&
      FrontendBoolSimulation.region (FrontendBoolProgram.sourceTypes defs) [] s.body t.body
  | _, _, _ => false

def check (src tgt : Module) (name : String) (stop count : Nat) : Bool :=
  FrontendBoolSimulation.check src tgt && checkpoint src tgt name stop count

/-- Acceptance connects execution of both actual initial segments to the approved
frontend repair policy. It does not assert ordinary equivalence to the raw fptosi source. -/
theorem check_sound {src tgt : Module} {name : String} {stop count : Nat}
    (h : check src tgt name stop count = true) :
    FrontendBoolSimulation.check src tgt = true ∧
    ∃ defs, ∃ (s : Scope src name stop count) (t : Scope tgt name stop count), ∃ a rs,
      FrontendBoolProgram.defsOps 64 src.ops = some defs ∧
      project? src name stop count = some s ∧ project? tgt name stop count = some t ∧
      t.selection.p.cut.block.args.mapM (fun v => vtyOf 64 v.ty) = some a ∧ t.resultFacts? = some rs ∧
      ∀ (limit : Nat) (base : FloatModel) (ctx : Ctx) (call : CallSem) (args : List RVal),
        ctx.indexBits = 64 → 1024 ≤ limit → HasTys 64 args a → ResultsDefined args →
        ∃ results,
          evalRegion (FrontendMaskRuntime.sem limit) {ctx with M := FrontendF32Comparison.model base}
            call [] t.body args = .ok results ∧
          evalRegion (FrontendBoolSimulationWith.policySem (FrontendMaskRuntime.sem limit)
              (FrontendBoolProgram.sourceTypes defs)) {ctx with M := FrontendF32Comparison.model base}
            call [] s.body args = .ok results ∧
          Definedness.Refined.FactsHold model 64 results rs := by
  unfold check checkpoint at h
  simp only [Bool.and_eq_true] at h
  refine ⟨h.1, ?_⟩
  rcases h with ⟨_, h⟩
  split at h
  · next defs s t hdefs hs ht =>
    simp only [Bool.and_eq_true] at h
    obtain ⟨a, rs, ha, hr, hexec⟩ := FrontendPositionExecution.scope_sound t h.1
    refine ⟨defs, s, t, a, rs, hdefs, hs, ht, ha, hr, ?_⟩
    intro limit base ctx call args hib hl hty hd
    obtain ⟨results, hv, hf⟩ := hexec limit base ctx call args hib hl hty hd
    have he := FrontendBoolSimulationWith.region_sound (FrontendMaskRuntime.sem limit)
      (FrontendBoolSimulationWith.candidateLaws limit) _ call [] s.body t.body
      {ctx with M := FrontendF32Comparison.model base} [] [] h.2
      (FrontendF32Comparison.model_correct base) (by intro x y hxy; simp [Rho.find] at hxy)
    exact ⟨results, hv, (congrFun he args).trans hv, hf⟩
  · cases h

#print axioms check_sound
end MerlinVerify.Rules.FrontendPositionVerification
