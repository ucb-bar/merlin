/- Sound complete boundary checker. Intent repair is provisional; invalid-source equivalence is
neither a hypothesis nor a conclusion. Shared driver integration is deliberately separate. -/
import MerlinVerify.Check.OverrankReference
import MerlinVerify.Check.ParametricModule
import MerlinVerify.Rules.OverrankKernel
namespace MerlinVerify.Check.Overrank
open MerlinVerify Rules.OverrankIntent OverrankReference

theorem laws : ParametricAlpha.Laws Sem.Ops.opSem where
  operation h _ _ := Sem.Ops.opSem_blind (blind_of_opHeadEq h)
  call_local h := Sem.Ops.opSem_call_local h

/-- Exact complete syntax is a linear fast path for unchanged retained boundaries. -/
def completeCheck (left right : Module) : Bool :=
  ExactSyntax.modules left right || ParametricModule.check left right

theorem completeCheck_sound {left right : Module} (h : completeCheck left right = true) :
    ∀ ctx name args, denote Sem.Ops.opSem ctx left name args = denote Sem.Ops.opSem ctx right name args := by
  simp only [completeCheck,Bool.or_eq_true] at h
  rcases h with exact | alpha
  · have same := ExactSyntax.modules_sound exact
    subst right
    intros
    rfl
  · exact ParametricModule.check_sound laws alpha

def check (before after : Module) : Bool :=
  (plan before).valid && completeCheck (reference before) after

/-- Whole module, every function and call, all nested regions and surrounding SSA. This relates
the output to the *repair reference*, never to the invalid original source operation. -/
theorem check_sound {before after : Module} (h : check before after = true) :
    (plan before).valid = true ∧
    ∀ ctx name args, denote Sem.Ops.opSem ctx after name args =
      denote Sem.Ops.opSem ctx (reference before) name args := by
  simp only [check,Bool.and_eq_true] at h
  exact ⟨h.1,fun ctx name args => (completeCheck_sound h.2 ctx name args).symm⟩

/-- Every recorded site's geometry is discharged by Lean's sound affine checker; no certificate
is allowed to assert the body or shape's correctness. -/
theorem site_intent (site : Site) (ctx : Ctx) (call : CallSem) (env : Env)
    {actual : Region} (body : alphaRegion [] (Rules.OverrankBody.region site.kind) actual = true)
    (A B : Idx → Option (FBits site.kind)) :
    Sem.Ops.Linalg.genericSem ctx (maps site.shape) (iters site.shape)
      [.tensor (Rules.OverrankTensor.floats site.kind site.shape.left A),
       .tensor (Rules.OverrankTensor.floats site.kind site.shape.right B)]
      [.tensor (Tensor.full site.shape.out (.float site.kind) (value site.kind (some (zero site.kind))))]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (tensor ctx.M site.kind site.shape A B)] :=
  Rules.OverrankTensor.generic_sound ctx call env site.kind site.shape site.layout body A B
/-- Additional complete-function qualification for standalone actual-pass cases. Unlike general
context correspondence, this check proves concrete success and initialized outputs outright. -/
def kernelCheck (kind : FKind) (s : Shape) (after : Module) : Bool :=
  layoutCheck kind s && ParametricModule.check (Rules.OverrankKernel.program kind s) after

theorem kernelCheck_sound {kind : FKind} {s : Shape} {after : Module}
    (h : kernelCheck kind s after = true) (ctx : Ctx)
    (A B : Idx → Option (FBits kind)) (domain : Domain kind s A B) (old : RVal) :
    denote Sem.Ops.opSem ctx after "subject"
      [.tensor (Rules.OverrankTensor.floats kind s.left A),
       .tensor (Rules.OverrankTensor.floats kind s.right B),old] =
      .ok [.tensor (tensor ctx.M kind s A B)] ∧ ResultsDefined [.tensor (tensor ctx.M kind s A B)] := by
  simp only [kernelCheck,Bool.and_eq_true] at h
  rw [← ParametricModule.check_sound laws h.2]
  exact Rules.OverrankKernel.program_defined ctx kind s h.1 A B domain old

end MerlinVerify.Check.Overrank
