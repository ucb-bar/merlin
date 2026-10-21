/- Whole-module translation validation against a fixed policy transformer. A policy is part of
the checker, never a certificate supplied by the candidate output. Rejection of an unsupported
source expansion cannot be replaced by an identity or by the proposed target. -/
import MerlinVerify.Check.Module
import MerlinVerify.Check.Chain
import MerlinVerify.Check.ExactSyntax

namespace MerlinVerify.Check.PolicyModule

open MerlinVerify

/-- Specifications may refuse unsupported source programs. The function only receives the
actual source module; neither the candidate target nor its hints can choose the reference. -/
structure Policy where
  name : String
  expand : Module → Option Module

/-- Policy validation relates the actual target to this exact source-derived reference. As
with C1, strict refinement additionally needs support of the reference on the claimed domain. -/
def Correct (policy : Policy) (before after : Module) : Prop :=
  ∃ reference, policy.expand before = some reference ∧
    ∀ fn ∈ Module.definedNames reference,
      FuncRefinesC Sem.Ops.opSem (fun _ => True) (fun _ => True) after fn reference fn

def check (policy : Policy) (before after : Module) : Bool :=
  match policy.expand before with
  | none => false
  | some reference => ExactSyntax.modules reference after ||
      (moduleCheck reference after && (outsideChanges reference after).unverified.isEmpty)

theorem check_sound {policy : Policy} {before after : Module}
    (h : check policy before after = true) : Correct policy before after := by
  unfold check at h
  cases expansion : policy.expand before with
  | none => simp [expansion] at h
  | some reference =>
    simp only [expansion, Bool.or_eq_true, Bool.and_eq_true] at h
    refine ⟨reference, expansion, ?_⟩
    rcases h with identical | checked
    · have same := ExactSyntax.modules_sound identical
      subst reference
      exact fun fn _ => FuncRefinesC.refl after fn
    · exact moduleCheck_sound checked.1

/-- No support fact about `before` can replace support of the policy reference. In particular,
this theorem cannot push a syntactic numerical policy backwards through an earlier rewrite. -/
theorem checked_refines {policy : Policy} {before after reference : Module} {fn : String}
    (h : check policy before after = true) (expanded : policy.expand before = some reference)
    (member : fn ∈ Module.definedNames reference)
    {ctxOk : Ctx → Prop} {domain : List RVal → Prop}
    (supported : FuncNotStuck Sem.Ops.opSem ctxOk domain reference fn) :
    FuncRefines Sem.Ops.opSem ctxOk domain after fn reference fn := by
  obtain ⟨actualReference, hexpanded, correct⟩ := check_sound h
  have same : actualReference = reference := Option.some.inj (hexpanded.symm.trans expanded)
  subst actualReference
  exact FuncRefinesC.strict (fun ctx _ args _ => correct fn member ctx trivial args trivial) supported

/-- The post-policy suffix composes with the checked policy boundary. Earlier rewrites keep
their own claim against the original source; no monotonicity of the policy is assumed. -/
theorem checked_suffix {policy : Policy} {before after reference : Module} {suffix : List Module}
    {fn : String} {ctxOk : Ctx → Prop} {domain : List RVal → Prop}
    (h : check policy before after = true)
    (expanded : policy.expand before = some reference) (member : fn ∈ Module.definedNames reference)
    (post : ChainRefinesC Sem.Ops.opSem ctxOk domain fn (after :: suffix)) :
    FuncRefinesC Sem.Ops.opSem ctxOk domain
      ((after :: suffix).getLast (List.cons_ne_nil _ _)) fn reference fn := by
  obtain ⟨actualReference, hexpanded, correct⟩ := check_sound h
  have same : actualReference = reference := Option.some.inj (hexpanded.symm.trans expanded)
  subst actualReference
  exact FuncRefinesC.trans (chain_refinesC after suffix post)
    (fun ctx _ args _ => correct fn member ctx trivial args trivial)

/-- Strict refinement already excludes an unmodelled source, even when that source has UB.
This fact must not be confused with successful, nonpoison execution. -/
theorem source_not_stuck {α : Type} {R : α → α → Prop} {target source : Outcome α}
    (refined : Refines R target source) : source.NotStuck := by
  cases source <;> cases target <;> simp_all [Refines, Outcome.NotStuck]

/-- A qualified policy result composes with a conditional downstream chain on the same domain.
The reference is fixed; this theorem does not move the policy across preceding rewrites. -/
theorem refining_suffix {sem : OpSem} {after reference : Module} {suffix : List Module}
    {fn : String} {ctxOk : Ctx → Prop} {domain : List RVal → Prop}
    (policyResult : FuncRefines sem ctxOk domain after fn reference fn)
    (post : ChainRefinesC sem ctxOk domain fn (after :: suffix)) :
    FuncRefines sem ctxOk domain
      ((after :: suffix).getLast (List.cons_ne_nil _ _)) fn reference fn := by
  apply FuncRefinesC.strict
    (FuncRefinesC.trans (chain_refinesC after suffix post)
      (fun ctx hc args ha => .inr (policyResult ctx hc args ha)))
  exact fun ctx hc args ha => source_not_stuck (policyResult ctx hc args ha)

#print axioms check_sound
#print axioms checked_refines
#print axioms checked_suffix
#print axioms refining_suffix

end MerlinVerify.Check.PolicyModule
