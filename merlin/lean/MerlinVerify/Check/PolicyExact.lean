/- Exact source-derived policy matching, generic in the program semantics. This route proves
equality of executions even for an additive source contract. Strict qualification independently
requires source definedness and support of the policy reference on the stated input domain. -/
import MerlinVerify.Check.PolicyModule
import MerlinVerify.Sem.Defined

namespace MerlinVerify.Check.PolicyExact

open MerlinVerify PolicyModule

def check (policy : Policy) (before after : Module) : Bool :=
  match policy.expand before with
  | none => false
  | some reference => ExactSyntax.modules reference after

/-- Matching the entire AST identifies executions under every semantics, context and input.
This theorem deliberately makes no claim that either execution is defined or modelled. -/
theorem check_sound {policy : Policy} {before after : Module}
    (checked : check policy before after = true) :
    ∃ reference, policy.expand before = some reference ∧
      ∀ sem ctx fn args, denote sem ctx after fn args = denote sem ctx reference fn args := by
  unfold check at checked
  cases expanded : policy.expand before with
  | none => simp [expanded] at checked
  | some reference =>
    have same := ExactSyntax.modules_sound (by simpa [expanded] using checked)
    subst reference
    exact ⟨after, rfl, fun _ _ _ _ => rfl⟩

def Qualified (sem : OpSem) (ctxOk : Ctx → Prop) (domain : List RVal → Prop)
    (policy : Policy) (before after : Module) (fn : String) : Prop :=
  ∃ reference, policy.expand before = some reference ∧
    FuncDefined sem ctxOk domain before fn ∧
    FuncNotStuck sem ctxOk domain reference fn ∧
    FuncRefines sem ctxOk domain after fn reference fn

/-- Source definedness and reference support are proofs quantified over the same context and
input domain. Neither is replaced by successful samples, matching syntax or a review flag. -/
theorem qualify_sound {policy : Policy} {before after reference : Module} {fn : String}
    {sem : OpSem} {ctxOk : Ctx → Prop} {domain : List RVal → Prop}
    (checked : check policy before after = true)
    (expanded : policy.expand before = some reference)
    (sourceDefined : FuncDefined sem ctxOk domain before fn)
    (referenceSupported : FuncNotStuck sem ctxOk domain reference fn) :
    Qualified sem ctxOk domain policy before after fn := by
  obtain ⟨actual, expansion, equal⟩ := check_sound checked
  have same : actual = reference := Option.some.inj (expansion.symm.trans expanded)
  subst actual
  refine ⟨reference, expanded, sourceDefined, referenceSupported, ?_⟩
  apply FuncRefinesC.strict ?_ referenceSupported
  intro ctx _ args _
  rw [equal sem ctx fn args]
  exact RefinesC.refl ResultsRefine.refl _

/-- A downstream proof uses the identical semantic interpretation and domain. The generic
theorem cannot import a frozen-semantics suffix theorem into a provisional source contract. -/
theorem qualified_suffix {policy : Policy} {before after : Module} {fn : String}
    {sem : OpSem} {ctxOk : Ctx → Prop} {domain : List RVal → Prop} {suffix : List Module}
    (qualified : Qualified sem ctxOk domain policy before after fn)
    (post : ChainRefinesC sem ctxOk domain fn (after :: suffix)) :
    ∃ reference, policy.expand before = some reference ∧
      FuncDefined sem ctxOk domain before fn ∧
      FuncRefines sem ctxOk domain
        ((after :: suffix).getLast (List.cons_ne_nil _ _)) fn reference fn := by
  obtain ⟨reference, expanded, defined, _, refined⟩ := qualified
  exact ⟨reference, expanded, defined, PolicyModule.refining_suffix refined post⟩

#print axioms check_sound
#print axioms qualify_sound
#print axioms qualified_suffix

end MerlinVerify.Check.PolicyExact
