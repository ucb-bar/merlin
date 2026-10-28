/- Source-derived policy validation through semantics-generic SSA equivalence. A renamed
compiler export cannot choose the policy reference or erase a separate domain obligation. -/
import MerlinVerify.Check.ParametricModule
import MerlinVerify.Check.PolicyExact

namespace MerlinVerify.Check.PolicyParametric

open MerlinVerify PolicyModule

def check (policy : Policy) (before after : Module) : Bool :=
  match policy.expand before with
  | none => false
  | some reference => ParametricModule.check reference after

theorem check_sound {sem : OpSem} (laws : ParametricAlpha.Laws sem)
    {policy : Policy} {before after : Module} (checked : check policy before after = true) :
    ∃ reference, policy.expand before = some reference ∧
      ∀ ctx fn args, denote sem ctx after fn args = denote sem ctx reference fn args := by
  unfold check at checked
  cases expanded : policy.expand before with
  | none => simp [expanded] at checked
  | some reference =>
    refine ⟨reference, rfl, ?_⟩
    exact fun ctx fn args => (ParametricModule.check_sound laws
      (by simpa [expanded] using checked) ctx fn args).symm

theorem qualify_sound {sem : OpSem} (laws : ParametricAlpha.Laws sem)
    {policy : Policy} {before after reference : Module} {fn : String}
    {ctxOk : Ctx → Prop} {domain : List RVal → Prop}
    (checked : check policy before after = true)
    (expanded : policy.expand before = some reference)
    (sourceDefined : FuncDefined sem ctxOk domain before fn)
    (referenceSupported : FuncNotStuck sem ctxOk domain reference fn) :
    PolicyExact.Qualified sem ctxOk domain policy before after fn := by
  obtain ⟨actual, expansion, equal⟩ := check_sound laws checked
  have same : actual = reference := Option.some.inj (expansion.symm.trans expanded)
  subst actual
  refine ⟨reference, expanded, sourceDefined, referenceSupported, ?_⟩
  apply FuncRefinesC.strict ?_ referenceSupported
  intro ctx _ args _
  rw [equal ctx fn args]
  exact RefinesC.refl ResultsRefine.refl _

#print axioms check_sound
#print axioms qualify_sound

end MerlinVerify.Check.PolicyParametric
