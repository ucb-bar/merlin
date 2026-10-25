/- Complete output equality to the source-derived contract under explicit operation semantics.
Unlike conditional refinement, this result does not discard a stuck source case. It says exactly
that both programs have the same execution, including errors; successful execution and the
numerical meaning of the emitted sites remain separate, independently proved obligations. -/
import MerlinVerify.Rules.Bf16F32AccReference
import MerlinVerify.Check.PolicyParametric

namespace MerlinVerify.Check.Bf16F32AccExact

open MerlinVerify Rules.Bf16F32AccReference

def check (before after : Module) : Bool :=
  match hints? before after with
  | none => false
  | some hints => PolicyParametric.check (policy hints) before after

/-- Every accepted complete output executes exactly its source-derived policy reference under
any interpretation satisfying SSA renaming and call locality. Neither the output nor a receipt
supplies an interpretation or a reference program. -/
theorem check_sound {sem : OpSem} (laws : ParametricAlpha.Laws sem)
    {before after : Module} (accepted : check before after = true) :
    ∃ hints reference, expand hints before = some reference ∧
      ∀ ctx fn args, denote sem ctx after fn args = denote sem ctx reference fn args := by
  unfold check at accepted
  split at accepted
  · cases accepted
  · next hints _ =>
    obtain ⟨reference, expanded, same⟩ := PolicyParametric.check_sound laws accepted
    exact ⟨hints, reference, expanded, same⟩

/-- Execution equality transfers actual successful, initialized reference executions; equality
alone never supplies such an execution. This theorem is generic in the complete interpreter. -/
theorem defined_execution {sem : OpSem} (laws : ParametricAlpha.Laws sem)
    {before after : Module} (accepted : check before after = true)
    {hints : List SiteHint} {reference : Module}
    (selected : hints? before after = some hints)
    (expanded : expand hints before = some reference)
    {ctx : Ctx} {fn : String} {args results : List RVal}
    (executed : denote sem ctx reference fn args = .ok results)
    (initialized : ResultsDefined results) :
    denote sem ctx after fn args = .ok results ∧ ResultsDefined results := by
  have checked : PolicyParametric.check (policy hints) before after = true := by
    simpa [check, selected] using accepted
  obtain ⟨actual, actualExpanded, same⟩ := PolicyParametric.check_sound laws checked
  have identical : actual = reference := Option.some.inj (actualExpanded.symm.trans expanded)
  subst actual
  exact ⟨(same ctx fn args).trans executed, initialized⟩

#print axioms check_sound
#print axioms defined_execution

end MerlinVerify.Check.Bf16F32AccExact
