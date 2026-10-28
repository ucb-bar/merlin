/- Complete-module correspondence to the source-derived W8A8 draft policy, independently
of successful source execution. Matching every body, declaration, signature and outer
attribute does not establish that the policy is independently approved or that execution
succeeds. Those stronger obligations remain separate. -/
import MerlinVerify.Check.PolicyParametric
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8SourceAlpha

namespace MerlinVerify.Check.PolicyW8A8Correspondence

open MerlinVerify Rules

local notation "sourceSem" => W8A8SourceSemantics.opSem

/-- The entire final program has exactly the outcomes of this fixed source-derived reference.
There is no hypothesis about the original program executing successfully. -/
def Correct (config : W8A8Reference.Config) (before after : Module) (suffix : List Module) : Prop :=
  ∃ reference, W8A8Reference.expand config before = some reference ∧
    ∀ ctx fn args, denote sourceSem ctx ((after :: suffix).getLast (List.cons_ne_nil _ _)) fn args =
      denote sourceSem ctx reference fn args

def suffixCheck : Module → List Module → Bool
  | _, [] => true
  | before, after :: rest => ParametricModule.check before after && suffixCheck after rest

theorem suffixCheck_sound {before : Module} {suffix : List Module}
    (checked : suffixCheck before suffix = true) :
    ∀ ctx fn args, denote sourceSem ctx ((before :: suffix).getLast (List.cons_ne_nil _ _)) fn args =
      denote sourceSem ctx before fn args := by
  induction suffix generalizing before with
  | nil => intros; rfl
  | cons after rest ih =>
    simp only [suffixCheck, Bool.and_eq_true] at checked
    intro ctx fn args
    exact (ih checked.2 ctx fn args).trans
      (ParametricModule.check_sound W8A8SourceAlpha.laws checked.1 ctx fn args).symm

def check (config : W8A8Reference.Config) (before after : Module) (suffix : List Module) : Bool :=
  PolicyParametric.check (W8A8Reference.policy config) before after && suffixCheck after suffix

theorem check_sound {config : W8A8Reference.Config} {before after : Module} {suffix : List Module}
    (checked : check config before after suffix = true) : Correct config before after suffix := by
  simp only [check, Bool.and_eq_true] at checked
  obtain ⟨reference, expanded, matched⟩ := PolicyParametric.check_sound W8A8SourceAlpha.laws checked.1
  exact ⟨reference, expanded, fun ctx fn args => (suffixCheck_sound checked.2 ctx fn args).trans
    (matched ctx fn args)⟩

/-- A successful reference execution transfers every returned value, under the same context
and input. Neither original-source definedness nor ten numerical laws are required here. -/
theorem successful_reference {config : W8A8Reference.Config} {before after reference : Module}
    {suffix : List Module} {ctx : Ctx} {fn : String} {args results : List RVal}
    (checked : check config before after suffix = true)
    (expanded : W8A8Reference.expand config before = some reference)
    (success : denote sourceSem ctx reference fn args = .ok results) :
    denote sourceSem ctx ((after :: suffix).getLast (List.cons_ne_nil _ _)) fn args = .ok results := by
  obtain ⟨actual, expansion, matched⟩ := check_sound checked
  have same : actual = reference := Option.some.inj (expansion.symm.trans expanded)
  subst actual
  exact (matched ctx fn args).trans success

/-- Successful, nonpoison execution on a whole input domain is a stronger conditional
corollary. The reference-definedness premise stays visible rather than being inferred from
structural correspondence or a human review flag. -/
theorem defined_reference {config : W8A8Reference.Config} {before after reference : Module}
    {suffix : List Module} {contexts : Ctx → Prop} {domain : List RVal → Prop} {fn : String}
    (checked : check config before after suffix = true)
    (expanded : W8A8Reference.expand config before = some reference)
    (defined : FuncDefined sourceSem contexts domain reference fn) :
    FuncDefined sourceSem contexts domain
      ((after :: suffix).getLast (List.cons_ne_nil _ _)) fn := by
  intro ctx permitted args inputs
  obtain ⟨results, success, initialized⟩ := defined ctx permitted args inputs
  exact ⟨results, successful_reference checked expanded success, initialized⟩

#print axioms suffixCheck_sound
#print axioms check_sound
#print axioms successful_reference
#print axioms defined_reference

end MerlinVerify.Check.PolicyW8A8Correspondence
