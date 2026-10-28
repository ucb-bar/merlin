/- Whole-function W8A8 draft qualification. The conditional reference match, defined original
source, and modelled policy reference are independent obligations. Specification review is not
inferred from any Boolean or native test; the public driver keeps that obligation explicit. -/
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Check.TypedModule
import MerlinVerify.Check.DefinedExtended

namespace MerlinVerify.Check.PolicyW8A8Module

open MerlinVerify

def check (config : Rules.W8A8Reference.Config) (before after : Module) : Bool :=
  PolicyModule.check (Rules.W8A8Reference.policy config) before after

theorem check_sound {config : Rules.W8A8Reference.Config} {before after : Module}
    (h : check config before after = true) :
    PolicyModule.Correct (Rules.W8A8Reference.policy config) before after :=
  PolicyModule.check_sound h

def Qualified (config : Rules.W8A8Reference.Config) (ib : Nat) (before after : Module) : Prop :=
  ∃ reference, Rules.W8A8Reference.expand config before = some reference ∧
    ∀ fn ∈ Module.definedNames reference, ∃ argTys resTys,
      TypedModule.signature? ib before fn = some (argTys, resTys) ∧
      TypedModule.signature? ib reference fn = some (argTys, resTys) ∧
      FuncDefined Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args) before fn ∧
      FuncRefines Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args) after fn reference fn

def domainCheck (ib : Nat) (before reference : Module) (fn : String) : Bool :=
  match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
  | some (a, r), some (a', r') =>
    decide (a = a' ∧ r = r') && DefinedExtended.check ib before fn a r &&
      supportCheck (merlinBook ib) reference fn a r
  | _, _ => false

def qualify (config : Rules.W8A8Reference.Config) (ib : Nat) (before after : Module) : Bool :=
  check config before after &&
    match Rules.W8A8Reference.expand config before with
    | none => false
    | some reference => (Module.definedNames reference).all (domainCheck ib before reference)

theorem qualify_sound {config : Rules.W8A8Reference.Config} {ib : Nat} {before after : Module}
    (h : qualify config ib before after = true) : Qualified config ib before after := by
  simp only [qualify, Bool.and_eq_true] at h
  obtain ⟨matched, domains⟩ := h
  cases expansion : Rules.W8A8Reference.expand config before with
  | none => simp [expansion] at domains
  | some reference =>
    simp only [expansion, List.all_eq_true] at domains
    refine ⟨reference, expansion, ?_⟩
    intro fn member
    have domain := domains fn member
    unfold domainCheck at domain
    split at domain
    · next a r a' r' hs hr =>
      simp only [Bool.and_eq_true, decide_eq_true_eq] at domain
      obtain ⟨⟨⟨rfl, rfl⟩, sourceDefined⟩, referenceSupported⟩ := domain
      refine ⟨a, r, hs, hr, DefinedExtended.check_sound sourceDefined, ?_⟩
      apply PolicyModule.checked_refines matched expansion member
      intro ctx hib args hargs
      exact supportCheck_sound (merlinBook ib) referenceSupported ctx hib args hargs.1
    · cases domain

#print axioms check_sound
#print axioms qualify_sound

end MerlinVerify.Check.PolicyW8A8Module
