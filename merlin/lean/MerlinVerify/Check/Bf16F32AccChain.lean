/-
Composition after the f32-accumulation policy boundary. The last program of a checked downstream
suffix refines the contract reference `P(before)`, never the original 16-bit program: a policy
is not monotone, so it is not pushed through earlier stages. A chain segment ends where the next
numerics-changing boundary begins; that boundary needs its own policy check.
-/
import MerlinVerify.Check.Bf16F32Acc
import MerlinVerify.Check.PolicyW8A8Chain

namespace MerlinVerify.Check.Bf16F32AccChain

open MerlinVerify Rules.Bf16F32AccReference

def last (after : Module) (rest : List Module) : Module := (after :: rest).getLast (List.cons_ne_nil _ _)

def suffixes (ib : Nat) (reference after : Module) (rest : List Module) : Bool :=
  (Module.definedNames reference).all fun fn =>
    match TypedModule.signature? ib reference fn with
    | some (args, results) => PolicyW8A8Chain.suffixCheck ib args results fn after rest
    | none => false

/-- Conditional segment claim: the end of the suffix refines the contract reference wherever that
reference is modelled, on the declared typed domain. -/
def Correct (ib : Nat) (before after : Module) (rest : List Module) : Prop :=
  ∃ hints reference, expand hints before = some reference ∧
    ∀ fn ∈ Module.definedNames reference, ∃ args results,
      TypedModule.signature? ib reference fn = some (args, results) ∧
      FuncRefinesC Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
        (fun values => HasTys ib values args ∧ ResultsDefined values) (last after rest) fn reference fn

def check (ib : Nat) (before after : Module) (rest : List Module) : Bool :=
  match hints? before after with
  | none => false
  | some hints =>
    PolicyModule.check (policy hints) before after &&
      match expand hints before with
      | none => false
      | some reference => suffixes ib reference after rest

theorem check_sound {ib : Nat} {before after : Module} {rest : List Module}
    (h : check ib before after rest = true) : Correct ib before after rest := by
  unfold check at h
  split at h
  · cases h
  · next hints _ =>
    simp only [Bool.and_eq_true] at h
    obtain ⟨matched, suffix⟩ := h
    cases expansion : expand hints before with
    | none => simp [expansion] at suffix
    | some reference =>
      simp only [expansion, suffixes, List.all_eq_true] at suffix
      refine ⟨hints, reference, expansion, ?_⟩
      intro fn member
      have post := suffix fn member
      split at post
      · next args results signature =>
        exact ⟨args, results, signature,
          PolicyModule.checked_suffix (policy := policy hints) matched expansion member
            (PolicyW8A8Chain.suffixCheck_sound post)⟩
      · cases post

/-- Strict segment claim: additionally, the original source is defined on the typed domain and
the end of the suffix refines the contract reference everywhere on it. -/
def Qualified (ib : Nat) (before after : Module) (rest : List Module) : Prop :=
  Bf16F32Acc.Qualified ib before (last after rest)

def qualify (ib : Nat) (before after : Module) (rest : List Module) : Bool :=
  match hints? before after with
  | none => false
  | some hints =>
    PolicyModule.check (policy hints) before after &&
      match expand hints before with
      | none => false
      | some reference =>
        (Module.definedNames reference).all (Bf16F32Acc.domainCheck ib before reference) &&
          suffixes ib reference after rest

theorem qualify_sound {ib : Nat} {before after : Module} {rest : List Module}
    (h : qualify ib before after rest = true) : Qualified ib before after rest := by
  unfold qualify at h
  split at h
  · cases h
  · next hints _ =>
    simp only [Bool.and_eq_true] at h
    obtain ⟨matched, checked⟩ := h
    cases expansion : expand hints before with
    | none => simp [expansion] at checked
    | some reference =>
      simp only [expansion, Bool.and_eq_true] at checked
      obtain ⟨domains, suffix⟩ := checked
      have functions := Bf16F32Acc.domains_sound matched expansion domains
      simp only [suffixes, List.all_eq_true] at suffix
      refine ⟨hints, reference, expansion, ?_⟩
      intro fn member
      obtain ⟨args, results, sourceSignature, referenceSignature, sourceDefined, refined⟩ :=
        functions fn member
      refine ⟨args, results, sourceSignature, referenceSignature, sourceDefined, ?_⟩
      apply PolicyModule.refining_suffix refined
      have post := suffix fn member
      simp only [referenceSignature] at post
      exact PolicyW8A8Chain.suffixCheck_sound post

#print axioms check_sound
#print axioms qualify_sound

end MerlinVerify.Check.Bf16F32AccChain
