/- Composition after a source-derived numerical policy boundary. The final program refines
P(original), not the original floating program. Earlier stages cannot be moved through P. -/
import MerlinVerify.Check.PolicyW8A8Module

namespace MerlinVerify.Check.PolicyW8A8Chain

open MerlinVerify

/-- Universal C1 remains sufficient. Typed normalization additionally requires this exact input
domain, not merely a target annotation or a similarly named function. -/
def stepCheck (ib : Nat) (args results : List VTy) (fn : String) (before after : Module) : Bool :=
  c1Check (synthC1 before after fn) ||
    (decide (fn ∈ Module.definedNames before) &&
      decide (TypedModule.signature? ib before fn = some (args, results)) &&
      TypedModule.check ib before after)

theorem stepCheck_sound {ib : Nat} {args results : List VTy} {fn : String} {before after : Module}
    (h : stepCheck ib args results fn before after = true) :
    FuncRefinesC Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
      (fun values => HasTys ib values args ∧ ResultsDefined values) after fn before fn := by
  simp only [stepCheck, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with universal | ⟨⟨member, signature⟩, typed⟩
  · exact fun ctx _ values _ => c1Check_sound (synthC1 before after fn) universal ctx trivial values trivial
  · exact fun ctx hc values hv => TypedModule.check_sound typed member signature ctx hc values hv.1

/-- Check every actual downstream boundary, including outside-body changes, on the common
typed input domain. Each intermediate program is the previous target and the next source. -/
def suffixCheck (ib : Nat) (args results : List VTy) (fn : String) : Module → List Module → Bool
  | _, [] => true
  | before, after :: rest =>
    stepCheck ib args results fn before after &&
      (outsideChanges before after).unverified.isEmpty && suffixCheck ib args results fn after rest

theorem suffixCheck_sound {ib : Nat} {args results : List VTy} {fn : String}
    {before : Module} {rest : List Module}
    (h : suffixCheck ib args results fn before rest = true) :
    ChainRefinesC Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
      (fun values => HasTys ib values args ∧ ResultsDefined values) fn (before :: rest) := by
  induction rest generalizing before with
  | nil => trivial
  | cons after rest ih =>
    simp only [suffixCheck, Bool.and_eq_true] at h
    exact ⟨stepCheck_sound h.1.1, ih h.2⟩

/-- The intended claim retains original-source definedness and exact policy-reference identity
and replaces only the target with the end of the checked suffix. -/
def Qualified (config : Rules.W8A8Reference.Config) (ib : Nat)
    (before after : Module) (rest : List Module) : Prop :=
  PolicyW8A8Module.Qualified config ib before
    ((after :: rest).getLast (List.cons_ne_nil _ _))

def check (config : Rules.W8A8Reference.Config) (ib : Nat)
    (before after : Module) (rest : List Module) : Bool :=
  PolicyW8A8Module.qualify config ib before after &&
    match Rules.W8A8Reference.expand config before with
    | none => false
    | some reference => (Module.definedNames reference).all fun fn =>
      match TypedModule.signature? ib reference fn with
      | some (args, results) => suffixCheck ib args results fn after rest
      | none => false

theorem check_sound {config : Rules.W8A8Reference.Config} {ib : Nat}
    {before after : Module} {rest : List Module}
    (h : check config ib before after rest = true) : Qualified config ib before after rest := by
  simp only [check, Bool.and_eq_true] at h
  obtain ⟨reference, expanded, qualified⟩ := PolicyW8A8Module.qualify_sound h.1
  simp only [expanded, List.all_eq_true] at h
  refine ⟨reference, expanded, ?_⟩
  intro fn member
  obtain ⟨args, results, sourceSignature, referenceSignature, sourceDefined, refined⟩ := qualified fn member
  refine ⟨args, results, sourceSignature, referenceSignature, sourceDefined, ?_⟩
  apply PolicyModule.refining_suffix refined
  have post := h.2 fn member
  simp only [referenceSignature] at post
  exact suffixCheck_sound post

#print axioms stepCheck_sound
#print axioms suffixCheck_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8Chain
