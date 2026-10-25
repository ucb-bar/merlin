/-
Checker for `lower_bf16_matmul_f32acc` against the f32-accumulation contract (review candidate,
`Rules.Bf16F32AccContract`). The policy reference is derived from the source alone; the only
choices recovered from the candidate output are per-site permitted fast-math flags and
semantics-free annotations, and each is re-validated by the reference transformer.

`check_sound`: acceptance means the actual output refines the contract's reference wherever that
reference is modelled, for every float model, external environment and input. `qualify_sound`
adds the strict, typed-domain claim from independently checked source definedness and reference
support. What the reference computes is stated by `Rules.Bf16F32AccTensor`.
-/
import MerlinVerify.Rules.Bf16F32AccReference
import MerlinVerify.Check.TypedModule
import MerlinVerify.Check.DefinedExtended

namespace MerlinVerify.Check.Bf16F32Acc

open MerlinVerify Rules.Bf16F32AccReference

/-- Some admissible hint list expands the source to a reference the output refines. -/
def Correct (before after : Module) : Prop :=
  ∃ hints, PolicyModule.Correct (policy hints) before after

def check (before after : Module) : Bool :=
  match hints? before after with
  | none => false
  | some hints => PolicyModule.check (policy hints) before after

theorem check_sound {before after : Module} (h : check before after = true) : Correct before after := by
  unfold check at h
  split at h
  · cases h
  · next hints _ => exact ⟨hints, PolicyModule.check_sound h⟩

def Qualified (ib : Nat) (before after : Module) : Prop :=
  ∃ hints reference, expand hints before = some reference ∧
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

def qualify (ib : Nat) (before after : Module) : Bool :=
  match hints? before after with
  | none => false
  | some hints =>
    PolicyModule.check (policy hints) before after &&
      match expand hints before with
      | none => false
      | some reference => (Module.definedNames reference).all (domainCheck ib before reference)

/-- The per-function obligations shared by the single-boundary and chain qualifications. -/
theorem domains_sound {ib : Nat} {before after reference : Module} {hints : List SiteHint}
    (matched : PolicyModule.check (policy hints) before after = true)
    (expansion : expand hints before = some reference)
    (domains : (Module.definedNames reference).all (domainCheck ib before reference) = true) :
    ∀ fn ∈ Module.definedNames reference, ∃ argTys resTys,
      TypedModule.signature? ib before fn = some (argTys, resTys) ∧
      TypedModule.signature? ib reference fn = some (argTys, resTys) ∧
      FuncDefined Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args) before fn ∧
      FuncRefines Sem.Ops.opSem (fun ctx => ctx.indexBits = ib)
        (fun args => HasTys ib args argTys ∧ ResultsDefined args) after fn reference fn := by
  simp only [List.all_eq_true] at domains
  intro fn member
  have domain := domains fn member
  unfold domainCheck at domain
  split at domain
  · next a r a' r' hs hr =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at domain
    obtain ⟨⟨⟨rfl, rfl⟩, sourceDefined⟩, referenceSupported⟩ := domain
    refine ⟨a, r, hs, hr, DefinedExtended.check_sound sourceDefined, ?_⟩
    apply PolicyModule.checked_refines (policy := policy hints) matched expansion member
    intro ctx hib args hargs
    exact supportCheck_sound (merlinBook ib) referenceSupported ctx hib args hargs.1
  · cases domain

theorem qualify_sound {ib : Nat} {before after : Module} (h : qualify ib before after = true) :
    Qualified ib before after := by
  unfold qualify at h
  split at h
  · cases h
  · next hints _ =>
    simp only [Bool.and_eq_true] at h
    obtain ⟨matched, domains⟩ := h
    cases expansion : expand hints before with
    | none => simp [expansion] at domains
    | some reference =>
      simp only [expansion] at domains
      exact ⟨hints, reference, expansion, domains_sound matched expansion domains⟩

#print axioms check_sound
#print axioms qualify_sound

end MerlinVerify.Check.Bf16F32Acc
