/- Prefix certificates execute the actual checked operations in order and preserve actual
SSA binding facts. They make no claim about operations after the first unsupported producer. -/
import MerlinVerify.Check.PolicyW8A8RotaryDefined
import MerlinVerify.Check.Step

namespace MerlinVerify.Check.PolicyW8A8RotaryPrefix

open MerlinVerify Rules Definedness PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def infer (ib : Nat) (Γ : Refined.FEnv Info) : List Op → Option (Refined.FEnv Info)
  | [] => some Γ
  | op :: rest => do
    if op.isTerminator = false ∧ op.successors = [] then
      let inputs ← op.operands.mapM (fun id => Γ.lookup id)
      let outputs ← (PolicyW8A8RotaryDefined.book ib).rule Γ op inputs
      if outputs.length == op.results.length then
        infer ib ((op.results.map (·.id)).zip outputs ++ Γ) rest
      else none
    else none

theorem infer_sound (ib : Nat) : ∀ Γ ops final ctx call env,
    infer ib Γ ops = some final → Refined.EnvHolds model ib Γ env →
    ctx.indexBits = ib → allowed ctx call →
    ∃ actual, execOps sourceSem ctx call env ops = .ok actual ∧ Refined.EnvHolds model ib final actual
  | Γ, [], final, _ctx, _call, env, checked, valid, _width, _permitted => by
    cases checked
    exact ⟨env, rfl, valid⟩
  | Γ, op :: rest, final, ctx, call, env, checked, valid, width, permitted => by
    simp only [infer] at checked
    split at checked
    · next header =>
      cases fetched : op.operands.mapM (fun id => Γ.lookup id) with
      | none => simp [fetched] at checked
      | some inputs =>
        simp only [fetched, bind, Option.bind] at checked
        cases selected : (PolicyW8A8RotaryDefined.book ib).rule Γ op inputs with
        | none => simp [selected] at checked
        | some outputs =>
          simp only [selected] at checked
          split at checked
          · next arity =>
            obtain ⟨args, bound, holds⟩ := valid.getAll op.operands inputs fetched
            obtain ⟨values, executed, results⟩ :=
              (PolicyW8A8RotaryDefined.book ib).sound Γ op inputs outputs selected ctx call env args
                width permitted valid holds
            have count : op.results.length = values.length :=
              (eq_of_beq arity).symm.trans (List.Forall₂.length_eq results).symm
            obtain ⟨actual, remaining, finalHolds⟩ :=
              infer_sound ib _ rest final ctx call _ checked (valid.extend _ values outputs results) width permitted
            refine ⟨actual, ?_, finalHolds⟩
            rw [execOps_cons_step sourceSem ctx call env op rest header.1, stepOp, bound, Outcome.bind_ok,
              executed, Outcome.bind_ok, bindAll_ok count, Outcome.bind_ok]
            exact remaining
          · cases checked
    · cases checked

theorem tracked_defined {ib : Nat} {Γ : Refined.FEnv Info} {ops : List Op} {final : Refined.FEnv Info}
    (checked : infer ib Γ ops = some final) {ctx : Ctx} {call : CallSem} {env : Env}
    (valid : Refined.EnvHolds model ib Γ env) (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    {id : ValueId} {fact : Refined.Fact Info} (found : final.lookup id = some fact)
    (initialized : fact.base.initialized = true) :
    ∃ actual value, execOps sourceSem ctx call env ops = .ok actual ∧ actual.get? id = some value ∧ value.Defined := by
  obtain ⟨actual, executed, finalHolds⟩ := infer_sound ib Γ ops final ctx call env checked valid width permitted
  obtain ⟨value, lookup, holds⟩ := finalHolds id fact found
  exact ⟨actual, value, executed, lookup, holds.1.2 initialized⟩

/-- A diagnostic chooses only a prefix length. Acceptance independently rechecks that exact
prefix, the actual entry binders, input facts and every source operation in order. -/
def entry (ib : Nat) (m : Module) (fn : String) (types : List VTy)
    (ranges : List (Option IndexRanges.Interval)) (caps : List (Option Nat)) (count : Nat) : Option (Refined.FEnv Info) := do
  let function ← m.func? fn
  if function.isDecl then none else
    match function.body with
    | .mk [.mk arguments ops] => do
      let facts ← PolicyW8A8RotaryDomain.inputFacts types ranges caps
      if arguments.mapM (fun arg => vtyOf ib arg.ty) == some (facts.map (·.base.ty)) ∧ count ≤ ops.length ∧
          PolicyW8A8RotaryDefined.validInputs ib types ranges caps = true then
        infer ib ((arguments.map (·.id)).zip facts) (ops.take count)
      else none
    | _ => none

theorem entry_sound {ib : Nat} {m : Module} {fn : String} {types : List VTy}
    {ranges : List (Option IndexRanges.Interval)} {caps : List (Option Nat)} {count : Nat} {final : Refined.FEnv Info}
    (checked : entry ib m fn types ranges caps count = some final) {ctx : Ctx} {call : CallSem} {args : List RVal}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    (domain : PolicyW8A8RotaryDomain.Domain ib types ranges caps args) :
    ∃ function arguments ops initial actual, m.func? fn = some function ∧
      function.body = .mk [.mk arguments ops] ∧ count ≤ ops.length ∧
      MerlinVerify.Env.bindAll [] arguments args = .ok initial ∧
      execOps sourceSem ctx call initial (ops.take count) = .ok actual ∧ Refined.EnvHolds model ib final actual := by
  unfold entry at checked
  cases found : m.func? fn with
  | none => simp [found] at checked
  | some function =>
    simp only [found, bind, Option.bind] at checked
    split at checked
    · cases checked
    · split at checked
      · next arguments ops body =>
        cases selected : PolicyW8A8RotaryDomain.inputFacts types ranges caps with
        | none => simp [selected] at checked
        | some facts =>
          simp only [selected] at checked
          split at checked
          · next conditions =>
            obtain ⟨argumentTypes, bounded, _inhabited⟩ := conditions
            have argumentTypes := eq_of_beq argumentTypes
            obtain ⟨actualFacts, actualSelected, holds⟩ := domain
            have equal : actualFacts = facts := Option.some.inj (actualSelected.symm.trans selected)
            subst actualFacts
            have countArgs : arguments.length = args.length := by
              have length := optionMapM_length _ _ _ argumentTypes
              simp only [List.length_map] at length
              exact length.symm.trans (List.Forall₂.length_eq holds).symm
            have empty : Refined.EnvHolds model ib [] [] := fun _ _ absent => by cases absent
            have initial := empty.extend (arguments.map (·.id)) args facts holds
            simp only [List.append_nil] at initial
            obtain ⟨actual, executed, finalHolds⟩ := infer_sound ib _ _ _ ctx call _ checked initial width permitted
            exact ⟨function, arguments, ops, _, actual, rfl, body, bounded,
              by simpa only [List.append_nil] using (bindAll_ok (env := []) countArgs),
              executed, finalHolds⟩
          · cases checked
      · cases checked

theorem entry_tracked_defined {ib : Nat} {m : Module} {fn : String} {types : List VTy}
    {ranges : List (Option IndexRanges.Interval)} {caps : List (Option Nat)} {count : Nat} {final : Refined.FEnv Info}
    (checked : entry ib m fn types ranges caps count = some final) {ctx : Ctx} {call : CallSem} {args : List RVal}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    (domain : PolicyW8A8RotaryDomain.Domain ib types ranges caps args)
    {id : ValueId} {fact : Refined.Fact Info} (found : final.lookup id = some fact)
    (initialized : fact.base.initialized = true) :
    ∃ function arguments ops initial actual value, m.func? fn = some function ∧
      function.body = .mk [.mk arguments ops] ∧ count ≤ ops.length ∧
      MerlinVerify.Env.bindAll [] arguments args = .ok initial ∧
      execOps sourceSem ctx call initial (ops.take count) = .ok actual ∧
      actual.get? id = some value ∧ value.Defined := by
  obtain ⟨function, arguments, ops, initial, actual, declared, body, bounded, bound, executed, holds⟩ :=
    entry_sound checked width permitted domain
  obtain ⟨value, lookup, facts⟩ := holds id fact found
  exact ⟨function, arguments, ops, initial, actual, value, declared, body, bounded, bound,
    executed, lookup, facts.1.2 initialized⟩

#print axioms infer_sound
#print axioms tracked_defined
#print axioms entry_sound
#print axioms entry_tracked_defined

/-- Requested facts use live inferred bounds. Dropping to an initialized/unknown
fallback cannot satisfy a numerical demand. -/
def bound (final : Refined.FEnv Info) (id : ValueId) (upper : Nat) : Bool :=
  match final.lookup id with
  | some fact => fact.base.initialized && fact.info.exponent.any (· ≤ upper)
  | none => false

theorem bound_sound {ib : Nat} {final : Refined.FEnv Info} {env : Env} {id : ValueId} {upper : Nat}
    (checked : bound final id upper = true) (valid : Refined.EnvHolds model ib final env) :
    ∃ value, env.get? id = some value ∧ value.Defined ∧ PolicyW8A8RotaryFacts.Holds (some upper) value := by
  unfold bound at checked
  cases found : final.lookup id with
  | none => simp [found] at checked
  | some fact =>
    simp only [found, Bool.and_eq_true] at checked
    obtain ⟨value, fetched, holds⟩ := valid id fact found
    refine ⟨value, fetched, holds.1.2 checked.1, ?_⟩
    cases capFound : fact.info.exponent with
    | none => simp [capFound] at checked
    | some valueCap =>
      have smaller : valueCap ≤ upper := by
        simpa only [capFound, Option.any_some, decide_eq_true_eq] using checked.2
      exact PolicyW8A8RotaryFacts.weaken (show exponentImplies (some valueCap) (some upper) = true from
        decide_eq_true smaller) (by simpa only [capFound] using holds.2.2)

theorem entry_bound {ib : Nat} {m : Module} {fn : String} {types : List VTy}
    {ranges : List (Option IndexRanges.Interval)} {caps : List (Option Nat)} {count : Nat}
    {final : Refined.FEnv Info} (checked : entry ib m fn types ranges caps count = some final)
    {id : ValueId} {upper : Nat} (demand : bound final id upper = true)
    {ctx : Ctx} {call : CallSem} {args : List RVal}
    (width : ctx.indexBits = ib) (permitted : allowed ctx call)
    (domain : PolicyW8A8RotaryDomain.Domain ib types ranges caps args) :
    ∃ function arguments ops initial actual value, m.func? fn = some function ∧
      function.body = .mk [.mk arguments ops] ∧ count ≤ ops.length ∧
      MerlinVerify.Env.bindAll [] arguments args = .ok initial ∧
      execOps sourceSem ctx call initial (ops.take count) = .ok actual ∧ actual.get? id = some value ∧
      value.Defined ∧ PolicyW8A8RotaryFacts.Holds (some upper) value := by
  obtain ⟨function, arguments, ops, initial, actual, found, body, bounded, binding, executed, holds⟩ :=
    entry_sound checked width permitted domain
  obtain ⟨value, fetched, initialized, finite⟩ := bound_sound demand holds
  exact ⟨function, arguments, ops, initial, actual, value, found, body, bounded, binding, executed,
    fetched, initialized, finite⟩

#print axioms bound_sound
#print axioms entry_bound

end MerlinVerify.Check.PolicyW8A8RotaryPrefix
