/- Actual-entry rounded-real trig bounds. Acceptance checks live finite operands, exact keys,
SSA bindings and structured regions under the separately named mathematical interpretation. -/
import MerlinVerify.Check.PolicyW8A8TrigDefined
import MerlinVerify.Check.PolicyW8A8RotaryPrefix
import MerlinVerify.Check.DefinedContextPrefix

namespace MerlinVerify.Check.PolicyW8A8TrigPrefix

open MerlinVerify Rules Definedness PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8TrigScalar.allowed

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  PolicyW8A8TrigDefined.book ib

def infer (ib : Nat) (Γ : Refined.FEnv Info) (ops : List Op) : Option (Refined.FEnv Info) :=
  Contextual.inferPrefix (book ib) Γ ops

/-- Prefix length is only a request. The checker rechecks that exact prefix from its actual
entry binders and declared input domain, including all SSA lookups and operation facts. -/
def entry (ib : Nat) (m : Module) (fn : String) (types : List VTy)
    (ranges : List (Option IndexRanges.Interval)) (caps : List (Option Nat)) (count : Nat) :
    Option (Refined.FEnv Info) := do
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
            obtain ⟨actual, executed, finalHolds⟩ := Contextual.inferPrefix_sound (book ib)
              _ _ _ ctx call _ checked initial width permitted
            exact ⟨function, arguments, ops, _, actual, rfl, body, bounded,
              by simpa only [List.append_nil] using (bindAll_ok (env := []) countArgs),
              executed, finalHolds⟩
          · cases checked
      · cases checked

abbrev bound := PolicyW8A8RotaryPrefix.bound

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
  obtain ⟨value, fetched, initialized, finite⟩ := PolicyW8A8RotaryPrefix.bound_sound demand holds
  exact ⟨function, arguments, ops, initial, actual, value, found, body, bounded, binding, executed,
    fetched, initialized, finite⟩

/-- The same mathematical model and accepted input contract have a joint witness.
This does not establish native fidelity or execution past a checked prefix. -/
theorem domain_inhabited {ib : Nat} {types : List VTy} {ranges : List (Option IndexRanges.Interval)}
    {caps : List (Option Nat)} (checked : PolicyW8A8RotaryDomain.check ib types ranges caps = true) :
    ∃ ctx args, ctx.indexBits = ib ∧ PolicyW8A8TrigScalar.allowed ctx
      (fun _ _ => .stuck "uninterpreted call") ∧ PolicyW8A8RotaryDomain.Domain ib types ranges caps args := by
  obtain ⟨args, inputs⟩ := PolicyW8A8RotaryDomain.check_inhabited checked
  refine ⟨W8A8RoundedReal.context W8A8IEEEPrimitives.unsupported
    (fun _ _ => .stuck "uninterpreted external") ib, args, rfl, ?_, inputs⟩
  exact ⟨W8A8IEEEPrimitives.unsupported, rfl⟩


end MerlinVerify.Check.PolicyW8A8TrigPrefix
