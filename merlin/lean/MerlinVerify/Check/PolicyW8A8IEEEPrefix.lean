/- The actual-entry/prefix bound theorem specialized to the explicit numerical family.
No source/dynamic law record or arithmetic Agreement remains a theorem hypothesis. -/
import MerlinVerify.Check.PolicyW8A8IEEEQualified
import MerlinVerify.Check.PolicyW8A8RotaryPrefix

namespace MerlinVerify.Check.PolicyW8A8IEEEPrefix

open MerlinVerify Rules Definedness PolicyW8A8RotaryFacts

theorem entry_bound {ib : Nat} {m : Module} {fn : String} {types : List VTy}
    {ranges : List (Option IndexRanges.Interval)} {caps : List (Option Nat)} {count : Nat}
    {final : Refined.FEnv Info} (checked : PolicyW8A8RotaryPrefix.entry ib m fn types ranges caps count = some final)
    {id : ValueId} {upper : Nat} (demand : PolicyW8A8RotaryPrefix.bound final id upper = true)
    {ctx : Ctx} {call : CallSem} {args : List RVal}
    (width : ctx.indexBits = ib) (model : PolicyW8A8IEEEQualified.ModelContext ctx)
    (domain : PolicyW8A8RotaryDomain.Domain ib types ranges caps args) :
    ∃ function arguments ops initial actual value, m.func? fn = some function ∧
      function.body = .mk [.mk arguments ops] ∧ count ≤ ops.length ∧
      MerlinVerify.Env.bindAll [] arguments args = .ok initial ∧
      execOps W8A8SourceSemantics.opSem ctx call initial (ops.take count) = .ok actual ∧
      actual.get? id = some value ∧ value.Defined ∧ PolicyW8A8RotaryFacts.Holds (some upper) value :=
  PolicyW8A8RotaryPrefix.entry_bound checked demand width (PolicyW8A8IEEEQualified.context_ok model) domain

#print axioms entry_bound

end MerlinVerify.Check.PolicyW8A8IEEEPrefix
