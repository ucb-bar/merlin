/- Closed captures have no foreign-result premises. This still proves only operational
typing: a typed poison is permitted and stuck/UB outcomes remain possible. -/
import MerlinVerify.Rules.ScalarSquareTypingReady

namespace MerlinVerify.Rules.ScalarSquareTyping
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops ScalarSquareClaims

def closedCheck (ib : Nat) (source : Module) : Bool :=
  programCheck ib source && source.funcs.all (fun f => !f.isDecl)

theorem no_external_results {source : Module}
    (closed : source.funcs.all (fun f => !f.isDecl) = true) (ib : Nat) (ctx : Ctx) :
    ExternalResults ib source ctx := by
  intro name f sig found declaration
  have h := List.all_eq_true.1 closed f (List.mem_of_find?_eq_some found)
  simp [declaration] at h

/-- Soundness for an executable closed-capture certificate: no operational typing
hypothesis, numerical law, external contract, progress assertion, or approval is assumed. -/
theorem closedCheck_sound {ib : Nat} {source : Module} (checked : closedCheck ib source = true)
    (M : FloatModel) : SourceTyping source (Family M ib) ib := by
  simp only [closedCheck, Bool.and_eq_true] at checked
  exact source_typing checked.1 M (fun ctx _ => no_external_results checked.2 ib ctx)

theorem closed_results {ib : Nat} {source : Module} (checked : closedCheck ib source = true)
    (ctx : Ctx) (hib : ctx.indexBits = ib) (fuel : Nat) :
    CallTyped ib (signatures ib source) (evalFunc opSem ctx source fuel) := by
  simp only [closedCheck, Bool.and_eq_true] at checked
  exact calls_preserve checked.1 ctx hib (no_external_results checked.2 ib ctx) fuel

/-- Operational typing is discharged; the numerical law and source definedness remain
separate premises. This theorem cannot be instantiated for the refuted native law. -/
theorem strict_source_with_checked_typing {ib : Nat} {source target : Module}
    (typed : closedCheck ib source = true)
    (boundary : Check.ScalarSquareProgram.check source target = true) (M : FloatModel)
    (numerical : ScalarSquareComposition.Numerical (Family M ib)) (entry : String)
    (contexts : ContextDomain (Family M ib) ib) (domain : EntryDomain source ib entry)
    (defined : FuncNotStuck opSem (Family M ib) (Inputs source ib entry) source entry) :
    StrictSourcePreservation source target (Family M ib) ib entry :=
  strict_source_preservation boundary (Family M ib) ib numerical (closedCheck_sound typed M)
    entry contexts domain defined

#print axioms strict_source_with_checked_typing
#print axioms closedCheck_sound
#print axioms closed_results
end MerlinVerify.Rules.ScalarSquareTyping
