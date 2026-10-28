/- Candidate arithmetic interpretation, separate from the original finite-input context.
IEEE operation agreement is explicit; it is not inferred from a receipt or a model name. -/
import MerlinVerify.Check.PolicyW8A8DynamicContext
import MerlinVerify.Rules.W8A8ArithmeticModel

namespace MerlinVerify.Check.PolicyW8A8NumericalContext

open MerlinVerify Rules Definedness

def ContextOk (ctx : Ctx) : Prop :=
  ∃ base, W8A8DynamicLaws.Laws base ∧ W8A8ArithmeticModel.Agreement base ∧
    ctx.M = W8A8ByteModel.extend base

def allowed (ctx : Ctx) (_ : CallSem) : Prop := ContextOk ctx

theorem dynamic_context {ctx : Ctx} (permitted : ContextOk ctx) : PolicyW8A8DynamicContext.ContextOk ctx := by
  obtain ⟨base, laws, _, interpretation⟩ := permitted
  exact ⟨base, laws, interpretation⟩

theorem loopStable : Contextual.LoopStable allowed := by
  intro _ _ _ permitted
  exact permitted

#print axioms dynamic_context
#print axioms loopStable

end MerlinVerify.Check.PolicyW8A8NumericalContext
