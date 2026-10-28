/- The dynamic route has its own explicit numerical context. Earlier source-quantizer
theorems keep their original, weaker context unchanged. -/
import MerlinVerify.Check.PolicyW8A8FiniteQuantize
import MerlinVerify.Check.DefinedContextLoops
import MerlinVerify.Rules.W8A8DynamicLaws

namespace MerlinVerify.Check.PolicyW8A8DynamicContext

open MerlinVerify Rules Definedness

def ContextOk (ctx : Ctx) : Prop :=
  ∃ base, W8A8DynamicLaws.Laws base ∧ ctx.M = W8A8ByteModel.extend base

def allowed (ctx : Ctx) (_ : CallSem) : Prop := ContextOk ctx

theorem source_context {ctx : Ctx} (permitted : ContextOk ctx) : PolicyW8A8FiniteQuantize.ContextOk ctx := by
  obtain ⟨base, laws, model⟩ := permitted
  exact ⟨base, laws.source, model⟩

theorem loopStable : Contextual.LoopStable allowed := by
  intro ctx call point permitted
  exact permitted

#print axioms source_context
#print axioms loopStable

end MerlinVerify.Check.PolicyW8A8DynamicContext
