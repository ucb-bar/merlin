/- Whole-function initialized execution on a declared finite-f32 input domain. Numerical
facts must be proved at each SSA producer. Ordinary initialization rules erase numerical
facts rather than treating initialized floating results as finite. -/
import MerlinVerify.Check.FiniteProducers
import MerlinVerify.Check.PolicyW8A8FiniteQuantize
import MerlinVerify.Check.FiniteStructured

namespace MerlinVerify.Check.PolicyW8A8FiniteDefined

open MerlinVerify Rules Definedness

local notation "sourceSem" => W8A8SourceSemantics.opSem

def Domain (ib : Nat) (types : List VTy) (args : List RVal) : Prop :=
  Refined.FactsHold FiniteFacts.model ib args (FiniteFacts.inputFacts types)

def Correct (ib : Nat) (m : Module) (fn : String) (types : List VTy) : Prop :=
  FuncDefined sourceSem (fun ctx => ctx.indexBits = ib ∧ PolicyW8A8FiniteQuantize.ContextOk ctx)
    (Domain ib types) m fn

def leafBook (ib : Nat) : Contextual.Book FiniteFacts.model sourceSem ib PolicyW8A8FiniteQuantize.allowed :=
  (PolicyW8A8FiniteQuantize.book ib).orElse
    (Contextual.Book.lift ((FiniteFacts.Producers.book ib).orElse
      (Refined.Book.lift FiniteFacts.model (PolicyW8A8Defined.book ib))))

theorem loopStable : Contextual.LoopStable PolicyW8A8FiniteQuantize.allowed := by
  intro ctx call point permitted
  exact permitted

theorem fillAgreement : Contextual.Fill.Agreement sourceSem := by
  intro ctx call op args regions named
  apply W8A8SourceSemantics.unchanged
  simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]

def book (ib : Nat) : Contextual.Book FiniteFacts.model sourceSem ib PolicyW8A8FiniteQuantize.allowed :=
  Contextual.Structured.recursiveBook FiniteFacts.Structured.elementwise loopStable
    PolicyW8A8Defined.genericAgreement fillAgreement (FiniteFacts.Structured.invariants ib) (leafBook ib)

def check (ib : Nat) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  Contextual.check (book ib) m fn (FiniteFacts.inputFacts inputs) outputs

/-- Finiteness is an explicit input-domain requirement, not a theorem about arbitrary typed
f32 values. The same admitted floating model is used throughout the actual program execution. -/
theorem check_sound {ib : Nat} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check ib m fn inputs outputs = true) : Correct ib m fn inputs :=
  Contextual.check_sound (book ib) checked (fun _ _ permitted => permitted)

#print axioms check_sound
#print axioms loopStable
#print axioms fillAgreement

end MerlinVerify.Check.PolicyW8A8FiniteDefined
