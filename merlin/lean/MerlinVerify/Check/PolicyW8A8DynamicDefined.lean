/- Initialized dynamic W8A8 execution on the existing finite-input domain under the separately
declared dynamic numerical context. Every tensor stage uses actual checked geometry and body. -/
import MerlinVerify.Check.PolicyW8A8DynamicCaptured
import MerlinVerify.Check.FloatProducers
import MerlinVerify.Check.FloatImplication
import MerlinVerify.Check.PolicyW8A8FiniteDefined

namespace MerlinVerify.Check.PolicyW8A8DynamicDefined

open MerlinVerify Rules Definedness FloatFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8DynamicContext.allowed

def contracts (ib : Nat) : Contextual.Regions model sourceSem ib allowed where
  infer Γ body inputs :=
    if W8A8SourceFree.region body then
      ((PolicyW8A8DynamicBodies.absMaxBook ib).orElse
        ((PolicyW8A8DynamicBodies.reciprocalBook ib).orElse
          ((PolicyW8A8DynamicCaptured.scaleBook ib).orElse
            (PolicyW8A8DynamicCaptured.quantizeBook ib)))).infer Γ body inputs
    else none
  sound Γ body inputs outputs checked ctx call env args width permitted valid holds := by
    split at checked
    · next ordinary =>
      obtain ⟨values, executed, facts, initialized⟩ :=
        ((PolicyW8A8DynamicBodies.absMaxBook ib).orElse
          ((PolicyW8A8DynamicBodies.reciprocalBook ib).orElse
            ((PolicyW8A8DynamicCaptured.scaleBook ib).orElse
              (PolicyW8A8DynamicCaptured.quantizeBook ib)))).sound Γ body inputs outputs checked
                ctx call env args width permitted valid holds
      exact ⟨values, (W8A8SourceFree.region_eq ordinary ctx call env args).trans executed, facts, initialized⟩
    · cases checked

def leafBook (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  (Contextual.Book.lift (FloatFacts.Producers.book ib)).orElse
    (Refined.adaptBook finiteMap ((PolicyW8A8FiniteDefined.leafBook ib).restrict
      (fun _ _ permitted => PolicyW8A8DynamicContext.source_context permitted)))

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  Contextual.Structured.recursiveBookWith (contracts ib) FloatFacts.elementwise
    PolicyW8A8DynamicContext.loopStable PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (FloatFacts.Implication.invariants ib) (leafBook ib)

def inputFacts (types : List VTy) : List (Refined.Fact Info) :=
  (FiniteFacts.inputFacts types).map (Refined.mapFact fromClass)

def Correct (ib : Nat) (m : Module) (fn : String) (inputs : List VTy) : Prop :=
  FuncDefined sourceSem (fun ctx => ctx.indexBits = ib ∧ PolicyW8A8DynamicContext.ContextOk ctx)
    (PolicyW8A8FiniteDefined.Domain ib inputs) m fn

def check (ib : Nat) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  Contextual.check (book ib) m fn (inputFacts inputs) outputs

theorem check_sound {ib : Nat} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check ib m fn inputs outputs = true) : Correct ib m fn inputs := by
  have initialized := Contextual.check_sound (book ib) checked (fun _ _ permitted => permitted)
  intro ctx permitted args holds
  exact initialized ctx permitted args (holds.map fromClass finiteMap.embed_sound)

#print axioms contracts
#print axioms leafBook
#print axioms book
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8DynamicDefined
