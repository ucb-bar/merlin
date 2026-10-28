/- Dynamic numerical facts and actual call contracts compose inside structured bodies. The
shared staged construction closes every contract against the real evaluator and module fuel. -/
import MerlinVerify.Check.PolicyW8A8DynamicDefined
import MerlinVerify.Check.PolicyW8A8Called

namespace MerlinVerify.Check.PolicyW8A8DynamicCalled

open MerlinVerify Rules Definedness FloatFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
abbrev Contract := Calls.Contract Info

def allowed (ib : Nat) (known : List Contract) : Contextual.Conditions :=
  ClosedCalls.allowed model ib PolicyW8A8DynamicContext.ContextOk known

def book (ib : Nat) (known : List Contract) : Contextual.Book model sourceSem ib (allowed ib known) :=
  let calls := Calls.book model ib PolicyW8A8Called.callAgreement FloatFacts.Implication.check
    (fun _ _ checked _ holds => FloatFacts.Implication.check_sound checked holds) known
  let leaf := (calls.restrict (fun _ _ permitted => permitted.2)).orElse
    ((PolicyW8A8DynamicDefined.leafBook ib).restrict (fun _ _ permitted => permitted.1))
  let contracts := (PolicyW8A8DynamicDefined.contracts ib).restrict (fun _ _ permitted => permitted.1)
  Contextual.Structured.recursiveBookWith contracts FloatFacts.elementwise
    (fun _ _ _ permitted => permitted) PolicyW8A8Defined.genericAgreement
    PolicyW8A8FiniteDefined.fillAgreement (FloatFacts.Implication.invariants ib) leaf

def system (ib : Nat) : ClosedCalls.System model sourceSem ib PolicyW8A8DynamicContext.ContextOk where
  arguments := PolicyW8A8DynamicDefined.inputFacts
  implies := FloatFacts.Implication.check
  implies_sound _ _ checked _ holds := FloatFacts.Implication.check_sound checked holds
  book := book ib

def stage (ib : Nat) (m : Module) : Nat → List Contract := ClosedCalls.stage (system ib) m

theorem stage_sound (ib : Nat) (m : Module) : ∀ fuel ctx,
    ctx.indexBits = ib → PolicyW8A8DynamicContext.ContextOk ctx →
    Calls.Holds model ib (stage ib m fuel) (evalFunc sourceSem ctx m fuel) :=
  ClosedCalls.stage_sound (system ib) m

def check (ib : Nat) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  ClosedCalls.check (system ib) m fn inputs outputs

theorem check_sound {ib : Nat} {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check ib m fn inputs outputs = true) : PolicyW8A8DynamicDefined.Correct ib m fn inputs := by
  have initialized := ClosedCalls.check_sound (system ib) checked
  intro ctx permitted args holds
  exact initialized ctx permitted args (holds.map fromClass finiteMap.embed_sound)

#print axioms book
#print axioms stage_sound
#print axioms check_sound

end MerlinVerify.Check.PolicyW8A8DynamicCalled
