/- Close conditional call contracts over actual function bodies at the evaluator's exact
fuel. The construction is generic in the numerical/refinement model and operation book. -/
import MerlinVerify.Check.DefinedNamedCalls

namespace MerlinVerify.Check.Definedness.ClosedCalls

open MerlinVerify Refined

def allowed {Info : Type} (M : Model Info) (ib : Nat) (contexts : Ctx → Prop)
    (known : List (Calls.Contract Info)) (ctx : Ctx) (call : CallSem) : Prop :=
  contexts ctx ∧ Calls.Holds M ib known call

structure System {Info : Type} (M : Model Info) (sem : OpSem) (ib : Nat) (contexts : Ctx → Prop) where
  arguments : List VTy → List (Refined.Fact Info)
  implies : List (Refined.Fact Info) → List (Refined.Fact Info) → Bool
  implies_sound : ∀ actual required, implies actual required = true →
    ∀ values, Refined.FactsHold M ib values actual → Refined.FactsHold M ib values required
  book : ∀ known, Contextual.Book M sem ib (allowed M ib contexts known)

variable {Info : Type} {M : Model Info} {sem : OpSem} {ib : Nat} {contexts : Ctx → Prop}

/-- Existing clients use the same input facts for equally typed functions. They retain
their original API and domain while sharing the function-indexed closure proof. -/
def System.named (system : System M sem ib contexts) : NamedCalls.System M sem ib contexts where
  arguments _ inputs := some (system.arguments inputs)
  implies := system.implies
  implies_sound := system.implies_sound
  book := system.book

def stage (system : System M sem ib contexts) (m : Module) : Nat → List (Calls.Contract Info) :=
  NamedCalls.stage system.named m

theorem stage_sound (system : System M sem ib contexts) (m : Module) : ∀ fuel ctx,
    ctx.indexBits = ib → contexts ctx → Calls.Holds M ib (stage system m fuel) (evalFunc sem ctx m fuel) :=
  NamedCalls.stage_sound system.named m

def checkKnown (system : System M sem ib contexts) (known : List (Calls.Contract Info))
    (fn : String) (inputs outputs : List VTy) : Bool :=
  NamedCalls.checkKnown system.named known fn inputs outputs

def check (system : System M sem ib contexts) (m : Module) (fn : String) (inputs outputs : List VTy) : Bool :=
  NamedCalls.check system.named m fn inputs outputs

theorem check_sound (system : System M sem ib contexts) {m : Module} {fn : String} {inputs outputs : List VTy}
    (checked : check system m fn inputs outputs = true) :
    Contextual.Correct M sem ib contexts m fn (system.arguments inputs) := by
  have correct := NamedCalls.check_sound system.named checked
  intro ctx permitted args holds
  exact correct ctx permitted args ⟨system.arguments inputs, rfl, holds⟩

#print axioms System.named
#print axioms stage_sound
#print axioms check_sound

end MerlinVerify.Check.Definedness.ClosedCalls
