/- Shared-pass schedule contract, extending StateEffects without replacing it.
The IR projection remains a separately bound host correspondence obligation.
Action totality, complete frames/locality and sequential observations are explicit. -/
import Merlin.StateEffects

namespace Merlin.GraphState
open StateEffects
set_option autoImplicit false

/-- Each actually indexed action has the existing complete Action semantics.
The existence of a footprint declaration alone does not establish this premise. -/
def Models {Value : Type} (footprints : List Footprint)
    (run : Nat → State Value → State Value) : Prop :=
  ∀ index effects, footprints[index]? = some effects →
    ∃ action : Action effects Value, action.step = run index

/-- The supported actual schedule boundary preserves ALL modeled state in any
sequential context. This does not admit partial, asynchronous or unknown actions. -/
def Preserves (footprints : List Footprint) (target : List Nat) : Prop :=
  ∀ (Value : Type) (run : Nat → State Value → State Value), Models footprints run →
    ∀ (earlier later : List (State Value → State Value)) (initial : State Value),
      execute (earlier ++ (List.range footprints.length).map run ++ later) initial =
        execute (earlier ++ target.map run ++ later) initial

end Merlin.GraphState
