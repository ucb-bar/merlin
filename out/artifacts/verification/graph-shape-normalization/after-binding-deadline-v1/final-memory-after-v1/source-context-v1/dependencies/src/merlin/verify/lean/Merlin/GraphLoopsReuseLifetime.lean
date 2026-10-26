/- Actual source-operation lifetime/initialization boundaries, extending the
existing disjointLifetime/ResourceGeometry temporal relation. -/
import Merlin.GraphLoopsReuseGeometry

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse

structure Lifetime where
  object : Nat
  allocation : Nat
  initialized : Nat
  release : Nat
  retainedAllocation : Bool
  retainedRelease : Bool
  deriving DecidableEq, Repr

def isInitializer (object : Nat) : Option Instruction → Bool
  | some (.initialize actual _) => actual == object
  | _ => false

def Lifetime.Valid (objects : Objects) (instructions : List Instruction) (life : Lifetime) : Prop :=
  life.object < objects.length ∧ life.allocation < life.initialized ∧ life.initialized < life.release ∧
  instructions[life.allocation]? = some (.allocate life.object life.retainedAllocation) ∧
  instructions[life.release]? = some (.release life.object life.retainedRelease) ∧
  isInitializer life.object instructions[life.initialized]? = true

instance (objects : Objects) (instructions : List Instruction) (life : Lifetime) :
    Decidable (life.Valid objects instructions) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def LifetimesValid (objects : Objects) (instructions : List Instruction) (lives : List Lifetime) : Prop :=
  lives.map Lifetime.object = List.range objects.length ∧
  (∀ life ∈ lives, life.Valid objects instructions) ∧
  ∀ first ∈ lives, ∀ second ∈ lives, first.object ≠ second.object → slotOf objects first.object = slotOf objects second.object →
    first.release < second.allocation ∨ second.release < first.allocation

instance (objects : Objects) (instructions : List Instruction) (lives : List Lifetime) :
    Decidable (LifetimesValid objects instructions lives) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ ∀ first ∈ lives, ∀ second ∈ lives, first.object ≠ second.object →
    slotOf objects first.object = slotOf objects second.object → _))

/-- The same actual physical slot cannot represent two live logical epochs. -/
theorem LifetimesValid.not_simultaneously_live {objects : Objects} {instructions : List Instruction} {lives : List Lifetime}
    (valid : LifetimesValid objects instructions lives) {first second : Lifetime}
    (firstMember : first ∈ lives) (secondMember : second ∈ lives) (different : first.object ≠ second.object)
    (sameSlot : slotOf objects first.object = slotOf objects second.object) (time : Nat) :
    ¬ (first.allocation ≤ time ∧ time ≤ first.release ∧ second.allocation ≤ time ∧ time ≤ second.release) := by
  have separate := valid.2.2 first firstMember second secondMember different sameSlot
  have qualified : ResourceGeometry.Separated 0 1 first.allocation first.release 0 1 second.allocation second.release := by
    rcases separate with before | after
    · exact Or.inl before
    · exact Or.inr (Or.inl after)
  intro live
  exact ResourceGeometry.separatedLiveRanges 0 1 first.allocation first.release 0 1 second.allocation second.release time 0
    qualified ⟨live.1, live.2.1, live.2.2.1, live.2.2.2, by omega, by omega, by omega, by omega⟩

end Merlin.GraphLoopsReuse
