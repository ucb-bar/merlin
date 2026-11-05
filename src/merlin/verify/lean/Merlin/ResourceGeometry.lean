/-
Geometry of an explicitly supplied sequential resource model. These predicates
do not establish that logical SSA values have the declared physical sizes, or
that hardware implements the supplied read/completion and overwrite contract.
-/
import Lean.Elab.Tactic.Omega

namespace Merlin.ResourceGeometry

/-- A half-open range is inside its bank and begins at an aligned address. -/
abbrev Fits (address extent capacity alignment : Nat) : Prop :=
  0 < alignment ∧ address + extent ≤ capacity ∧ address % alignment = 0

/-- Physical ranges share no address. Lifetimes use closed epoch intervals. -/
abbrev Disjoint (aAddress aExtent bAddress bExtent : Nat) : Prop :=
  aAddress + aExtent ≤ bAddress ∨ bAddress + bExtent ≤ aAddress

abbrev Separated (aAddress aExtent aStart aEnd bAddress bExtent bStart bEnd : Nat) : Prop :=
  aEnd < bStart ∨ bEnd < aStart ∨ Disjoint aAddress aExtent bAddress bExtent

/-- A qualified final-use operand is read before its exact-address overwrite. -/
abbrev FinalUseOverwrite
    (childAddress parentAddress childExtent parentExtent references uses output
      childLastRead parentCompletion : Nat) : Prop :=
  childAddress = parentAddress ∧ childExtent = parentExtent ∧
    references = 1 ∧ uses = 1 ∧ output = 0 ∧ childLastRead < parentCompletion

/-- Includes result-free state events: their inputs remain live until read. -/
abbrev ReadWithin (ready issue offset completion liveEnd : Nat) : Prop :=
  ready ≤ issue + offset ∧ issue + offset ≤ completion ∧ issue + offset ≤ liveEnd

abbrev EqualOffset (left right : Nat) (offset : Int) : Prop :=
  (left : Int) = (right : Int) + offset

theorem fitsContains (address extent capacity alignment point : Nat)
    (fits : Fits address extent capacity alignment)
    (inside : address ≤ point ∧ point < address + extent) : point < capacity := by
  unfold Fits at fits
  omega

theorem disjointRanges (aAddress aExtent bAddress bExtent point : Nat)
    (disjoint : Disjoint aAddress aExtent bAddress bExtent) :
    ¬ (aAddress ≤ point ∧ point < aAddress + aExtent ∧
      bAddress ≤ point ∧ point < bAddress + bExtent) := by
  unfold Disjoint at disjoint
  omega

theorem separatedLiveRanges
    (aAddress aExtent aStart aEnd bAddress bExtent bStart bEnd time point : Nat)
    (separated : Separated aAddress aExtent aStart aEnd bAddress bExtent bStart bEnd) :
    ¬ (aStart ≤ time ∧ time ≤ aEnd ∧ bStart ≤ time ∧ time ≤ bEnd ∧
      aAddress ≤ point ∧ point < aAddress + aExtent ∧
      bAddress ≤ point ∧ point < bAddress + bExtent) := by
  unfold Separated Disjoint at separated
  omega

theorem overwriteAfterFinalRead
    (childAddress parentAddress childExtent parentExtent references uses output
      childLastRead parentCompletion : Nat)
    (qualified : FinalUseOverwrite childAddress parentAddress childExtent parentExtent
      references uses output childLastRead parentCompletion) :
    childLastRead < parentCompletion := by
  exact qualified.2.2.2.2.2

theorem readPreservedUntilUse (ready issue offset completion liveEnd : Nat)
    (within : ReadWithin ready issue offset completion liveEnd) :
    issue + offset ≤ liveEnd := within.2.2

end Merlin.ResourceGeometry
