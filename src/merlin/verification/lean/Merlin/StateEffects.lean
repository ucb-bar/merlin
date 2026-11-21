/-
Conditional sequential state/SSA preservation. An admitted action is deterministic
and total, changes only its writes, and determines those writes from its reads.
These are semantic premises, not inferred hardware or timing facts. The Python
projection binds concrete footprints and each adjacent exchange to actual IR.
-/
import Init.Data.List.Lemmas

namespace Merlin.StateEffects

structure Footprint where
  reads : List Nat
  writes : List Nat
  deriving DecidableEq

abbrev State (Value : Type) := Nat → Value

structure Action (effects : Footprint) (Value : Type) where
  step : State Value → State Value
  frame : ∀ state key, key ∉ effects.writes → step state key = state key
  locality : ∀ first second,
    (∀ key, key ∈ effects.reads → first key = second key) →
    ∀ key, key ∈ effects.writes → step first key = step second key

def Disjoint (first second : List Nat) : Prop := ∀ key ∈ first, key ∉ second

instance (first second : List Nat) : Decidable (Disjoint first second) :=
  inferInstanceAs (Decidable (∀ key ∈ first, key ∉ second))

def Independent (first second : Footprint) : Prop :=
  Disjoint first.writes second.reads ∧
  Disjoint second.writes first.reads ∧
  Disjoint first.writes second.writes

instance (first second : Footprint) : Decidable (Independent first second) :=
  inferInstanceAs (Decidable (Disjoint first.writes second.reads ∧
    Disjoint second.writes first.reads ∧ Disjoint first.writes second.writes))

/-- Disjoint read/write frames imply commutation of complete state transitions. -/
theorem commuteActions {firstEffects secondEffects : Footprint} {Value : Type}
    (first : Action firstEffects Value) (second : Action secondEffects Value)
    (independent : Independent firstEffects secondEffects) (state : State Value) :
    second.step (first.step state) = first.step (second.step state) := by
  funext key
  rcases independent with ⟨firstRead, secondRead, writes⟩
  by_cases inFirst : key ∈ firstEffects.writes
  · have outsideSecond : key ∉ secondEffects.writes := fun h => writes key inFirst h
    rw [second.frame _ key outsideSecond]
    apply first.locality state (second.step state) _ key inFirst
    intro read inReads
    exact (second.frame state read (fun h => secondRead read h inReads)).symm
  · by_cases inSecond : key ∈ secondEffects.writes
    · rw [first.frame _ key inFirst]
      apply second.locality (first.step state) state _ key inSecond
      intro read inReads
      exact first.frame state read (fun h => firstRead read h inReads)
    · rw [second.frame _ key inSecond, first.frame _ key inFirst,
        first.frame _ key inFirst, second.frame _ key inSecond]

def execute (actions : List (State Value → State Value)) (initial : State Value) : State Value :=
  actions.foldl (fun state action => action state) initial

/-- A checked adjacent exchange preserves every surrounding sequential context. -/
theorem exchangeAdjacent (earlier later : List (State Value → State Value))
    (first second : State Value → State Value)
    (commute : ∀ state, second (first state) = first (second state)) (initial : State Value) :
    execute (earlier ++ first :: second :: later) initial =
      execute (earlier ++ second :: first :: later) initial := by
  simp only [execute, List.foldl_append, List.foldl_cons]
  rw [commute]

/-- A composed sequence of justified adjacent exchanges preserves the full state. -/
inductive Exchanges (run : Nat → State Value → State Value) : List Nat → List Nat → Prop where
  | refl (order) : Exchanges run order order
  | adjacent (earlier later : List Nat) (first second : Nat)
      (commute : ∀ state, run second (run first state) = run first (run second state)) :
      Exchanges run (earlier ++ first :: second :: later) (earlier ++ second :: first :: later)
  | trans {first middle last} : Exchanges run first middle → Exchanges run middle last → Exchanges run first last

theorem preserveExchanges {run : Nat → State Value → State Value} {before after : List Nat}
    (changes : Exchanges run before after) (initial : State Value) :
    execute (before.map run) initial = execute (after.map run) initial := by
  induction changes with
  | refl => rfl
  | adjacent earlier later first second commute =>
      simpa only [List.map_append, List.map_cons] using
        exchangeAdjacent (earlier.map run) (later.map run) (run first) (run second) commute initial
  | trans _ _ left right => exact left.trans right

end Merlin.StateEffects
