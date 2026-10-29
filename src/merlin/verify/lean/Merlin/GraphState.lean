/- Checked composition of actual projected adjacent exchanges. This closes the
sequence/order link left to Python by per-exchange obligations and reuses the
existing preserveExchanges root. No candidate hint supplies a semantic fact. -/
import Merlin.GraphStateContract

namespace Merlin.GraphState
open StateEffects
set_option autoImplicit false

structure Exchange where
  position : Nat
  first : Nat
  second : Nat
  deriving DecidableEq, Repr

def swapAt (footprints : List Footprint) : Nat → Nat → Nat → List Nat → Option (List Nat)
  | 0,first,second,x::y::tail =>
    if x == first && y == second then
      match footprints[x]?,footprints[y]? with
      | some a,some b => if Independent a b then some (y::x::tail) else none
      | _,_ => none
    else none
  | position+1,first,second,x::tail => (swapAt footprints position first second tail).map (x::·)
  | _,_,_,_ => none

def replay (footprints : List Footprint) : List Nat → List Exchange → Option (List Nat)
  | order,[] => some order
  | order,exchange::rest => do
    let next ← swapAt footprints exchange.position exchange.first exchange.second order
    replay footprints next rest

/-- The source roster is complete and dense. Acceptance recomputes every actual
neighbor, checks both footprints, their independence, and the EXACT final order. -/
def check (footprints : List Footprint) (target : List Nat) (exchanges : List Exchange) : Bool :=
  match replay footprints (List.range footprints.length) exchanges with
  | some actual => actual == target
  | none => false

theorem context_exchanges {Value : Type} {run : Nat → State Value → State Value}
    {before after : List Nat} (changes : Exchanges run before after) (earlierContext laterContext : List Nat) :
    Exchanges run (earlierContext ++ before ++ laterContext) (earlierContext ++ after ++ laterContext) := by
  induction changes with
  | refl order => exact .refl _
  | adjacent earlier later first second commute =>
    simpa only [List.append_assoc,List.cons_append] using
      (Exchanges.adjacent (run := run) (earlierContext++earlier) (later++laterContext) first second commute)
  | trans _ _ left right => exact .trans left right

theorem swapAt_sound {Value : Type} {footprints : List Footprint}
    {run : Nat → State Value → State Value} (model : Models footprints run) :
    ∀ position first second before after,
      swapAt footprints position first second before = some after → Exchanges run before after := by
  intro position
  induction position with
  | zero =>
    intro first second before after checked
    cases before with
    | nil => cases checked
    | cons x tail =>
      cases tail with
      | nil => cases checked
      | cons y tail =>
        simp only [swapAt] at checked
        split at checked
        · cases left : footprints[x]? with
          | none => simp [left] at checked
          | some a =>
            cases right : footprints[y]? with
            | none => simp [left,right] at checked
            | some b =>
              simp only [left,right] at checked
              split at checked
              · next independent =>
                cases checked
                obtain ⟨firstAction,firstStep⟩ := model x a left
                obtain ⟨secondAction,secondStep⟩ := model y b right
                apply Exchanges.adjacent [] tail x y
                intro state
                simpa only [firstStep,secondStep] using commuteActions firstAction secondAction independent state
              · cases checked
        · cases checked
  | succ position ih =>
    intro first second before after checked
    cases before with
    | nil => cases checked
    | cons x tail =>
      simp only [swapAt] at checked
      cases next : swapAt footprints position first second tail with
      | none => simp [next] at checked
      | some result =>
        simp only [next,Option.map_some,Option.some.injEq] at checked
        subst after
        simpa using context_exchanges (ih first second tail result next) [x] []

theorem replay_sound {Value : Type} {footprints : List Footprint}
    {run : Nat → State Value → State Value} (model : Models footprints run) :
    ∀ exchanges before after, replay footprints before exchanges = some after →
      Exchanges run before after
  | [],before,after,checked => by cases checked; exact .refl _
  | exchange::rest,before,after,checked => by
    cases next : swapAt footprints exchange.position exchange.first exchange.second before with
    | none => simp [replay,next] at checked
    | some middle =>
      simp only [replay,next,bind,Option.bind] at checked
      exact .trans (swapAt_sound model _ _ _ _ _ next) (replay_sound model rest middle after checked)

/-- Acceptance now yields one composed statement about the complete selected
schedule, not merely independent theorems about individual proposed swaps. -/
theorem check_sound {footprints : List Footprint} {target : List Nat} {exchanges : List Exchange}
    (accepted : check footprints target exchanges = true) : Preserves footprints target := by
  unfold check at accepted
  cases replayed : replay footprints (List.range footprints.length) exchanges with
  | none => simp [replayed] at accepted
  | some result =>
    have same : result = target := by simpa only [replayed,beq_iff_eq] using accepted
    subst result
    intro Value run model earlier later initial
    have preserved := preserveExchanges (replay_sound model exchanges _ _ replayed)
    simp only [execute,List.foldl_append]
    exact congrArg (fun state => later.foldl (fun s action => action s) state)
      (preserved (earlier.foldl (fun s action => action s) initial))

theorem observations_preserved {footprints : List Footprint} {target : List Nat} {exchanges : List Exchange}
    (accepted : check footprints target exchanges = true) {Value Result : Type}
    (run : Nat → State Value → State Value) (model : Models footprints run)
    (earlier later : List (State Value → State Value)) (initial : State Value)
    (observe : State Value → Result) :
    observe (execute (earlier ++ (List.range footprints.length).map run ++ later) initial) =
      observe (execute (earlier ++ target.map run ++ later) initial) :=
  congrArg observe (check_sound accepted Value run model earlier later initial)

#print axioms swapAt_sound
#print axioms replay_sound
#print axioms check_sound
#print axioms observations_preserved
end Merlin.GraphState
