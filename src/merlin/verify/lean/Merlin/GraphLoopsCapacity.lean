/- Whole-trace pending-work capacity, under the exact same memory/completion
semantics as GraphLoopsEpochs. The selected capacity remains a target declaration;
the theorem does not derive its physical truth or infer allowed concurrency pairs. -/
import Merlin.GraphLoopsEpochs

set_option autoImplicit false

namespace Merlin.GraphLoops.Capacity

open Storage Bound

def execute (limit : Nat) (windows privateObjects : List Region) : State → List Effect → Option State
  | state, effects =>
    if state.pending.length ≤ limit then
      match effects with
      | [] => some state
      | effect :: rest => do
        let next ← step windows privateObjects (Epochs.activeLedger state) effect
        execute limit windows privateObjects next rest
    else none

inductive Trace (limit : Nat) (windows privateObjects : List Region) : State → List Effect → State → Prop where
  | nil (state : State) (bounded : state.pending.length ≤ limit) :
      Trace limit windows privateObjects state [] state
  | cons {before middle after : State} {effect : Effect} {rest : List Effect}
      (bounded : before.pending.length ≤ limit)
      (event : SafeStep windows privateObjects (Epochs.activeLedger before) effect middle)
      (remaining : Trace limit windows privateObjects middle rest after) :
      Trace limit windows privateObjects before (effect :: rest) after

theorem execute_sound (limit : Nat) (windows privateObjects : List Region) : ∀ state effects final,
    execute limit windows privateObjects state effects = some final →
      Trace limit windows privateObjects state effects final := by
  intro state effects
  induction effects generalizing state with
  | nil =>
    intro final checked
    simp only [execute] at checked
    split at checked
    next bounded => cases checked; exact .nil state bounded
    next => cases checked
  | cons effect rest induction =>
    intro final checked
    simp only [execute] at checked
    split at checked
    next bounded =>
      cases selected : step windows privateObjects (Epochs.activeLedger state) effect with
      | none => simp [selected] at checked
      | some next =>
        simp only [selected, bind, Option.bind] at checked
        exact .cons bounded (step_safe selected) (induction _ _ checked)
    next => cases checked

theorem Trace.erase {limit : Nat} {windows privateObjects : List Region} {state final : State}
    {effects : List Effect} (trace : Trace limit windows privateObjects state effects final) :
    Epochs.Trace windows privateObjects state effects final := by
  induction trace with
  | nil state _ => exact .nil state
  | cons _ event _ induction => exact .cons event induction

theorem Trace.state_bounds {limit : Nat} {windows privateObjects : List Region} {state final : State}
    {effects : List Effect} (trace : Trace limit windows privateObjects state effects final) :
    state.pending.length ≤ limit ∧ final.pending.length ≤ limit := by
  induction trace with
  | nil _ bounded => exact ⟨bounded, bounded⟩
  | cons bounded _ _ induction => exact ⟨bounded, induction.2⟩

theorem Trace.event {limit : Nat} {windows privateObjects : List Region} {state final : State}
    {effects : List Effect} (trace : Trace limit windows privateObjects state effects final) :
    ∀ preceding effect suffix, effects = preceding ++ effect :: suffix →
      ∃ before after, Epochs.Trace windows privateObjects state preceding before ∧
        SafeStep windows privateObjects (Epochs.activeLedger before) effect after ∧
        before.pending.length ≤ limit ∧ after.pending.length ≤ limit ∧
        Trace limit windows privateObjects after suffix final := by
  induction trace with
  | nil state bounded =>
    intro preceding effect suffix equal
    have : 0 = preceding.length + (suffix.length + 1) := by simpa using congrArg List.length equal
    omega
  | @cons before middle after current rest bounded event remaining induction =>
    intro preceding effect suffix equal
    cases preceding with
    | nil =>
      simp only [List.nil_append, List.cons.injEq] at equal
      rcases equal with ⟨rfl, rfl⟩
      exact ⟨before, middle, .nil before, event, bounded, remaining.state_bounds.1, remaining⟩
    | cons first preceding =>
      simp only [List.cons_append, List.cons.injEq] at equal
      obtain ⟨rfl, equal⟩ := equal
      obtain ⟨pointState, next, earlier, safe, beforeBound, afterBound, later⟩ :=
        induction preceding effect suffix equal
      exact ⟨pointState, next, .cons event earlier, safe, beforeBound, afterBound, later⟩

def check (loop : Counted) (actual : List ClosedCall) (allocations : List Allocation)
    (external initialized : List Region) (events : List (List Event)) (limit : Nat) : Bool :=
  Epochs.check loop actual allocations external initialized events && decide (0 < limit) &&
    match execute limit (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
        ⟨initialized, [], []⟩ (events.flatten.map Event.effect) with
    | none => false
    | some final => decide (final.pending = [])

theorem check_sound {loop : Counted} {actual : List ClosedCall} {allocations : List Allocation}
    {external initialized : List Region} {events : List (List Event)} {limit : Nat}
    (accepted : check loop actual allocations external initialized events limit = true) :
    Epochs.check loop actual allocations external initialized events = true ∧ 0 < limit ∧
    ∃ final, Trace limit (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
      ⟨initialized, [], []⟩ (events.flatten.map Event.effect) final ∧ final.pending = [] := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at accepted
  obtain ⟨⟨boundary, positive⟩, accepted⟩ := accepted
  refine ⟨boundary, positive, ?_⟩
  cases completed : execute limit (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
      ⟨initialized, [], []⟩ (events.flatten.map Event.effect) with
  | none => simp only [completed] at accepted; cases accepted
  | some final =>
    simp only [completed, decide_eq_true_eq] at accepted
    exact ⟨final, execute_sound _ _ _ _ _ _ completed, accepted⟩

end Merlin.GraphLoops.Capacity
