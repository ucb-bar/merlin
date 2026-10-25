/- Complete dynamic memory/completion execution across period boundaries.
Only the auxiliary seen-token history is refreshed: initialized bytes and actual
pending work are NEVER cleared by a loop boundary. A physical token may be reused
after its exact wait. The stricter per-period checks still require fresh private
initialization and drain, independently of this whole-trace check. -/
import Merlin.GraphLoopsBound

set_option autoImplicit false

namespace Merlin.GraphLoops.Epochs

open Storage Bound

def activeLedger (state : State) : State :=
  { state with seen := state.pending.map Job.token }

def execute (windows privateObjects : List Region) : State → List Effect → Option State
  | state, [] => some state
  | state, effect :: rest => do
    let next ← step windows privateObjects (activeLedger state) effect
    execute windows privateObjects next rest

inductive Trace (windows privateObjects : List Region) : State → List Effect → State → Prop where
  | nil (state : State) : Trace windows privateObjects state [] state
  | cons {before middle after : State} {effect : Effect} {rest : List Effect}
      (event : SafeStep windows privateObjects (activeLedger before) effect middle)
      (remaining : Trace windows privateObjects middle rest after) :
      Trace windows privateObjects before (effect :: rest) after

theorem execute_sound (windows privateObjects : List Region) : ∀ state effects final,
    execute windows privateObjects state effects = some final → Trace windows privateObjects state effects final := by
  intro state effects
  induction effects generalizing state with
  | nil => intro final checked; cases checked; exact .nil state
  | cons effect rest induction =>
    intro final checked
    simp only [execute] at checked
    cases selected : step windows privateObjects (activeLedger state) effect with
    | none => simp [selected] at checked
    | some next =>
      simp only [selected, bind, Option.bind] at checked
      exact .cons (step_safe selected) (induction _ _ checked)

theorem active_token_refused {windows privateObjects : List Region} {state next : State}
    {token : Nat} {accesses : List Access}
    (checked : step windows privateObjects (activeLedger state) (.asynchronous token accesses) = some next) :
    ∀ job ∈ state.pending, job.token ≠ token := by
  have fresh := (asynchronous_ready checked).2
  intro job present equal
  apply fresh
  exact List.mem_map.mpr ⟨job, present, equal⟩

/-- The full actual trace is checked from a SINGLE initial pending set. No
unobserved completion or boundary-based cancellation is introduced. -/
def check (loop : Counted) (actual : List ClosedCall) (allocations : List Allocation)
    (external initialized : List Region) (events : List (List Event)) : Bool :=
  Bound.check loop actual allocations external initialized events &&
    match execute (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
        ⟨initialized, [], []⟩ (events.flatten.map Event.effect) with
    | none => false
    | some final => decide (final.pending = [])

theorem check_sound {loop : Counted} {actual : List ClosedCall} {allocations : List Allocation}
    {external initialized : List Region} {events : List (List Event)}
    (accepted : check loop actual allocations external initialized events = true) :
    Bound.check loop actual allocations external initialized events = true ∧
    ∃ final, Trace (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
      ⟨initialized, [], []⟩ (events.flatten.map Event.effect) final ∧ final.pending = [] := by
  simp only [check, Bool.and_eq_true] at accepted
  obtain ⟨boundary, accepted⟩ := accepted
  refine ⟨boundary, ?_⟩
  cases completed : execute (external ++ allocations.map Allocation.region) (allocations.map Allocation.region)
      ⟨initialized, [], []⟩ (events.flatten.map Event.effect) with
  | none => simp only [completed] at accepted; cases accepted
  | some final =>
    simp only [completed, decide_eq_true_eq] at accepted
    exact ⟨final, execute_sound _ _ _ _ _ completed, accepted⟩

theorem Trace.event {windows privateObjects : List Region} {state final : State}
    {effects : List Effect} (trace : Trace windows privateObjects state effects final) :
    ∀ preceding effect suffix, effects = preceding ++ effect :: suffix →
      ∃ before after, Trace windows privateObjects state preceding before ∧
        SafeStep windows privateObjects (activeLedger before) effect after ∧
        Trace windows privateObjects after suffix final := by
  induction trace with
  | nil state =>
    intro preceding effect suffix equal
    have : 0 = preceding.length + (suffix.length + 1) := by simpa using congrArg List.length equal
    omega
  | @cons before middle after current rest transition remaining induction =>
    intro preceding effect suffix equal
    cases preceding with
    | nil =>
      simp only [List.nil_append, List.cons.injEq] at equal
      rcases equal with ⟨rfl, rfl⟩
      exact ⟨before, middle, .nil before, transition, remaining⟩
    | cons first preceding =>
      simp only [List.cons_append, List.cons.injEq] at equal
      obtain ⟨rfl, equal⟩ := equal
      obtain ⟨pointState, next, earlier, safe, later⟩ := induction preceding effect suffix equal
      exact ⟨pointState, next, .cons transition earlier, safe, later⟩

end Merlin.GraphLoops.Epochs
