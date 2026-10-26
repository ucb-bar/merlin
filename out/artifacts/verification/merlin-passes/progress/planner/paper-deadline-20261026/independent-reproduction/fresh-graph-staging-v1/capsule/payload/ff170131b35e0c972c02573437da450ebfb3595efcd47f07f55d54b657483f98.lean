/- Necessary-premise countermodels. These are NOT claims that allocation-changing
reuse preserves observable allocator failures, pointer identity or stale cells. -/
import Merlin.GraphLoopsReuseModel

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse.Countermodels

def attemptAllocate (capacity used bytes : Nat) : Option Nat :=
  if used + bytes ≤ capacity then some (used + bytes) else none

/-- A retained slot during a lifetime gap can exhaust a finite allocator even
when freeing the source's earlier object permits the intervening allocation. -/
theorem resource_exhaustion : attemptAllocate 8 0 8 = some 8 ∧ attemptAllocate 8 8 8 = none := by cbv

/-- Two observable fresh object identities differ from one reused identity. -/
theorem pointer_identity : ([0, 1] : List Nat) ≠ [0, 0] := by decide

/-- Removing an allocation can remove a possible allocation failure. -/
theorem allocation_call_budget : (2 ≤ 1 ↔ 1 ≤ 1) = False := by decide

/-- Without complete initialization, a stale candidate value cannot be equated
with an uninitialized source cell. The program protocol forbids this load. -/
theorem stale_cell : (none : Option Nat) ≠ some 37 := by decide

/-- Sharing a slot while BOTH epochs are live destroys a prior nonzero value. -/
theorem overlapping_live_values :
    ¬ Related (Value := Nat) (Public := Unit) (fun _ => 0) (fun _ => 1) [0, 1]
      ⟨(), fun object _ => some (if object = 0 then 17 else 29)⟩
      ⟨(), fun _ _ => some 29⟩ := by
  intro related
  have contradiction := related.read (object := 0) (index := 0) (by simp) (by decide)
  simp only at contradiction
  cases contradiction

end Merlin.GraphLoopsReuse.Countermodels

/- Explicit inhabited nonzero model, additional to actual xDSL witnesses.
It demonstrates the declared contract is not only a vacuous no-success relation;
it does NOT assert a raw-SCF/initializer evaluator bridge. -/
namespace Merlin.GraphLoopsReuse.Witness

def objects : Objects := [⟨0, [2, 3], 32, 16⟩, ⟨0, [2, 3], 32, 16⟩]
def instructions : List Instruction := [
  .allocate 0 true, .initialize 0 "first", .load 0 "read", .release 0 false,
  .allocate 1 false, .initialize 1 "second", .load 1 "read", .release 1 true]

def primitives : Primitives Nat (List Nat) where
  initialization key _ values := values = fun _ => if key = "first" then 17 else 29
  address _ _ index := index = 0
  load _ value before after := after = before ++ [value]
  store _ _ _ _ := False
  dimension _ _ _ _ := False
  retained _ _ _ := False

def source : Memory Nat (List Nat) := ⟨[], fun _ _ => none⟩
def target : Memory Nat (List Nat) := ⟨[], fun _ _ => some 91⟩

theorem protocol : checkProgram objects emptyStatus instructions = true := by cbv

def firstInitialized : Memory Nat (List Nat) :=
  initializeMemory objects false 0 (fun _ => 17) (clearMemory objects false 0 true source)
def secondInitialized : Memory Nat (List Nat) :=
  initializeMemory objects false 1 (fun _ => 29)
    (clearMemory objects false 1 false (clearMemory objects false 0 false ⟨[17], firstInitialized.heap⟩))
def sourceFinal : Memory Nat (List Nat) := clearMemory objects false 1 true ⟨[17, 29], secondInitialized.heap⟩

theorem source_inhabited : ∃ final, Execution objects primitives false instructions source final ∧ final.observed = [17, 29] := by
  refine ⟨sourceFinal, ?_, rfl⟩
  apply Execution.more (.allocate 0 true source)
  apply Execution.more (.initialized 0 "first" _ (fun _ => 17) rfl)
  apply Execution.more (.load 0 0 "read" _ 17 [17] rfl (by decide) rfl rfl)
  apply Execution.more (.release 0 false _)
  apply Execution.more (.allocate 1 false _)
  apply Execution.more (.initialized 1 "second" _ (fun _ => 29) rfl)
  apply Execution.more (.load 1 0 "read" _ 29 [17, 29] rfl (by decide) rfl rfl)
  apply Execution.more (.release 1 true _)
  exact .done _

/-- Stale physical91 is fully reinitialized, and BOTH nonzero load observations
17 then29 survive the real epoch-to-one-slot model protocol. -/
theorem target_inhabited : ∃ final, Execution objects primitives true instructions target final ∧ final.observed = [17, 29] :=
  (checked_observations objects instructions protocol primitives source target rfl [17, 29]).mp source_inhabited

end Merlin.GraphLoopsReuse.Witness
