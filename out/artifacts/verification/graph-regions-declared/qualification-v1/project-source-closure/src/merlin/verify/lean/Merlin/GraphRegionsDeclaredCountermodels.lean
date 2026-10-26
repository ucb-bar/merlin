/- Small independent countermodels to premise elision. These are mathematical
models, not executions of the selected provider or claims of native failure. -/
import Merlin.GraphRegionsDeclaredMemory

namespace Merlin.GraphRegionsDeclared

theorem nofault_premise_required {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    {primitive : Primitive → List α → α} {call : BoundCall} {program : Program}
    (memory : M) (arguments : List (Tensor elements α))
    (initialized : call.reads.map (read memory) = arguments.map some)
    (fault : invoke call memory = none) : ¬InstructionContract read invoke primitive call program := by
  intro contract
  obtain ⟨next, ran, _⟩ := contract.capture_compute_commit memory arguments initialized
  rw [fault] at ran
  cases ran

theorem completion_premise_required {α M : Type} [Inhabited α] {elements : Nat}
    {read : M → Window → Option (Tensor elements α)} {invoke : BoundCall → M → Option M}
    {primitive : Primitive → List α → α} {call : BoundCall} {program : Program}
    (memory next : M) (arguments : List (Tensor elements α))
    (initialized : call.reads.map (read memory) = arguments.map some)
    (returned : invoke call memory = some next)
    (notCommitted : read next call.output ≠ some (tensorEvaluate program primitive arguments)) :
    ¬InstructionContract read invoke primitive call program := by
  intro contract
  obtain ⟨actual, ran, computed, _⟩ := contract.capture_compute_commit memory arguments initialized
  have equal : actual = next := Option.some.inj (ran.symm.trans returned)
  subst actual
  exact notCommitted computed

/-- If the producer overwrites an external input, a later consumer can read a
different same-typed value. Source: p=2*3=6, result=p+3=9. Aliased result:6+6=12. -/
theorem alias_changes_consumer : (2 * 3 + 3 : Nat) ≠ 2 * 3 + 2 * 3 := by decide

/-- Returning before the scratch write commits can expose its old value to the
consumer: source=9, stale-scratch consumer=3. -/
theorem async_changes_consumer : (2 * 3 + 3 : Nat) ≠ 0 + 3 := by decide

/-- Suppressing an externally observed intermediate transfer changes an actual
observation even if the two requested output values still agree. -/
theorem private_observer_required : ([9, 18, 6] : List Nat) ≠ [9, 18, 0] := by decide

/-- Removing a transfer that faults cannot preserve error behavior. Equal
successful output payloads are insufficient when source execution fails. -/
theorem suppressed_transfer_fault_required : (none : Option (Nat × Nat)) ≠ some (9, 18) := by decide

end Merlin.GraphRegionsDeclared
