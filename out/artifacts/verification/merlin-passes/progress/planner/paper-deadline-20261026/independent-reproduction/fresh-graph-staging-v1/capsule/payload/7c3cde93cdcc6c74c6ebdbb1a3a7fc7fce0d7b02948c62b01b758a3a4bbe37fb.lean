/- Per-statement correspondence for retained constants and ordered partial
binary operations. Public ghost register updates are derived from concrete
operand/result bindings, and the entire caller/private world is framed. -/
import Merlin.GraphLoopsShapeEpochState
import Merlin.GraphLoopsShapeEpochMeaning

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem retained_step_iff {Value User : Type} (bound : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (spec : Spec)
    (selected : specAt specifications key = some spec)
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (before after : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
      (.retained key) before after ↔
      ∃ observed, Retained bound scalar spec before.observed observed ∧ after = ⟨observed, before.heap⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary ordinary run =>
      cases run with
      | retained key initial observed action =>
        obtain ⟨actual, found, action⟩ := action
        have same : actual = spec := Option.some.inj (found.symm.trans selected)
        subst actual
        exact ⟨observed, action, rfl⟩
  · rintro ⟨observed, action, rfl⟩
    apply GraphLoopsMemorySCF.Step.ordinary (by trivial)
    apply GraphLoopsReuse.Step.retained
    exact ⟨spec, selected, action⟩

theorem constant_atom {Value User : Type} (bound next : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (result : Binding) (value : Nat)
    (selected : specAt specifications key = some (.constant result value))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) (env : Env Value)
    (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.constant result value).step meaning before.observed.registers (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
        (.retained key) before after ∧ env = after.observed.registers ∧ world = clockWorld bound next after := by
  simp only [constant_step_iff bound specifications scalar mapping key result value selected objects configs,
    exists_eq_left, Atom.step, writeRegister, clockWorld]

/-- Every scalar result is derived by the SAME partial binary relation; the old
instruction cannot invent a value, skip a failure, reorder inputs or change heap. -/
theorem binary_atom {Value User : Type} (bound next : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key operation : String)
    (left right result : Binding) (element : String)
    (selected : specAt specifications key = some (.binary operation left right result element))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (meaning : Interpretation Value (MemoryWorld Value (AllocationPublic User)))
    (binary : meaning.action operation = Binary element (scalar operation))
    (before : GraphLoopsReuse.Memory Value (PublicState Value User)) (env : Env Value)
    (world : MemoryWorld Value (AllocationPublic User)) :
    (Atom.action operation [left, right] [result]).step meaning before.observed.registers
      (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
        (.retained key) before after ∧ env = after.observed.registers ∧ world = clockWorld bound next after := by
  simp only [retained_step_iff bound specifications scalar mapping key _ selected objects configs]
  constructor
  · rintro ⟨values, action, typed, env⟩
    rw [binary] at action
    obtain ⟨a, b, value, inputs, computed, rfl, rfl⟩ := action
    have leftRead : before.observed.registers left.name = .scalar element a := (List.cons.inj inputs).1
    have rightRead : before.observed.registers right.name = .scalar element b := (List.cons.inj (List.cons.inj inputs).2).1
    have holds : result.Holds (.scalar element value) := by cases typed with | cons holds tail => exact holds
    refine ⟨⟨writeRegister before.observed result.name (.scalar element value), before.heap⟩,
      ⟨_, ⟨a, b, value, leftRead, rightRead, computed, holds, rfl⟩, rfl⟩, env, rfl⟩
  · rintro ⟨after, ⟨observed, ⟨a, b, value, leftRead, rightRead, computed, holds, rfl⟩, rfl⟩, rfl, rfl⟩
    refine ⟨[.scalar element value], ?_, .cons holds .nil, rfl⟩
    rw [binary]
    refine ⟨a, b, value, ?_, computed, rfl, rfl⟩
    change [before.observed.registers left.name, before.observed.registers right.name] = _
    rw [leftRead, rightRead]

/-- The final return reads the actual current SSA operands. The private epoch
heap is retained, while the public observation receives exactly those values. -/
theorem returned_step_iff {Value User : Type} (bound : Nat) (specifications : Specifications)
    (scalar : BinaryMeaning Value) (mapping : Aliases) (key : String) (values : List Binding)
    (selected : specAt specifications key = some (.returned values))
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (before after : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications mapping scalar) false false
      (.retained key) before after ↔
      after = ⟨{ before.observed with returned := readArguments false [] before.observed.registers values }, before.heap⟩ := by
  rw [retained_step_iff bound specifications scalar mapping key _ selected objects configs]
  simp only [Retained, exists_eq_left]

end Merlin.GraphLoopsShape.Epoch
