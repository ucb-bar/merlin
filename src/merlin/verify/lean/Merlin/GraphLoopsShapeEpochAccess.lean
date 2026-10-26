/- One-instruction typed SSA reduction rules. Load/store behavior is unfolded
from descriptors; unchanged scalar arithmetic remains one explicit partial
relation in its original order. No numeric identity or totality is assumed. -/
import Merlin.GraphLoopsShapeEpochCells

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem scalar_result {Value : Type} (result : Binding) (element : String)
    (kind : result.kind = .scalar element) (plain : result.known = none) (value : Value) :
    ValuesHold [result] [Runtime.scalar element value] :=
  .cons (Binding.holds_plain result (.scalar element value) kind.symm plain) .nil

theorem eval_load {Value User : Type} (meaning : Interpretation Value (MemoryWorld Value User))
    (key : String) (load : meaning.action key = Load) (aliases : Aliases) (env : Env Value)
    (operands : List Binding) (result : Binding) (rest : Code) (object : Nat)
    (shape coordinates : List Nat) (element : String)
    (arguments : readArguments false aliases env operands = view object shape element :: coordinates.map Runtime.index)
    (kind : result.kind = .scalar element) (plain : result.known = none)
    (before after : MemoryWorld Value User) (outputs : List (Runtime Value)) :
    Eval meaning false aliases (.action key operands [result] rest) env before outputs after ↔
      ∃ value, GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
        before.heap object (GraphLoopsReuse.Geometry.flatten shape coordinates) = some value ∧
        Eval meaning false (extendAliases aliases [result]) rest
          (put env result.name (.scalar element value)) before outputs after := by
  simp only [Eval, load, arguments, load_iff]
  constructor
  · rintro ⟨values, middle, ⟨value, inside, read, rfl, rfl⟩, _, suffix⟩
    exact ⟨value, inside, read, suffix⟩
  · rintro ⟨value, inside, read, suffix⟩
    exact ⟨[.scalar element value], before, ⟨value, inside, read, rfl, rfl⟩,
      scalar_result result element kind plain value, suffix⟩

theorem eval_store {Value User : Type} (meaning : Interpretation Value (MemoryWorld Value User))
    (key : String) (store : meaning.action key = Store) (aliases : Aliases) (env : Env Value)
    (operands : List Binding) (rest : Code) (object : Nat) (shape coordinates : List Nat) (element : String) (value : Value)
    (arguments : readArguments false aliases env operands = .scalar element value :: view object shape element ::
      coordinates.map Runtime.index)
    (before after : MemoryWorld Value User) (outputs : List (Runtime Value)) :
    Eval meaning false aliases (.action key operands [] rest) env before outputs after ↔
      GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧
        Eval meaning false aliases rest env
          ⟨before.observed, GraphLoopsReuse.replaceCell before.heap object
            (GraphLoopsReuse.Geometry.flatten shape coordinates) value⟩ outputs after := by
  simp only [Eval, store, arguments, store_iff]
  constructor
  · rintro ⟨values, middle, ⟨inside, rfl, rfl⟩, _, suffix⟩
    exact ⟨inside, suffix⟩
  · rintro ⟨inside, suffix⟩
    exact ⟨[], _, ⟨inside, rfl, rfl⟩, .nil, suffix⟩

/-- Typed binary scalar meaning. Partiality and non-determinism are retained. -/
def Binary {Value World : Type} (element : String) (operation : Value → Value → Value → Prop)
    (inputs : List (Runtime Value)) (before : World) (outputs : List (Runtime Value)) (after : World) : Prop :=
  ∃ left right result, inputs = [.scalar element left, .scalar element right] ∧
    operation left right result ∧ outputs = [.scalar element result] ∧ after = before

theorem binary_arguments {Value World : Type} (element : String) (operation : Value → Value → Value → Prop)
    (left right : Value) (before after : World) (outputs : List (Runtime Value)) :
    Binary element operation [.scalar element left, .scalar element right] before outputs after ↔
      ∃ result, operation left right result ∧ outputs = [.scalar element result] ∧ after = before := by
  constructor
  · rintro ⟨a, b, result, inputs, run, outputs, after⟩
    have first := Runtime.scalar.inj (List.cons.inj inputs).1
    have second := Runtime.scalar.inj (List.cons.inj (List.cons.inj inputs).2).1
    cases first.2
    cases second.2
    exact ⟨result, run, outputs, after⟩
  · rintro ⟨result, run, outputs, after⟩
    exact ⟨left, right, result, rfl, run, outputs, after⟩

theorem eval_binary {Value World : Type} (meaning : Interpretation Value World)
    (key : String) (element : String) (operation : Value → Value → Value → Prop)
    (binary : meaning.action key = Binary element operation) (aliases : Aliases) (env : Env Value)
    (operands : List Binding) (result : Binding) (rest : Code) (left right : Value)
    (arguments : readArguments false aliases env operands = [.scalar element left, .scalar element right])
    (kind : result.kind = .scalar element) (plain : result.known = none)
    (before after : World) (outputs : List (Runtime Value)) :
    Eval meaning false aliases (.action key operands [result] rest) env before outputs after ↔
      ∃ value, operation left right value ∧
        Eval meaning false (extendAliases aliases [result]) rest
          (put env result.name (.scalar element value)) before outputs after := by
  simp only [Eval, binary, arguments, binary_arguments]
  constructor
  · rintro ⟨values, middle, ⟨value, run, rfl, rfl⟩, _, suffix⟩
    exact ⟨value, run, suffix⟩
  · rintro ⟨value, run, suffix⟩
    exact ⟨[.scalar element value], before, ⟨value, run, rfl, rfl⟩,
      scalar_result result element kind plain value, suffix⟩

end Merlin.GraphLoopsShape.Epoch
