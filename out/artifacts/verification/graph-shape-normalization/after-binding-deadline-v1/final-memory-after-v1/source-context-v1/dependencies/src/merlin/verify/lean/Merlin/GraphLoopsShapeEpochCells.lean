/- Derive the initializer's typed SSA store from actual operand bindings.
The lemma unfolds the concrete memory operation; it never assumes whole-body
or whole-initializer agreement. -/
import Merlin.GraphLoopsShapeEpochSCF

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure Cell where
  key : String
  seed : Binding
  buffer : Binding
  indices : List Binding
  deriving DecidableEq, Repr

def Cell.code (cell : Cell) : Code :=
  .action cell.key (cell.seed :: cell.buffer :: cell.indices) [] (.returned [])

def Cell.Inputs {Value : Type} (cell : Cell) (env : Env Value) (object : Nat) (shape : List Nat)
    (element : String) (coordinates : List Nat) (value : Value) : Prop :=
  env cell.seed.name = .scalar element value ∧ env cell.buffer.name = view object shape element ∧
    readArguments false [] env cell.indices = coordinates.map Runtime.index

theorem Cell.arguments {Value : Type} {cell : Cell} {env : Env Value} {object : Nat} {shape coordinates : List Nat}
    {element : String} {value : Value} (inputs : cell.Inputs env object shape element coordinates value)
    (aliases : Aliases) :
    readArguments false aliases env (cell.seed :: cell.buffer :: cell.indices) =
      .scalar element value :: view object shape element :: coordinates.map Runtime.index := by
  change env cell.seed.name :: env cell.buffer.name :: readArguments false [] env cell.indices = _
  rw [inputs.1, inputs.2.1, inputs.2.2]

theorem Cell.eval {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (cell : Cell) (store : classify cell.key = .store)
    (aliases : Aliases) (env : Env Value) (object : Nat) (shape coordinates : List Nat) (element : String)
    (value : Value) (inputs : cell.Inputs env object shape element coordinates value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases cell.code env before outputs after ↔
      Merlin.GraphLoopsReuse.Geometry.Coordinate shape coordinates ∧ outputs = [] ∧
        after = ⟨before.observed, Merlin.GraphLoopsReuse.replaceCell before.heap object
          (Merlin.GraphLoopsReuse.Geometry.flatten shape coordinates) value⟩ := by
  change (∃ values middle,
    (withMemory base classify).action cell.key (readArguments false aliases env (cell.seed :: cell.buffer :: cell.indices))
      before values middle ∧ ValuesHold [] values ∧ outputs = [] ∧ after = middle) ↔ _
  rw [Cell.arguments inputs aliases]
  simp only [withMemory, store, store_iff]
  constructor
  · rintro ⟨values, middle, ⟨inside, rfl, rfl⟩, _, rfl, rfl⟩
    exact ⟨inside, rfl, rfl⟩
  · rintro ⟨inside, rfl, rfl⟩
    exact ⟨[], _, ⟨inside, rfl, rfl⟩, .nil, rfl, rfl⟩

/-- Bound outer captures survive one fresh induction-variable definition. -/
theorem lookup_assign_index {Value : Type} (env : Env Value) (binding : Binding) (index name : Nat)
    (different : name ≠ binding.name) : assign env [binding] [.index index] name = env name := by
  simp only [assign, put, different, ↓reduceIte]

theorem lookup_assign_index_at {Value : Type} (env : Env Value) (binding : Binding) (index : Nat) :
    assign env [binding] [.index index] binding.name = .index index := by
  simp only [assign, put, ↓reduceIte]

theorem index_parameter {Value : Type} (binding : Binding) (index : Nat)
    (kind : binding.kind = .index) (plain : binding.known = none) :
    ValuesHold [binding] [Runtime.index (Value := Value) index] :=
  .cons (Binding.holds_plain binding (.index index) kind.symm plain) .nil

/-- The uniform source body reads the actual seed/view/row/column SSA operands.
These local scope inequalities are discharged from the checked binding map. -/
theorem uniform_cell_inputs {Value : Type} (env : Env Value) (seed buffer outer inner : Binding)
    (key : String) (object rows columns row column : Nat) (element : String) (value : Value)
    (seedValue : env seed.name = .scalar element value)
    (bufferValue : env buffer.name = view object [rows, columns] element)
    (seedOuter : seed.name ≠ outer.name) (seedInner : seed.name ≠ inner.name)
    (bufferOuter : buffer.name ≠ outer.name) (bufferInner : buffer.name ≠ inner.name)
    (separate : outer.name ≠ inner.name) :
    (Cell.mk key seed buffer [outer, inner]).Inputs
      (assign (assign env [outer] [.index row]) [inner] [.index column]) object [rows, columns] element [row, column] value := by
  refine ⟨?_, ?_, ?_⟩
  · rw [lookup_assign_index _ _ _ _ seedInner, lookup_assign_index _ _ _ _ seedOuter]
    exact seedValue
  · rw [lookup_assign_index _ _ _ _ bufferInner, lookup_assign_index _ _ _ _ bufferOuter]
    exact bufferValue
  · change [assign (assign env [outer] [.index row]) [inner] [.index column] outer.name,
      assign (assign env [outer] [.index row]) [inner] [.index column] inner.name] = [.index row, .index column]
    rw [lookup_assign_index _ _ _ _ separate, lookup_assign_index_at, lookup_assign_index_at]

end Merlin.GraphLoopsShape.Epoch
