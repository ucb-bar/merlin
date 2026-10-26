/- Actual uniform rank-two SCF initializer adapter. Both models execute the
same bound seed and induction variables; pure bodies are empty here. -/
import Merlin.GraphLoopsShapeEpochCells

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure Uniform where
  storeKey : String
  loopKey : String
  seed : Binding
  buffer : Binding
  outer : Binding
  inner : Binding
  zero : Binding
  rows : Binding
  columns : Binding
  one : Binding
  deriving DecidableEq, Repr

def Uniform.cell (program : Uniform) : Cell :=
  ⟨program.storeKey, program.seed, program.buffer, [program.outer, program.inner]⟩

def Uniform.rowCode (program : Uniform) : Code :=
  .region program.loopKey [program.zero, program.columns, program.one] [program.inner] []
    program.cell.code (.returned [])

def Uniform.code (program : Uniform) : Code :=
  .region program.loopKey [program.zero, program.rows, program.one] [program.outer] []
    program.rowCode (.returned [])

def regionBody {Value Public : Type} (meaning : Interpretation Value (MemoryWorld Value Public))
    (aliases : Aliases) (parameters : List Binding) (code : Code) (env : Env Value) :
    RegionMeaning Value (MemoryWorld Value Public) :=
  fun inputs before outputs after => ValuesHold parameters inputs ∧
    Eval meaning false (extendAliases aliases parameters) code (assign env parameters inputs) before outputs after

theorem eval_region {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (key : String) (loop : classify key = .loop)
    (aliases : Aliases) (operands parameters : List Binding) (code : Code) (env : Env Value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases (.region key operands parameters [] code (.returned []))
      env before outputs after ↔
    outputs = [] ∧ SCF (regionBody (withMemory base classify) aliases parameters code env)
      (readArguments false aliases env operands) before [] after := by
  change (∃ values middle, (withMemory base classify).region key (readArguments false aliases env operands)
    (regionBody (withMemory base classify) aliases parameters code env) before values middle ∧
      ValuesHold [] values ∧ outputs = [] ∧ after = middle) ↔ _
  simp only [withMemory, loop]
  constructor
  · rintro ⟨values, middle, run, typed, rfl, rfl⟩
    cases typed
    exact ⟨rfl, run⟩
  · rintro ⟨rfl, run⟩
    exact ⟨[], after, run, .nil, rfl, rfl⟩

theorem uniform_store_iff {Value : Type} (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (outer inner seed object rows columns row column : Nat) (captures : GraphLoopsMemorySCF.Env Value)
    (value : Value) (seedValue : captures seed = some (.scalar value))
    (separate : outer ≠ inner) (seedOuter : seed ≠ outer) (seedInner : seed ≠ inner)
    (before after : Heap Value) :
    GraphLoopsMemorySCF.StoreBody meaning ⟨[], seed, [outer, inner]⟩ [rows, columns] inner object
      (GraphLoopsMemorySCF.put captures outer (.index row)) column before after ↔
    Merlin.GraphLoopsReuse.Geometry.Coordinate [rows, columns] [row, column] ∧
      after = Merlin.GraphLoopsReuse.replaceCell before object
        (Merlin.GraphLoopsReuse.Geometry.flatten [rows, columns] [row, column]) value := by
  have arguments : GraphLoopsMemorySCF.read
      (GraphLoopsMemorySCF.put (GraphLoopsMemorySCF.put captures outer (.index row)) inner (.index column)) [outer, inner] =
        some ([row, column].map GraphLoopsMemorySCF.Datum.index) := by
    simp only [GraphLoopsMemorySCF.read, GraphLoopsMemorySCF.put, separate, ↓reduceIte,
      List.map_cons, List.map_nil, Option.pure_def]
    rfl
  constructor
  · rintro ⟨coordinates, stored, result, inside, final⟩
    have coordinatesEqual : coordinates = [row, column] :=
      result.coordinates (by simp [GraphLoopsMemorySCF.definitions]) arguments
    subst coordinates
    rcases result with ⟨env, computed, scalar, _⟩
    cases computed
    have storedEqual : stored = value := by
      simp only [GraphLoopsMemorySCF.put, seedInner, seedOuter, ↓reduceIte, seedValue] at scalar
      exact (GraphLoopsMemorySCF.Datum.scalar.inj (Option.some.inj scalar)).symm
    subst stored
    exact ⟨inside, final⟩
  · rintro ⟨inside, final⟩
    exact ⟨[row, column], value,
      GraphLoopsMemorySCF.Witness.leaf_result meaning outer inner seed row column separate seedOuter seedInner
        captures value seedValue, inside, final⟩

/-- Reconstruct a cell's SSA inputs after both actual lexical IV definitions. -/
theorem Uniform.inner_corresponds {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (program : Uniform) (store : classify program.storeKey = .store)
    (aliases : Aliases) (env : Env Value) (object rows columns row : Nat) (element : String) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [rows, columns] element)
    (seedOuter : program.seed.name ≠ program.outer.name) (seedInner : program.seed.name ≠ program.inner.name)
    (bufferOuter : program.buffer.name ≠ program.outer.name) (bufferInner : program.buffer.name ≠ program.inner.name)
    (separate : program.outer.name ≠ program.inner.name)
    (innerKind : program.inner.kind = .index) (innerPlain : program.inner.known = none)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (observed : Public) :
    HeapBodyCorresponds
      (regionBody (withMemory base classify) (extendAliases aliases [program.outer]) [program.inner] program.cell.code
        (assign env [program.outer] [.index row]))
      (GraphLoopsMemorySCF.StoreBody meaning ⟨[], program.seed.name, [program.outer.name, program.inner.name]⟩
        [rows, columns] program.inner.name object
        (GraphLoopsMemorySCF.put (eraseEnv env) program.outer.name (.index row))) observed := by
  intro column before outputs after
  have inputs := uniform_cell_inputs env program.seed program.buffer program.outer program.inner program.storeKey
    object rows columns row column element value seedValue bufferValue seedOuter seedInner bufferOuter bufferInner separate
  have typed := index_parameter (Value := Value) program.inner column innerKind innerPlain
  change (ValuesHold [program.inner] [.index column] ∧ _) ↔ _
  simp only [typed, true_and]
  rw [Cell.eval base classify program.cell store _ _ object [rows, columns] [row, column] element value inputs]
  have captured : eraseEnv env program.seed.name = some (.scalar value) := by simp only [eraseEnv, seedValue, erase]
  rw [uniform_store_iff meaning program.outer.name program.inner.name program.seed.name object rows columns row column
    (eraseEnv env) value captured separate seedOuter seedInner]
  rcases after with ⟨afterPublic, afterHeap⟩
  simp only [GraphLoopsReuse.Memory.mk.injEq]
  constructor
  · rintro ⟨inside, outputs, samePublic, sameHeap⟩
    exact ⟨outputs, samePublic, inside, sameHeap⟩
  · rintro ⟨outputs, samePublic, inside, sameHeap⟩
    exact ⟨inside, outputs, samePublic, sameHeap⟩

theorem uniform_nested_iff {Value : Type} (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (outer inner seed object rows columns row : Nat) (captures : GraphLoopsMemorySCF.Env Value)
    (before after : Heap Value) :
    GraphLoopsMemorySCF.NestedStores meaning [] ⟨[], seed, [outer, inner]⟩ rows columns outer inner object
      captures row before after ↔
      ∃ trace, GraphLoopsSegment.Execution
        (GraphLoopsMemorySCF.StoreBody meaning ⟨[], seed, [outer, inner]⟩ [rows, columns] inner object
          (GraphLoopsMemorySCF.put captures outer (.index row))) columns 1 0 before trace after := by
  constructor
  · rintro ⟨env, trace, pure, run⟩
    cases pure
    exact ⟨trace, run⟩
  · rintro ⟨trace, run⟩
    exact ⟨_, trace, .done _, run⟩

theorem Uniform.outer_corresponds {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (program : Uniform)
    (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (object rows columns : Nat) (element : String) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [rows, columns] element)
    (zeroValue : env program.zero.name = .index 0) (columnsValue : env program.columns.name = .index columns)
    (oneValue : env program.one.name = .index 1)
    (seedOuter : program.seed.name ≠ program.outer.name) (seedInner : program.seed.name ≠ program.inner.name)
    (bufferOuter : program.buffer.name ≠ program.outer.name) (bufferInner : program.buffer.name ≠ program.inner.name)
    (separate : program.outer.name ≠ program.inner.name)
    (zeroOuter : program.zero.name ≠ program.outer.name) (columnsOuter : program.columns.name ≠ program.outer.name)
    (oneOuter : program.one.name ≠ program.outer.name)
    (outerKind : program.outer.kind = .index) (outerPlain : program.outer.known = none)
    (innerKind : program.inner.kind = .index) (innerPlain : program.inner.known = none)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (observed : Public) :
    HeapBodyCorresponds
      (regionBody (withMemory base classify) aliases [program.outer] program.rowCode env)
      (GraphLoopsMemorySCF.NestedStores meaning [] ⟨[], program.seed.name, [program.outer.name, program.inner.name]⟩
        rows columns program.outer.name program.inner.name object (eraseEnv env)) observed := by
  intro row before outputs after
  have typed := index_parameter (Value := Value) program.outer row outerKind outerPlain
  change (ValuesHold [program.outer] [.index row] ∧ _) ↔ _
  simp only [typed, true_and]
  rw [Uniform.rowCode, eval_region base classify program.loopKey loop]
  have controls : readArguments false (extendAliases aliases [program.outer])
      (assign env [program.outer] [.index row]) [program.zero, program.columns, program.one] =
        [.index 0, .index columns, .index 1] := by
    change [assign env [program.outer] [.index row] program.zero.name,
      assign env [program.outer] [.index row] program.columns.name,
      assign env [program.outer] [.index row] program.one.name] = _
    rw [lookup_assign_index _ _ _ _ zeroOuter, lookup_assign_index _ _ _ _ columnsOuter,
      lookup_assign_index _ _ _ _ oneOuter, zeroValue, columnsValue, oneValue]
  rw [controls]
  have bodyLaw := program.inner_corresponds base classify store aliases env object rows columns row element value
    seedValue bufferValue seedOuter seedInner bufferOuter bufferInner separate innerKind innerPlain meaning observed
  rw [scf_heap_iff _ _ observed bodyLaw]
  rw [uniform_nested_iff]
  simp only [Nat.zero_lt_succ, true_and]

/-- The complete typed two-loop body equals the old operation-level HeapRun.
All SSA inputs and scoped IV effects are derived above. Shape/volume remains a
parameter; this theorem introduces no whole-initializer agreement assumption. -/
theorem Uniform.eval_iff {Value Public : Type} (base : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (program : Uniform)
    (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (object rows columns tile : Nat) (element : String) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [rows, columns] element)
    (zeroValue : env program.zero.name = .index 0) (rowsValue : env program.rows.name = .index rows)
    (columnsValue : env program.columns.name = .index columns) (oneValue : env program.one.name = .index 1)
    (seedOuter : program.seed.name ≠ program.outer.name) (seedInner : program.seed.name ≠ program.inner.name)
    (bufferOuter : program.buffer.name ≠ program.outer.name) (bufferInner : program.buffer.name ≠ program.inner.name)
    (separate : program.outer.name ≠ program.inner.name)
    (zeroOuter : program.zero.name ≠ program.outer.name) (columnsOuter : program.columns.name ≠ program.outer.name)
    (oneOuter : program.one.name ≠ program.outer.name)
    (outerKind : program.outer.kind = .index) (outerPlain : program.outer.known = none)
    (innerKind : program.inner.kind = .index) (innerPlain : program.inner.known = none)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (before after : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        GraphLoopsMemorySCF.HeapRun meaning
          (GraphLoopsMemorySCF.Witness.uniform rows columns program.outer.name program.inner.name program.seed.name)
          tile object false (eraseEnv env) before.heap after.heap := by
  rw [Uniform.code, eval_region base classify program.loopKey loop]
  have controls : readArguments false aliases env [program.zero, program.rows, program.one] =
      [.index 0, .index rows, .index 1] := by
    change [env program.zero.name, env program.rows.name, env program.one.name] = _
    rw [zeroValue, rowsValue, oneValue]
  rw [controls]
  have bodyLaw := program.outer_corresponds base classify store loop aliases env object rows columns element value
    seedValue bufferValue zeroValue columnsValue oneValue seedOuter seedInner bufferOuter bufferInner separate
    zeroOuter columnsOuter oneOuter outerKind outerPlain innerKind innerPlain meaning before.observed
  have result := scf_heap_iff _ _ before.observed bodyLaw 0 rows 1 before.heap [] after
  rcases before with ⟨observed, heap⟩
  rw [result]
  simp only [Nat.zero_lt_succ, true_and, GraphLoopsMemorySCF.HeapRun,
    GraphLoopsMemorySCF.Witness.uniform, GraphLoopsMemorySCF.HeapBody,
    GraphLoopsMemorySCF.Initializer.rows, List.headD_cons, Bool.false_eq_true, ↓reduceIte]

end Merlin.GraphLoopsShape.Epoch
