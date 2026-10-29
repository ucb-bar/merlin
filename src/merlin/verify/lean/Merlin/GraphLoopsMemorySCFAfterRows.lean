/- UNCOMPILED. Remove only irrelevant temporary SSA captures from a uniform
initializer's row relation, by deriving each actual store from its inputs.
The heap is never replaced by an assumed final equality or an initialized flag.
-/
import Merlin.GraphLoopsMemorySCFAfterSCF
import Merlin.GraphLoopsShapeEpochModel

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def CellWrite {Value : Type} (object rows columns : Nat) (value : Value) (row column : Nat)
    (before after : Epoch.Heap Value) : Prop :=
  GraphLoopsReuse.Geometry.Coordinate [rows, columns] [row, column] ∧
    after = GraphLoopsReuse.replaceCell before object
      (GraphLoopsReuse.Geometry.flatten [rows, columns] [row, column]) value

def RowWrite {Value : Type} (object rows columns : Nat) (value : Value) (row : Nat)
    (before after : Epoch.Heap Value) : Prop :=
  ∃ trace, GraphLoopsSegment.Execution (CellWrite object rows columns value row) columns 1 0 before trace after

/-- The erased environment may contain extra tile/add/min bindings. The only
scalar read by this row is the actual bound seed, proved at every cell.
-/
theorem nested_row_iff {Value : Type} (meaning : PureMeaning Value)
    (outer inner seed object rows columns row : Nat) (captures : GraphLoopsMemorySCF.Env Value)
    (value : Value) (seedValue : captures seed = some (.scalar value))
    (separate : outer ≠ inner) (seedOuter : seed ≠ outer) (seedInner : seed ≠ inner)
    (before after : Epoch.Heap Value) :
    NestedStores meaning [] ⟨[], seed, [outer, inner]⟩ rows columns outer inner object
      captures row before after ↔ RowWrite object rows columns value row before after := by
  rw [Epoch.uniform_nested_iff]
  apply exists_congr
  intro trace
  apply execution_live_iff
  intro column _ first last
  exact Epoch.uniform_store_iff meaning outer inner seed object rows columns row column captures value
    seedValue separate seedOuter seedInner first last

/-- Rejoin the derived rows to the old operation-level initializer relation.
This uses the same bound seed and heap, including every untouched cell.
-/
theorem rows_heap_iff {Value : Type} (meaning : PureMeaning Value)
    (outer inner seed object rows columns tile : Nat) (captures : GraphLoopsMemorySCF.Env Value)
    (value : Value) (seedValue : captures seed = some (.scalar value))
    (separate : outer ≠ inner) (seedOuter : seed ≠ outer) (seedInner : seed ≠ inner)
    (before after : Epoch.Heap Value) :
    (∃ trace, GraphLoopsSegment.Execution (RowWrite object rows columns value) rows 1 0 before trace after) ↔
      HeapRun meaning (Witness.uniform rows columns outer inner seed) tile object false captures before after := by
  simp only [HeapRun, Witness.uniform, HeapBody, Initializer.rows, List.headD_cons,
    Bool.false_eq_true, ↓reduceIte]
  apply exists_congr
  intro trace
  apply execution_live_iff
  intro row _ first last
  exact (nested_row_iff meaning outer inner seed object rows columns row captures value
    seedValue separate seedOuter seedInner first last).symm

/-- A checked ordinary row body has a capture-independent physical heap
meaning, so fresh generated controls need not be equated to old registers.
-/
theorem row_corresponds {Value Public : Type} (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (program : Epoch.Uniform)
    (rows columns : Nat) (element : String) (facts : program.Facts rows columns element)
    (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (object : Nat) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = Epoch.view object [rows, columns] element)
    (zeroValue : env program.zero.name = .index 0)
    (columnsValue : env program.columns.name = .index columns)
    (oneValue : env program.one.name = .index 1) (observed : Public) :
    Epoch.HeapBodyCorresponds
      (Epoch.regionBody (withMemory base classify) aliases [program.outer] program.rowCode env)
      (RowWrite object rows columns value) observed := by
  rcases facts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, outerKind, outerPlain, innerKind, innerPlain,
    seedOuter, seedInner, bufferOuter, bufferInner, separate, zeroOuter, columnsOuter, oneOuter⟩
  have rowsLaw := program.outer_corresponds base classify store loop aliases env object rows columns element value
    seedValue bufferValue zeroValue columnsValue oneValue seedOuter seedInner bufferOuter bufferInner separate
    zeroOuter columnsOuter oneOuter outerKind outerPlain innerKind innerPlain
    (fun _ _ _ => False) observed
  have captured : Epoch.eraseEnv env program.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, seedValue, Epoch.erase]
  intro row before outputs after
  rw [rowsLaw row before outputs after]
  exact and_congr_right (fun _ => and_congr_right (fun _ =>
    nested_row_iff (fun _ _ _ => False) program.outer.name program.inner.name program.seed.name
      object rows columns row (Epoch.eraseEnv env) value captured separate seedOuter seedInner before after.heap))

/-- The actual dynamic inner lower/upper SSA operands select the exact heap
execution interval. The upper bound is read from the emitted min result, never
replaced by the full shape or a Python-provided loop count.
-/
theorem TiledInitializer.inner_heap_iff {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledInitializer) (rows columns : Nat) (element : String)
    (facts : program.uniform.Facts rows columns element)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (object lower upper : Nat) (value : Value)
    (seedValue : env program.uniform.seed.name = .scalar element value)
    (bufferValue : env program.uniform.buffer.name = Epoch.view object [rows, columns] element)
    (zeroValue : env program.uniform.zero.name = .index 0)
    (columnsValue : env program.uniform.columns.name = .index columns)
    (oneValue : env program.uniform.one.name = .index 1)
    (lowerValue : env program.tileIV.name = .index lower)
    (upperValue : env program.bound.name = .index upper)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.inner env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        ∃ trace, GraphLoopsSegment.Execution (RowWrite object rows columns value)
          upper 1 lower before.heap trace after.heap := by
  rw [TiledInitializer.inner, Epoch.eval_region base classify program.uniform.loopKey loop]
  have controls : readArguments false aliases env [program.tileIV, program.bound, program.uniform.one] =
      [.index lower, .index upper, .index 1] := by
    change [env program.tileIV.name, env program.bound.name, env program.uniform.one.name] = _
    rw [lowerValue, upperValue, oneValue]
  rw [controls]
  have rowsLaw := row_corresponds base classify program.uniform rows columns element facts store loop
    aliases env object value seedValue bufferValue zeroValue columnsValue oneValue before.observed
  have result := Epoch.scf_heap_iff _ _ before.observed rowsLaw lower upper 1 before.heap [] after
  rcases before with ⟨observed, heap⟩
  rw [result]
  simp only [Nat.zero_lt_succ, true_and]

end Merlin.GraphLoopsMemorySCF.After
