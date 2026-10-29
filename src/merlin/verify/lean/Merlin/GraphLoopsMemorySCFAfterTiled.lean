/- UNCOMPILED. Candidate operational bridge for the actual generated tiled
rank-two initializer. Its add/min, dynamic inner bounds, empty yields, scoped
SSA and complete heap are interpreted; no whole-loop agreement is a premise.
-/
import Merlin.GraphLoopsMemorySCFAfterRows

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def TiledInitializer.captures (program : TiledInitializer) : List Binding :=
  [program.uniform.seed, program.uniform.buffer, program.uniform.zero, program.uniform.rows,
    program.uniform.columns, program.uniform.one, program.tile]

structure TiledInitializer.Inputs {Value : Type} (program : TiledInitializer)
    (env : GraphLoopsShape.Env Value) (object rows columns tile : Nat) (element : String) (value : Value) : Prop where
  seed : env program.uniform.seed.name = .scalar element value
  buffer : env program.uniform.buffer.name = Epoch.view object [rows, columns] element
  zero : env program.uniform.zero.name = .index 0
  rowsValue : env program.uniform.rows.name = .index rows
  columnsValue : env program.uniform.columns.name = .index columns
  one : env program.uniform.one.name = .index 1
  tileValue : env program.tile.name = .index tile

theorem TiledInitializer.Inputs.put {Value : Type} {program : TiledInitializer}
    {env : GraphLoopsShape.Env Value} {object rows columns tile : Nat} {element : String} {value : Value}
    (inputs : program.Inputs env object rows columns tile element value)
    (name : Nat) (replacement : Runtime Value) (fresh : name ∉ names program.captures) :
    program.Inputs (GraphLoopsShape.put env name replacement) object rows columns tile element value := by
  have read : ∀ binding ∈ program.captures, GraphLoopsShape.put env name replacement binding.name = env binding.name := by
    intro binding member
    have distinct : binding.name ≠ name := fun equal => fresh (equal ▸ List.mem_map_of_mem member)
    exact ite_eq_right distinct
  constructor
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.seed
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.buffer
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.zero
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.rowsValue
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.columnsValue
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.one
  · exact (read _ (by simp [TiledInitializer.captures])).trans inputs.tileValue

/-- All obligations here are finite syntax/binding facts, distinct from the
runtime input values and the parameterized execution proof.
-/
def TiledInitializer.ControlFacts (program : TiledInitializer) : Prop :=
  program.tileIV.kind = .index ∧ program.tileIV.known = none ∧
  program.sum.kind = .index ∧ program.sum.known = none ∧
  program.bound.kind = .index ∧ program.bound.known = none ∧
  program.tileIV.name ∉ names program.captures ∧
  program.sum.name ∉ names program.captures ∧
  program.bound.name ∉ names program.captures ∧
  program.tileIV.name ≠ program.sum.name ∧ program.tileIV.name ≠ program.bound.name

instance (program : TiledInitializer) : Decidable program.ControlFacts := by
  unfold TiledInitializer.ControlFacts
  infer_instance

theorem TiledInitializer.live_body {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledInitializer) (bits rows columns tile object : Nat) (element : String) (value : Value)
    (valid : (GraphLoopsSegment.Bounds.mk bits 0 rows 1 tile).Valid)
    (facts : program.uniform.Facts rows columns element) (controls : program.ControlFacts)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : program.Inputs env object rows columns tile element value) (observed : Public) :
    LiveHeapBodyCorresponds
      (Epoch.regionBody (withMemory base classify) aliases [program.tileIV] program.body env)
      (Chunk (RowWrite object rows columns value) rows 1 tile) observed rows := by
  rcases controls with ⟨ivKind, ivPlain, sumKind, sumPlain, boundKind, boundPlain,
    ivFresh, sumFresh, boundFresh, ivSum, ivBound⟩
  intro outer live before outputs after
  have typed := Epoch.index_parameter (Value := Value) program.tileIV outer ivKind ivPlain
  change (ValuesHold [program.tileIV] [.index outer] ∧ _) ↔ _
  simp only [typed, true_and]
  let first := GraphLoopsShape.put env program.tileIV.name (.index outer)
  have firstInputs := inputs.put program.tileIV.name (.index outer) ivFresh
  have ivValue : first program.tileIV.name = .index outer := by simp only [first, GraphLoopsShape.put, ↓reduceIte]
  have rowsFresh : program.uniform.rows.name ≠ program.sum.name := by
    intro equal
    apply sumFresh
    simp [names, TiledInitializer.captures, ← equal]
  have controlPrefix := program.control_prefix (withMemory base classify)
    ⟨bits, 0, rows, 1, tile⟩ valid add minimum sumKind sumPlain boundKind boundPlain rowsFresh
    (extendAliases aliases [program.tileIV]) first outer live ivValue
    (by simpa only [Nat.one_mul] using firstInputs.tileValue) firstInputs.rowsValue
    ⟨observed, before⟩ outputs after
  simp only [Nat.one_mul] at controlPrefix
  change Eval (withMemory base classify) false (extendAliases aliases [program.tileIV])
    program.body first ⟨observed, before⟩ outputs after ↔ _
  rw [controlPrefix]
  let second := GraphLoopsShape.put first program.sum.name (.index (outer + tile))
  let third := GraphLoopsShape.put second program.bound.name (.index (min (outer + tile) rows))
  have secondInputs := firstInputs.put program.sum.name (.index (outer + tile)) sumFresh
  have thirdInputs := secondInputs.put program.bound.name (.index (min (outer + tile) rows)) boundFresh
  have lowerValue : third program.tileIV.name = .index outer := by
    simp only [third, second, GraphLoopsShape.put, ivSum, ivBound, ↓reduceIte, ivValue]
  have upperValue : third program.bound.name = .index (min (outer + tile) rows) := by
    simp only [third, GraphLoopsShape.put, ↓reduceIte]
  have result := program.inner_heap_iff base classify rows columns element facts store loop
    (extendAliases (extendAliases (extendAliases aliases [program.tileIV]) [program.sum]) [program.bound])
    third object outer
    (min (outer + tile) rows) value thirdInputs.seed thirdInputs.buffer thirdInputs.zero
    thirdInputs.columnsValue thirdInputs.one lowerValue upperValue ⟨observed, before⟩ after outputs
  simpa only [Chunk, Nat.one_mul] using result

/-- Actual outer+inner SCF execution is the same ordered physical row/cell
execution for arbitrary fitting dimensions and tails, including empty shapes.
This remains a local initializer theorem, not a complete two-epoch program claim.
-/
theorem TiledInitializer.eval_rows {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledInitializer) (bits rows columns tile object : Nat) (element : String) (value : Value)
    (valid : (GraphLoopsSegment.Bounds.mk bits 0 rows 1 tile).Valid)
    (facts : program.uniform.Facts rows columns element) (controls : program.ControlFacts)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : program.Inputs env object rows columns tile element value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        ∃ trace, GraphLoopsSegment.Execution (RowWrite object rows columns value)
          rows 1 0 before.heap trace after.heap := by
  rw [TiledInitializer.code, Epoch.eval_region base classify program.uniform.loopKey loop]
  have arguments : readArguments false aliases env [program.uniform.zero, program.uniform.rows, program.tile] =
      [.index 0, .index rows, .index tile] := by
    change [env program.uniform.zero.name, env program.uniform.rows.name, env program.tile.name] = _
    rw [inputs.zero, inputs.rowsValue, inputs.tileValue]
  rw [arguments]
  have localStep := program.live_body base classify bits rows columns tile object element value valid facts controls
    store loop add minimum aliases env inputs before.observed
  have heapResult := scf_live_heap_iff _ _ before.observed 0 rows tile localStep before.heap [] after
  rcases before with ⟨observed, heap⟩
  rw [heapResult]
  have tilePositive : 0 < tile := Nat.lt_trans (by decide) valid.2.2.2.1
  have ordered := outer_source_iff (RowWrite object rows columns value) rows 1 tile 0
    (by decide) tilePositive heap after.heap
  simp only [Nat.one_mul] at ordered
  rw [ordered]
  simp only [tilePositive, true_and]

/-- The concrete generated initializer has the old segmented physical-heap
meaning, derived from its syntax/SSA instructions and ordered rows. This root
does not equate the surrounding actual final function with an epoch blueprint.
-/
theorem TiledInitializer.eval_epoch {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledInitializer) (bits rows columns tile object : Nat) (element : String) (value : Value)
    (valid : (GraphLoopsSegment.Bounds.mk bits 0 rows 1 tile).Valid)
    (facts : program.uniform.Facts rows columns element) (controls : program.ControlFacts)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : program.Inputs env object rows columns tile element value)
    (meaning : PureMeaning Value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        HeapRun meaning
          (Witness.uniform rows columns program.uniform.outer.name program.uniform.inner.name program.uniform.seed.name)
          tile object true (Epoch.eraseEnv env) before.heap after.heap := by
  rw [program.eval_rows base classify bits rows columns tile object element value valid facts controls
    store loop add minimum aliases env inputs before after outputs]
  rcases facts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    seedOuter, seedInner, _, _, separate, _⟩
  have captured : Epoch.eraseEnv env program.uniform.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, inputs.seed, Epoch.erase]
  have flat := rows_heap_iff meaning program.uniform.outer.name program.uniform.inner.name program.uniform.seed.name
    object rows columns tile (Epoch.eraseEnv env) value captured separate seedOuter seedInner before.heap after.heap
  have tilePositive : 0 < tile := Nat.lt_trans (by decide) valid.2.2.2.1
  have segmented := segmentation_exact_heaps meaning
    (Witness.uniform rows columns program.uniform.outer.name program.uniform.inner.name program.uniform.seed.name)
    tile object tilePositive (Epoch.eraseEnv env) before.heap after.heap
  exact and_congr_right (fun _ => and_congr_right (fun _ => flat.trans segmented))

end Merlin.GraphLoopsMemorySCF.After
