/- Actual tiled rank-one initializer. Every control and store is derived from its SSA operands. -/
import Merlin.GraphLoopsShapeEpochLine
import Merlin.GraphLoopsMemorySCFAfterTiled

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

structure TiledLine where
  uniform : Epoch.UniformLine
  tile : Binding
  tileIV : Binding
  sum : Binding
  bound : Binding
  addKey : String
  minKey : String
  deriving DecidableEq, Repr

def TiledLine.inner (program : TiledLine) : Code :=
  .region program.uniform.loopKey [program.tileIV, program.bound, program.uniform.one]
    [program.uniform.index] [] program.uniform.cell.code (.returned [])

def TiledLine.body (program : TiledLine) : Code :=
  .action program.addKey [program.tileIV, program.tile] [program.sum]
    (.action program.minKey [program.sum, program.uniform.extent] [program.bound] program.inner)

def TiledLine.code (program : TiledLine) : Code :=
  .region program.uniform.loopKey [program.uniform.zero, program.uniform.extent, program.tile]
    [program.tileIV] [] program.body (.returned [])

theorem TiledLine.control_prefix {Value World : Type} (program : TiledLine)
    (meaning : GraphLoopsShape.Interpretation Value World) (bounds : GraphLoopsSegment.Bounds) (valid : bounds.Valid)
    (add : meaning.action program.addKey = IndexAction bounds.bits .add)
    (minimum : meaning.action program.minKey = IndexAction bounds.bits .minimum)
    (sumKind : program.sum.kind = .index) (sumPlain : program.sum.known = none)
    (boundKind : program.bound.kind = .index) (boundPlain : program.bound.known = none)
    (rowsFresh : program.uniform.extent.name ≠ program.sum.name)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (outer : Nat) (live : outer < bounds.upper)
    (outerValue : env program.tileIV.name = .index outer)
    (tileValue : env program.tile.name = .index (bounds.step * bounds.tile))
    (rowsValue : env program.uniform.extent.name = .index bounds.upper)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false aliases program.body env before outputs after ↔
      Eval meaning false (extendAliases (extendAliases aliases [program.sum]) [program.bound]) program.inner
        (GraphLoopsShape.put (GraphLoopsShape.put env program.sum.name (.index (outer + bounds.step * bounds.tile))) program.bound.name
          (.index (min (outer + bounds.step * bounds.tile) bounds.upper))) before outputs after := by
  have constants := GraphLoopsSegment.Bounds.constants valid
  have controls := GraphLoopsSegment.Bounds.outer_control valid live
  rw [TiledLine.body]
  rw [eval_index_action meaning bounds.bits .add program.addKey add program.tileIV program.tile program.sum
    sumKind sumPlain aliases _ env outer (bounds.step * bounds.tile) outerValue tileValue
    controls.1 constants.2.2.2 controls.2.1]
  have sumValue : GraphLoopsShape.put env program.sum.name (.index (outer + bounds.step * bounds.tile)) program.sum.name =
      .index (outer + bounds.step * bounds.tile) := by simp only [GraphLoopsShape.put, ↓reduceIte]
  have upperValue : GraphLoopsShape.put env program.sum.name (.index (outer + bounds.step * bounds.tile)) program.uniform.extent.name =
      .index bounds.upper := by simp only [GraphLoopsShape.put, rowsFresh, ↓reduceIte, rowsValue]
  exact eval_index_action meaning bounds.bits .minimum program.minKey minimum program.sum program.uniform.extent program.bound
    boundKind boundPlain _ program.inner _ (outer + bounds.step * bounds.tile) bounds.upper sumValue upperValue
    controls.2.1 constants.2.1 controls.2.2 before outputs after


def PointWrite {Value : Type} (object extent : Nat) (value : Value) (coordinate : Nat)
    (before after : Epoch.Heap Value) : Prop :=
  GraphLoopsReuse.Geometry.Coordinate [extent] [coordinate] ∧
    after = GraphLoopsReuse.replaceCell before object
      (GraphLoopsReuse.Geometry.flatten [extent] [coordinate]) value

theorem line_heap_iff {Value : Type} (meaning : PureMeaning Value)
    (index seed object extent tile : Nat) (captures : GraphLoopsMemorySCF.Env Value)
    (value : Value) (seedValue : captures seed = some (.scalar value)) (seedFresh : seed ≠ index)
    (before after : Epoch.Heap Value) :
    (∃ trace, GraphLoopsSegment.Execution (PointWrite object extent value) extent 1 0 before trace after) ↔
      HeapRun meaning (Epoch.uniformLine extent index seed) tile object false captures before after := by
  simp only [HeapRun, Epoch.uniformLine, HeapBody, Initializer.rows, List.headD_cons, Bool.false_eq_true, ↓reduceIte]
  apply exists_congr
  intro trace
  apply execution_live_iff
  intro coordinate _ first last
  exact (Epoch.line_store_iff meaning index seed object extent coordinate captures value seedValue seedFresh first last).symm

theorem line_point_corresponds {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : Epoch.UniformLine) (extent : Nat) (element : String) (facts : program.Facts extent element)
    (store : classify program.storeKey = .store) (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (object : Nat) (value : Value) (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = Epoch.view object [extent] element) (observed : Public) :
    Epoch.HeapBodyCorresponds
      (Epoch.regionBody (withMemory base classify) aliases [program.index] program.cell.code env)
      (PointWrite object extent value) observed := by
  rcases facts with ⟨_, _, _, _, _, _, _, _, _, _, indexKind, indexPlain, seedFresh, bufferFresh⟩
  have bodyLaw := program.body_corresponds base classify store aliases env object extent element value
    seedValue bufferValue seedFresh bufferFresh indexKind indexPlain (fun _ _ _ => False) observed
  have captured : Epoch.eraseEnv env program.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, seedValue, Epoch.erase]
  intro coordinate before outputs after
  rw [bodyLaw coordinate before outputs after]
  exact and_congr_right (fun _ => and_congr_right (fun _ =>
    Epoch.line_store_iff (fun _ _ _ => False) program.index.name program.seed.name object extent coordinate
      (Epoch.eraseEnv env) value captured seedFresh before after.heap))

theorem TiledLine.inner_heap_iff {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledLine) (extent : Nat) (element : String) (facts : program.uniform.Facts extent element)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value) (object lower upper : Nat) (value : Value)
    (seedValue : env program.uniform.seed.name = .scalar element value)
    (bufferValue : env program.uniform.buffer.name = Epoch.view object [extent] element)
    (oneValue : env program.uniform.one.name = .index 1)
    (lowerValue : env program.tileIV.name = .index lower) (upperValue : env program.bound.name = .index upper)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.inner env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        ∃ trace, GraphLoopsSegment.Execution (PointWrite object extent value) upper 1 lower before.heap trace after.heap := by
  rw [TiledLine.inner, Epoch.eval_region base classify program.uniform.loopKey loop]
  have controls : readArguments false aliases env [program.tileIV, program.bound, program.uniform.one] =
      [.index lower, .index upper, .index 1] := by
    change [env program.tileIV.name, env program.bound.name, env program.uniform.one.name] = _
    rw [lowerValue, upperValue, oneValue]
  rw [controls]
  have pointsLaw := line_point_corresponds base classify program.uniform extent element facts store aliases env
    object value seedValue bufferValue before.observed
  have result := Epoch.scf_heap_iff _ _ before.observed pointsLaw lower upper 1 before.heap [] after
  rcases before with ⟨observed, heap⟩
  rw [result]
  simp only [Nat.zero_lt_succ, true_and]
def TiledLine.captures (program : TiledLine) : List Binding :=
  [program.uniform.seed, program.uniform.buffer, program.uniform.zero, program.uniform.extent,
    program.uniform.one, program.tile]

structure TiledLine.Inputs {Value : Type} (program : TiledLine)
    (env : GraphLoopsShape.Env Value) (object extent tile : Nat) (element : String) (value : Value) : Prop where
  seed : env program.uniform.seed.name = .scalar element value
  buffer : env program.uniform.buffer.name = Epoch.view object [extent] element
  zero : env program.uniform.zero.name = .index 0
  extentValue : env program.uniform.extent.name = .index extent
  one : env program.uniform.one.name = .index 1
  tileValue : env program.tile.name = .index tile

theorem TiledLine.Inputs.put {Value : Type} {program : TiledLine}
    {env : GraphLoopsShape.Env Value} {object extent tile : Nat} {element : String} {value : Value}
    (inputs : program.Inputs env object extent tile element value)
    (name : Nat) (replacement : Runtime Value) (fresh : name ∉ names program.captures) :
    program.Inputs (GraphLoopsShape.put env name replacement) object extent tile element value := by
  have read : ∀ binding ∈ program.captures, GraphLoopsShape.put env name replacement binding.name = env binding.name := by
    intro binding member
    have distinct : binding.name ≠ name := fun equal => fresh (equal ▸ List.mem_map_of_mem member)
    exact ite_eq_right distinct
  constructor
  · exact (read _ (by simp [TiledLine.captures])).trans inputs.seed
  · exact (read _ (by simp [TiledLine.captures])).trans inputs.buffer
  · exact (read _ (by simp [TiledLine.captures])).trans inputs.zero
  · exact (read _ (by simp [TiledLine.captures])).trans inputs.extentValue
  · exact (read _ (by simp [TiledLine.captures])).trans inputs.one
  · exact (read _ (by simp [TiledLine.captures])).trans inputs.tileValue

/-- All obligations here are finite syntax/binding facts, distinct from the
runtime input values and the parameterized execution proof.
-/
def TiledLine.ControlFacts (program : TiledLine) : Prop :=
  program.tileIV.kind = .index ∧ program.tileIV.known = none ∧
  program.sum.kind = .index ∧ program.sum.known = none ∧
  program.bound.kind = .index ∧ program.bound.known = none ∧
  program.tileIV.name ∉ names program.captures ∧
  program.sum.name ∉ names program.captures ∧
  program.bound.name ∉ names program.captures ∧
  program.tileIV.name ≠ program.sum.name ∧ program.tileIV.name ≠ program.bound.name

instance tiledLineControlFactsDecidable (program : TiledLine) : Decidable program.ControlFacts := by
  unfold TiledLine.ControlFacts
  infer_instance

theorem TiledLine.live_body {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledLine) (bits extent tile object : Nat) (element : String) (value : Value)
    (valid : (GraphLoopsSegment.Bounds.mk bits 0 extent 1 tile).Valid)
    (facts : program.uniform.Facts extent element) (controls : program.ControlFacts)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : program.Inputs env object extent tile element value) (observed : Public) :
    LiveHeapBodyCorresponds
      (Epoch.regionBody (withMemory base classify) aliases [program.tileIV] program.body env)
      (Chunk (PointWrite object extent value) extent 1 tile) observed extent := by
  rcases controls with ⟨ivKind, ivPlain, sumKind, sumPlain, boundKind, boundPlain,
    ivFresh, sumFresh, boundFresh, ivSum, ivBound⟩
  intro outer live before outputs after
  have typed := Epoch.index_parameter (Value := Value) program.tileIV outer ivKind ivPlain
  change (ValuesHold [program.tileIV] [.index outer] ∧ _) ↔ _
  simp only [typed, true_and]
  let first := GraphLoopsShape.put env program.tileIV.name (.index outer)
  have firstInputs := inputs.put program.tileIV.name (.index outer) ivFresh
  have ivValue : first program.tileIV.name = .index outer := by simp only [first, GraphLoopsShape.put, ↓reduceIte]
  have extentFresh : program.uniform.extent.name ≠ program.sum.name := by
    intro equal
    apply sumFresh
    simp [names, TiledLine.captures, ← equal]
  have controlPrefix := program.control_prefix (withMemory base classify)
    ⟨bits, 0, extent, 1, tile⟩ valid add minimum sumKind sumPlain boundKind boundPlain extentFresh
    (extendAliases aliases [program.tileIV]) first outer live ivValue
    (by simpa only [Nat.one_mul] using firstInputs.tileValue) firstInputs.extentValue
    ⟨observed, before⟩ outputs after
  simp only [Nat.one_mul] at controlPrefix
  change Eval (withMemory base classify) false (extendAliases aliases [program.tileIV])
    program.body first ⟨observed, before⟩ outputs after ↔ _
  rw [controlPrefix]
  let second := GraphLoopsShape.put first program.sum.name (.index (outer + tile))
  let third := GraphLoopsShape.put second program.bound.name (.index (min (outer + tile) extent))
  have secondInputs := firstInputs.put program.sum.name (.index (outer + tile)) sumFresh
  have thirdInputs := secondInputs.put program.bound.name (.index (min (outer + tile) extent)) boundFresh
  have lowerValue : third program.tileIV.name = .index outer := by
    simp only [third, second, GraphLoopsShape.put, ivSum, ivBound, ↓reduceIte, ivValue]
  have upperValue : third program.bound.name = .index (min (outer + tile) extent) := by
    simp only [third, GraphLoopsShape.put, ↓reduceIte]
  have result := program.inner_heap_iff base classify extent element facts store loop
    (extendAliases (extendAliases (extendAliases aliases [program.tileIV]) [program.sum]) [program.bound])
    third object outer
    (min (outer + tile) extent) value thirdInputs.seed thirdInputs.buffer thirdInputs.one lowerValue upperValue ⟨observed, before⟩ after outputs
  simpa only [Chunk, Nat.one_mul] using result

/-- Actual outer+inner SCF execution is the same ordered physical one-dimensional
execution for arbitrary fitting extents and tails, including empty shapes.
This remains a local initializer theorem, not a complete two-epoch program claim.
-/
theorem TiledLine.eval_points {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledLine) (bits extent tile object : Nat) (element : String) (value : Value)
    (valid : (GraphLoopsSegment.Bounds.mk bits 0 extent 1 tile).Valid)
    (facts : program.uniform.Facts extent element) (controls : program.ControlFacts)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : program.Inputs env object extent tile element value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        ∃ trace, GraphLoopsSegment.Execution (PointWrite object extent value)
          extent 1 0 before.heap trace after.heap := by
  rw [TiledLine.code, Epoch.eval_region base classify program.uniform.loopKey loop]
  have arguments : readArguments false aliases env [program.uniform.zero, program.uniform.extent, program.tile] =
      [.index 0, .index extent, .index tile] := by
    change [env program.uniform.zero.name, env program.uniform.extent.name, env program.tile.name] = _
    rw [inputs.zero, inputs.extentValue, inputs.tileValue]
  rw [arguments]
  have localStep := program.live_body base classify bits extent tile object element value valid facts controls
    store loop add minimum aliases env inputs before.observed
  have heapResult := scf_live_heap_iff _ _ before.observed 0 extent tile localStep before.heap [] after
  rcases before with ⟨observed, heap⟩
  rw [heapResult]
  have tilePositive : 0 < tile := Nat.lt_trans (by decide) valid.2.2.2.1
  have ordered := outer_source_iff (PointWrite object extent value) extent 1 tile 0
    (by decide) tilePositive heap after.heap
  simp only [Nat.one_mul] at ordered
  rw [ordered]
  simp only [tilePositive, true_and]

/-- The concrete generated initializer has the old segmented physical-heap
meaning, derived from its syntax/SSA instructions and ordered extent. This root
does not equate the surrounding actual final function with an epoch blueprint.
-/
theorem TiledLine.eval_epoch {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : TiledLine) (bits extent tile object : Nat) (element : String) (value : Value)
    (valid : (GraphLoopsSegment.Bounds.mk bits 0 extent 1 tile).Valid)
    (facts : program.uniform.Facts extent element) (controls : program.ControlFacts)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : program.Inputs env object extent tile element value)
    (meaning : PureMeaning Value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        HeapRun meaning
          (Epoch.uniformLine extent program.uniform.index.name program.uniform.seed.name)
          tile object true (Epoch.eraseEnv env) before.heap after.heap := by
  rw [program.eval_points base classify bits extent tile object element value valid facts controls
    store loop add minimum aliases env inputs before after outputs]
  rcases facts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, seedFresh, _⟩
  have captured : Epoch.eraseEnv env program.uniform.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, inputs.seed, Epoch.erase]
  have flat := line_heap_iff meaning program.uniform.index.name program.uniform.seed.name
    object extent tile (Epoch.eraseEnv env) value captured seedFresh before.heap after.heap
  have tilePositive : 0 < tile := Nat.lt_trans (by decide) valid.2.2.2.1
  have segmented := segmentation_exact_heaps meaning
    (Epoch.uniformLine extent program.uniform.index.name program.uniform.seed.name)
    tile object tilePositive (Epoch.eraseEnv env) before.heap after.heap
  exact and_congr_right (fun _ => and_congr_right (fun _ => flat.trans segmented))


end Merlin.GraphLoopsMemorySCF.After
