/- Actual one-loop rank-one initialization, with derived store/SCF correspondence.
Its concrete stores and loop bounds are checked against the actual typed source. -/
import Merlin.GraphLoopsShapeEpochModel

set_option autoImplicit false
namespace Merlin.GraphLoopsShape.Epoch

structure UniformLine where
  storeKey : String
  loopKey : String
  seed : Binding
  buffer : Binding
  index : Binding
  zero : Binding
  extent : Binding
  one : Binding
  deriving DecidableEq, Repr

def UniformLine.cell (program : UniformLine) : Cell :=
  ⟨program.storeKey, program.seed, program.buffer, [program.index]⟩

def UniformLine.code (program : UniformLine) : Code :=
  .region program.loopKey [program.zero, program.extent, program.one] [program.index] []
    program.cell.code (.returned [])

def uniformLine (extent index seed : Nat) : GraphLoopsMemorySCF.Initializer :=
  ⟨[extent], [index], [], ⟨[], seed, [index]⟩⟩

theorem line_leaf_result {Value : Type} (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (index seed coordinate : Nat) (different : seed ≠ index)
    (captures : GraphLoopsMemorySCF.Env Value) (value : Value)
    (captured : captures seed = some (.scalar value)) :
    GraphLoopsMemorySCF.LeafResult meaning ⟨[], seed, [index]⟩
      (GraphLoopsMemorySCF.put captures index (.index coordinate)) [coordinate] value := by
  refine ⟨_, .done _, ?_, ?_⟩
  · simp only [GraphLoopsMemorySCF.put, ite_eq_right different, captured]
  · simp only [GraphLoopsMemorySCF.read, GraphLoopsMemorySCF.put, ↓reduceIte,
      List.map_cons, List.map_nil, Option.pure_def]
    rfl

theorem line_store_iff {Value : Type} (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (index seed object extent coordinate : Nat) (captures : GraphLoopsMemorySCF.Env Value)
    (value : Value) (seedValue : captures seed = some (.scalar value)) (different : seed ≠ index)
    (before after : Heap Value) :
    GraphLoopsMemorySCF.StoreBody meaning ⟨[], seed, [index]⟩ [extent] index object
      captures coordinate before after ↔
    GraphLoopsReuse.Geometry.Coordinate [extent] [coordinate] ∧
      after = GraphLoopsReuse.replaceCell before object
        (GraphLoopsReuse.Geometry.flatten [extent] [coordinate]) value := by
  have arguments : GraphLoopsMemorySCF.read
      (GraphLoopsMemorySCF.put captures index (.index coordinate)) [index] =
        some ([coordinate].map GraphLoopsMemorySCF.Datum.index) := by
    simp only [GraphLoopsMemorySCF.read, GraphLoopsMemorySCF.put, ↓reduceIte,
      List.map_cons, List.map_nil, Option.pure_def]
    rfl
  constructor
  · rintro ⟨coordinates, stored, result, inside, final⟩
    have coordinatesEqual : coordinates = [coordinate] :=
      result.coordinates (by simp [GraphLoopsMemorySCF.definitions]) arguments
    subst coordinates
    rcases result with ⟨env, computed, scalar, _⟩
    cases computed
    have storedEqual : stored = value := by
      simp only [GraphLoopsMemorySCF.put, ite_eq_right different, seedValue] at scalar
      exact (GraphLoopsMemorySCF.Datum.scalar.inj (Option.some.inj scalar)).symm
    subst stored
    exact ⟨inside, final⟩
  · rintro ⟨inside, final⟩
    exact ⟨[coordinate], value,
      line_leaf_result meaning index seed coordinate different captures value seedValue, inside, final⟩

theorem line_cell_inputs {Value : Type} (env : Env Value) (seed buffer index : Binding)
    (key : String) (object extent coordinate : Nat) (element : String) (value : Value)
    (seedValue : env seed.name = .scalar element value)
    (bufferValue : env buffer.name = view object [extent] element)
    (seedFresh : seed.name ≠ index.name) (bufferFresh : buffer.name ≠ index.name) :
    (Cell.mk key seed buffer [index]).Inputs
      (assign env [index] [.index coordinate]) object [extent] element [coordinate] value := by
  refine ⟨?_, ?_, ?_⟩
  · rw [lookup_assign_index _ _ _ _ seedFresh]
    exact seedValue
  · rw [lookup_assign_index _ _ _ _ bufferFresh]
    exact bufferValue
  · change [assign env [index] [.index coordinate] index.name] = [.index coordinate]
    rw [lookup_assign_index_at]

theorem UniformLine.body_corresponds {Value Public : Type}
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : UniformLine) (store : classify program.storeKey = .store)
    (aliases : Aliases) (env : Env Value) (object extent : Nat) (element : String) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [extent] element)
    (seedFresh : program.seed.name ≠ program.index.name)
    (bufferFresh : program.buffer.name ≠ program.index.name)
    (indexKind : program.index.kind = .index) (indexPlain : program.index.known = none)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (observed : Public) :
    HeapBodyCorresponds
      (regionBody (withMemory base classify) aliases [program.index] program.cell.code env)
      (GraphLoopsMemorySCF.StoreBody meaning ⟨[], program.seed.name, [program.index.name]⟩
        [extent] program.index.name object (eraseEnv env)) observed := by
  intro coordinate before outputs after
  have inputs := line_cell_inputs env program.seed program.buffer program.index program.storeKey
    object extent coordinate element value seedValue bufferValue seedFresh bufferFresh
  have typed := index_parameter (Value := Value) program.index coordinate indexKind indexPlain
  change (ValuesHold [program.index] [.index coordinate] ∧ _) ↔ _
  simp only [typed, true_and]
  rw [Cell.eval base classify program.cell store _ _ object [extent] [coordinate] element value inputs]
  have captured : eraseEnv env program.seed.name = some (.scalar value) := by
    simp only [eraseEnv, seedValue, erase]
  rw [line_store_iff meaning program.index.name program.seed.name object extent coordinate
    (eraseEnv env) value captured seedFresh]
  rcases after with ⟨afterPublic, afterHeap⟩
  simp only [GraphLoopsReuse.Memory.mk.injEq]
  constructor
  · rintro ⟨inside, outputs, samePublic, sameHeap⟩
    exact ⟨outputs, samePublic, inside, sameHeap⟩
  · rintro ⟨outputs, samePublic, inside, sameHeap⟩
    exact ⟨inside, outputs, samePublic, sameHeap⟩

theorem UniformLine.eval_iff {Value Public : Type}
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : UniformLine) (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (object extent tile : Nat) (element : String) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [extent] element)
    (zeroValue : env program.zero.name = .index 0) (extentValue : env program.extent.name = .index extent)
    (oneValue : env program.one.name = .index 1)
    (seedFresh : program.seed.name ≠ program.index.name)
    (bufferFresh : program.buffer.name ≠ program.index.name)
    (indexKind : program.index.kind = .index) (indexPlain : program.index.known = none)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (before after : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        GraphLoopsMemorySCF.HeapRun meaning (uniformLine extent program.index.name program.seed.name)
          tile object false (eraseEnv env) before.heap after.heap := by
  rw [UniformLine.code, eval_region base classify program.loopKey loop]
  have controls : readArguments false aliases env [program.zero, program.extent, program.one] =
      [.index 0, .index extent, .index 1] := by
    change [env program.zero.name, env program.extent.name, env program.one.name] = _
    rw [zeroValue, extentValue, oneValue]
  rw [controls]
  have bodyLaw := program.body_corresponds base classify store aliases env object extent element value
    seedValue bufferValue seedFresh bufferFresh indexKind indexPlain meaning before.observed
  have result := scf_heap_iff _ _ before.observed bodyLaw 0 extent 1 before.heap [] after
  rcases before with ⟨observed, heap⟩
  rw [result]
  simp only [Nat.zero_lt_succ, true_and, GraphLoopsMemorySCF.HeapRun,
    uniformLine, GraphLoopsMemorySCF.HeapBody, GraphLoopsMemorySCF.Initializer.rows,
    List.headD_cons, Bool.false_eq_true, ↓reduceIte]


def UniformLine.Facts (program : UniformLine) (extent : Nat) (element : String) : Prop :=
  program.seed.kind = .scalar element ∧ program.seed.known = none ∧
  program.buffer.kind = .view (type [extent] element) ∧ program.buffer.known = none ∧
  program.zero.kind = .index ∧ program.zero.known = some 0 ∧
  program.extent.kind = .index ∧ program.extent.known = some extent ∧
  program.one.kind = .index ∧ program.one.known = some 1 ∧
  program.index.kind = .index ∧ program.index.known = none ∧
  program.seed.name ≠ program.index.name ∧ program.buffer.name ≠ program.index.name

instance uniformLineFactsDecidable (program : UniformLine) (extent : Nat) (element : String) :
    Decidable (program.Facts extent element) := by
  unfold UniformLine.Facts
  infer_instance

theorem UniformLine.checked_eval {Value Public : Type}
    (base : Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : UniformLine) (extent : Nat) (element : String)
    (checked : decide (program.Facts extent element) = true)
    (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (object tile : Nat) (value : Value)
    (seedValue : env program.seed.name = .scalar element value)
    (bufferValue : env program.buffer.name = view object [extent] element)
    (zeroValue : env program.zero.name = .index 0) (extentValue : env program.extent.name = .index extent)
    (oneValue : env program.one.name = .index 1)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value) (before after : MemoryWorld Value Public)
    (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        GraphLoopsMemorySCF.HeapRun meaning (uniformLine extent program.index.name program.seed.name)
          tile object false (eraseEnv env) before.heap after.heap := by
  have facts := of_decide_eq_true checked
  rcases facts with ⟨_, _, _, _, _, _, _, _, _, _, indexKind, indexPlain, seedFresh, bufferFresh⟩
  exact program.eval_iff base classify store loop aliases env object extent tile element value
    seedValue bufferValue zeroValue extentValue oneValue seedFresh bufferFresh indexKind indexPlain meaning before after outputs

theorem uniformLine_wellFormed (extent index seed : Nat) (positive : 0 < extent) :
    (uniformLine extent index seed).WellFormed := by
  refine ⟨Or.inl ?_, rfl, ?_, ?_⟩
  · exact ⟨rfl, rfl, positive, rfl⟩
  · simp [uniformLine, GraphLoopsMemorySCF.definitions]
  · simp [uniformLine, GraphLoopsMemorySCF.definitions]


theorem line_store_alpha {Value : Type}
    (first second : GraphLoopsMemorySCF.PureMeaning Value)
    (aIndex aSeed bIndex bSeed object extent : Nat)
    (aCaptures bCaptures : GraphLoopsMemorySCF.Env Value) (value : Value)
    (aRead : aCaptures aSeed = some (.scalar value)) (bRead : bCaptures bSeed = some (.scalar value))
    (aFresh : aSeed ≠ aIndex) (bFresh : bSeed ≠ bIndex) :
    GraphLoopsMemorySCF.StoreBody first ⟨[], aSeed, [aIndex]⟩ [extent] aIndex object aCaptures =
    GraphLoopsMemorySCF.StoreBody second ⟨[], bSeed, [bIndex]⟩ [extent] bIndex object bCaptures := by
  funext coordinate before after
  apply propext
  rw [line_store_iff first aIndex aSeed object extent coordinate aCaptures value aRead aFresh]
  rw [line_store_iff second bIndex bSeed object extent coordinate bCaptures value bRead bFresh]

theorem line_heap_alpha {Value : Type}
    (first second : GraphLoopsMemorySCF.PureMeaning Value)
    (aIndex aSeed bIndex bSeed object extent tile : Nat) (segmented : Bool)
    (aCaptures bCaptures : GraphLoopsMemorySCF.Env Value) (value : Value)
    (aRead : aCaptures aSeed = some (.scalar value)) (bRead : bCaptures bSeed = some (.scalar value))
    (aFresh : aSeed ≠ aIndex) (bFresh : bSeed ≠ bIndex) (before after : Heap Value) :
    GraphLoopsMemorySCF.HeapRun first (uniformLine extent aIndex aSeed) tile object segmented aCaptures before after ↔
    GraphLoopsMemorySCF.HeapRun second (uniformLine extent bIndex bSeed) tile object segmented bCaptures before after := by
  simp only [GraphLoopsMemorySCF.HeapRun, uniformLine, GraphLoopsMemorySCF.HeapBody,
    GraphLoopsMemorySCF.Initializer.rows, List.headD_cons]
  rw [line_store_alpha first second aIndex aSeed bIndex bSeed object extent aCaptures bCaptures value
    aRead bRead aFresh bFresh]

end Merlin.GraphLoopsShape.Epoch
