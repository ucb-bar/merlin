/- Initializer SSA alpha correspondence between the actual raw middle and the
source canonical names. Equality follows from per-cell operand reads and the
same ordered loops; no whole-initializer agreement is assumed. -/
import Merlin.GraphLoopsShapeEpochInitializer

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def eraseMapped {Value : Type} (mapping : Nat → Nat) (env : Env Value) : GraphLoopsMemorySCF.Env Value :=
  fun name => erase (env (mapping name))

theorem mapped_seed {Value : Type} (mapping : Nat → Nat) (env : Env Value) (raw source : Nat)
    (same : mapping raw = source) (element : String) (value : Value)
    (read : env source = .scalar element value) : eraseMapped mapping env raw = some (.scalar value) := by
  simp only [eraseMapped, same, read, erase]

/-- Each model reads its actual scoped seed/row/column names, then makes the
same concrete private store. Pure meanings are irrelevant for the empty body. -/
theorem uniform_store_alpha {Value : Type}
    (first second : GraphLoopsMemorySCF.PureMeaning Value)
    (aOuter aInner aSeed bOuter bInner bSeed object rows columns row : Nat)
    (aCaptures bCaptures : GraphLoopsMemorySCF.Env Value) (value : Value)
    (aRead : aCaptures aSeed = some (.scalar value)) (bRead : bCaptures bSeed = some (.scalar value))
    (aSeparate : aOuter ≠ aInner) (aSeedOuter : aSeed ≠ aOuter) (aSeedInner : aSeed ≠ aInner)
    (bSeparate : bOuter ≠ bInner) (bSeedOuter : bSeed ≠ bOuter) (bSeedInner : bSeed ≠ bInner) :
    GraphLoopsMemorySCF.StoreBody first ⟨[], aSeed, [aOuter, aInner]⟩ [rows, columns] aInner object
      (GraphLoopsMemorySCF.put aCaptures aOuter (.index row)) =
    GraphLoopsMemorySCF.StoreBody second ⟨[], bSeed, [bOuter, bInner]⟩ [rows, columns] bInner object
      (GraphLoopsMemorySCF.put bCaptures bOuter (.index row)) := by
  funext column before after
  apply propext
  rw [uniform_store_iff first aOuter aInner aSeed object rows columns row column aCaptures value aRead
    aSeparate aSeedOuter aSeedInner]
  rw [uniform_store_iff second bOuter bInner bSeed object rows columns row column bCaptures value bRead
    bSeparate bSeedOuter bSeedInner]

/-- Preserve the complete nested effect trace across raw/source SSA naming. -/
theorem uniform_nested_alpha {Value : Type}
    (first second : GraphLoopsMemorySCF.PureMeaning Value)
    (aOuter aInner aSeed bOuter bInner bSeed object rows columns : Nat)
    (aCaptures bCaptures : GraphLoopsMemorySCF.Env Value) (value : Value)
    (aRead : aCaptures aSeed = some (.scalar value)) (bRead : bCaptures bSeed = some (.scalar value))
    (aSeparate : aOuter ≠ aInner) (aSeedOuter : aSeed ≠ aOuter) (aSeedInner : aSeed ≠ aInner)
    (bSeparate : bOuter ≠ bInner) (bSeedOuter : bSeed ≠ bOuter) (bSeedInner : bSeed ≠ bInner) :
    GraphLoopsMemorySCF.NestedStores first [] ⟨[], aSeed, [aOuter, aInner]⟩ rows columns aOuter aInner object aCaptures =
      GraphLoopsMemorySCF.NestedStores second [] ⟨[], bSeed, [bOuter, bInner]⟩ rows columns bOuter bInner object bCaptures := by
  funext row before after
  apply propext
  rw [uniform_nested_iff, uniform_nested_iff]
  rw [uniform_store_alpha first second aOuter aInner aSeed bOuter bInner bSeed object rows columns row
    aCaptures bCaptures value aRead bRead aSeparate aSeedOuter aSeedInner bSeparate bSeedOuter bSeedInner]

theorem uniform_heap_alpha {Value : Type}
    (first second : GraphLoopsMemorySCF.PureMeaning Value)
    (aOuter aInner aSeed bOuter bInner bSeed object rows columns tile : Nat) (segmented : Bool)
    (aCaptures bCaptures : GraphLoopsMemorySCF.Env Value) (value : Value)
    (aRead : aCaptures aSeed = some (.scalar value)) (bRead : bCaptures bSeed = some (.scalar value))
    (aSeparate : aOuter ≠ aInner) (aSeedOuter : aSeed ≠ aOuter) (aSeedInner : aSeed ≠ aInner)
    (bSeparate : bOuter ≠ bInner) (bSeedOuter : bSeed ≠ bOuter) (bSeedInner : bSeed ≠ bInner)
    (before after : Heap Value) :
    GraphLoopsMemorySCF.HeapRun first (GraphLoopsMemorySCF.Witness.uniform rows columns aOuter aInner aSeed)
      tile object segmented aCaptures before after ↔
    GraphLoopsMemorySCF.HeapRun second (GraphLoopsMemorySCF.Witness.uniform rows columns bOuter bInner bSeed)
      tile object segmented bCaptures before after := by
  simp only [GraphLoopsMemorySCF.HeapRun, GraphLoopsMemorySCF.Witness.uniform,
    GraphLoopsMemorySCF.HeapBody, GraphLoopsMemorySCF.Initializer.rows, List.headD_cons]
  rw [uniform_nested_alpha first second aOuter aInner aSeed bOuter bInner bSeed object rows columns
    aCaptures bCaptures value aRead bRead aSeparate aSeedOuter aSeedInner bSeparate bSeedOuter bSeedInner]

end Merlin.GraphLoopsShape.Epoch
