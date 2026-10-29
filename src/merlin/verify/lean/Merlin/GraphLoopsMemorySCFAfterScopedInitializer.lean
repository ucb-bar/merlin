/- UNCOMPILED. The two emitted initializer fragments execute their actual
tile constant and nested SCF body. The physical heap/caller relation is derived
from target_initializer_link; generated scratch does not enter logical scope.
-/
import Merlin.GraphLoopsMemorySCFAfterInitializerLink
import Merlin.GraphLoopsMemorySCFAfterPlan

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem tiled_fragments_eval {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (link : Epoch.InitializerLink) (program : TiledInitializer)
    (actual next : GraphLoopsShape.Env Value) (before after : World) :
    FragmentsRun meaning (Expansion.tiled link program).fragments actual before next after ↔
      next = GraphLoopsShape.put actual program.tile.name (.index link.config.tile) ∧
      Eval meaning false [] program.code
        (GraphLoopsShape.put actual program.tile.name (.index link.config.tile)) before [] after := by
  constructor
  · intro run
    cases run with
    | more first suffix =>
      obtain ⟨rfl, rfl⟩ := first
      cases suffix with
      | more region tail =>
        cases tail
        exact (Epoch.region_empty_step meaning program.uniform.loopKey
          [program.uniform.zero, program.uniform.rows, program.tile] [program.tileIV] program.body
          _ _ next after).mp region
  · rintro ⟨rfl, run⟩
    refine .more ⟨rfl, rfl⟩ (.more ?_ (.done _ _))
    exact (Epoch.region_empty_step meaning program.uniform.loopKey
      [program.uniform.zero, program.uniform.rows, program.tile] [program.tileIV] program.body
      before _ _ after).mpr ⟨rfl, run⟩

theorem scoped_initializer_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (specifications : Epoch.Specifications) (scalar : Epoch.BinaryMeaning Value)
    (link : Epoch.InitializerLink) (facts : link.Facts)
    (member : link.config ∈ configs)
    (unique : ∀ config ∈ configs, config.key = link.config.key → config = link.config)
    (program : TiledInitializer) (sameUniform : program.uniform = link.canonical)
    (controlFacts : program.ControlFacts) (bits clock : Nat)
    (validBounds : (GraphLoopsSegment.Bounds.mk bits 0 link.rows 1 link.config.tile).Valid)
    (slotInside : GraphLoopsReuse.slotOf objects link.config.object < objects.length)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (context : Context) (fresh : program.tile.name ∉ names context)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated objects context before.observed.registers actual)
    (value : Value) (sourceSeed : before.observed.registers link.canonical.seed.name = .scalar link.element value)
    (inputs : program.Inputs (GraphLoopsShape.put actual program.tile.name (.index link.config.tile))
      (GraphLoopsReuse.slotOf objects link.config.object) link.rows link.columns link.config.tile link.element value)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    FragmentsRun (withMemory base classify) (Expansion.tiled link program).fragments actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs
        (Epoch.interpretation objects.length specifications link.mapping scalar) true true
        (.initialize link.config.object link.config.key) before after ∧
        next = GraphLoopsShape.put actual program.tile.name (.index link.config.tile) ∧
        world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects context after.observed.registers next := by
  rw [tiled_fragments_eval]
  rw [target_initializer_link objects link facts program sameUniform controlFacts bits objects.length clock
    validBounds slotInside base classify store loop add minimum _ value inputs
    (Epoch.interpretation objects.length specifications link.mapping scalar).pure before sourceSeed [] world]
  simp only [true_and, candidate_initialized_iff objects configs
    (Epoch.interpretation objects.length specifications link.mapping scalar) link.config member unique]
  constructor
  · rintro ⟨rfl, heap, run, rfl⟩
    exact ⟨⟨before.observed, heap⟩, ⟨heap, run, rfl⟩, rfl, rfl,
      related.scratch program.tile.name (.index link.config.tile) fresh⟩
  · rintro ⟨after, ⟨heap, run, rfl⟩, assigned, world, _⟩
    exact ⟨assigned, heap, run, world⟩

end Merlin.GraphLoopsMemorySCF.After
