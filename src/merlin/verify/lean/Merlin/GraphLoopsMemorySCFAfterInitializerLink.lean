/- UNCOMPILED. Actual tiled SSA inputs → canonical stores → raw epoch capture
names → physical private heap, preserving the entire caller/clock world. Extra
target scratch registers are irrelevant by the already derived seed/store laws.
-/
import Merlin.GraphLoopsMemorySCFAfterScope
import Merlin.GraphLoopsShapeEpochInitializerSteps

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem segmented_world_partition {Value User : Type} (bound object tile : Nat)
    (inside : object < bound) (positive : 0 < tile) (meaning : PureMeaning Value)
    (program : Initializer) (valid : program.WellFormed) (captures : GraphLoopsMemorySCF.Env Value)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) (after : MemoryWorld Value User) :
    (after.observed = before.observed.observed ∧
      HeapRun meaning program tile object true captures (Epoch.fullWorld bound before).heap after.heap) ↔
      ∃ next, HeapRun meaning program tile object true captures before.heap next ∧
        after = Epoch.fullWorld bound ⟨before.observed, next⟩ := by
  refine (and_congr Iff.rfl
    (segmentation_exact_heaps meaning program tile object positive captures
      (Epoch.fullWorld bound before).heap after.heap).symm).trans ?_
  refine (Epoch.initializer_world_partition bound object tile inside meaning program valid captures before after).trans ?_
  apply exists_congr
  intro next
  exact and_congr (segmentation_exact_heaps meaning program tile object positive captures before.heap next) Iff.rfl

theorem target_initializer_link {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.InitializerLink) (facts : link.Facts) (program : TiledInitializer)
    (sameUniform : program.uniform = link.canonical) (controlFacts : program.ControlFacts)
    (bits bound clock : Nat) (validBounds : (GraphLoopsSegment.Bounds.mk bits 0 link.rows 1 link.config.tile).Valid)
    (inside : GraphLoopsReuse.slotOf objects link.config.object < bound)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (actual : GraphLoopsShape.Env Value) (value : Value)
    (inputs : program.Inputs actual (GraphLoopsReuse.slotOf objects link.config.object)
      link.rows link.columns link.config.tile link.element value)
    (meaning : PureMeaning Value)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (sourceSeed : before.observed.registers link.canonical.seed.name = .scalar link.element value)
    (outputs : List (Runtime Value)) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Eval (withMemory base classify) false [] program.code actual (Epoch.clockWorld bound clock before) outputs world ↔
      outputs = [] ∧ ∃ next,
        HeapRun meaning link.config.program link.config.tile (GraphLoopsReuse.slotOf objects link.config.object) true
          (Epoch.eraseMapped (resolve link.mapping) before.observed.registers) before.heap next ∧
        world = Epoch.clockWorld bound clock ⟨before.observed, next⟩ := by
  have canonicalFacts : program.uniform.Facts link.rows link.columns link.element := by
    rw [sameUniform]
    exact facts.2.1
  rw [program.eval_epoch base classify bits link.rows link.columns link.config.tile
    (GraphLoopsReuse.slotOf objects link.config.object) link.element value validBounds canonicalFacts controlFacts
    store loop add minimum [] actual inputs meaning]
  have targetSeed : Epoch.eraseEnv actual program.uniform.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, inputs.seed, Epoch.erase]
  have rawSeed := Epoch.mapped_seed (resolve link.mapping) before.observed.registers
    link.raw.seed.name link.canonical.seed.name (link.seed_mapping facts) link.element value sourceSeed
  rcases canonicalFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    targetSeedOuter, targetSeedInner, _, _, targetSeparate, _⟩
  have rawFacts := facts.1
  rcases rawFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    rawSeedOuter, rawSeedInner, _, _, rawSeparate, _⟩
  have alpha := Epoch.uniform_heap_alpha meaning meaning
    program.uniform.outer.name program.uniform.inner.name program.uniform.seed.name
    link.raw.outer.name link.raw.inner.name link.raw.seed.name (GraphLoopsReuse.slotOf objects link.config.object)
    link.rows link.columns link.config.tile true (Epoch.eraseEnv actual)
    (Epoch.eraseMapped (resolve link.mapping) before.observed.registers) value targetSeed rawSeed
    targetSeparate targetSeedOuter targetSeedInner rawSeparate rawSeedOuter rawSeedInner
    (Epoch.clockWorld bound clock before).heap world.heap
  rw [alpha, ← facts.2.2.2.1]
  have positive : 0 < link.config.tile := Nat.lt_trans (by decide) validBounds.2.2.2.1
  exact and_congr_right (fun _ => segmented_world_partition bound
    (GraphLoopsReuse.slotOf objects link.config.object) link.config.tile inside positive meaning
    link.config.program facts.2.2.2.2.2 (Epoch.eraseMapped (resolve link.mapping) before.observed.registers)
    (Epoch.atClock clock before) world)

theorem candidate_initialized_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (config : Configuration) (member : config ∈ configs)
    (unique : ∀ actual ∈ configs, actual.key = config.key → actual = config)
    (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Step objects configs meaning true true (.initialize config.object config.key) before after ↔
      ∃ heap, HeapRun meaning.pure config.program config.tile (GraphLoopsReuse.slotOf objects config.object) true
        (meaning.captures before.observed) before.heap heap ∧ after = ⟨before.observed, heap⟩ := by
  constructor
  · intro run
    generalize instructionEq : GraphLoopsReuse.Instruction.initialize config.object config.key = instruction at run
    cases run with
    | ordinary ordinary run =>
      rw [← instructionEq] at ordinary
      exact False.elim ordinary
    | @initialized actual actualMember initial heap run =>
      have keyEqual := (GraphLoopsReuse.Instruction.initialize.inj instructionEq).2
      have same := unique actual actualMember keyEqual.symm
      subst actual
      exact ⟨heap, run, rfl⟩
  · rintro ⟨heap, run, rfl⟩
    exact .initialized member before run

end Merlin.GraphLoopsMemorySCF.After
