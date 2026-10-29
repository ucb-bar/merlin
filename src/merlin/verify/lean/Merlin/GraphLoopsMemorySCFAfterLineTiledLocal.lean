/- Actual tiled rank-one initialization joins, with derived input/world/local simulation. -/
import Merlin.GraphLoopsMemorySCFAfterRetained
import Merlin.GraphLoopsMemorySCFAfterLine
import Merlin.GraphLoopsMemorySCFAfterLineTiled
set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

theorem tiled_line_target_link {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.LineInitializerLink) (facts : link.Facts) (program : TiledLine)
    (sameUniform : program.uniform = link.canonical) (controlFacts : program.ControlFacts)
    (bits bound clock : Nat) (validBounds : (GraphLoopsSegment.Bounds.mk bits 0 link.extent 1 link.config.tile).Valid)
    (inside : GraphLoopsReuse.slotOf objects link.config.object < bound)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum)
    (actual : GraphLoopsShape.Env Value) (value : Value)
    (inputs : program.Inputs actual (GraphLoopsReuse.slotOf objects link.config.object)
      link.extent link.config.tile link.element value)
    (meaning : PureMeaning Value)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (sourceSeed : before.observed.registers link.canonical.seed.name = .scalar link.element value)
    (outputs : List (Runtime Value)) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Eval (withMemory base classify) false [] program.code actual (Epoch.clockWorld bound clock before) outputs world ↔
      outputs = [] ∧ ∃ next,
        HeapRun meaning link.config.program link.config.tile (GraphLoopsReuse.slotOf objects link.config.object) true
          (Epoch.eraseMapped (resolve link.mapping) before.observed.registers) before.heap next ∧
        world = Epoch.clockWorld bound clock ⟨before.observed, next⟩ := by
  have canonicalFacts : program.uniform.Facts link.extent link.element := by
    rw [sameUniform]
    exact facts.2.1
  rw [program.eval_epoch base classify bits link.extent link.config.tile
    (GraphLoopsReuse.slotOf objects link.config.object) link.element value validBounds canonicalFacts controlFacts
    store loop add minimum [] actual inputs meaning]
  have targetSeed : Epoch.eraseEnv actual program.uniform.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, inputs.seed, Epoch.erase]
  have rawSeed := Epoch.mapped_seed (resolve link.mapping) before.observed.registers
    link.raw.seed.name link.canonical.seed.name (link.seed_mapping facts) link.element value sourceSeed
  rcases canonicalFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, targetFresh, _⟩
  have rawFacts := facts.1
  rcases rawFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, rawFresh, _⟩
  have alpha := Epoch.line_heap_alpha meaning meaning
    program.uniform.index.name program.uniform.seed.name link.raw.index.name link.raw.seed.name
    (GraphLoopsReuse.slotOf objects link.config.object) link.extent link.config.tile true
    (Epoch.eraseEnv actual) (Epoch.eraseMapped (resolve link.mapping) before.observed.registers)
    value targetSeed rawSeed targetFresh rawFresh (Epoch.clockWorld bound clock before).heap world.heap
  rw [alpha, ← facts.2.2.2.1]
  have positive : 0 < link.config.tile := Nat.lt_trans (by decide) validBounds.2.2.2.1
  exact and_congr_right (fun _ => segmented_world_partition bound
    (GraphLoopsReuse.slotOf objects link.config.object) link.config.tile inside positive meaning
    link.config.program facts.2.2.2.2.2 (Epoch.eraseMapped (resolve link.mapping) before.observed.registers)
    (Epoch.atClock clock before) world)

theorem tiled_line_fragments_eval {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (link : Epoch.LineInitializerLink) (program : TiledLine)
    (actual next : GraphLoopsShape.Env Value) (before after : World) :
    FragmentsRun meaning (Expansion.tiledLine link program).fragments actual before next after ↔
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
          [program.uniform.zero, program.uniform.extent, program.tile] [program.tileIV] program.body
          _ _ next after).mp region
  · rintro ⟨rfl, run⟩
    refine .more ⟨rfl, rfl⟩ (.more ?_ (.done _ _))
    exact (Epoch.region_empty_step meaning program.uniform.loopKey
      [program.uniform.zero, program.uniform.extent, program.tile] [program.tileIV] program.body
      before _ _ after).mpr ⟨rfl, run⟩

theorem scoped_tiled_line_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (specifications : Epoch.Specifications) (scalar : Epoch.BinaryMeaning Value)
    (link : Epoch.LineInitializerLink) (facts : link.Facts)
    (member : link.config ∈ configs)
    (unique : ∀ config ∈ configs, config.key = link.config.key → config = link.config)
    (program : TiledLine) (sameUniform : program.uniform = link.canonical)
    (controlFacts : program.ControlFacts) (bits clock : Nat)
    (validBounds : (GraphLoopsSegment.Bounds.mk bits 0 link.extent 1 link.config.tile).Valid)
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
      (GraphLoopsReuse.slotOf objects link.config.object) link.extent link.config.tile link.element value)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    FragmentsRun (withMemory base classify) (Expansion.tiledLine link program).fragments actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs
        (Epoch.interpretation objects.length specifications link.mapping scalar) true true
        (.initialize link.config.object link.config.key) before after ∧
        next = GraphLoopsShape.put actual program.tile.name (.index link.config.tile) ∧
        world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects context after.observed.registers next := by
  rw [tiled_line_fragments_eval]
  rw [tiled_line_target_link objects link facts program sameUniform controlFacts bits objects.length clock
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

theorem scoped_tiled_line_inputs {Value : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.LineInitializerLink) (facts : link.Facts) (program : TiledLine)
    (sameUniform : program.uniform = link.canonical)
    (context : Context) (views : List Epoch.PrivateView) (bound : link.Bound context views)
    (bufferBound : link.canonical.buffer ∈ context) (inside : link.config.object < objects.length)
    (logical actual : GraphLoopsShape.Env Value) (valid : Epoch.RegistersValid context views logical)
    (related : RegistersRelated objects context logical actual) (fresh : program.tile.name ∉ names context) :
    ∃ value, logical link.canonical.seed.name = .scalar link.element value ∧
      program.Inputs (GraphLoopsShape.put actual program.tile.name (.index link.config.tile))
        (GraphLoopsReuse.slotOf objects link.config.object) link.extent link.config.tile link.element value := by
  have scratch := related.scratch program.tile.name (.index link.config.tile) fresh
  obtain ⟨value,sourceSeed,inputs⟩ := line_inputs_from_related objects link facts context views bound
    bufferBound inside logical _ valid scratch
  have target : LineInputs program.uniform (GraphLoopsShape.put actual program.tile.name (.index link.config.tile))
      (GraphLoopsReuse.slotOf objects link.config.object) link.extent link.element value := by
    simpa only [sameUniform] using inputs
  exact ⟨value,sourceSeed,target.seed,target.buffer,target.zero,target.extentValue,target.one,
    by simp only [GraphLoopsShape.put, ↓reduceIte]⟩

theorem tiled_line_initializer_local {Value User : Type} (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (context : Context) (sourceClock clock : Nat) (link : Epoch.LineInitializerLink) (program : TiledLine)
    (checked : (Epoch.Command.initializeLine link).Valid plan context sourceClock)
    (sameUniform : program.uniform = link.canonical) (controls : program.ControlFacts)
    (bits : Nat) (bounds : (GraphLoopsSegment.Bounds.mk bits 0 link.extent 1 link.config.tile).Valid)
    (fresh : program.tile.name ∉ names context)
    (slotInside : GraphLoopsReuse.slotOf plan.objects link.config.object < plan.objects.length)
    (scalar : Epoch.BinaryMeaning Value)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum) :
    LocalSimulation plan scalar (withMemory base classify) context clock (.tiledLine link program) := by
  rcases checked.1 with ⟨facts, inputBound, configMember, mapping, inside, _, _⟩
  have bufferBound := checked.2.1 link.canonical.buffer (by simp [Epoch.Command.inputs])
  have unique : ∀ config ∈ plan.configs, config.key = link.config.key → config = link.config :=
    fun config member same => planValid.1 config member link.config configMember same
  have localLaw (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
      (actual next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User))
      (registers : Epoch.RegistersValid context plan.views before.observed.registers)
      (related : RegistersRelated plan.objects context before.observed.registers actual) :
      FragmentsRun (withMemory base classify) (Expansion.tiledLine link program).fragments actual
        (Epoch.clockWorld plan.objects.length clock before) next world ↔
        ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
          (.initialize link.config.object link.config.key) before after ∧
          next = GraphLoopsShape.put actual program.tile.name (.index link.config.tile) ∧
          world = Epoch.clockWorld plan.objects.length clock after ∧
          RegistersRelated plan.objects context after.observed.registers next := by
    obtain ⟨value, sourceSeed, inputs⟩ := scoped_tiled_line_inputs plan.objects link facts program sameUniform context
      plan.views inputBound bufferBound inside before.observed.registers actual registers related fresh
    simpa only [mapping, Epoch.PlanContext.oldMeaning] using
      scoped_tiled_line_iff plan.objects plan.configs plan.specifications scalar link facts configMember unique program
        sameUniform controls bits clock bounds slotInside base classify store loop add minimum context fresh before actual
        related value sourceSeed inputs next world
  constructor
  · intro before after actual registers callers related run
    have publicSame := candidate_initializer_public plan.objects plan.configs (plan.oldMeaning scalar)
      link.config.object link.config.key before after run
    have finalRelated : RegistersRelated plan.objects context after.observed.registers
        (GraphLoopsShape.put actual program.tile.name (.index link.config.tile)) := by
      rw [publicSame]
      exact related.scratch program.tile.name (.index link.config.tile) fresh
    exact ⟨_, (localLaw before actual _ _ registers related).mpr ⟨after, run, rfl, rfl, finalRelated⟩, finalRelated⟩
  · intro before actual next world registers callers related run
    obtain ⟨after, oldRun, _, finalWorld, finalRelated⟩ := (localLaw before actual next world registers related).mp run
    exact ⟨after, oldRun, finalWorld, finalRelated⟩

end Merlin.GraphLoopsMemorySCF.After
