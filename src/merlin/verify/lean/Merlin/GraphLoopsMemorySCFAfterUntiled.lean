/- Actual untiled rank-two initializers. The mathematical epoch tile parameter
is an internal proof parameter; no segmentation transform or generated tile
constant occurs in the actual initializer term. -/
import Merlin.GraphLoopsMemorySCFAfterScopedInitializer

set_option autoImplicit false
namespace Merlin.GraphLoopsMemorySCF.After
open GraphLoopsShape

structure UntiledInputs {Value : Type} (program : Epoch.Uniform)
    (env : GraphLoopsShape.Env Value) (object rows columns : Nat) (element : String) (value : Value) : Prop where
  seed : env program.seed.name = .scalar element value
  buffer : env program.buffer.name = Epoch.view object [rows, columns] element
  zero : env program.zero.name = .index 0
  rowsValue : env program.rows.name = .index rows
  columnsValue : env program.columns.name = .index columns
  one : env program.one.name = .index 1

theorem untiled_epoch {Value Public : Type}
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value Public)) (classify : String → MemoryOpcode)
    (program : Epoch.Uniform) (rows columns tile object : Nat) (positive : 0 < tile)
    (element : String) (value : Value) (facts : program.Facts rows columns element)
    (store : classify program.storeKey = .store) (loop : classify program.loopKey = .loop)
    (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (inputs : UntiledInputs program env object rows columns element value)
    (meaning : PureMeaning Value)
    (before after : MemoryWorld Value Public) (outputs : List (Runtime Value)) :
    Eval (withMemory base classify) false aliases program.code env before outputs after ↔
      outputs = [] ∧ after.observed = before.observed ∧
        HeapRun meaning (Witness.uniform rows columns program.outer.name program.inner.name program.seed.name)
          tile object true (Epoch.eraseEnv env) before.heap after.heap := by
  have checked : decide (program.Facts rows columns element) = true := by simpa using facts
  rw [program.checked_eval base classify rows columns element checked store loop aliases env object tile value
    inputs.seed inputs.buffer inputs.zero inputs.rowsValue inputs.columnsValue inputs.one meaning before after outputs]
  exact and_congr_right (fun _ => and_congr_right (fun _ =>
    segmentation_exact_heaps meaning
      (Witness.uniform rows columns program.outer.name program.inner.name program.seed.name)
      tile object positive (Epoch.eraseEnv env) before.heap after.heap))

theorem untiled_target_link {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.InitializerLink) (facts : link.Facts) (positive : 0 < link.config.tile)
    (bound clock : Nat) (inside : GraphLoopsReuse.slotOf objects link.config.object < bound)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (actual : GraphLoopsShape.Env Value) (value : Value)
    (inputs : UntiledInputs link.canonical actual (GraphLoopsReuse.slotOf objects link.config.object)
      link.rows link.columns link.element value)
    (meaning : PureMeaning Value)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (sourceSeed : before.observed.registers link.canonical.seed.name = .scalar link.element value)
    (outputs : List (Runtime Value)) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    Eval (withMemory base classify) false [] link.canonical.code actual (Epoch.clockWorld bound clock before) outputs world ↔
      outputs = [] ∧ ∃ next,
        HeapRun meaning link.config.program link.config.tile (GraphLoopsReuse.slotOf objects link.config.object) true
          (Epoch.eraseMapped (resolve link.mapping) before.observed.registers) before.heap next ∧
        world = Epoch.clockWorld bound clock ⟨before.observed, next⟩ := by
  rw [untiled_epoch base classify link.canonical link.rows link.columns link.config.tile
    (GraphLoopsReuse.slotOf objects link.config.object) positive link.element value facts.2.1 store loop [] actual inputs meaning]
  have targetSeed : Epoch.eraseEnv actual link.canonical.seed.name = some (.scalar value) := by
    simp only [Epoch.eraseEnv, inputs.seed, Epoch.erase]
  have rawSeed := Epoch.mapped_seed (resolve link.mapping) before.observed.registers
    link.raw.seed.name link.canonical.seed.name (link.seed_mapping facts) link.element value sourceSeed
  have canonicalFacts := facts.2.1
  rcases canonicalFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    targetSeedOuter, targetSeedInner, _, _, targetSeparate, _⟩
  have rawFacts := facts.1
  rcases rawFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    rawSeedOuter, rawSeedInner, _, _, rawSeparate, _⟩
  have alpha := Epoch.uniform_heap_alpha meaning meaning
    link.canonical.outer.name link.canonical.inner.name link.canonical.seed.name
    link.raw.outer.name link.raw.inner.name link.raw.seed.name (GraphLoopsReuse.slotOf objects link.config.object)
    link.rows link.columns link.config.tile true (Epoch.eraseEnv actual)
    (Epoch.eraseMapped (resolve link.mapping) before.observed.registers) value targetSeed rawSeed
    targetSeparate targetSeedOuter targetSeedInner rawSeparate rawSeedOuter rawSeedInner
    (Epoch.clockWorld bound clock before).heap world.heap
  rw [alpha, ← facts.2.2.2.1]
  exact and_congr_right (fun _ => segmented_world_partition bound
    (GraphLoopsReuse.slotOf objects link.config.object) link.config.tile inside positive meaning
    link.config.program facts.2.2.2.2.2 (Epoch.eraseMapped (resolve link.mapping) before.observed.registers)
    (Epoch.atClock clock before) world)

theorem untiled_inputs_from_related {Value : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.InitializerLink) (facts : link.Facts)
    (context : Context) (views : List Epoch.PrivateView) (bound : link.Bound context views)
    (bufferBound : link.canonical.buffer ∈ context) (inside : link.config.object < objects.length)
    (logical actual : GraphLoopsShape.Env Value) (valid : Epoch.RegistersValid context views logical)
    (related : RegistersRelated objects context logical actual) :
    ∃ value, logical link.canonical.seed.name = .scalar link.element value ∧
      UntiledInputs link.canonical actual (GraphLoopsReuse.slotOf objects link.config.object)
        link.rows link.columns link.element value := by
  obtain ⟨value, seed, buffer, zero, rows, columns, one⟩ := link.inputs facts context views bound logical valid
  refine ⟨value, seed, ?_⟩
  constructor
  · rw [related _ bound.1, seed]; rfl
  · exact related.private_view link.canonical.buffer bufferBound link.config.object inside
      [link.rows,link.columns] link.element buffer
  · rw [related _ bound.2.1, zero]; rfl
  · rw [related _ bound.2.2.1, rows]; rfl
  · rw [related _ bound.2.2.2.1, columns]; rfl
  · rw [related _ bound.2.2.2.2.1, one]; rfl

theorem untiled_fragments_eval {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (link : Epoch.InitializerLink) (actual next : GraphLoopsShape.Env Value) (before after : World) :
    FragmentsRun meaning [.atom link.canonical.atom] actual before next after ↔
      next = actual ∧ Eval meaning false [] link.canonical.code actual before [] after := by
  constructor
  · intro run
    cases run with
    | more region tail =>
      cases tail
      exact (Epoch.region_empty_step meaning link.canonical.loopKey
        [link.canonical.zero,link.canonical.rows,link.canonical.one] [link.canonical.outer]
        link.canonical.rowCode before actual next after).mp region
  · rintro ⟨same,run⟩
    subst next
    refine .more ?_ (.done _ _)
    exact (Epoch.region_empty_step meaning link.canonical.loopKey
      [link.canonical.zero,link.canonical.rows,link.canonical.one] [link.canonical.outer]
      link.canonical.rowCode before actual actual after).mpr ⟨rfl,run⟩

theorem scoped_untiled_iff {Value User : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (specifications : Epoch.Specifications) (scalar : Epoch.BinaryMeaning Value)
    (link : Epoch.InitializerLink) (facts : link.Facts) (positive : 0 < link.config.tile)
    (member : link.config ∈ configs)
    (unique : ∀ config ∈ configs, config.key = link.config.key → config = link.config)
    (clock : Nat) (slotInside : GraphLoopsReuse.slotOf objects link.config.object < objects.length)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User)))
    (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (context : Context)
    (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (actual : GraphLoopsShape.Env Value) (related : RegistersRelated objects context before.observed.registers actual)
    (value : Value) (sourceSeed : before.observed.registers link.canonical.seed.name = .scalar link.element value)
    (inputs : UntiledInputs link.canonical actual (GraphLoopsReuse.slotOf objects link.config.object)
      link.rows link.columns link.element value)
    (next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    FragmentsRun (withMemory base classify) [.atom link.canonical.atom] actual
      (Epoch.clockWorld objects.length clock before) next world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs
        (Epoch.interpretation objects.length specifications link.mapping scalar) true true
        (.initialize link.config.object link.config.key) before after ∧
        next = actual ∧ world = Epoch.clockWorld objects.length clock after ∧
        RegistersRelated objects context after.observed.registers next := by
  rw [untiled_fragments_eval]
  rw [untiled_target_link objects link facts positive objects.length clock slotInside base classify
    store loop actual value inputs (Epoch.interpretation objects.length specifications link.mapping scalar).pure
    before sourceSeed [] world]
  simp only [true_and,candidate_initialized_iff objects configs
    (Epoch.interpretation objects.length specifications link.mapping scalar) link.config member unique]
  constructor
  · rintro ⟨rfl,heap,run,rfl⟩
    exact ⟨⟨before.observed,heap⟩,⟨heap,run,rfl⟩,rfl,rfl,related⟩
  · rintro ⟨after,⟨heap,run,rfl⟩,assigned,world,_⟩
    exact ⟨assigned,heap,run,world⟩

end Merlin.GraphLoopsMemorySCF.After
