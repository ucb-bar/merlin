/- UNCOMPILED. Discharge the tiled initializer constructor from finite source
bindings, fresh scratch, signed bounds and actual selected operation meanings.
The seed and every control operand are reconstructed from the live invariant.
-/
import Merlin.GraphLoopsMemorySCFAfterRetained

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem scoped_initializer_inputs {Value : Type} (objects : GraphLoopsReuse.Objects)
    (link : Epoch.InitializerLink) (facts : link.Facts) (program : TiledInitializer)
    (sameUniform : program.uniform = link.canonical)
    (context : Context) (views : List Epoch.PrivateView) (bound : link.Bound context views)
    (bufferBound : link.canonical.buffer ∈ context) (inside : link.config.object < objects.length)
    (logical actual : GraphLoopsShape.Env Value) (valid : Epoch.RegistersValid context views logical)
    (related : RegistersRelated objects context logical actual) (fresh : program.tile.name ∉ names context) :
    ∃ value, logical link.canonical.seed.name = .scalar link.element value ∧
      program.Inputs (GraphLoopsShape.put actual program.tile.name (.index link.config.tile))
        (GraphLoopsReuse.slotOf objects link.config.object) link.rows link.columns link.config.tile link.element value := by
  have scratch := related.scratch program.tile.name (.index link.config.tile) fresh
  obtain ⟨value, inputs⟩ := initializer_inputs_from_related objects link facts program sameUniform context views bound
    bufferBound inside logical _ valid scratch (by simp only [GraphLoopsShape.put, ↓reduceIte])
  obtain ⟨sourceValue, sourceRead, _⟩ := link.inputs facts context views bound logical valid
  have actualRead : GraphLoopsShape.put actual program.tile.name (.index link.config.tile)
      link.canonical.seed.name = .scalar link.element sourceValue := by
    rw [scratch _ bound.1, sourceRead]
    rfl
  have seedRead : GraphLoopsShape.put actual program.tile.name (.index link.config.tile)
      link.canonical.seed.name = .scalar link.element value := by
    simpa only [sameUniform] using inputs.seed
  have same := (Runtime.scalar.inj (actualRead.symm.trans seedRead)).2
  subst sourceValue
  exact ⟨value, sourceRead, inputs⟩

theorem initializer_local {Value User : Type} (plan : Epoch.PlanContext) (planValid : plan.Valid)
    (context : Context) (sourceClock clock : Nat) (link : Epoch.InitializerLink) (program : TiledInitializer)
    (checked : (Epoch.Command.initialize link).Valid plan context sourceClock)
    (sameUniform : program.uniform = link.canonical) (controls : program.ControlFacts)
    (bits : Nat) (bounds : (GraphLoopsSegment.Bounds.mk bits 0 link.rows 1 link.config.tile).Valid)
    (fresh : program.tile.name ∉ names context)
    (slotInside : GraphLoopsReuse.slotOf plan.objects link.config.object < plan.objects.length)
    (scalar : Epoch.BinaryMeaning Value)
    (base : GraphLoopsShape.Interpretation Value (MemoryWorld Value (Epoch.AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify program.uniform.storeKey = .store) (loop : classify program.uniform.loopKey = .loop)
    (add : (withMemory base classify).action program.addKey = IndexAction bits .add)
    (minimum : (withMemory base classify).action program.minKey = IndexAction bits .minimum) :
    LocalSimulation plan scalar (withMemory base classify) context clock (.tiled link program) := by
  rcases checked.1 with ⟨facts, inputBound, configMember, mapping, inside, _, _⟩
  have bufferBound := checked.2.1 link.canonical.buffer (by simp [Epoch.Command.inputs])
  have unique : ∀ config ∈ plan.configs, config.key = link.config.key → config = link.config :=
    fun config member same => planValid.1 config member link.config configMember same
  have localLaw (before : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
      (actual next : GraphLoopsShape.Env Value) (world : MemoryWorld Value (Epoch.AllocationPublic User))
      (registers : Epoch.RegistersValid context plan.views before.observed.registers)
      (related : RegistersRelated plan.objects context before.observed.registers actual) :
      FragmentsRun (withMemory base classify) (Expansion.tiled link program).fragments actual
        (Epoch.clockWorld plan.objects.length clock before) next world ↔
        ∃ after, GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
          (.initialize link.config.object link.config.key) before after ∧
          next = GraphLoopsShape.put actual program.tile.name (.index link.config.tile) ∧
          world = Epoch.clockWorld plan.objects.length clock after ∧
          RegistersRelated plan.objects context after.observed.registers next := by
    obtain ⟨value, sourceSeed, inputs⟩ := scoped_initializer_inputs plan.objects link facts program sameUniform context
      plan.views inputBound bufferBound inside before.observed.registers actual registers related fresh
    simpa only [mapping, Epoch.PlanContext.oldMeaning] using
      scoped_initializer_iff plan.objects plan.configs plan.specifications scalar link facts configMember unique program
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
