/- One actual region/store initializer step, derived from checked SSA registers. -/
import Merlin.GraphLoopsShapeEpochLineLink
import Merlin.GraphLoopsShapeEpochInitializerSteps

set_option autoImplicit false
namespace Merlin.GraphLoopsShape.Epoch

/-- The actual single SCF region, with its one coordinate/store body intact.
This definition lives above the checked epoch plan and introduces no execution
or whole-entry equivalence assumption. -/
def UniformLine.atom (program : UniformLine) : Atom :=
  .region program.loopKey [program.zero, program.extent, program.one]
    [program.index] [] program.cell.code

theorem line_initializer_atom {Value User : Type} (link : LineInitializerLink)
    (checked : decide link.Facts = true) (context : Context) (views : List PrivateView)
    (boundChecked : decide (link.Bound context views) = true) (bound next : Nat)
    (inside : link.config.object < bound)
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (member : link.config ∈ configs)
    (unique : ∀ actual ∈ configs, actual.key = link.config.key → actual = link.config)
    (specifications : Specifications) (scalar : BinaryMeaning Value)
    (base : Interpretation Value (MemoryWorld Value (AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (valid : RegistersValid context views before.observed.registers)
    (env : Env Value) (world : MemoryWorld Value (AllocationPublic User)) :
    link.canonical.atom.step (withMemory base classify) before.observed.registers (clockWorld bound next before) env world ↔
      ∃ after, GraphLoopsMemorySCF.Step objects configs (interpretation bound specifications link.mapping scalar) false false
        (.initialize link.config.object link.config.key) before after ∧
        env = after.observed.registers ∧ world = clockWorld bound next after := by
  rw [UniformLine.atom, region_empty_step]
  change (env = before.observed.registers ∧ Eval (withMemory base classify) false [] link.canonical.code
    before.observed.registers (clockWorld bound next before) [] world) ↔ _
  rw [link.clocked checked context views boundChecked bound next inside base classify store loop []
    (interpretation bound specifications link.mapping scalar).pure before valid [] world]
  simp only [true_and, initialized_step_iff objects configs (interpretation bound specifications link.mapping scalar)
    link.config member unique]
  constructor
  · rintro ⟨env, heap, run, world⟩
    exact ⟨⟨before.observed, heap⟩, ⟨heap, run, rfl⟩, env, world⟩
  · rintro ⟨after, ⟨heap, run, rfl⟩, env, world⟩
    exact ⟨env, heap, run, world⟩

end Merlin.GraphLoopsShape.Epoch
