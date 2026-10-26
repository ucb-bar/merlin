/- Actual nested initializer statements correspond to exactly the selected old
configuration. Store/SCF/SSA/public effects are derived by prior local lemmas;
configuration uniqueness prevents an alternative hidden initializer meaning. -/
import Merlin.GraphLoopsShapeEpochState
import Merlin.GraphLoopsShapeEpochSequence

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

theorem initialized_step_iff {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List GraphLoopsMemorySCF.Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (config : GraphLoopsMemorySCF.Configuration) (member : config ∈ configs)
    (unique : ∀ actual ∈ configs, actual.key = config.key → actual = config)
    (before after : GraphLoopsReuse.Memory Value Public) :
    GraphLoopsMemorySCF.Step objects configs meaning false false (.initialize config.object config.key) before after ↔
      ∃ heap, GraphLoopsMemorySCF.HeapRun meaning.pure config.program config.tile config.object false
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

def Uniform.atom (program : Uniform) : Atom :=
  .region program.loopKey [program.zero, program.rows, program.one] [program.outer] [] program.rowCode

theorem region_empty_step {Value World : Type} (meaning : Interpretation Value World)
    (key : String) (operands parameters : List Binding) (body : Code) (before : World) (env nextEnv : Env Value)
    (after : World) :
    (Atom.region key operands parameters [] body).step meaning env before nextEnv after ↔
      nextEnv = env ∧ Eval meaning false [] (.region key operands parameters [] body (.returned [])) env before [] after := by
  constructor
  · intro run
    have same : nextEnv = env := by obtain ⟨_, _, _, same⟩ := run; exact same
    refine ⟨same, ?_⟩
    exact (Atom.eval_suffix meaning (.region key operands parameters [] body) (.returned []) env before [] after).mpr
      ⟨nextEnv, after, run, rfl, rfl⟩
  · rintro ⟨sameEnv, run⟩
    subst nextEnv
    obtain ⟨middleEnv, middle, first, _, equalWorld⟩ :=
      (Atom.eval_suffix meaning (.region key operands parameters [] body) (.returned []) env before [] after).mp run
    have same : middleEnv = env := by obtain ⟨_, _, _, same⟩ := first; exact same
    subst middleEnv
    subst middle
    exact first

theorem initializer_atom {Value User : Type} (link : InitializerLink) (checked : decide link.Facts = true)
    (context : Context) (views : List PrivateView) (boundChecked : decide (link.Bound context views) = true)
    (bound next : Nat) (inside : link.config.object < bound)
    (objects : GraphLoopsReuse.Objects) (configs : List GraphLoopsMemorySCF.Configuration)
    (member : link.config ∈ configs) (unique : ∀ actual ∈ configs, actual.key = link.config.key → actual = link.config)
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
  rw [Uniform.atom, region_empty_step]
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
