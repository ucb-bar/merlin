/- Actual raw/source names for one-loop initializers; no whole-run equality premise. -/
import Merlin.GraphLoopsShapeEpochLine
import Merlin.GraphLoopsShapeEpochState

set_option autoImplicit false
namespace Merlin.GraphLoopsShape.Epoch

def renameUniformLine (mapping : Aliases) (program : UniformLine) : UniformLine :=
  { program with
    seed := rename mapping program.seed
    buffer := rename mapping program.buffer
    index := rename mapping program.index
    zero := rename mapping program.zero
    extent := rename mapping program.extent
    one := rename mapping program.one }

structure LineInitializerLink where
  raw : UniformLine
  canonical : UniformLine
  config : GraphLoopsMemorySCF.Configuration
  mapping : Aliases
  extent : Nat
  element : String
  sourceBuffer : Nat
  deriving DecidableEq, Repr

def LineInitializerLink.Facts (link : LineInitializerLink) : Prop :=
  link.raw.Facts link.extent link.element ∧ link.canonical.Facts link.extent link.element ∧
  link.canonical = renameUniformLine link.mapping link.raw ∧
  link.config.program = uniformLine link.extent link.raw.index.name link.raw.seed.name ∧
  link.raw.buffer.name = link.sourceBuffer ∧ link.config.program.WellFormed

instance lineInitializerLinkFactsDecidable (link : LineInitializerLink) : Decidable link.Facts := by
  unfold LineInitializerLink.Facts
  infer_instance

theorem LineInitializerLink.seed_mapping (link : LineInitializerLink) (facts : link.Facts) :
    resolve link.mapping link.raw.seed.name = link.canonical.seed.name := by
  rw [facts.2.2.1]
  rfl

theorem LineInitializerLink.checked_execution {Value User : Type} (link : LineInitializerLink)
    (checked : decide link.Facts = true) (bound : Nat) (inside : link.config.object < bound)
    (base : Interpretation Value (MemoryWorld Value User)) (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (value : Value)
    (seedValue : env link.canonical.seed.name = .scalar link.element value)
    (bufferValue : env link.canonical.buffer.name = view link.config.object [link.extent] link.element)
    (zeroValue : env link.canonical.zero.name = .index 0)
    (extentValue : env link.canonical.extent.name = .index link.extent)
    (oneValue : env link.canonical.one.name = .index 1)
    (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (outputs : List (Runtime Value)) (after : MemoryWorld Value User) :
    Eval (withMemory base classify) false aliases link.canonical.code env (fullWorld bound before) outputs after ↔
      outputs = [] ∧ ∃ next,
        GraphLoopsMemorySCF.HeapRun meaning link.config.program link.config.tile link.config.object false
          (eraseMapped (resolve link.mapping) env) before.heap next ∧
        after = fullWorld bound ⟨before.observed, next⟩ := by
  have facts := of_decide_eq_true checked
  have canonicalFacts := facts.2.1
  have rawFacts := facts.1
  have rawSeed := mapped_seed (resolve link.mapping) env link.raw.seed.name link.canonical.seed.name
    (link.seed_mapping facts) link.element value seedValue
  have canonicalSeed : eraseEnv env link.canonical.seed.name = some (.scalar value) := by
    simp only [eraseEnv, seedValue, erase]
  rcases rawFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, rawFresh, _⟩
  rcases canonicalFacts with ⟨_, _, _, _, _, _, _, _, _, _, _, _, canonicalFresh, _⟩
  rw [link.canonical.checked_eval base classify link.extent link.element (decide_eq_true facts.2.1)
    store loop aliases env link.config.object link.config.tile value seedValue bufferValue zeroValue extentValue oneValue meaning]
  have alpha := line_heap_alpha meaning meaning link.canonical.index.name link.canonical.seed.name
    link.raw.index.name link.raw.seed.name link.config.object link.extent link.config.tile false
    (eraseEnv env) (eraseMapped (resolve link.mapping) env) value canonicalSeed rawSeed canonicalFresh rawFresh
    (fullWorld bound before).heap after.heap
  rw [alpha, ← facts.2.2.2.1]
  exact and_congr_right (fun _ => initializer_world_partition bound link.config.object link.config.tile inside
    meaning link.config.program facts.2.2.2.2.2 (eraseMapped (resolve link.mapping) env) before after)


def LineInitializerLink.Bound (link : LineInitializerLink) (context : Context) (views : List PrivateView) : Prop :=
  link.canonical.seed ∈ context ∧ link.canonical.zero ∈ context ∧ link.canonical.extent ∈ context ∧
  link.canonical.one ∈ context ∧
  PrivateView.mk link.canonical.buffer link.config.object [link.extent] link.element ∈ views

instance lineInitializerLinkBoundDecidable (link : LineInitializerLink) (context : Context) (views : List PrivateView) :
    Decidable (link.Bound context views) := by
  unfold LineInitializerLink.Bound
  infer_instance

theorem LineInitializerLink.inputs {Value : Type} (link : LineInitializerLink) (facts : link.Facts)
    (context : Context) (views : List PrivateView) (bound : link.Bound context views)
    (env : Env Value) (valid : RegistersValid context views env) :
    ∃ value, env link.canonical.seed.name = .scalar link.element value ∧
      env link.canonical.buffer.name = view link.config.object [link.extent] link.element ∧
      env link.canonical.zero.name = .index 0 ∧ env link.canonical.extent.name = .index link.extent ∧
      env link.canonical.one.name = .index 1 := by
  rcases facts.2.1 with ⟨seedKind, _, _, _, _, zeroKnown, _, extentKnown, _, oneKnown, _⟩
  obtain ⟨value, scalar⟩ := scalar_of_kind (env link.canonical.seed.name) link.element
    ((valid.1 _ bound.1).1.trans seedKind)
  exact ⟨value, scalar, valid.2 _ bound.2.2.2.2,
    known_index (valid.1 _ bound.2.1) zeroKnown,
    known_index (valid.1 _ bound.2.2.1) extentKnown,
    known_index (valid.1 _ bound.2.2.2.1) oneKnown⟩

theorem LineInitializerLink.from_registers {Value User : Type} (link : LineInitializerLink)
    (checked : decide link.Facts = true) (context : Context) (views : List PrivateView)
    (boundChecked : decide (link.Bound context views) = true) (bound : Nat) (inside : link.config.object < bound)
    (base : Interpretation Value (MemoryWorld Value User)) (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (aliases : Aliases) (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (valid : RegistersValid context views before.observed.registers)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value User) :
    Eval (withMemory base classify) false aliases link.canonical.code before.observed.registers
      (fullWorld bound before) outputs after ↔
      outputs = [] ∧ ∃ next,
        GraphLoopsMemorySCF.HeapRun meaning link.config.program link.config.tile link.config.object false
          (eraseMapped (resolve link.mapping) before.observed.registers) before.heap next ∧
        after = fullWorld bound ⟨before.observed, next⟩ := by
  obtain ⟨value, seed, buffer, zero, extent, one⟩ := link.inputs (of_decide_eq_true checked)
    context views (of_decide_eq_true boundChecked) before.observed.registers valid
  exact link.checked_execution checked bound inside base classify store loop aliases before.observed.registers value
    seed buffer zero extent one meaning before outputs after

theorem LineInitializerLink.clocked {Value User : Type} (link : LineInitializerLink)
    (checked : decide link.Facts = true) (context : Context) (views : List PrivateView)
    (boundChecked : decide (link.Bound context views) = true) (bound next : Nat)
    (inside : link.config.object < bound)
    (base : Interpretation Value (MemoryWorld Value (AllocationPublic User))) (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (aliases : Aliases) (meaning : GraphLoopsMemorySCF.PureMeaning Value)
    (before : GraphLoopsReuse.Memory Value (PublicState Value User))
    (valid : RegistersValid context views before.observed.registers)
    (outputs : List (Runtime Value)) (after : MemoryWorld Value (AllocationPublic User)) :
    Eval (withMemory base classify) false aliases link.canonical.code before.observed.registers
      (clockWorld bound next before) outputs after ↔
      outputs = [] ∧ ∃ heap,
        GraphLoopsMemorySCF.HeapRun meaning link.config.program link.config.tile link.config.object false
          (eraseMapped (resolve link.mapping) before.observed.registers) before.heap heap ∧
        after = clockWorld bound next ⟨before.observed, heap⟩ :=
  link.from_registers checked context views boundChecked bound inside base classify store loop aliases meaning
    (atClock next before) valid outputs after

end Merlin.GraphLoopsShape.Epoch
