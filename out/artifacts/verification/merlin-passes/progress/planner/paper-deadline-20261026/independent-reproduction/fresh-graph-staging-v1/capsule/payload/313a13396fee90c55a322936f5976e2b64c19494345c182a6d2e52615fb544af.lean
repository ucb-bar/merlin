/- Checked raw-middle/canonical initializer binding. Raw SSA names remain linked
to the exact old configuration; scoped stores, alpha conversion and complete
heap framing derive the bridge. No initializer agreement is postulated. -/
import Merlin.GraphLoopsShapeEpochFrame
import Merlin.GraphLoopsShapeEpochModel

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def renameUniform (mapping : Aliases) (program : Uniform) : Uniform :=
  { program with
    seed := rename mapping program.seed
    buffer := rename mapping program.buffer
    outer := rename mapping program.outer
    inner := rename mapping program.inner
    zero := rename mapping program.zero
    rows := rename mapping program.rows
    columns := rename mapping program.columns
    one := rename mapping program.one }

structure InitializerLink where
  raw : Uniform
  canonical : Uniform
  config : GraphLoopsMemorySCF.Configuration
  mapping : Aliases
  rows : Nat
  columns : Nat
  element : String
  sourceBuffer : Nat
  deriving DecidableEq, Repr

def InitializerLink.Facts (link : InitializerLink) : Prop :=
  link.raw.Facts link.rows link.columns link.element ∧
  link.canonical.Facts link.rows link.columns link.element ∧
  link.canonical = renameUniform link.mapping link.raw ∧
  link.config.program = GraphLoopsMemorySCF.Witness.uniform link.rows link.columns
    link.raw.outer.name link.raw.inner.name link.raw.seed.name ∧
  link.raw.buffer.name = link.sourceBuffer ∧ link.config.program.WellFormed

instance (link : InitializerLink) : Decidable link.Facts := by
  unfold InitializerLink.Facts
  infer_instance

theorem InitializerLink.seed_mapping (link : InitializerLink) (facts : link.Facts) :
    resolve link.mapping link.raw.seed.name = link.canonical.seed.name := by
  rw [facts.2.2.1]
  rfl

/-- Connect the actual raw configuration's capture names to source names and
split the operational initializer's full heap into private/caller components. -/
theorem InitializerLink.checked_execution {Value User : Type} (link : InitializerLink)
    (checked : decide link.Facts = true) (bound : Nat) (inside : link.config.object < bound)
    (base : Interpretation Value (MemoryWorld Value User)) (classify : String → MemoryOpcode)
    (store : classify link.canonical.storeKey = .store) (loop : classify link.canonical.loopKey = .loop)
    (aliases : Aliases) (env : Env Value) (value : Value)
    (seedValue : env link.canonical.seed.name = .scalar link.element value)
    (bufferValue : env link.canonical.buffer.name = view link.config.object [link.rows, link.columns] link.element)
    (zeroValue : env link.canonical.zero.name = .index 0)
    (rowsValue : env link.canonical.rows.name = .index link.rows)
    (columnsValue : env link.canonical.columns.name = .index link.columns)
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
  have rawSeparate := rawFacts.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have rawSeedOuter := rawFacts.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have rawSeedInner := rawFacts.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have canonicalSeparate := canonicalFacts.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have canonicalSeedOuter := canonicalFacts.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have canonicalSeedInner := canonicalFacts.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  rw [link.canonical.checked_eval base classify link.rows link.columns link.element
    (decide_eq_true canonicalFacts) store loop aliases env link.config.object link.config.tile value
    seedValue bufferValue zeroValue rowsValue columnsValue oneValue meaning]
  have alpha := uniform_heap_alpha meaning meaning
    link.canonical.outer.name link.canonical.inner.name link.canonical.seed.name
    link.raw.outer.name link.raw.inner.name link.raw.seed.name link.config.object
    link.rows link.columns link.config.tile false (eraseEnv env) (eraseMapped (resolve link.mapping) env)
    value canonicalSeed rawSeed canonicalSeparate canonicalSeedOuter canonicalSeedInner
    rawSeparate rawSeedOuter rawSeedInner (fullWorld bound before).heap after.heap
  rw [alpha, ← facts.2.2.2.1]
  exact and_congr_right (fun _ => initializer_world_partition bound link.config.object link.config.tile inside
    meaning link.config.program facts.2.2.2.2.2 (eraseMapped (resolve link.mapping) env) before after)

end Merlin.GraphLoopsShape.Epoch
