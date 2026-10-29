/- Scoped typed registers and prebound private descriptors. The invariant covers
actual bound names only; future descriptor ghosts are protected from scalar
writes and are proved irrelevant to earlier free reads by EpochScope. -/
import Merlin.GraphLoopsShapeEpochLink
import Merlin.GraphLoopsShapeEpochPrimitives

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

structure PrivateView where
  binding : Binding
  object : Nat
  shape : List Nat
  element : String
  deriving DecidableEq, Repr

def Pinned {Value : Type} (views : List PrivateView) (env : Env Value) : Prop :=
  ∀ item ∈ views, env item.binding.name = view item.object item.shape item.element

def RegistersValid {Value : Type} (context : Context) (views : List PrivateView) (env : Env Value) : Prop :=
  Typed context env ∧ Pinned views env

theorem typed_define {Value : Type} (context : Context) (env : Env Value) (typed : Typed context env)
    (result : Binding) (fresh : result.name ∉ names context) (value : Runtime Value) (holds : result.Holds value) :
    Typed (result :: context) (put env result.name value) := by
  intro binding member
  rcases List.mem_cons.mp member with rfl | previous
  · simpa only [put, ↓reduceIte] using holds
  · have distinct : binding.name ≠ result.name := fun equal => fresh (equal ▸ List.mem_map_of_mem previous)
    simpa only [put, distinct, ↓reduceIte] using typed binding previous

theorem typed_assign {Value : Type} (context : Context) (env : Env Value) (typed : Typed context env)
    (results : List Binding) (fresh : Fresh context results) (values : List (Runtime Value))
    (holds : ValuesHold results values) : Typed (extendContext context results) (assign env results values) := by
  induction holds generalizing context env with
  | nil => exact typed
  | @cons binding value rest remaining holds typedRest induction =>
    exact induction _ _ (typed_define context env typed binding fresh.1 value holds) fresh.2

theorem pinned_assign {Value : Type} (views : List PrivateView) (env : Env Value) (pinned : Pinned views env)
    (results : List Binding) (values : List (Runtime Value))
    (viewsFresh : ∀ item ∈ views, item.binding.name ∉ names results) :
    Pinned views (assign env results values) := by
  intro item member
  rw [assign_at_other env results values item.binding.name (viewsFresh item member)]
  exact pinned item member

theorem registers_assign {Value : Type} (context : Context) (views : List PrivateView) (env : Env Value)
    (valid : RegistersValid context views env) (results : List Binding) (fresh : Fresh context results)
    (values : List (Runtime Value)) (holds : ValuesHold results values)
    (viewsFresh : ∀ item ∈ views, item.binding.name ∉ names results) :
    RegistersValid (extendContext context results) views (assign env results values) :=
  ⟨typed_assign context env valid.1 results fresh values holds, pinned_assign views env valid.2 results values viewsFresh⟩

theorem scalar_of_kind {Value : Type} (runtime : Runtime Value) (element : String)
    (kind : runtime.kind = .scalar element) : ∃ value, runtime = .scalar element value := by
  cases runtime with
  | index => cases kind
  | view => cases kind
  | scalar actual value =>
    have same := Kind.scalar.inj kind
    subst actual
    exact ⟨value, rfl⟩

def InitializerLink.Bound (link : InitializerLink) (context : Context) (views : List PrivateView) : Prop :=
  link.canonical.seed ∈ context ∧ link.canonical.zero ∈ context ∧ link.canonical.rows ∈ context ∧
  link.canonical.columns ∈ context ∧ link.canonical.one ∈ context ∧
  (PrivateView.mk link.canonical.buffer link.config.object [link.rows, link.columns] link.element) ∈ views

instance initializerLinkBoundDecidable (link : InitializerLink) (context : Context) (views : List PrivateView) :
    Decidable (link.Bound context views) := by
  unfold InitializerLink.Bound
  infer_instance

/-- All runtime initializer inputs follow from the scoped register invariant:
known constants, seed scalar typing and the actual canonical private view. -/
theorem InitializerLink.inputs {Value : Type} (link : InitializerLink) (facts : link.Facts)
    (context : Context) (views : List PrivateView) (bound : link.Bound context views)
    (env : Env Value) (valid : RegistersValid context views env) :
    ∃ value, env link.canonical.seed.name = .scalar link.element value ∧
      env link.canonical.buffer.name = view link.config.object [link.rows, link.columns] link.element ∧
      env link.canonical.zero.name = .index 0 ∧ env link.canonical.rows.name = .index link.rows ∧
      env link.canonical.columns.name = .index link.columns ∧ env link.canonical.one.name = .index 1 := by
  rcases facts.2.1 with ⟨seedKind, _, _, _, _, zeroKnown, _, rowsKnown, _, columnsKnown, _, oneKnown, _⟩
  obtain ⟨value, scalar⟩ := scalar_of_kind (env link.canonical.seed.name) link.element
    ((valid.1 _ bound.1).1.trans seedKind)
  exact ⟨value, scalar, valid.2 _ bound.2.2.2.2.2,
    known_index (valid.1 _ bound.2.1) zeroKnown,
    known_index (valid.1 _ bound.2.2.1) rowsKnown,
    known_index (valid.1 _ bound.2.2.2.1) columnsKnown,
    known_index (valid.1 _ bound.2.2.2.2.1) oneKnown⟩

/-- This derived form removes individual input-value premises from the local
initializer bridge. The whole program must establish and preserve the invariant. -/
theorem InitializerLink.from_registers {Value User : Type} (link : InitializerLink)
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
  obtain ⟨value, seed, buffer, zero, rows, columns, one⟩ := link.inputs (of_decide_eq_true checked)
    context views (of_decide_eq_true boundChecked) before.observed.registers valid
  exact link.checked_execution checked bound inside base classify store loop aliases before.observed.registers value
    seed buffer zero rows columns one meaning before outputs after

/-- Private logical allocation naming is reconstructed separately from caller
observations. The register and caller-heap components are unchanged by this lift. -/
def atClock {Value User : Type} (next : Nat) (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    GraphLoopsReuse.Memory Value (PublicState Value (AllocationPublic User)) :=
  ⟨⟨memory.observed.registers, memory.observed.callerHeap, ⟨next, memory.observed.observed⟩,
    memory.observed.returned⟩, memory.heap⟩

theorem atClock_world {Value User : Type} (bound next : Nat)
    (memory : GraphLoopsReuse.Memory Value (PublicState Value User)) :
    fullWorld bound (atClock next memory) = clockWorld bound next memory := rfl

/-- The exact derived initializer bridge also frames private allocation naming.
It executes no allocations and changes no caller-visible cells or observations. -/
theorem InitializerLink.clocked {Value User : Type} (link : InitializerLink)
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
