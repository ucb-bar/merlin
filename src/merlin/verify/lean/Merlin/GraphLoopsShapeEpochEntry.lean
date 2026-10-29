/- Function-entry ghost descriptors are constructed, not assumed to be runtime
inputs. They have no free reads before their actual allocation definitions.
Caller descriptors remain explicit input-domain facts and may alias each other. -/
import Merlin.GraphLoopsShapeEpochInvariant

set_option autoImplicit false

namespace Merlin.GraphLoopsShape.Epoch

def seedViews {Value : Type} (env : Env Value) : List PrivateView → Env Value
  | [] => env
  | item :: rest => put (seedViews env rest) item.binding.name (view item.object item.shape item.element)

theorem seed_other {Value : Type} (env : Env Value) (views : List PrivateView) (name : Nat)
    (outside : name ∉ views.map (fun item => item.binding.name)) : seedViews env views name = env name := by
  induction views with
  | nil => rfl
  | cons item rest induction =>
    have different : name ≠ item.binding.name := fun same => outside (by simp [same])
    have remaining : name ∉ rest.map (fun item => item.binding.name) :=
      fun member => outside (List.mem_cons_of_mem _ member)
    change (if name = item.binding.name then _ else _) = _
    rw [ite_eq_right different]
    exact induction remaining

theorem seed_pinned {Value : Type} (env : Env Value) (views : List PrivateView)
    (unique : (views.map (fun item => item.binding.name)).Nodup) : Pinned views (seedViews env views) := by
  induction views with
  | nil => intro item member; cases member
  | cons first rest induction =>
    have facts := List.nodup_cons.mp unique
    intro item member
    rcases List.mem_cons.mp member with rfl | later
    · simp only [seedViews, put, ↓reduceIte]
    · have different : item.binding.name ≠ first.binding.name := fun same =>
        facts.1 (List.mem_map.mpr ⟨item, later, same⟩)
      simp only [seedViews, put, different, ↓reduceIte]
      exact induction facts.2 item later

theorem seed_typed {Value : Type} (env : Env Value) (context : Context) (views : List PrivateView)
    (typed : Typed context env) (fresh : ∀ item ∈ views, item.binding.name ∉ names context) :
    Typed context (seedViews env views) := by
  intro binding member
  have outside : binding.name ∉ views.map (fun item => item.binding.name) := by
    rintro mapped
    obtain ⟨item, itemMember, equal⟩ := List.mem_map.mp mapped
    exact fresh item itemMember (equal ▸ List.mem_map_of_mem member)
  rw [seed_other env views binding.name outside]
  exact typed binding member

theorem seed_callers {Value : Type} (env : Env Value) (bound : Nat) (context : Context)
    (views : List PrivateView) (callers : List CallerView)
    (fresh : ∀ item ∈ views, item.binding.name ∉ names context)
    (boundCallers : CallersBound context callers) (valid : CallersValid bound callers env) :
    CallersValid bound callers (seedViews env views) := by
  intro caller member
  obtain ⟨object, outside, read⟩ := valid caller member
  refine ⟨object, outside, ?_⟩
  rw [seed_other env views caller.binding.name]
  · exact read
  · rintro mapped
    obtain ⟨item, itemMember, equal⟩ := List.mem_map.mp mapped
    exact fresh item itemMember (equal ▸ List.mem_map_of_mem (boundCallers caller member))

/-- The whole typed body, including nested captures, cannot inspect these
future descriptor ghosts. No retained-operation behavior is assumed here. -/
theorem seed_eval {Value World : Type} (meaning : Interpretation Value World) (code : Code)
    (env : Env Value) (views : List PrivateView)
    (unused : ∀ item ∈ views, item.binding.name ∉ freeReads code)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval meaning false [] code env before outputs after ↔
      Eval meaning false [] code (seedViews env views) before outputs after := by
  apply eval_free_reads
  intro name member
  apply Eq.symm
  apply seed_other
  intro mapped
  obtain ⟨item, itemMember, equal⟩ := List.mem_map.mp mapped
  exact unused item itemMember (equal ▸ member)

def EntryFacts (plan : PlanContext) (function : Function) : Prop :=
  Fresh [] function.parameters ∧
  (plan.views.map (fun item => item.binding.name)).Nodup ∧
  (∀ item ∈ plan.views, item.binding.name ∉ names function.parameters) ∧
  (∀ item ∈ plan.views, item.binding.name ∉ freeReads function.code) ∧
  CallersBound function.parameters plan.callers

instance entryFactsDecidable (plan : PlanContext) (function : Function) : Decidable (EntryFacts plan function) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def entryMemory {Value User : Type} (plan : PlanContext) (function : Function)
    (inputs : List (Runtime Value)) (heap : Heap Value) (user : User) :
    GraphLoopsReuse.Memory Value (PublicState Value User) :=
  ⟨⟨seedViews (assign emptyEnv function.parameters inputs) plan.views, heap, user, []⟩, heap⟩

theorem merge_same {Value : Type} (bound : Nat) (heap : Heap Value) : mergeHeap bound heap heap = heap := by
  funext object index
  simp only [mergeHeap, ite_self]

theorem entry_world {Value User : Type} (plan : PlanContext) (function : Function)
    (inputs : List (Runtime Value)) (heap : Heap Value) (user : User) :
    clockWorld plan.objects.length 0 (entryMemory plan function inputs heap user) = ⟨⟨0, user⟩, heap⟩ := by
  simp only [clockWorld, entryMemory, merge_same]

theorem mem_extend_context (context bindings : Context) (binding : Binding) :
    binding ∈ extendContext context bindings ↔ binding ∈ bindings ∨ binding ∈ context := by
  induction bindings generalizing context with
  | nil => simp only [extendContext, List.not_mem_nil, false_or]
  | cons first rest induction =>
    simp only [extendContext, induction, List.mem_cons]
    constructor
    · rintro (later | first | prior)
      · exact Or.inl (Or.inr later)
      · exact Or.inl (Or.inl first)
      · exact Or.inr prior
    · rintro ((first | later) | prior)
      · exact Or.inr (Or.inl first)
      · exact Or.inl later
      · exact Or.inr (Or.inr prior)

theorem entry_registers {Value User : Type} (plan : PlanContext) (function : Function)
    (facts : EntryFacts plan function) (inputs : List (Runtime Value)) (typed : ValuesHold function.parameters inputs)
    (heap : Heap Value) (user : User) :
    RegistersValid function.parameters plan.views
      (entryMemory plan function inputs heap user).observed.registers := by
  have assigned := typed_assign [] emptyEnv (by simp [Typed]) _ facts.1 inputs typed
  have parameters : Typed function.parameters (assign emptyEnv function.parameters inputs) := by
    intro binding member
    exact assigned binding ((mem_extend_context [] function.parameters binding).mpr (Or.inl member))
  exact ⟨seed_typed _ _ _ parameters facts.2.2.1, seed_pinned _ _ facts.2.1⟩

theorem entry_callers {Value User : Type} (plan : PlanContext) (function : Function)
    (facts : EntryFacts plan function) (inputs : List (Runtime Value))
    (valid : CallersValid plan.objects.length plan.callers (assign emptyEnv function.parameters inputs))
    (heap : Heap Value) (user : User) :
    CallersValid plan.objects.length plan.callers (entryMemory plan function inputs heap user).observed.registers :=
  seed_callers _ _ _ _ _ facts.2.2.1 facts.2.2.2.2 valid

end Merlin.GraphLoopsShape.Epoch
