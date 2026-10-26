/- Live initialized logical epochs related to physical reuse slots.
Only live, in-bounds cells are compared. Dead/private heap cells and pointer
identities are NOT asserted equal. Public state includes caller memory, scalar
results and retained observations. Allocation is private, total and unobservable
in this model; concrete allocator failure/observability remains a premise. -/
import Merlin.ResourceGeometry
import Merlin.StateEffects

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse

abbrev Cells (Value : Type) := Nat → Option Value
abbrev Heap (Value : Type) := Nat → Cells Value

structure Memory (Value Public : Type) where
  observed : Public
  heap : Heap Value

def replaceObject {Value : Type} (heap : Heap Value) (object : Nat) (cells : Cells Value) : Heap Value :=
  fun other => if other = object then cells else heap other

def replaceCell {Value : Type} (heap : Heap Value) (object index : Nat) (value : Value) : Heap Value :=
  replaceObject heap object (fun other => if other = index then some value else heap object other)

def initializedCells {Value : Type} (extent : Nat) (values : Nat → Value) : Cells Value :=
  fun index => if index < extent then some (values index) else none

def Related {Value Public : Type} (slot extent : Nat → Nat) (ready : List Nat)
    (source target : Memory Value Public) : Prop :=
  source.observed = target.observed ∧
  ∀ object ∈ ready, ∀ index, index < extent object → source.heap object index = target.heap (slot object) index

def SlotUnique (slot : Nat → Nat) (live : List Nat) : Prop :=
  ∀ first ∈ live, ∀ second ∈ live, slot first = slot second → first = second

instance (slot : Nat → Nat) (live : List Nat) : Decidable (SlotUnique slot live) :=
  inferInstanceAs (Decidable (∀ first ∈ live, ∀ second ∈ live, slot first = slot second → first = second))

theorem Related.read {Value Public : Type} {slot extent : Nat → Nat} {ready : List Nat}
    {source target : Memory Value Public} (related : Related slot extent ready source target)
    {object index : Nat} (initialized : object ∈ ready) (inside : index < extent object) :
    source.heap object index = target.heap (slot object) index := related.2 object initialized index inside

/-- Retained partial/effectful actions operate on exactly the same public state.
They may alter caller memory and observations. They cannot introspect private
allocator history or private pointers unless separately modeled and proved. -/
theorem Related.observed_step {Value Public : Type} {slot extent : Nat → Nat} {ready : List Nat}
    {source target : Memory Value Public} (related : Related slot extent ready source target)
    (after : Public) : Related slot extent ready ⟨after, source.heap⟩ ⟨after, target.heap⟩ :=
  ⟨rfl, related.2⟩

/-- A newly live epoch is uninitialized. Stale physical contents are unconstrained
until complete initialization; no load is admitted during that interval. -/
theorem Related.allocate {Value Public : Type} {slot extent : Nat → Nat} {ready : List Nat}
    {source target : Memory Value Public} (related : Related slot extent ready source target) (object : Nat)
    (fresh : object ∉ ready) (separate : ∀ other ∈ ready, slot other ≠ slot object) (retain : Bool) :
    Related slot extent ready
      ⟨source.observed, replaceObject source.heap object (fun _ => none)⟩
      ⟨target.observed, if retain then replaceObject target.heap (slot object) (fun _ => none) else target.heap⟩ := by
  refine ⟨related.1, ?_⟩
  intro other initialized index inside
  have different : other ≠ object := by intro equal; subst other; exact fresh initialized
  simp only [replaceObject, ite_eq_right different]
  cases retain with
  | false => exact related.read initialized inside
  | true => simpa only [Bool.true_eq, ↓reduceIte, replaceObject, ite_eq_right (separate other initialized)] using related.read initialized inside

/-- Complete initialization relates every valid coordinate in the new epoch,
independently of arbitrary stale prior-slot contents. -/
theorem Related.initialize {Value Public : Type} {slot extent : Nat → Nat} {ready : List Nat}
    {source target : Memory Value Public} (related : Related slot extent ready source target)
    (object : Nat) (values : Nat → Value) (separate : ∀ other ∈ ready, other ≠ object → slot other ≠ slot object) :
    Related slot extent (object :: ready)
      ⟨source.observed, replaceObject source.heap object (initializedCells (extent object) values)⟩
      ⟨target.observed, replaceObject target.heap (slot object) (initializedCells (extent object) values)⟩ := by
  refine ⟨related.1, ?_⟩
  intro other initialized index inside
  rcases List.mem_cons.mp initialized with rfl | previous
  · simp only [replaceObject, ↓reduceIte]
  · by_cases equal : other = object
    · subst other; simp only [replaceObject, ↓reduceIte]
    · simpa only [replaceObject, ite_eq_right equal, ite_eq_right (separate other previous equal)] using related.read previous inside

/-- Store simulation preserves ALL simultaneously live initialized epochs,
not equality of every independently named logical and physical cell. -/
theorem Related.store {Value Public : Type} {slot extent : Nat → Nat} {ready : List Nat}
    {source target : Memory Value Public} (related : Related slot extent ready source target)
    (unique : SlotUnique slot ready) {object : Nat} (initialized : object ∈ ready) (index : Nat) (value : Value) :
    Related slot extent ready ⟨source.observed, replaceCell source.heap object index value⟩
      ⟨target.observed, replaceCell target.heap (slot object) index value⟩ := by
  refine ⟨related.1, ?_⟩
  intro other previous cell inside
  by_cases equal : other = object
  · subst other
    simp only [replaceCell, replaceObject, ↓reduceIte]
    split
    · rfl
    · exact related.read initialized inside
  · have separate : slot other ≠ slot object := fun same => equal (unique other previous object initialized same)
    simpa only [replaceCell, replaceObject, ite_eq_right equal, ite_eq_right separate] using related.read previous inside

/-- Source free and a retained-or-erased target free preserve all other live
values; exact final reservation validity is checked by the program protocol. -/
theorem Related.free {Value Public : Type} {slot extent : Nat → Nat} {ready : List Nat}
    {source target : Memory Value Public} (related : Related slot extent ready source target)
    (unique : SlotUnique slot ready) (object : Nat) (initializedObject : object ∈ ready) (retain : Bool) :
    Related slot extent (ready.filter (fun other => other != object))
      ⟨source.observed, replaceObject source.heap object (fun _ => none)⟩
      ⟨target.observed, if retain then replaceObject target.heap (slot object) (fun _ => none) else target.heap⟩ := by
  refine ⟨related.1, ?_⟩
  intro other initialized index inside
  obtain ⟨previous, different⟩ := List.mem_filter.mp initialized
  have different : other ≠ object := by simpa only [bne_iff_ne] using different
  have separate : slot other ≠ slot object := fun same =>
    different (unique other previous object initializedObject same)
  simp only [replaceObject, ite_eq_right different]
  cases retain with
  | false => exact related.read previous inside
  | true => simpa only [Bool.true_eq, ↓reduceIte, replaceObject, ite_eq_right separate] using related.read previous inside

end Merlin.GraphLoopsReuse
