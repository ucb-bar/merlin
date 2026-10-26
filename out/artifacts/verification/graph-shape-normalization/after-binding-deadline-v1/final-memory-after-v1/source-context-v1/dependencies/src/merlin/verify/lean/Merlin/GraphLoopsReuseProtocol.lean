/- Finite protocol for actual logical epochs, initializer-before-read and the
retained physical allocation/final-free boundary. Tensor cells are not enumerated. -/
import Merlin.GraphLoopsReuse

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse

structure Object where
  slot : Nat
  shape : List Nat
  elementBits : Nat
  alignment : Nat
  deriving DecidableEq, Repr, Inhabited

def Object.extent (object : Object) : Nat := object.shape.foldr (· * ·) 1
abbrev Objects := List Object

def objectAt (objects : Objects) (object : Nat) : Object := objects[object]?.getD default
def slotOf (objects : Objects) (object : Nat) : Nat := (objectAt objects object).slot
def extentOf (objects : Objects) (object : Nat) : Nat := (objectAt objects object).extent

inductive Instruction where
  | allocate (object : Nat) (retained : Bool)
  | initialize (object : Nat) (key : String)
  | load (object : Nat) (key : String)
  | store (object : Nat) (key : String)
  | dimension (object : Nat) (key : String)
  | release (object : Nat) (retained : Bool)
  | retained (key : String)
  deriving DecidableEq, Repr

structure Status where
  live : List Nat
  ready : List Nat
  reserved : List Nat
  deriving DecidableEq, Repr

def emptyStatus : Status := ⟨[], [], []⟩

def Invariant (objects : Objects) (status : Status) : Prop :=
  status.live.Nodup ∧ status.ready.Nodup ∧ status.reserved.Nodup ∧
  (∀ object ∈ status.live, object < objects.length) ∧
  (∀ object ∈ status.ready, object ∈ status.live) ∧
  SlotUnique (slotOf objects) status.live ∧
  ∀ object ∈ status.live, slotOf objects object ∈ status.reserved

instance (objects : Objects) (status : Status) : Decidable (Invariant objects status) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def next (objects : Objects) (status : Status) : Instruction → Status
  | .allocate object retained => ⟨object :: status.live, status.ready,
      if retained then slotOf objects object :: status.reserved else status.reserved⟩
  | .initialize object _ => ⟨status.live, object :: status.ready, status.reserved⟩
  | .release object retained => ⟨status.live.filter (· != object), status.ready.filter (· != object),
      if retained then status.reserved.filter (· != slotOf objects object) else status.reserved⟩
  | _ => status

def Guard (objects : Objects) (status : Status) : Instruction → Prop
  | .allocate object retained => object < objects.length ∧ object ∉ status.live ∧ object ∉ status.ready ∧
      (if retained then slotOf objects object ∉ status.reserved else slotOf objects object ∈ status.reserved)
  | .initialize object _ => object ∈ status.live ∧ object ∉ status.ready
  | .load object _ | .store object _ | .dimension object _ => object ∈ status.ready
  | .release object _ => object ∈ status.ready
  | .retained _ => True

instance (objects : Objects) (status : Status) (instruction : Instruction) :
    Decidable (Guard objects status instruction) := by
  cases instruction <;> unfold Guard <;> infer_instance

def Admitted (objects : Objects) (status : Status) (instruction : Instruction) : Prop :=
  Invariant objects status ∧ Guard objects status instruction ∧ Invariant objects (next objects status instruction)

instance (objects : Objects) (status : Status) (instruction : Instruction) :
    Decidable (Admitted objects status instruction) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

inductive Safe (objects : Objects) : Status → List Instruction → Prop where
  | done : Safe objects emptyStatus []
  | step {status : Status} {instruction : Instruction} {rest : List Instruction}
      (admitted : Admitted objects status instruction)
      (remaining : Safe objects (next objects status instruction) rest) : Safe objects status (instruction :: rest)

def checkProgram (objects : Objects) (status : Status) : List Instruction → Bool
  | [] => decide (status = emptyStatus)
  | instruction :: rest => decide (Admitted objects status instruction) &&
      checkProgram objects (next objects status instruction) rest

theorem checkProgram_sound (objects : Objects) (instructions : List Instruction) :
    ∀ status, checkProgram objects status instructions = true → Safe objects status instructions := by
  induction instructions with
  | nil =>
    intro status checked
    have equal : status = emptyStatus := of_decide_eq_true checked
    subst status
    exact .done
  | cons instruction rest induction =>
    intro status checked
    obtain ⟨first, remaining⟩ := Bool.and_eq_true_iff.mp checked
    exact .step (of_decide_eq_true first) (induction _ remaining)

theorem Invariant.ready_unique {objects : Objects} {status : Status} (valid : Invariant objects status) :
    SlotUnique (slotOf objects) status.ready := by
  intro first initializedFirst second initializedSecond equal
  exact valid.2.2.2.2.2.1 first (valid.2.2.2.2.1 first initializedFirst)
    second (valid.2.2.2.2.1 second initializedSecond) equal

theorem Admitted.allocate_separate {objects : Objects} {status : Status} {object : Nat} {retain : Bool}
    (admitted : Admitted objects status (.allocate object retain)) :
    ∀ other ∈ status.ready, slotOf objects other ≠ slotOf objects object := by
  intro other initialized equal
  have previousLive := admitted.1.2.2.2.2.1 other initialized
  have unique := admitted.2.2.2.2.2.2.2.1
  have same := unique other (List.mem_cons_of_mem object previousLive) object (List.mem_cons_self) equal
  exact admitted.2.1.2.1 (same ▸ previousLive)

theorem Admitted.initialize_separate {objects : Objects} {status : Status} {object : Nat} {key : String}
    (admitted : Admitted objects status (.initialize object key)) :
    ∀ other ∈ status.ready, other ≠ object → slotOf objects other ≠ slotOf objects object := by
  intro other initialized different equal
  have otherLive := admitted.1.2.2.2.2.1 other initialized
  have objectLive := admitted.2.1.1
  exact different (admitted.1.2.2.2.2.2.1 other otherLive object objectLive equal)

end Merlin.GraphLoopsReuse
