/- Checked actual buffer-reuse projection. Python/raw-SCF/allocator conformance
are explicit trust boundaries, not assumptions added as Lean axioms. -/
import Merlin.GraphLoopsReuseLifetime

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse.Model

structure Initializer where
  object : Nat
  controls : List (Nat × Nat × Nat)
  ivs : List Nat
  indices : List Nat
  sourceBody : String
  targetBody : String
  deriving DecidableEq, Repr

structure Projection where
  bits : Nat
  objects : Objects
  instructions : List Instruction
  initializers : List Initializer
  lifetimes : List Lifetime
  sourceContext : String
  targetContext : String
  allocatorMode : String
  initializerMode : String
  deriving DecidableEq, Repr

def outcomes {Value Public : Type} (p : Projection) (primitives : Primitives Value Public) (candidate : Bool)
    (initial : Memory Value Public) (outcome : Public) : Prop :=
  ∃ final, Execution p.objects primitives candidate p.instructions initial final ∧ final.observed = outcome

def ValidObject (bits : Nat) (object : Object) : Prop :=
  Geometry.Positive object.shape ∧ object.shape ≠ [] ∧ object.elementBits ∈ [8, 16, 32, 64] ∧
  0 < object.alignment ∧ object.alignment = 2 ^ (Nat.log2 object.alignment) ∧
  Geometry.volume object.shape * (object.elementBits / 8) < 2 ^ (bits - 1)

instance (bits : Nat) (object : Object) : Decidable (ValidObject bits object) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

theorem ValidObject.element_bytes_positive {bits : Nat} {object : Object} (valid : ValidObject bits object) :
    0 < object.elementBits / 8 := by
  have widths := valid.2.2.1
  simp only [List.mem_cons, List.not_mem_nil, or_false] at widths
  rcases widths with eight | sixteen | thirtyTwo | sixtyFour <;> omega

/-- Relative byte offsets fit. No physical base address/capacity or allocator
ABI relation is invented by this index arithmetic theorem. -/
theorem ValidObject.byte_offset {bits : Nat} {object : Object} (valid : ValidObject bits object)
    {coordinates : List Nat} (inside : Geometry.Coordinate object.shape coordinates) :
    Geometry.flatten object.shape coordinates * (object.elementBits / 8) < 2 ^ (bits - 1) := by
  have coordinate := Geometry.coordinate_inside inside
  exact Nat.lt_trans (Nat.mul_lt_mul_of_pos_right coordinate valid.element_bytes_positive) valid.2.2.2.2.2

theorem ValidObject.initializer_control {bits : Nat} {object : Object} (valid : ValidObject bits object)
    {dimension index : Nat} (member : dimension ∈ object.shape) (inside : index < dimension) :
    index < 2 ^ (bits - 1) ∧ index + 1 < 2 ^ (bits - 1) := by
  have dimensionBound := Geometry.member_le_volume valid.1 member
  have bytes := valid.element_bytes_positive
  have volumeBound := Nat.mul_le_mul_left (Geometry.volume object.shape) (show 1 ≤ object.elementBits / 8 by omega)
  simp only [Nat.mul_one] at volumeBound
  have size := valid.2.2.2.2.2
  omega

def CompatibleObjects (objects : Objects) : Prop :=
  ∀ first ∈ objects, ∀ second ∈ objects, first.slot = second.slot →
    first.shape = second.shape ∧ first.elementBits = second.elementBits ∧ first.alignment = second.alignment

instance (objects : Objects) : Decidable (CompatibleObjects objects) :=
  inferInstanceAs (Decidable (∀ first ∈ objects, ∀ second ∈ objects, first.slot = second.slot →
    first.shape = second.shape ∧ first.elementBits = second.elementBits ∧ first.alignment = second.alignment))

def Initializer.Valid (objects : Objects) (init : Initializer) : Prop :=
  init.object < objects.length ∧
  init.controls = (objectAt objects init.object).shape.map (fun extent => (0, extent, 1)) ∧
  init.ivs.length = (objectAt objects init.object).shape.length ∧ init.indices = init.ivs ∧
  init.sourceBody = init.targetBody

instance (objects : Objects) (init : Initializer) : Decidable (init.Valid objects) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def InitializerBound (projection : Projection) (instruction : Instruction) : Prop :=
  match instruction with
  | .initialize object key => ∃ init ∈ projection.initializers, init.object = object ∧ init.sourceBody = key
  | _ => True

instance (projection : Projection) (instruction : Instruction) : Decidable (InitializerBound projection instruction) := by
  cases instruction <;> unfold InitializerBound <;> infer_instance

def Valid (p : Projection) : Prop :=
  (p.bits = 32 ∨ p.bits = 64) ∧ (∀ object ∈ p.objects, ValidObject p.bits object) ∧ CompatibleObjects p.objects ∧
  p.sourceContext = p.targetContext ∧ p.allocatorMode = "total_private_unobservable" ∧
  p.initializerMode = "partial_complete_rectangular_pure" ∧
  (∀ init ∈ p.initializers, init.Valid p.objects) ∧ (∀ instruction ∈ p.instructions, InitializerBound p instruction) ∧
  LifetimesValid p.objects p.instructions p.lifetimes

instance (p : Projection) : Decidable (Valid p) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))

def check (p : Projection) : Bool := decide (Valid p) && checkProgram p.objects emptyStatus p.instructions

theorem check_sound {p : Projection} (accepted : check p = true) :
    Valid p ∧ Safe p.objects emptyStatus p.instructions := by
  obtain ⟨valid, safe⟩ := Bool.and_eq_true_iff.mp accepted
  exact ⟨of_decide_eq_true valid, checkProgram_sound _ _ _ safe⟩

/-- Every valid actual initializer loop visits a coordinate for each in-bounds
linear offset. Its raw SCF operational semantics is explicitly admitted separately. -/
theorem checked_initializer_coverage {p : Projection} (accepted : check p = true)
    {init : Initializer} (member : init ∈ p.initializers) :
    ∀ index, index < Geometry.volume (objectAt p.objects init.object).shape →
      ∃ coordinates, Geometry.Coordinate (objectAt p.objects init.object).shape coordinates ∧
        Geometry.flatten (objectAt p.objects init.object).shape coordinates = index := by
  have valid := (check_sound accepted).1
  have initValid := valid.2.2.2.2.2.2.1 init member
  have objectMember : objectAt p.objects init.object ∈ p.objects := by
    unfold objectAt
    rw [List.getElem?_eq_getElem initValid.1]
    exact List.mem_of_getElem rfl
  exact fun _ inside => Geometry.every_cell_covered (valid.2.1 _ objectMember).1 inside

/-- Complete possible-observation equivalence under the declared PRIVATE TOTAL
allocator model and the SAME admitted partial primitive interpretations. -/
theorem checked_complete_observations {p : Projection} (accepted : check p = true) {Value Public : Type}
    (primitives : Primitives Value Public) (source target : Memory Value Public)
    (samePublic : source.observed = target.observed) (outcome : Public) :
    (∃ final, Execution p.objects primitives false p.instructions source final ∧ final.observed = outcome) ↔
    (∃ final, Execution p.objects primitives true p.instructions target final ∧ final.observed = outcome) := by
  exact checked_observations _ _ (Bool.and_eq_true_iff.mp accepted).2 primitives source target samePublic outcome

theorem checked_context {p : Projection} (accepted : check p = true) {Value Public Outcome : Type}
    (primitives : Primitives Value Public) (source target : Memory Value Public)
    (samePublic : source.observed = target.observed) (context : String → (Public → Prop) → Outcome) :
    context p.sourceContext (outcomes p primitives false source) =
      context p.targetContext (outcomes p primitives true target) := by
  have contexts := (check_sound accepted).1.2.2.2.1
  have possible : outcomes p primitives false source = outcomes p primitives true target := by
    funext outcome
    exact propext (checked_complete_observations accepted primitives source target samePublic outcome)
  rw [contexts, possible]

end Merlin.GraphLoopsReuse.Model
