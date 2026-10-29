/- Parameterized row-major rectangular initializer coverage.
This establishes coordinate coverage, not a fabricated frozen-evaluator/SCF
execution route. Complete rectangular-loop semantics is an explicit admission. -/
import Merlin.GraphLoopsReuseSemantics

set_option autoImplicit false

namespace Merlin.GraphLoopsReuse.Geometry

def volume : List Nat → Nat
  | [] => 1
  | extent :: rest => extent * volume rest

def Positive (shape : List Nat) : Prop := ∀ extent ∈ shape, 0 < extent

instance (shape : List Nat) : Decidable (Positive shape) :=
  inferInstanceAs (Decidable (∀ extent ∈ shape, 0 < extent))

theorem volume_positive {shape : List Nat} (positive : Positive shape) : 0 < volume shape := by
  induction shape with
  | nil => simp only [volume, Nat.zero_lt_succ]
  | cons extent rest induction =>
    exact Nat.mul_pos (positive extent (List.mem_cons_self))
      (induction (fun other member => positive other (List.mem_cons_of_mem extent member)))

def Coordinate : List Nat → List Nat → Prop
  | [], [] => True
  | extent :: shape, index :: rest => index < extent ∧ Coordinate shape rest
  | _, _ => False

def flatten : List Nat → List Nat → Nat
  | _ :: shape, index :: rest => index * volume shape + flatten shape rest
  | _, _ => 0

def decode : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: shape, index => index / volume shape :: decode shape (index % volume shape)

theorem decode_valid {shape : List Nat} (positive : Positive shape) {index : Nat} (inside : index < volume shape) :
    Coordinate shape (decode shape index) := by
  induction shape generalizing index with
  | nil => trivial
  | cons extent rest induction =>
    have positiveRest : Positive rest := fun other member => positive other (List.mem_cons_of_mem extent member)
    have stridePositive := volume_positive positiveRest
    change index / volume rest < extent ∧ Coordinate rest (decode rest (index % volume rest))
    exact ⟨(Nat.div_lt_iff_lt_mul stridePositive).mpr inside, induction positiveRest (Nat.mod_lt _ stridePositive)⟩

theorem flatten_decode {shape : List Nat} (positive : Positive shape) {index : Nat} (inside : index < volume shape) :
    flatten shape (decode shape index) = index := by
  induction shape generalizing index with
  | nil => simp only [volume] at inside; simp only [flatten]; omega
  | cons extent rest induction =>
    have positiveRest : Positive rest := fun other member => positive other (List.mem_cons_of_mem extent member)
    have stridePositive := volume_positive positiveRest
    change index / volume rest * volume rest + flatten rest (decode rest (index % volume rest)) = index
    rw [induction positiveRest (Nat.mod_lt _ stridePositive)]
    simpa only [Nat.mul_comm] using Nat.div_add_mod index (volume rest)

theorem every_cell_covered {shape : List Nat} (positive : Positive shape) {index : Nat} (inside : index < volume shape) :
    ∃ coordinates, Coordinate shape coordinates ∧ flatten shape coordinates = index :=
  ⟨decode shape index, decode_valid positive inside, flatten_decode positive inside⟩

/-- The exact static bounds 0..extent step1 of each nested initializer loop.
No tensor-volume-sized list or proof object is produced. -/
def Rectangular (writes : List Nat → Prop) : List Nat → Prop
  | [] => writes []
  | extent :: shape => ∀ index, index < extent → Rectangular (fun rest => writes (index :: rest)) shape

theorem rectangular_iff_coordinates (shape : List Nat) : ∀ writes,
    Rectangular writes shape ↔ ∀ coordinates, Coordinate shape coordinates → writes coordinates := by
  induction shape with
  | nil =>
    intro writes
    constructor
    · intro written coordinates valid
      cases coordinates with
      | nil => exact written
      | cons => cases valid
    · intro written; exact written [] True.intro
  | cons extent rest induction =>
    intro writes
    constructor
    · intro rectangular coordinates valid
      cases coordinates with
      | nil => cases valid
      | cons index tail => exact (induction _).mp (rectangular index valid.1) tail valid.2
    · intro written index inside
      apply (induction _).mpr
      intro coordinates valid
      exact written (index :: coordinates) ⟨inside, valid⟩

/-- Every in-bounds cell has its initialization value after the complete
rectangular write. This removes stale prior-epoch contents at ALL valid offsets. -/
theorem rectangular_initializes {Value : Type} {shape : List Nat} (positive : Positive shape)
    (cells : Cells Value) (values : Nat → Value)
    (written : Rectangular (fun coordinate => cells (flatten shape coordinate) = some (values (flatten shape coordinate))) shape) :
    ∀ index, index < volume shape → cells index = some (values index) := by
  intro index inside
  obtain ⟨coordinate, valid, equal⟩ := every_cell_covered positive inside
  have atCoordinate := (rectangular_iff_coordinates shape _).mp written coordinate valid
  simpa only [equal] using atCoordinate

theorem coordinate_inside {shape coordinates : List Nat} (valid : Coordinate shape coordinates) :
    flatten shape coordinates < volume shape := by
  induction shape generalizing coordinates with
  | nil => cases coordinates <;> simp_all only [Coordinate, flatten, volume, Nat.zero_lt_succ]
  | cons extent rest induction =>
    cases coordinates with
    | nil => cases valid
    | cons index tail =>
      have headInside := valid.1
      have tailInside := induction valid.2
      have bound := Nat.mul_le_mul_right (volume rest) (show index + 1 ≤ extent by omega)
      simp only [Nat.add_mul, Nat.one_mul] at bound
      change index * volume rest + flatten rest tail < extent * volume rest
      omega

theorem member_le_volume {shape : List Nat} (positive : Positive shape) {extent : Nat} (member : extent ∈ shape) :
    extent ≤ volume shape := by
  induction shape with
  | nil => cases member
  | cons head rest induction =>
    have headPositive := positive head (List.mem_cons_self)
    have restPositive : Positive rest := fun other inside => positive other (List.mem_cons_of_mem head inside)
    have restVolumePositive := volume_positive restPositive
    rcases List.mem_cons.mp member with rfl | later
    · simpa only [volume, Nat.mul_one] using Nat.mul_le_mul_left extent (show 1 ≤ volume rest by omega)
    · have previous := induction restPositive later
      have growth := Nat.mul_le_mul_right (volume rest) (show 1 ≤ head by omega)
      simp only [Nat.one_mul] at growth
      exact Nat.le_trans previous growth

end Merlin.GraphLoopsReuse.Geometry

namespace Merlin.GraphLoopsReuse

theorem Object.extent_geometry (object : Object) : object.extent = Geometry.volume object.shape := by
  unfold Object.extent
  induction object.shape with
  | nil => rfl
  | cons head rest induction => simpa only [List.foldr_cons, Geometry.volume] using congrArg (head * ·) induction

end Merlin.GraphLoopsReuse
