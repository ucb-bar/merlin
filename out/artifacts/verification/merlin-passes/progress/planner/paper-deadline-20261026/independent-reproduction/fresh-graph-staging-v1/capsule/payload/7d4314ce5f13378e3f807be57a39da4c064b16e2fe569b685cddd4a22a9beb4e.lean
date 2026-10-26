/- Explicit static view descriptors for shape_view_normalization.
This is a new graph interpretation, not the frozen MerlinVerify evaluator.
Descriptor/type consistency is a LOCAL semantic premise, not supplied by hashes. -/
import Merlin.GraphLoopsMemorySCFBounds

set_option autoImplicit false

namespace Merlin.GraphLoopsShape

structure ViewType where
  shape : List Nat
  offset : Nat
  strides : List Nat
  element : String
  space : Nat
  deriving DecidableEq, Repr

/-- Allocation identity and aligned base remain explicit even when an element
address is offset from that base. Identical casts never rebuild descriptors. -/
structure Descriptor where
  allocation : Nat
  alignedBase : Nat
  offset : Nat
  shape : List Nat
  strides : List Nat
  element : String
  space : Nat
  deriving DecidableEq, Repr

def Conforms (type : ViewType) (descriptor : Descriptor) : Prop :=
  descriptor.shape = type.shape ∧ descriptor.offset = type.offset ∧
  descriptor.strides = type.strides ∧ descriptor.element = type.element ∧ descriptor.space = type.space

instance (type : ViewType) (descriptor : Descriptor) : Decidable (Conforms type descriptor) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

/-- No observable state or heap is touched. Native descriptor/operator fidelity
is separate; a logging/trapping cast is not this admitted operation. -/
def Cast (source target : ViewType) (before after : Descriptor) : Prop :=
  source = target ∧ Conforms source before ∧ after = before

def Dimension (type : ViewType) (descriptor : Descriptor) (axis result : Nat) : Prop :=
  Conforms type descriptor ∧ descriptor.shape[axis]? = some result

def ValidDimension (bits : Nat) (type : ViewType) (axis result : Nat) : Prop :=
  (bits = 32 ∨ bits = 64) ∧ type.shape[axis]? = some result ∧ result < 2 ^ (bits - 1)

instance (bits : Nat) (type : ViewType) (axis result : Nat) : Decidable (ValidDimension bits type axis result) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- Same-type casts preserve the ENTIRE descriptor, including allocation/base,
offset, layout and element/address space. Type equality alone is insufficient
without a valid input descriptor. -/
theorem identical_cast (type : ViewType) (descriptor : Descriptor) (valid : Conforms type descriptor) :
    Cast type type descriptor descriptor := ⟨rfl, valid, rfl⟩

theorem cast_descriptor {source target : ViewType} {before after : Descriptor} (run : Cast source target before after) :
    after = before := run.2.2

/-- Source metadata read and emitted index literal have the SAME fitting value.
This also covers zero dimensions independently of old positive-buffer lemmas. -/
theorem static_dimension {bits : Nat} {type : ViewType} {descriptor : Descriptor} {axis result : Nat}
    (valid : Conforms type descriptor) (checked : ValidDimension bits type axis result) :
    Dimension type descriptor axis result ∧ Merlin.GraphLoopsSegment.FitsSigned bits result := by
  refine ⟨⟨valid, ?_⟩, Merlin.GraphLoopsSegment.fits_signed_of_lt checked.2.2⟩
  rw [valid.1]
  exact checked.2.1

/-- The linear address uses actual descriptor offset/strides. This definition
makes preservation of allocation identity and arbitrary in-bounds addresses
visible rather than asserting equality between unrelated storage slots. -/
def linearOffset : List Nat → List Nat → Nat
  | stride :: strides, index :: indices => stride * index + linearOffset strides indices
  | _, _ => 0

def address (descriptor : Descriptor) (indices : List Nat) : Nat × Nat :=
  (descriptor.allocation, descriptor.alignedBase + descriptor.offset + linearOffset descriptor.strides indices)

theorem cast_address {source target : ViewType} {before after : Descriptor} (run : Cast source target before after)
    (indices : List Nat) : address after indices = address before indices := by rw [cast_descriptor run]

/-- Necessary-premise countermodel: a descriptor advertising a3-row runtime
shape cannot justify folding a2-row static type. -/
def inconsistent : Descriptor := ⟨7, 100, 0, [3], [1], "i32", 0⟩
def declared : ViewType := ⟨[2], 0, [1], "i32", 0⟩

theorem inconsistent_dimension : inconsistent.shape[0]? ≠ declared.shape[0]? := by decide

theorem observable_cast_countermodel : (0 + 1 : Nat) ≠ 0 := by decide

end Merlin.GraphLoopsShape
