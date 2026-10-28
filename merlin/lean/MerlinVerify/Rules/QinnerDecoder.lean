/- Concrete Qinner storage interpretation. Soundness establishes exact in-bounds bits, typing
and initialization for accepted little-endian row-major payloads. The archive's authenticity
and actual runtime delivery of these bytes remain external obligations. No float law is used. -/
import MerlinVerify.Check.QinnerRuntime
import MerlinVerify.Check.RulesTensor
namespace MerlinVerify.Rules.QinnerDecoder
open MerlinVerify MerlinVerify.Check

inductive Element where
  | i1 | i8 | i16 | i32 | i64 | f16 | f32 | f64
  deriving DecidableEq, Repr

def Element.ty : Element → Ty
  | .i1 => .int 1 | .i8 => .int 8 | .i16 => .int 16 | .i32 => .int 32 | .i64 => .int 64
  | .f16 => .float .f16 | .f32 => .float .f32 | .f64 => .float .f64

def Element.storage : Element → Nat
  | .i1 | .i8 => 1 | .i16 | .f16 => 2 | .i32 | .f32 => 4 | .i64 | .f64 => 8

def Element.bits : Element → Nat
  | .i1 => 1 | .i8 => 8 | .i16 | .f16 => 16 | .i32 | .f32 => 32 | .i64 | .f64 => 64

def Element.scalar : Element → Nat → Scalar
  | .i1,n => .bits 1 (BitVec.ofNat 1 n)
  | .i8,n => .bits 8 (BitVec.ofNat 8 n)
  | .i16,n => .bits 16 (BitVec.ofNat 16 n)
  | .i32,n => .bits 32 (BitVec.ofNat 32 n)
  | .i64,n => .bits 64 (BitVec.ofNat 64 n)
  | .f16,n => .flt .f16 (BitVec.ofNat 16 n)
  | .f32,n => .flt .f32 (BitVec.ofNat 32 n)
  | .f64,n => .flt .f64 (BitVec.ofNat 64 n)

def element? : Ty → Option Element
  | .int 1 => some .i1 | .int 8 => some .i8 | .int 16 => some .i16
  | .int 32 => some .i32 | .int 64 => some .i64
  | .float .f16 => some .f16 | .float .f32 => some .f32 | .float .f64 => some .f64
  | _ => none

theorem element_type {ty : Ty} {e : Element} (h : element? ty = some e) : e.ty = ty := by
  fun_cases element? ty <;> simp_all [element?] <;> cases h <;> rfl

theorem scalar_typed (e : Element) (ib n : Nat) : (e.scalar n).conforms ib e.ty = true := by
  cases e <;> rfl

structure Descriptor where
  shape : List Nat
  element : Element

def descriptor? : Ty → Option Descriptor
  | .tensor dims elt none => do
    let shape ← dims.mapM id
    let element ← element? elt
    return ⟨shape,element⟩
  | _ => none

theorem descriptor_typed {ty : Ty} {d : Descriptor} (h : descriptor? ty = some d) (ib : Nat) :
    vtyOf ib ty = some (.tensor d.shape d.element.ty) := by
  cases ty with
  | tensor dims elt enc =>
    cases enc with
    | some attr => simp [descriptor?] at h
    | none =>
      cases hd : dims.mapM id with
      | none => simp [descriptor?,hd] at h
      | some shape =>
        cases he : element? elt with
        | none => simp [descriptor?,hd,he] at h
        | some element =>
          simp only [descriptor?,hd,he,Option.bind_some,Option.pure_def,Option.some.injEq] at h
          cases h
          rw [← element_type he]
          cases element <;> simp [vtyOf,Element.ty,Ty.bitWidth?,hd]
  | _ => simp [descriptor?] at h

/-- Bool is stored as one full host byte, with canonical values exactly 0 and 1. -/
def canonical (e : Element) (bytes : List UInt8) : Bool :=
  if e == .i1 then bytes.all (fun b => b.toNat < 2) else true

/-- The byte slice for row-major element k; no host-endian cast or float conversion. -/
def chunk (e : Element) (bytes : List UInt8) (k : Nat) : List UInt8 :=
  (bytes.drop (k * e.storage)).take e.storage

def word (e : Element) (bytes : List UInt8) (k : Nat) : Nat := leNat (chunk e bytes k)

def tensor (d : Descriptor) (bytes : List UInt8) : Tensor :=
  Tensor.canon d.shape d.element.ty fun i =>
    .some (d.element.scalar (word d.element bytes (linearIndex d.shape i)))

/-- Representation tags are checked rather than inferred from raw bytes. Arbitrary byte vectors
cannot reveal whether their producer intended a different endianness. -/
def payloadCheck (order layout : String) (ty : Ty) (bytes : List UInt8) : Bool :=
  order == "little" && layout == "row_major" &&
    match descriptor? ty with
    | none => false
    | some d => bytes.length == shapeSize d.shape * d.element.storage && canonical d.element bytes

def decode? (order layout : String) (ty : Ty) (bytes : List UInt8) : Option RVal :=
  if payloadCheck order layout ty bytes then
    (descriptor? ty).map fun d => .tensor (tensor d bytes)
  else none

theorem check_descriptor {order layout : String} {ty : Ty} {bytes : List UInt8}
    (h : payloadCheck order layout ty bytes = true) :
    ∃ d, descriptor? ty = some d ∧ order = "little" ∧ layout = "row_major" ∧
      bytes.length = shapeSize d.shape * d.element.storage ∧ canonical d.element bytes = true := by
  unfold payloadCheck at h
  cases hd : descriptor? ty with
  | none => simp [hd] at h
  | some d =>
    simp only [hd,Bool.and_eq_true,beq_iff_eq] at h
    exact ⟨d,rfl,h.1.1,h.1.2,h.2.1,h.2.2⟩

theorem decode_success {order layout : String} {ty : Ty} {bytes : List UInt8}
    (h : payloadCheck order layout ty bytes = true) :
    ∃ d, descriptor? ty = some d ∧ decode? order layout ty bytes = some (.tensor (tensor d bytes)) := by
  obtain ⟨d,hd,_,_,_,_⟩ := check_descriptor h
  exact ⟨d,hd,by simp [decode?,h,hd]⟩

theorem tensor_typed (d : Descriptor) (bytes : List UInt8) (ib : Nat) :
    RVal.HasTy ib (.tensor (tensor d bytes)) (.tensor d.shape d.element.ty) := by
  refine ⟨rfl,rfl,?_⟩
  intro i hi
  simp only [tensor,Tensor.canon,hi,↓reduceIte,Val.HasTy]
  exact scalar_typed d.element ib _

theorem tensor_defined (d : Descriptor) (bytes : List UInt8) :
    (RVal.tensor (tensor d bytes)).Defined := by
  intro i hi
  change Idx.InBounds d.shape i at hi
  simp [tensor,Tensor.canon,hi]

theorem decode_typed_initialized {order layout : String} {ty : Ty} {bytes : List UInt8}
    (h : payloadCheck order layout ty bytes = true) (ib : Nat) :
    ∃ value vt, decode? order layout ty bytes = some value ∧ vtyOf ib ty = some vt ∧
      RVal.HasTy ib value vt ∧ value.Defined := by
  obtain ⟨d,hd,decoded⟩ := decode_success h
  exact ⟨.tensor (tensor d bytes),.tensor d.shape d.element.ty,decoded,descriptor_typed hd ib,
    tensor_typed d bytes ib,tensor_defined d bytes⟩

/-- Typed/initialized alone is insufficient for fidelity. This bound proves no high bits are
silently truncated, including i1's canonical host bytes. -/
theorem leNat_bound (bytes : List UInt8) : leNat bytes < 256 ^ bytes.length := by
  induction bytes with
  | nil => simp [leNat]
  | cons b bs ih =>
    have hb : b.toNat < 256 := b.toNat_lt
    simp only [leNat,List.length_cons,Nat.pow_succ]
    omega

theorem word_storage_bound (e : Element) (bytes : List UInt8) (k : Nat) :
    word e bytes k < 256 ^ e.storage := by
  exact Nat.lt_of_lt_of_le (leNat_bound (chunk e bytes k))
    (Nat.pow_le_pow_right (by decide) (List.length_take_le e.storage (bytes.drop (k * e.storage))))

theorem word_bound (e : Element) (bytes : List UInt8) (k : Nat)
    (valid : canonical e bytes = true) : word e bytes k < 2 ^ e.bits := by
  cases e with
  | i1 =>
    have all : ∀ b ∈ bytes, b.toNat < 2 := by simpa [canonical] using valid
    cases h : bytes.drop k with
    | nil => simp [word,chunk,Element.storage,h,leNat,Element.bits]
    | cons b bs =>
      have mem : b ∈ bytes := List.mem_of_mem_drop (by rw [h]; exact List.mem_cons_self ..)
      have hb := all b mem
      simpa [word,chunk,Element.storage,h,leNat,Element.bits] using hb
  | i8 => exact word_storage_bound .i8 bytes k
  | i16 => exact word_storage_bound .i16 bytes k
  | i32 => exact word_storage_bound .i32 bytes k
  | i64 => exact word_storage_bound .i64 bytes k
  | f16 => exact word_storage_bound .f16 bytes k
  | f32 => exact word_storage_bound .f32 bytes k
  | f64 => exact word_storage_bound .f64 bytes k

def scalarBits : Scalar → Nat
  | .bits _ v => v.toNat | .flt _ v => v.toNat

theorem scalar_exact (e : Element) (n : Nat) (bound : n < 2 ^ e.bits) :
    scalarBits (e.scalar n) = n := by
  cases e <;> change (BitVec.ofNat _ n).toNat = n <;>
    exact (BitVec.toNat_ofNat n _).trans (Nat.mod_eq_of_lt bound)

theorem exact_bits (d : Descriptor) (bytes : List UInt8) (valid : canonical d.element bytes = true)
    (i : Idx) (hi : Idx.InBounds d.shape i) :
    (tensor d bytes).get i = .some (d.element.scalar (word d.element bytes (linearIndex d.shape i))) ∧
    scalarBits (d.element.scalar (word d.element bytes (linearIndex d.shape i))) =
      leNat ((bytes.drop (linearIndex d.shape i * d.element.storage)).take d.element.storage) := by
  exact ⟨by simp [tensor,Tensor.canon,hi], scalar_exact _ _ (word_bound _ _ _ valid)⟩

theorem complete_chunk (d : Descriptor) (bytes : List UInt8)
    (size : bytes.length = shapeSize d.shape * d.element.storage)
    (i : Idx) (hi : Idx.InBounds d.shape i) :
    (chunk d.element bytes (linearIndex d.shape i)).length = d.element.storage := by
  have index := linearIndex_lt d.shape i hi
  have limit := Nat.mul_le_mul_right d.element.storage (Nat.succ_le_of_lt index)
  rw [Nat.succ_mul,← size] at limit
  apply List.length_take_of_le
  rw [List.length_drop]
  omega

theorem leNat_div (b : UInt8) (bs : List UInt8) : leNat (b::bs) / 256 = leNat bs := by
  simp only [leNat,Nat.add_mul_div_left _ _ (by decide : 0 < 256),
    Nat.div_eq_of_lt b.toNat_lt,Nat.zero_add]

/-- Each original byte is recovered by selecting its eight bits from the decoded word. -/
theorem leNat_byte : ∀ (bytes : List UInt8) (j : Nat) (h : j < bytes.length),
    leNat bytes / 256^j % 256 = (bytes[j]).toNat
  | [],_,h => by simp at h
  | b::bs,0,_ => by
    simp only [Nat.pow_zero,Nat.div_one,leNat,Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt b.toNat_lt,List.getElem_cons_zero]
  | b::bs,j+1,h => by
    rw [Nat.pow_succ',← Nat.div_div_eq_div_mul,leNat_div]
    exact leNat_byte bs j (by simpa using h)

theorem exact_byte (d : Descriptor) (bytes : List UInt8)
    (size : bytes.length = shapeSize d.shape * d.element.storage)
    (valid : canonical d.element bytes = true) (i : Idx) (hi : Idx.InBounds d.shape i)
    (j : Nat) (hj : j < d.element.storage) :
    scalarBits (d.element.scalar (word d.element bytes (linearIndex d.shape i))) / 256^j % 256 =
      ((chunk d.element bytes (linearIndex d.shape i))[j]'(by rw [complete_chunk d bytes size i hi]; exact hj)).toNat := by
  rw [(exact_bits d bytes valid i hi).2]
  have bound : j < (chunk d.element bytes (linearIndex d.shape i)).length := by
    rw [complete_chunk d bytes size i hi]
    exact hj
  exact leNat_byte _ j bound

/-- Complete concrete cell contract: successful decode, correct scalar type, exact word and
recovery of every original stored byte at the row-major offset. -/
theorem decoded_cells {order layout : String} {ty : Ty} {bytes : List UInt8}
    (h : payloadCheck order layout ty bytes = true) (ib : Nat) :
    ∃ d, descriptor? ty = some d ∧ decode? order layout ty bytes = some (.tensor (tensor d bytes)) ∧
      ∀ i, Idx.InBounds d.shape i → ∃ scalar,
        (tensor d bytes).get i = .some scalar ∧ scalar.conforms ib d.element.ty = true ∧
        scalarBits scalar = leNat (chunk d.element bytes (linearIndex d.shape i)) ∧
        ∀ j, j < d.element.storage → scalarBits scalar / 256^j % 256 =
          ((chunk d.element bytes (linearIndex d.shape i)).getD j 0).toNat := by
  obtain ⟨d,found,orderEq,layoutEq,size,valid⟩ := check_descriptor h
  refine ⟨d,found,by simp [decode?,h,found],?_⟩
  intro i hi
  refine ⟨d.element.scalar (word d.element bytes (linearIndex d.shape i)),
    (exact_bits d bytes valid i hi).1,scalar_typed _ ib _,(exact_bits d bytes valid i hi).2,?_⟩
  intro j hj
  have bound : j < (chunk d.element bytes (linearIndex d.shape i)).length := by
    rw [complete_chunk d bytes size i hi]; exact hj
  simpa [List.getD, List.getElem?_eq_getElem bound] using exact_byte d bytes size valid i hi j hj

#print axioms decoded_cells
#print axioms exact_byte
#print axioms decode_typed_initialized
#print axioms exact_bits
#print axioms complete_chunk
end MerlinVerify.Rules.QinnerDecoder
