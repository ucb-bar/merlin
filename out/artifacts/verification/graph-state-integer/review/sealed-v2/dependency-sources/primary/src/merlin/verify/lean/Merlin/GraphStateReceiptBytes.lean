/- Compact exact byte buffers for receipt binding. Size preserves leading zero
bytes. Kernel equality of packed values implies equality of decoded full bytes;
the encoder/decoder round trip is proved for every legal byte list. The host
I/O/encoding implementation is still identified trusted code, not proved runtime. -/
import Init.Data.List.Lemmas
import Init.Data.Nat.Div.Lemmas

namespace Merlin.GraphState.Receipt.Bytes
set_option autoImplicit false

structure Encoded where
  size : Nat
  value : Nat
  deriving DecidableEq, BEq, ReflBEq, LawfulBEq, Repr

def number : List Nat → Nat
  | [] => 0
  | byte::rest => byte + 256 * number rest

def encode (bytes : List Nat) : Encoded := ⟨bytes.length,number bytes⟩

def decodeNumber : Nat → Nat → List Nat
  | 0,_ => []
  | size+1,value => (value % 256)::decodeNumber size (value / 256)

def decode (encoded : Encoded) : List Nat := decodeNumber encoded.size encoded.value

theorem round_trip (bytes : List Nat) (legal : ∀ byte ∈ bytes, byte < 256) : decode (encode bytes) = bytes := by
  induction bytes with
  | nil => rfl
  | cons byte rest ih =>
    have inside : byte < 256 := legal byte (by simp)
    have tail := ih (fun value member => legal value (by simp [member]))
    simp only [decode,encode,List.length_cons,number,decodeNumber]
    rw [Nat.add_mul_mod_self_left,Nat.mod_eq_of_lt inside]
    rw [Nat.add_mul_div_left _ _ (by decide : 0 < 256),Nat.div_eq_of_lt inside,Nat.zero_add]
    exact congrArg (byte::·) tail

theorem encode_injective (first second : List Nat)
    (firstLegal : ∀ byte ∈ first, byte < 256) (secondLegal : ∀ byte ∈ second, byte < 256)
    (same : encode first = encode second) : first = second := by
  calc first = decode (encode first) := (round_trip first firstLegal).symm
       _ = decode (encode second) := congrArg decode same
       _ = second := round_trip second secondLegal

theorem decoded_equal {first second : Encoded} (same : first = second) : decode first = decode second :=
  congrArg decode same

#print axioms round_trip
#print axioms encode_injective
end Merlin.GraphState.Receipt.Bytes
