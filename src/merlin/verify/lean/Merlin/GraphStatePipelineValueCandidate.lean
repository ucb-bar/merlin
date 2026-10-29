/- SOURCE CANDIDATE ONLY; not compiled or eligible.
   Additive generic value carrier for opaque tensors/floats and exact indices.
   This supplies no numerical semantics or runtime/parser correspondence. -/
import Merlin.GraphStatePipelineIndex

namespace Merlin.GraphState.Pipeline.ValueCarrierCandidate
set_option autoImplicit false
open Scalar

/-- Indices inhabit a distinguished retract of the value carrier. Nonindex
values remain values of Value; they are never represented by their Int tags.
Static index typing and actual SSA control joins remain separate obligations. -/
structure Codec (Value : Type) where
  encode : Int → Value
  decode : Value → Int
  roundTrip : ∀ value, decode (encode value) = value

def IsIndex {Value : Type} (codec : Codec Value) (value : Value) : Prop :=
  ∃ index, value = codec.encode index

def RepresentsIndex {Value : Type} (codec : Codec Value) (bits : Nat) (value : Value) : Prop :=
  ∃ index, signedRepresentative bits index ∧ value = codec.encode index

theorem fitted_is_index {Value : Type} (codec : Codec Value) {bits : Nat} {index : Int}
    (fitted : signedRepresentative bits index) : RepresentsIndex codec bits (codec.encode index) :=
  ⟨index,fitted,rfl⟩

theorem encoded_is_index {Value : Type} (codec : Codec Value) (index : Int) :
    IsIndex codec (codec.encode index) := ⟨index,rfl⟩

theorem encode_injective {Value : Type} (codec : Codec Value) {left right : Int}
    (same : codec.encode left = codec.encode right) : left = right := by
  have decoded := congrArg codec.decode same
  simpa only [codec.roundTrip] using decoded

theorem decode_encoded_list {Value : Type} (codec : Codec Value) (indices : List Int) :
    (indices.map codec.encode).map codec.decode = indices := by
  induction indices with
  | nil => rfl
  | cons index later ih => simpa only [List.map_cons,codec.roundTrip] using congrArg (List.cons index) ih

structure External (Value World Error : Type) (registry : List Signature) where
  run : Nat → List Value → World → Except (Error × World) (List Value × World)
  arity : ∀ identity arguments world values final,
    run identity arguments world = .ok (values,final) →
      values.length = (signature registry identity).outputs.length

def effect {Value World Error : Type} {registry : List Signature}
    (codec : Codec Value) (controls : Controls)
    (external : External Value World Error registry)
    (identity : Nat) (arguments : List Value) (world : World) :
    Except (Error × World) (List Value × World) :=
  match controlAt controls identity with
  | some op => .ok ([codec.encode (op.eval (arguments.map codec.decode))],world)
  | none => external.run identity arguments world

/-- Partial opaque operations keep their complete Value results and World /
Error. Only the separately checked index-control operations use Codec. -/
def model {Value World Error : Type} {registry : List Signature} {controls : Controls}
    (codec : Codec Value) (valid : ControlValid registry controls)
    (default : Value) (primitive : Prim → List Value → Value)
    (external : External Value World Error registry) : Model Value World Error registry where
  default := default
  primitive := primitive
  index := codec.encode
  iterations := fun lower upper stride =>
    indexIterations (codec.decode lower) (codec.decode upper) (codec.decode stride)
  effect := effect codec controls external
  arity := by
    intro identity arguments world values final success
    cases found : controlAt controls identity with
    | none => exact external.arity _ _ _ _ _ (by simpa only [effect,found] using success)
    | some op =>
      have same : ([codec.encode (op.eval (arguments.map codec.decode))],world) = (values,final) := by
        simpa only [effect,found,Except.ok.injEq] using success
      cases same
      rw [(valid.2 identity op found).2.2]
      rfl

theorem control_effect {Value World Error : Type} {registry : List Signature} {controls : Controls}
    (codec : Codec Value) (valid : ControlValid registry controls)
    (default : Value) (primitive : Prim → List Value → Value)
    (external : External Value World Error registry) {identity : Nat} {op : IndexOp}
    (found : controlAt controls identity = some op) (arguments : List Value) (world : World) :
    opaqueStep (model codec valid default primitive external) identity arguments world =
      .ok ([codec.encode (op.eval (arguments.map codec.decode))],world) := by
  simp only [opaqueStep,model,effect,found]

/-- For the actual typed index operands, the lifted operation has exactly the
old checked Int meaning inside the distinguished index summand. -/
theorem control_on_indices {Value World Error : Type} {registry : List Signature} {controls : Controls}
    (codec : Codec Value) (valid : ControlValid registry controls)
    (default : Value) (primitive : Prim → List Value → Value)
    (external : External Value World Error registry) {identity : Nat} {op : IndexOp}
    (found : controlAt controls identity = some op) (arguments : List Int) (world : World) :
    opaqueStep (model codec valid default primitive external) identity (arguments.map codec.encode) world =
      .ok ([codec.encode (op.eval arguments)],world) := by
  rw [control_effect codec valid default primitive external found,decode_encoded_list]

theorem iterations_encoded {Value World Error : Type} {registry : List Signature} {controls : Controls}
    (codec : Codec Value) (valid : ControlValid registry controls)
    (default : Value) (primitive : Prim → List Value → Value)
    (external : External Value World Error registry) (lower upper stride : Int) :
    (model codec valid default primitive external).iterations
      (codec.encode lower) (codec.encode upper) (codec.encode stride) =
      indexIterations lower upper stride := by
  simp only [model,codec.roundTrip]

def sumCodec (Payload : Type) : Codec (Int ⊕ Payload) where
  encode := Sum.inl
  decode := fun value => match value with | .inl index => index | .inr _ => 0
  roundTrip := by intro value; rfl

end Merlin.GraphState.Pipeline.ValueCarrierCandidate
