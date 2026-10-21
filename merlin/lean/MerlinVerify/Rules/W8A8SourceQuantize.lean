/- Captured per-tensor quantization review candidate, following PyTorch v2.10.0
torch/ao/quantization/fx/_decomposed.py:42-128. Scalar tensor parameters are extracted;
the reciprocal is Python binary64, then converted to the f32 tensor's scalar type.
Zero-point addition and lower-then-upper clamping are retained even for literal zero.
This is not installed in frozen source semantics or approved as an independent spec. -/
import MerlinVerify.Rules.W8A8SourceByte
import MerlinVerify.Check.DefinedDequant

namespace MerlinVerify.Rules.W8A8SourceQuantize

open MerlinVerify MerlinVerify.Sem.Ops

def inverse (M : FloatModel) (scale : FBits .f32) : FBits .f32 :=
  M.convert (Spec.Dequant.defaultKey "arith.truncf") .f64 .f32
    (M.op (Spec.Dequant.defaultKey "arith.divf") .f64
      [0x3ff0000000000000#64,
       M.convert (Spec.Dequant.defaultKey "arith.extf") .f32 .f64 scale])

def shifted (width : Nat) (M : FloatModel) (x scale : FBits .f32)
    (zero : BitVec width) : FBits .f32 :=
  M.op (Spec.Dequant.defaultKey "arith.addf") .f32
    [M.op (Spec.Dequant.defaultKey "math.roundeven") .f32
      [M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, inverse M scale]],
     M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 zero]

def clamped (width : Nat) (M : FloatModel) (x scale : FBits .f32)
    (zero : BitVec width) : FBits .f32 :=
  M.op (Spec.Dequant.defaultKey "arith.minimumf") .f32
    [M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32
      [shifted width M x scale zero, 0xc3000000#32], 0x42fe0000#32]

def conversion (width : Nat) (M : FloatModel) (x scale : FBits .f32)
    (zero : BitVec width) : Option (BitVec 8) :=
  M.toInt (Spec.Dequant.defaultKey "arith.fptosi") .f32 8 (clamped width M x scale zero)

def scalar (width : Nat) (M : FloatModel) (x : Option (FBits .f32))
    (scale : FBits .f32) (zero : BitVec width) : Val :=
  match x with
  | none => .poison
  | some x => match conversion width M x scale zero with
    | none => .poison
    | some byte => .some (.bits 8 byte)

/-- Candidate eligibility is finite, strictly positive calibration scales. This runtime check
does not declare arbitrary dynamic scales valid or provide the missing policy review. -/
def scaleAdmissible (scale : FBits .f32) : Bool :=
  !W8A8ByteFloat.negative scale && decide (0 < W8A8ByteFloat.magnitudeWord scale) &&
    decide (W8A8ByteFloat.exponent scale < 255)

def result (width : Nat) (M : FloatModel) (shape : List Nat) (input : Tensor)
    (scale : FBits .f32) (zero : BitVec width) : Tensor :=
  Tensor.canon shape (.int 8) fun point => scalar width M (Dequant.float32 (input.get point)) scale zero

def validInputs (width : Nat) (shape : List Nat) (input scale zero : Tensor) : Bool :=
  (width == 32 || width == 64) &&
  decide (input.shape = shape ∧ scale.shape = [] ∧ zero.shape = []) &&
  Dequant.tensorTyped input (.float .f32) && Dequant.tensorTyped scale (.float .f32) &&
    Dequant.tensorTyped zero (.int width)

def run (width : Nat) (M : FloatModel) (shape : List Nat) : List RVal → Outcome (List RVal)
  | [.tensor input, .tensor scale, .tensor zero] =>
    if validInputs width shape input scale zero then
      match Dequant.float32 (scale.get []), Dequant.bits width (zero.get []) with
      | some s, some z => if scaleAdmissible s then .ok [.tensor (result width M shape input s z)]
          else .stuck "source quantize candidate requires a finite positive scale"
      | _, _ => .stuck "source quantize candidate requires initialized scalar parameters"
    else .stuck "source quantize candidate requires f32 input/scale and scalar i32/i64 zero point"
  | _ => .stuck "source quantize candidate expects three tensors"

theorem scalar_defined_iff (width : Nat) (M : FloatModel) (x scale : FBits .f32) (zero : BitVec width) :
    scalar width M (some x) scale zero ≠ .poison ↔ ∃ byte, conversion width M x scale zero = some byte := by
  cases converted : conversion width M x scale zero <;> simp [scalar, converted]

theorem shifted_extended (width : Nat) (base : FloatModel) (x scale : FBits .f32) (zero : BitVec width) :
    shifted width (W8A8ByteModel.extend base) x scale zero = shifted width base x scale zero := by
  simp [shifted, inverse, W8A8ByteModel.extend, Spec.Dequant.defaultKey]

theorem clamped_extended (width : Nat) (base : FloatModel) (x scale : FBits .f32) (zero : BitVec width) :
    clamped width (W8A8ByteModel.extend base) x scale zero =
      W8A8SourceByte.clamp (shifted width base x scale zero) := by
  have formula (y lo hi : FBits .f32) :
      (W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "arith.minimumf") .f32
        [(W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "arith.maximumf") .f32 [y, lo], hi] =
        W8A8ByteFloat.minimum (W8A8ByteFloat.maximum y lo) hi := by
    rw [W8A8ByteModel.minimum_eq, W8A8ByteModel.maximum_eq]
  exact (formula (shifted width (W8A8ByteModel.extend base) x scale zero) _ _).trans
    (congrArg W8A8SourceByte.clamp (shifted_extended width base x scale zero))

theorem conversion_domain (width : Nat) (base : FloatModel) (x scale : FBits .f32) (zero : BitVec width) :
    (conversion width (W8A8ByteModel.extend base) x scale zero).isSome = true ↔
      W8A8ByteFloat.isNaN (shifted width base x scale zero) = false := by
  unfold conversion
  rw [W8A8ByteModel.conversion_eq, clamped_extended]
  exact W8A8SourceByte.conversion_domain _

theorem scalar_defined (width : Nat) (base : FloatModel) (x scale : FBits .f32) (zero : BitVec width)
    (nonNaN : W8A8ByteFloat.isNaN (shifted width base x scale zero) = false) :
    scalar width (W8A8ByteModel.extend base) (some x) scale zero ≠ .poison := by
  apply (scalar_defined_iff ..).2
  cases converted : conversion width (W8A8ByteModel.extend base) x scale zero with
  | none => have h := (conversion_domain width base x scale zero).2 nonNaN; simp [converted] at h
  | some byte => exact ⟨byte, rfl⟩

theorem result_typed (ib width : Nat) (M : FloatModel) (shape : List Nat) (input : Tensor)
    (scale : FBits .f32) (zero : BitVec width) :
    Check.RVal.HasTy ib (.tensor (result width M shape input scale zero)) (.tensor shape (.int 8)) := by
  refine ⟨rfl, rfl, fun point hp => ?_⟩
  simp only [result, Tensor.canon, hp, ↓reduceIte]
  unfold scalar
  split
  · simp [Check.Val.HasTy]
  · split <;> simp [Check.Val.HasTy, Scalar.conforms]

/-- Exact semantic domain, stated over the actual input tensor's decoded observations. It is
not a syntactic initialization check, and is not asserted for all typed program inputs. -/
def Domain (width : Nat) (base : FloatModel) (shape : List Nat) (input : Tensor)
    (scale : FBits .f32) (zero : BitVec width) : Prop :=
  ∀ point, Idx.InBounds shape point → ∀ x, Dequant.float32 (input.get point) = some x →
    W8A8ByteFloat.isNaN (shifted width base x scale zero) = false

theorem result_defined {width : Nat} {base : FloatModel} {shape : List Nat} {input : Tensor}
    {scale : FBits .f32} {zero : BitVec width}
    (shapeEq : input.shape = shape) (typed : Dequant.tensorTyped input (.float .f32) = true)
    (initialized : (RVal.tensor input).Defined) (domain : Domain width base shape input scale zero) :
    (RVal.tensor (result width (W8A8ByteModel.extend base) shape input scale zero)).Defined := by
  intro point hp
  change Idx.InBounds shape point at hp
  have bounds : Idx.InBounds input.shape point := shapeEq ▸ hp
  obtain ⟨x, decoded⟩ := Check.Definedness.dequant_float_defined
    (DequantSource.tensorTyped_get typed bounds) (initialized point bounds)
  simp only [result, Tensor.canon, hp, ↓reduceIte, decoded]
  exact scalar_defined width base x scale zero (domain point hp x decoded)

/-- Successful execution of the candidate's actual runtime entry, with typed, initialized
output. Parameter extraction, runtime shape/type checks and the numerical domain remain
separate obligations; none is replaced by a trusted certificate flag. -/
theorem run_defined {ib width : Nat} {base : FloatModel} {shape : List Nat}
    {input scale zero : Tensor} {s : FBits .f32} {z : BitVec width}
    (valid : validInputs width shape input scale zero = true)
    (scaleDecoded : Dequant.float32 (scale.get []) = some s)
    (zeroDecoded : Dequant.bits width (zero.get []) = some z)
    (admissible : scaleAdmissible s = true) (initialized : (RVal.tensor input).Defined)
    (domain : Domain width base shape input s z) :
    ∃ output, run width (W8A8ByteModel.extend base) shape [.tensor input, .tensor scale, .tensor zero] =
      .ok [.tensor output] ∧
      Check.RVal.HasTy ib (.tensor output) (.tensor shape (.int 8)) ∧ (RVal.tensor output).Defined := by
  have data := valid
  simp only [validInputs, Bool.and_eq_true, decide_eq_true_eq] at data
  obtain ⟨⟨⟨⟨_, ⟨shapeEq, _, _⟩⟩, typed⟩, _⟩, _⟩ := data
  refine ⟨result width (W8A8ByteModel.extend base) shape input s z, ?_, result_typed ..,
    result_defined shapeEq typed initialized domain⟩
  simp [run, valid, scaleDecoded, zeroDecoded, admissible]

#print axioms scalar_defined_iff
#print axioms conversion_domain
#print axioms result_typed
#print axioms result_defined
#print axioms run_defined

end MerlinVerify.Rules.W8A8SourceQuantize
