/- Executable W8A8 tensor-stage checks: actual body, literal captures, maps, iterator kinds,
shapes and access bounds. This does not yet match a complete source contraction to its output. -/
import MerlinVerify.Check.PolicyW8A8Scalar
import MerlinVerify.Check.PolicyW8A8Geometry
import MerlinVerify.Rules.W8A8FloatTensor

namespace MerlinVerify.Check.PolicyW8A8Pointwise

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Rules W8A8TensorOperands W8A8Scalar
open PolicyW8A8Geometry

inductive UnaryKind where
  | scale | reciprocal

def unaryValue (kind : UnaryKind) : FloatModel → Option (FBits .f32) → Val :=
  match kind with
  | .scale => W8A8Scalar.scale
  | .reciprocal => W8A8Scalar.reciprocal

def UnaryCorrect (known : List ConstantFacts.Fact) (shape : List Nat) (kind : UnaryKind)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Prop :=
  ∀ env, BindingFacts.Valid ConstantFacts.model known env → ∀ ctx call get initial,
    genericSem ctx maps iters [(W8A8FloatTensor.floatInput shape get).value]
      [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) (fun p => unaryValue kind ctx.M (get p)))]

def QuantizeCorrect (known : List ConstantFacts.Fact) (shape dims : List Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Prop :=
  ∀ env, BindingFacts.Valid ConstantFacts.model known env → ∀ ctx call get reciprocal initial,
    genericSem ctx maps iters [(W8A8FloatTensor.floatInput shape get).value,
      (W8A8FloatTensor.floatInput (parameterShape shape dims) reciprocal).value]
      [.tensor ⟨shape, .int 8, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.int 8) (fun point => W8A8Scalar.quantize ctx.M (get point)
        (reciprocal (coordinate (parameterShape shape dims) (parameterMap shape.length dims) point))
        PolicyW8A8Scalar.upper PolicyW8A8Scalar.lower))]

def RequantizeCorrect (count : Nat) (shape leftDims rightDims : List Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Prop :=
  ∀ ctx call env get left right initial,
    genericSem ctx maps iters
      ((W8A8FloatTensor.intInput shape get ::
        ([W8A8FloatTensor.floatInput (parameterShape shape leftDims) left,
          W8A8FloatTensor.floatInput (parameterShape shape rightDims) right].take count)).map Input.value)
      [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float .f32) (fun point => W8A8Scalar.requantize ctx.M
        (get point) ([left (coordinate (parameterShape shape leftDims)
          (parameterMap shape.length leftDims) point), right (coordinate (parameterShape shape rightDims)
          (parameterMap shape.length rightDims) point)].take count)))]

def checkUnary (known : List ConstantFacts.Fact) (shape : List Nat) (kind : UnaryKind) (denom : Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Bool :=
  checkParallel [.tensor shape (.float .f32)] shape (.float .f32)
    [identityMap shape.length] maps iters &&
  match kind with
  | .scale => PolicyW8A8Scalar.checkScaleKnown known denom actual
  | .reciprocal => PolicyW8A8Scalar.checkReciprocal actual

theorem checkUnary_sound {known : List ConstantFacts.Fact} {shape : List Nat} {kind : UnaryKind}
    {denom : Nat} {maps : List AffineMap} {iters : List IterKind} {actual : Region}
    (h : checkUnary known shape kind denom maps iters actual = true) :
    UnaryCorrect known shape kind maps iters actual := by
  have hh := h
  simp only [checkUnary, Bool.and_eq_true] at hh
  obtain ⟨hm, _, geometry⟩ := checkParallel_sound hh.1
  intro env valid ctx call get initial
  apply W8A8FloatTensor.unary_sound ctx call env shape maps iters actual hm geometry (unaryValue kind)
    _ get initial
  intro c x old
  cases kind with
  | scale => exact PolicyW8A8Scalar.checkScaleKnown_sound hh.2 valid c call x old
  | reciprocal => exact PolicyW8A8Scalar.checkReciprocal_sound hh.2 c call env x old

def checkQuantize (known : List ConstantFacts.Fact) (shape dims : List Nat) (hi lo : Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Bool :=
  checkDimensions shape dims &&
  checkParallel [.tensor shape (.float .f32), .tensor (parameterShape shape dims) (.float .f32)]
    shape (.int 8) [identityMap shape.length, parameterMap shape.length dims] maps iters &&
  PolicyW8A8Scalar.checkQuantizeKnown known hi lo actual

theorem checkQuantize_sound {known : List ConstantFacts.Fact} {shape dims : List Nat} {hi lo : Nat}
    {maps : List AffineMap} {iters : List IterKind} {actual : Region}
    (h : checkQuantize known shape dims hi lo maps iters actual = true) :
    QuantizeCorrect known shape dims maps iters actual := by
  have hh := h
  simp only [checkQuantize, Bool.and_eq_true] at hh
  obtain ⟨hm, _, geometry⟩ := checkParallel_sound hh.1.2
  intro env valid ctx call get reciprocal initial
  apply W8A8FloatTensor.quantize_sound ctx call env shape (parameterShape shape dims)
    (parameterMap shape.length dims) maps iters actual hm geometry PolicyW8A8Scalar.upper
    PolicyW8A8Scalar.lower _ get reciprocal initial
  intro c x inv old
  exact PolicyW8A8Scalar.checkQuantizeKnown_sound hh.2 valid c call x inv old

def checkRequantize (count : Nat) (shape leftDims rightDims : List Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Bool :=
  checkDimensions shape leftDims && checkDimensions shape rightDims &&
  checkParallel ([.tensor shape (.int 32)] ++
    ([.tensor (parameterShape shape leftDims) (.float .f32),
      .tensor (parameterShape shape rightDims) (.float .f32)].take count)) shape (.float .f32)
    (identityMap shape.length :: ([parameterMap shape.length leftDims,
      parameterMap shape.length rightDims].take count)) maps iters &&
  PolicyW8A8Scalar.checkRequantize count actual

theorem checkRequantize_sound {count : Nat} {shape leftDims rightDims : List Nat}
    {maps : List AffineMap} {iters : List IterKind} {actual : Region}
    (h : checkRequantize count shape leftDims rightDims maps iters actual = true) :
    RequantizeCorrect count shape leftDims rightDims maps iters actual := by
  have hh := h
  simp only [checkRequantize, Bool.and_eq_true] at hh
  obtain ⟨hm, _, geometry⟩ := checkParallel_sound hh.1.2
  have hbody := hh.2
  simp only [PolicyW8A8Scalar.checkRequantize, Bool.and_eq_true] at hbody
  intro ctx call env get left right initial
  apply W8A8FloatTensor.requantize_sound ctx call env count (of_decide_eq_true hbody.1) shape
    (parameterShape shape leftDims) (parameterShape shape rightDims)
    (parameterMap shape.length leftDims) (parameterMap shape.length rightDims) maps iters actual
    (by simpa using hm) (by simpa using geometry) _ get left right initial
  intro c acc l r old
  exact PolicyW8A8Scalar.checkRequantize_sound hh.2 c call env acc l r old

#print axioms checkUnary_sound
#print axioms checkQuantize_sound
#print axioms checkRequantize_sound

end MerlinVerify.Check.PolicyW8A8Pointwise
