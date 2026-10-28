/- Review candidate for captured dequantization with i32 or i64 zero points. The operation
orders follow the versioned PyTorch source, rather than Merlin's lowering. The i32 specialization
is proved equal to the existing contract. This does not change frozen dispatch or Spec/ files. -/
import MerlinVerify.Check.DefinedDequant

namespace MerlinVerify.Rules.W8A8DequantWidth

open MerlinVerify MerlinVerify.Sem.Ops

/-- Per-tensor converts the byte and zero point separately before subtraction. -/
def perTensor (width : Nat) (M : FloatModel) (q : BitVec 8) (z : BitVec width)
    (scale : FBits .f32) : FBits .f32 :=
  M.op (Spec.Dequant.defaultKey "arith.mulf") .f32
    [M.op (Spec.Dequant.defaultKey "arith.subf") .f32
      [M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") 8 .f32 q,
       M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 z], scale]

/-- Per-channel promotes the byte, subtracts in the zero point's integer width, then converts.
The intended supported widths are 32 and 64, both wider than signed-byte storage. -/
def perChannel (width : Nat) (M : FloatModel) (q : BitVec 8) (z : BitVec width)
    (scale : FBits .f32) : FBits .f32 :=
  M.op (Spec.Dequant.defaultKey "arith.mulf") .f32
    [M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") width .f32 (q.signExtend width - z), scale]

def scalar (width : Nat) (channel : Bool) (M : FloatModel) (q : BitVec 8)
    (z : BitVec width) (scale : FBits .f32) : FBits .f32 :=
  if channel then perChannel width M q z scale else perTensor width M q z scale

def scalarValue (width : Nat) (channel : Bool) (M : FloatModel) (q : Option (BitVec 8))
    (z : Option (BitVec width)) (scale : Option (FBits .f32)) : Val :=
  match q, z, scale with
  | some q, some z, some scale => .some (.flt .f32 (scalar width channel M q z scale))
  | _, _, _ => .poison

theorem perTensor32 (M : FloatModel) (q : BitVec 8) (z : BitVec 32) (scale : FBits .f32) :
    perTensor 32 M q z scale = Spec.Dequant.perTensor M q z scale := rfl

theorem perChannel32 (M : FloatModel) (q : BitVec 8) (z : BitVec 32) (scale : FBits .f32) :
    perChannel 32 M q z scale = Spec.Dequant.perChannel M q z scale := rfl

theorem scalarValue32 (channel : Bool) (M : FloatModel) (q : Option (BitVec 8))
    (z : Option (BitVec 32)) (scale : Option (FBits .f32)) :
    scalarValue 32 channel M q z scale = Spec.Dequant.scalarValue channel M q z scale := by
  cases q <;> cases z <;> cases scale <;> cases channel <;> rfl

def result (width : Nat) (c : Dequant.Config) (M : FloatModel) (q s z : Tensor) : Tensor :=
  Tensor.canon c.shape (.float .f32) fun point =>
    let parameter := Spec.Dequant.parameterIndex c.channel c.axis point
    scalarValue width c.channel M (Dequant.bits 8 (q.get point))
      (Dequant.bits width (z.get parameter)) (Dequant.float32 (s.get parameter))

def validInputs (width : Nat) (c : Dequant.Config) (q s z : Tensor) : Bool :=
  decide (q.shape = c.shape ∧ s.shape = Dequant.paramShape c ∧ z.shape = Dequant.paramShape c) &&
  Dequant.tensorTyped q (.int 8) && Dequant.tensorTyped s (.float .f32) &&
    Dequant.tensorTyped z (.int width)

def run (width : Nat) (c : Dequant.Config) (M : FloatModel) : List RVal → Outcome (List RVal)
  | [.tensor q, .tensor s, .tensor z] =>
    if validInputs width c q s z then .ok [.tensor (result width c M q s z)]
    else .stuck "dequantization: unsupported runtime types or parameter shapes"
  | _ => .stuck "dequantization: expected three tensor operands"

theorem result32 (c : Dequant.Config) (M : FloatModel) (q s z : Tensor) :
    result 32 c M q s z = Dequant.result c M q s z := by
  simp only [result, Dequant.result, Spec.Dequant.tensor, scalarValue32]

/-- Where this i32 candidate succeeds the source semantics agrees (the source also admits
per-tensor i64 zero points). -/
theorem run32 (c : Dequant.Config) (M : FloatModel) (args : List RVal) (r : List RVal)
    (h : run 32 c M args = .ok r) : Dequant.run c M args = .ok r := by
  unfold run at h
  split at h
  · next q s z =>
    split at h
    · next hv =>
      have hv' : Dequant.validInputs c q s z = true := by
        simpa [validInputs, Dequant.validInputs] using hv
      simp only [Dequant.run, hv', ↓reduceIte, ← result32]
      exact h
    · cases h
  · cases h

theorem result_typed (ib width : Nat) (c : Dequant.Config) (M : FloatModel) (q s z : Tensor) :
    Check.RVal.HasTy ib (.tensor (result width c M q s z)) (.tensor c.shape (.float .f32)) := by
  refine ⟨rfl, rfl, fun point hp => ?_⟩
  simp only [result, Tensor.canon, hp, ↓reduceIte]
  unfold scalarValue
  split <;> simp [Check.Val.HasTy, Scalar.conforms]

/-- Initialized inputs produce initialized outputs under every current FloatModel. This is
typed totality, not a claim about concrete IEEE accuracy or about a lowering implementation. -/
theorem result_defined {width : Nat} {c : Dequant.Config} {q s z : Tensor} (M : FloatModel)
    (axis : c.channel = false ∨ c.axis < c.shape.length)
    (valid : validInputs width c q s z = true)
    (hq : (RVal.tensor q).Defined) (hs : (RVal.tensor s).Defined)
    (hz : (RVal.tensor z).Defined) : (RVal.tensor (result width c M q s z)).Defined := by
  simp only [validInputs, Bool.and_eq_true, decide_eq_true_eq] at valid
  obtain ⟨⟨⟨⟨qShape, sShape, zShape⟩, qt⟩, st⟩, zt⟩ := valid
  intro point hp
  change Idx.InBounds c.shape point at hp
  have parameterBounds : Idx.InBounds (Dequant.paramShape c)
      (Spec.Dequant.parameterIndex c.channel c.axis point) :=
    DequantSource.parameterIndex_bounds axis hp
  have qp : Idx.InBounds q.shape point := qShape ▸ hp
  have sp : Idx.InBounds s.shape (Spec.Dequant.parameterIndex c.channel c.axis point) :=
    sShape ▸ parameterBounds
  have zp : Idx.InBounds z.shape (Spec.Dequant.parameterIndex c.channel c.axis point) :=
    zShape ▸ parameterBounds
  obtain ⟨a, qa⟩ := Check.Definedness.dequant_bits_defined
    (DequantSource.tensorTyped_get qt qp) (hq point qp)
  obtain ⟨b, zb⟩ := Check.Definedness.dequant_bits_defined
    (DequantSource.tensorTyped_get zt zp) (hz _ zp)
  obtain ⟨scale, ss⟩ := Check.Definedness.dequant_float_defined
    (DequantSource.tensorTyped_get st sp) (hs _ sp)
  simp [result, Tensor.canon, hp, scalarValue, qa, zb, ss]

#print axioms scalarValue32
#print axioms run32
#print axioms result_typed
#print axioms result_defined

end MerlinVerify.Rules.W8A8DequantWidth
