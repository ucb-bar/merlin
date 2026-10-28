/- Bit-class facts for the separately named W8A8 numerical-domain candidate. Non-NaN
includes infinities. Extra facts are never inferred from initialization or an operation name. -/
import MerlinVerify.Check.BoundedFloatFacts
import MerlinVerify.Rules.W8A8IEEE32

namespace MerlinVerify.Check.PolicyW8A8NumericalFacts

open MerlinVerify Definedness Rules

inductive Class where
  | any | finite | nonNaN | nonnegative | nonzeroFinite
  deriving DecidableEq, Repr

def bitsCheck : Class → FBits .f32 → Bool
  | .any, _ => true
  | .finite, bits => decide (W8A8IEEE32.Finite bits)
  | .nonNaN, bits => !W8A8ByteFloat.isNaN bits
  | .nonnegative, bits => !W8A8ByteFloat.isNaN bits && !W8A8ByteFloat.negative bits
  | .nonzeroFinite, bits => decide (W8A8IEEE32.Finite bits) && decide (W8A8IEEE32.Nonzero bits)

def pointCheck : Class → Val → Bool
  | .any, _ => true
  | cls, value => match Sem.Ops.Dequant.float32 value with
    | some bits => bitsCheck cls bits
    | none => false

def Holds : Class → RVal → Prop
  | .any, _ => True
  | cls, .scalar value => pointCheck cls value = true
  | cls, .tensor tensor =>
    ∀ point, Idx.InBounds tensor.shape point → pointCheck cls (tensor.get point) = true

structure Info where
  prior : BoundedFloatFacts.Info := ⟨FloatFacts.model.empty, none⟩
  numeric : Class := .any
  deriving DecidableEq

def model : Refined.Model Info where
  holds info value := BoundedFloatFacts.model.holds info.prior value ∧ Holds info.numeric value
  empty := ⟨BoundedFloatFacts.model.empty, .any⟩
  empty_sound value := ⟨BoundedFloatFacts.model.empty_sound value, trivial⟩

def classify (bits : FBits .f32) : Class :=
  if bitsCheck .nonnegative bits then .nonnegative
  else if bitsCheck .nonzeroFinite bits then .nonzeroFinite
  else if bitsCheck .finite bits then .finite
  else if bitsCheck .nonNaN bits then .nonNaN else .any

theorem classify_sound (bits : FBits .f32) : bitsCheck (classify bits) bits = true := by
  unfold classify
  split
  · assumption
  · split
    · assumption
    · split
      · assumption
      · split
        · assumption
        · rfl

theorem point_literal {cls : Class} {bits : FBits .f32} {value : Val}
    (checked : bitsCheck cls bits = true) (decoded : value = .some (.flt .f32 bits)) :
    pointCheck cls value = true := by
  cases cls <;> simp [pointCheck, decoded, Sem.Ops.Dequant.float32, checked]

theorem literal_holds {cls : Class} {bits : FBits .f32} {value : RVal}
    (checked : bitsCheck cls bits = true) (literal : FloatFacts.Literal (some bits) value) : Holds cls value := by
  cases cls with
  | any => trivial
  | finite =>
    cases value with
    | scalar value => exact point_literal checked literal
    | tensor tensor => exact fun point bounds => point_literal checked (literal point bounds)
  | nonNaN =>
    cases value with
    | scalar value => exact point_literal checked literal
    | tensor tensor => exact fun point bounds => point_literal checked (literal point bounds)
  | nonnegative =>
    cases value with
    | scalar value => exact point_literal checked literal
    | tensor tensor => exact fun point bounds => point_literal checked (literal point bounds)
  | nonzeroFinite =>
    cases value with
    | scalar value => exact point_literal checked literal
    | tensor tensor => exact fun point bounds => point_literal checked (literal point bounds)

def fromPrior (prior : BoundedFloatFacts.Info) : Info :=
  ⟨prior, match prior.floating.literal with | some bits => classify bits | none => .any⟩

theorem fromPrior_sound (prior : BoundedFloatFacts.Info) (value : RVal)
    (holds : BoundedFloatFacts.model.holds prior value) : model.holds (fromPrior prior) value := by
  refine ⟨holds, ?_⟩
  cases decoded : prior.floating.literal with
  | none => simp only [fromPrior, decoded, Holds]
  | some bits =>
    simp only [fromPrior, decoded]
    apply literal_holds (classify_sound bits)
    simpa only [decoded] using holds.1.2

def priorMap : Refined.FactMap model BoundedFloatFacts.model where
  project := Info.prior
  embed := fromPrior
  project_sound _ _ holds := holds.1
  embed_sound := fromPrior_sound

theorem holds_elementwise (cls : Class) (tensor : Tensor) :
    Holds cls (.tensor tensor) ↔
      ∀ point, Idx.InBounds tensor.shape point → Holds cls (.scalar (tensor.get point)) := by
  cases cls with
  | any => exact ⟨fun _ _ _ => trivial, fun _ => trivial⟩
  | finite | nonNaN | nonnegative | nonzeroFinite => rfl

theorem elementwise : Refined.Elementwise model := by
  intro info tensor
  constructor
  · intro holds point bounds
    exact ⟨(BoundedFloatFacts.elementwise _ _).1 holds.1 point bounds,
      (holds_elementwise _ _).1 holds.2 point bounds⟩
  · intro holds
    exact ⟨(BoundedFloatFacts.elementwise _ _).2 (fun point bounds => (holds point bounds).1),
      (holds_elementwise _ _).2 (fun point bounds => (holds point bounds).2)⟩

theorem full_holds {info : Info} {value : Val} (holds : model.holds info (.scalar value))
    (shape : List Nat) (elt : Ty) : model.holds info (.tensor (Tensor.full shape elt value)) := by
  apply (elementwise _ _).2
  intro point bounds
  change Idx.InBounds shape point at bounds
  simpa only [Tensor.full, Tensor.canon, bounds, ↓reduceIte] using holds

#print axioms fromPrior_sound
#print axioms priorMap
#print axioms elementwise

end MerlinVerify.Check.PolicyW8A8NumericalFacts
