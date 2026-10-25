/- Scalar numerical requirements inspect proved value facts. The context maps exact primitive
keys to executable IEEE computations; typing and initialization remain separate obligations. -/
import MerlinVerify.Check.PolicyW8A8NumericalImplication
import MerlinVerify.Check.PolicyW8A8NumericalContext
import MerlinVerify.Check.FloatScalarFacts

namespace MerlinVerify.Check.PolicyW8A8NumericalScalar

open MerlinVerify Rules Definedness PolicyW8A8NumericalFacts

def fact (cls : Class) : Refined.Fact Info :=
  ⟨⟨.scalar (.float .f32), true⟩, ⟨BoundedFloatFacts.model.empty, cls⟩⟩

theorem result {ib : Nat} {cls : Class} {bits : FBits .f32} (checked : bitsCheck cls bits = true) :
    (fact cls).Holds model ib (.scalar (.some (.flt .f32 bits))) := by
  refine ⟨⟨by simp [fact, RVal.HasTy, Val.HasTy, Scalar.conforms], by simp [RVal.Defined]⟩,
    BoundedFloatFacts.model.empty_sound _, ?_⟩
  cases cls with
  | any => trivial
  | finite | nonNaN | nonnegative | nonzeroFinite =>
    simpa [fact, Holds, pointCheck, Sem.Ops.Dequant.float32, bitsCheck] using checked

theorem requirement {ib : Nat} {cls : Class} {info : Info} {bits : FBits .f32}
    (checked : accepts info cls = true)
    (holds : (⟨⟨.scalar (.float .f32), true⟩, info⟩ : Refined.Fact Info).Holds model ib
      (.scalar (.some (.flt .f32 bits)))) : bitsCheck cls bits = true := by
  have actual := accepts_sound checked holds.2
  cases cls with
  | any => rfl
  | finite | nonNaN | nonnegative | nonzeroFinite =>
    simpa [Holds, pointCheck, Sem.Ops.Dequant.float32] using actual

def requirements (info : Info) (needed : List Class) : Bool := needed.all (accepts info)
def bitRequirements (bits : FBits .f32) (needed : List Class) : Bool := needed.all (fun cls => bitsCheck cls bits)

theorem requirements_sound {ib : Nat} {info : Info} {bits : FBits .f32} {needed : List Class}
    (checked : requirements info needed = true)
    (holds : (⟨⟨.scalar (.float .f32), true⟩, info⟩ : Refined.Fact Info).Holds model ib
      (.scalar (.some (.flt .f32 bits)))) : bitRequirements bits needed = true := by
  apply List.all_eq_true.2
  intro cls member
  exact requirement (List.all_eq_true.1 checked cls member) holds

inductive BinaryKind where
  | multiply | add | divide
  deriving DecidableEq, Repr

def BinaryKind.name : BinaryKind → String
  | .multiply => "arith.mulf"
  | .add => "arith.addf"
  | .divide => "arith.divf"

/-- A static verified rule registry. Certificates do not construct these proof-bearing entries. -/
structure BinaryContract where
  kind : BinaryKind
  left : List Class
  right : List Class
  output : Class
  sound : ∀ base, W8A8ArithmeticModel.Agreement base → ∀ x y,
    bitRequirements x left = true → bitRequirements y right = true →
    bitsCheck output (base.op (Spec.Dequant.defaultKey kind.name) .f32 [x, y]) = true

def multiplyFinite : BinaryContract where
  kind := .multiply
  left := [.finite]
  right := [.finite]
  output := .nonNaN
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      decide_eq_true_eq] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.mul_finite_notNaN left right
    rw [agreement.multiply32 _ _ numerical]
    simpa [bitsCheck] using numerical

def multiplyLeft : BinaryContract where
  kind := .multiply
  left := [.nonzeroFinite]
  right := [.nonNaN]
  output := .nonNaN
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.mul_nonzeroFinite_nonNaN left.1 left.2 right
    rw [agreement.multiply32 _ _ numerical]
    simpa [bitsCheck] using numerical

def multiplyRight : BinaryContract where
  kind := .multiply
  left := [.nonNaN]
  right := [.nonzeroFinite]
  output := .nonNaN
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.mul_nonNaN_nonzeroFinite left right.1 right.2
    rw [agreement.multiply32 _ _ numerical]
    simpa [bitsCheck] using numerical

def addNonnegative : BinaryContract where
  kind := .add
  left := [.nonnegative]
  right := [.nonnegative]
  output := .nonnegative
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.add_nonnegative left right
    rw [agreement.add32 _ _ numerical.1]
    simpa [bitsCheck, W8A8IEEE32.Nonnegative] using numerical

def addFiniteRight : BinaryContract where
  kind := .add
  left := [.nonNaN]
  right := [.finite]
  output := .nonNaN
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.add_nonNaN_finite left right
    rw [agreement.add32 _ _ numerical]
    simpa [bitsCheck] using numerical

def divideNonnegative : BinaryContract where
  kind := .divide
  left := [.nonnegative]
  right := [.nonnegative, .nonzeroFinite]
  output := .nonnegative
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.div_nonnegative left ⟨right.1, right.2.2⟩ right.2.1
    rw [agreement.divide32 _ _ numerical.1]
    simpa [bitsCheck, W8A8IEEE32.Nonnegative] using numerical

def divideAny : BinaryContract where
  kind := .divide
  left := [.nonNaN]
  right := [.nonzeroFinite]
  output := .nonNaN
  sound base agreement x y left right := by
    simp only [bitRequirements, List.all_cons, List.all_nil, Bool.and_true, bitsCheck,
      Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true] at left right
    simp only [BinaryKind.name]
    have numerical := W8A8IEEE32.div_nonNaN_nonzeroFinite left right.1 right.2
    rw [agreement.divide32 _ _ numerical]
    simpa [bitsCheck] using numerical

def binaryContracts : List BinaryContract :=
  [multiplyFinite, multiplyLeft, multiplyRight, addNonnegative, addFiniteRight, divideNonnegative, divideAny]

#print axioms result
#print axioms requirements_sound
#print axioms binaryContracts

end MerlinVerify.Check.PolicyW8A8NumericalScalar
