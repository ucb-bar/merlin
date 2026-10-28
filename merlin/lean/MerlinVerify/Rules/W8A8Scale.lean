/-
The zero-row guard is a separate obligation under parametric floating-point semantics.
The unguarded expression follows design §5.5 and TorchAO v0.16.0 quant_primitives.py
(choose_qparams_affine_with_min_max). The guarded expression describes Merlin's extra select.
This file proves the precise missing identity; it does not add a floating-point assumption.
-/
import MerlinVerify.Spec.Dequant

namespace MerlinVerify.Rules.W8A8Scale

open MerlinVerify

def denominator : FBits .f32 := 0x42fe0000#32
def floor : FBits .f32 := 0x3727c5ac#32
def zero : FBits .f32 := 0#32

def equalKey : FloatOpKey :=
  { op := "arith.cmpf", modifiers := "fm1:0;rm4:none;pred12:attr 1 : i64" }

def declared (M : FloatModel) (amax : FBits .f32) : FBits .f32 :=
  M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32
    [M.op (Spec.Dequant.defaultKey "arith.divf") .f32 [amax, denominator], floor]

def guarded (M : FloatModel) (amax : FBits .f32) : FBits .f32 :=
  if M.cmp equalKey .f32 amax zero then floor else declared M amax

/-- A proposed proof obligation, not an assumed axiom or an approved float law. -/
def GuardAgreement (M : FloatModel) : Prop :=
  ∀ amax, M.cmp equalKey .f32 amax zero = true → declared M amax = floor

/-- The extra select agrees with the declared scale exactly when this identity holds. -/
theorem guarded_eq_iff (M : FloatModel) :
    (∀ amax, guarded M amax = declared M amax) ↔ GuardAgreement M := by
  constructor
  · intro h amax heq
    have hg := h amax
    simpa [guarded, heq] using hg.symm
  · intro h amax
    by_cases heq : M.cmp equalKey .f32 amax zero = true
    · simpa [guarded, heq] using (h amax heq).symm
    · simp [guarded, heq]

/-- A permitted abstract model that separates the formulas; it is deliberately not an IEEE model. -/
def separatingModel : FloatModel where
  op _ k _ := 0#k.width
  cmp _ _ _ _ := true
  convert _ _ k _ := 0#k.width
  toInt _ _ _ _ := none
  ofInt _ _ k _ := 0#k.width

/-- No theorem equating these expressions for every current FloatModel can be proved. -/
theorem guard_counterexample : guarded separatingModel zero ≠ declared separatingModel zero := by
  decide

#print axioms guarded_eq_iff
#print axioms guard_counterexample

end MerlinVerify.Rules.W8A8Scale
