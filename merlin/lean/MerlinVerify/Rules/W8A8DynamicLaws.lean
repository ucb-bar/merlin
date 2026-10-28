/- Additional primitive numerical premises for the generated dynamic W8A8 sequence.
These are separate from the earlier source-quantizer context, whose domain is unchanged.
No concrete model instance, reviewed assumption or new global axiom is introduced. -/
import MerlinVerify.Rules.W8A8FiniteLaws

namespace MerlinVerify.Rules.W8A8DynamicLaws

open MerlinVerify W8A8FiniteLaws

structure Laws (base : FloatModel) : Prop where
  source : W8A8FiniteLaws.Laws base
  absolute : ∀ x, Finite x → Finite (base.op (key "math.absf") .f32 [x])
  scaleDivide : ∀ x, Finite x →
    Finite (base.op (key "arith.divf") .f32 [x, W8A8Scale.denominator])
  reciprocal32 : ∀ scale, scaleCheck scale = true →
    Finite (base.op (key "arith.divf") .f32 [0x3f800000#32, scale])

theorem finite_nonNaN {x : FBits .f32} (finite : Finite x) : W8A8ByteFloat.isNaN x = false := by
  have different : W8A8ByteFloat.exponent x ≠ 255 := Nat.ne_of_lt finite
  simp [W8A8ByteFloat.isNaN, different]

theorem maximum_finite {x y : FBits .f32} (hx : Finite x) (hy : Finite y) :
    Finite (W8A8ByteFloat.maximum x y) := by
  simp only [W8A8ByteFloat.maximum, finite_nonNaN hx, finite_nonNaN hy,
    Bool.false_eq_true, ↓reduceIte]
  split <;> assumption

theorem minimum_finite {x y : FBits .f32} (hx : Finite x) (hy : Finite y) :
    Finite (W8A8ByteFloat.minimum x y) := by
  simp only [W8A8ByteFloat.minimum, finite_nonNaN hx, finite_nonNaN hy,
    Bool.false_eq_true, ↓reduceIte]
  split <;> assumption

theorem maximum_normal_right {x y : FBits .f32} (hx : Finite x) (normal : scaleCheck y = true) :
    scaleCheck (W8A8ByteFloat.maximum x y) = true := by
  have parts := normal
  simp only [scaleCheck, W8A8SourceQuantize.scaleAdmissible, Bool.and_eq_true,
    decide_eq_true_eq] at parts
  obtain ⟨⟨⟨positive, magnitude⟩, finite⟩, exponent⟩ := parts
  have signY : W8A8ByteFloat.negative y = false := by
    cases sign : W8A8ByteFloat.negative y with
    | false => rfl
    | true => simp [sign] at positive
  simp only [W8A8ByteFloat.maximum, finite_nonNaN hx, finite_nonNaN finite,
    Bool.false_eq_true, ↓reduceIte]
  split
  · exact normal
  · next order =>
    have sign : W8A8ByteFloat.negative x = false := by
      cases negative : W8A8ByteFloat.negative x with
      | false => rfl
      | true => simp [W8A8ByteFloat.precedes, negative, signY] at order
    have bound : W8A8ByteFloat.magnitudeWord y ≤ W8A8ByteFloat.magnitudeWord x := by
      have failed : ¬ W8A8ByteFloat.magnitudeWord x < W8A8ByteFloat.magnitudeWord y :=
        of_decide_eq_false (by simpa [W8A8ByteFloat.precedes, sign, signY] using order)
      omega
    have expPositive : 0 < W8A8ByteFloat.exponent x := by
      unfold W8A8ByteFloat.exponent at exponent ⊢
      omega
    have magPositive : 0 < W8A8ByteFloat.magnitudeWord x := by omega
    simp [scaleCheck, W8A8SourceQuantize.scaleAdmissible, sign, magPositive, hx, expPositive]

theorem maximum_normal_left {x y : FBits .f32} (normal : scaleCheck x = true) (hy : Finite y) :
    scaleCheck (W8A8ByteFloat.maximum x y) = true := by
  have parts := normal
  simp only [scaleCheck, W8A8SourceQuantize.scaleAdmissible, Bool.and_eq_true,
    decide_eq_true_eq] at parts
  obtain ⟨⟨⟨positive, magnitude⟩, finite⟩, exponent⟩ := parts
  have signX : W8A8ByteFloat.negative x = false := by
    cases sign : W8A8ByteFloat.negative x with
    | false => rfl
    | true => simp [sign] at positive
  simp only [W8A8ByteFloat.maximum, finite_nonNaN finite, finite_nonNaN hy,
    Bool.false_eq_true, ↓reduceIte]
  split
  · next order =>
    have sign : W8A8ByteFloat.negative y = false := by
      cases negative : W8A8ByteFloat.negative y with
      | false => rfl
      | true => simp [W8A8ByteFloat.precedes, negative, signX] at order
    have bound : W8A8ByteFloat.magnitudeWord x < W8A8ByteFloat.magnitudeWord y :=
      of_decide_eq_true (by simpa [W8A8ByteFloat.precedes, sign, signX] using order)
    have expPositive : 0 < W8A8ByteFloat.exponent y := by
      unfold W8A8ByteFloat.exponent at exponent ⊢
      omega
    have magPositive : 0 < W8A8ByteFloat.magnitudeWord y := by omega
    simp [scaleCheck, W8A8SourceQuantize.scaleAdmissible, sign, magPositive, hy, expPositive]
  · exact normal

theorem floor_normal : scaleCheck W8A8Scale.floor = true := by decide

theorem scale_normal {base : FloatModel} (laws : Laws base) {amax : FBits .f32} (finite : Finite amax) :
    scaleCheck (W8A8Scale.guarded (W8A8ByteModel.extend base) amax) = true := by
  unfold W8A8Scale.guarded
  split
  · exact floor_normal
  · unfold W8A8Scale.declared
    rw [W8A8ByteModel.maximum_eq]
    apply maximum_normal_right ?_ floor_normal
    simpa [W8A8ByteModel.extend, key, Spec.Dequant.defaultKey] using laws.scaleDivide amax finite

theorem reciprocal_finite {base : FloatModel} (laws : Laws base) {scale : FBits .f32}
    (normal : scaleCheck scale = true) :
    Finite ((W8A8ByteModel.extend base).op (key "arith.divf") .f32 [0x3f800000#32, scale]) := by
  simpa [W8A8ByteModel.extend, key, Spec.Dequant.defaultKey] using laws.reciprocal32 scale normal

theorem absmax_finite {base : FloatModel} (laws : Laws base) {x acc : FBits .f32}
    (finite : Finite x) (initial : Finite acc) :
    Finite ((W8A8ByteModel.extend base).op (key "arith.maximumf") .f32
      [(W8A8ByteModel.extend base).op (key "math.absf") .f32 [x], acc]) := by
  rw [W8A8ByteModel.maximum_eq]
  apply maximum_finite ?_ initial
  simpa [W8A8ByteModel.extend, key, Spec.Dequant.defaultKey] using laws.absolute x finite

theorem rounded_nonNaN {base : FloatModel} (laws : Laws base) {x inverse : FBits .f32}
    (finite : Finite x) (reciprocal : Finite inverse) :
    W8A8ByteFloat.isNaN (W8A8ByteModel.rounded base x inverse) = false :=
  laws.source.round _ (laws.source.multiply _ _ finite reciprocal)

theorem quantize_defined {base : FloatModel} (laws : Laws base) {x inverse : FBits .f32}
    (finite : Finite x) (reciprocal : Finite inverse) :
    W8A8Scalar.quantize (W8A8ByteModel.extend base) (some x) (some inverse)
      0x42fe0000#32 0xc2fe0000#32 ≠ .poison :=
  W8A8ByteModel.quantize_defined base x inverse (rounded_nonNaN laws finite reciprocal)

#print axioms finite_nonNaN
#print axioms maximum_finite
#print axioms minimum_finite
#print axioms maximum_normal_right
#print axioms maximum_normal_left
#print axioms scale_normal
#print axioms reciprocal_finite
#print axioms absmax_finite
#print axioms rounded_nonNaN
#print axioms quantize_defined

end MerlinVerify.Rules.W8A8DynamicLaws
