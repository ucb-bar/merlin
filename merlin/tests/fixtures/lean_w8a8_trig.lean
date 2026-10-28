/- Kernel controls for the named mathematical trig interpretation and actual SSA rules.
These do not establish native transcendental fidelity or complete attention execution. -/
import MerlinVerify.Check.PolicyW8A8TrigPrefix

open MerlinVerify Rules Check Definedness Sem.Ops

namespace W8A8TrigControls

def unary (name : String) (attributes : List (String × Attr) := [])
    (kind : FKind := .f32) : Op := .mk name [⟨1, .float kind⟩] [0] attributes [] [] []

def bounded (cap : Nat) : Refined.Fact PolicyW8A8RotaryFacts.Info := PolicyW8A8RotaryScalar.fact cap

def unknown : Refined.Fact PolicyW8A8RotaryFacts.Info :=
  ⟨⟨.scalar (.float .f32), true⟩, PolicyW8A8RotaryFacts.model.empty⟩

private theorem wrong_key_refuses (op : Op) (inputs : List (Refined.Fact PolicyW8A8RotaryFacts.Info))
    (different : Arith.floatKey op.blind ≠ Spec.Dequant.defaultKey op.name) :
    PolicyW8A8TrigScalar.rule [] op inputs = none := by
  simp [PolicyW8A8TrigScalar.rule, different]

theorem accepted_finite_trig :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin") [bounded 254] = some [bounded 127] ∧
    PolicyW8A8TrigScalar.rule [] (unary "math.cos") [bounded 0] = some [bounded 127] := by cbv

theorem reject_fastmath :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin" [("fastmath", .enum "arith" "fastmath" "fast")])
      [bounded 127] = none := by
  apply wrong_key_refuses
  cbv
  decide

theorem reject_roundingmode :
    PolicyW8A8TrigScalar.rule [] (unary "math.cos" [("roundingmode", .str "toward_zero")])
      [bounded 127] = none := by
  apply wrong_key_refuses
  cbv
  decide

theorem reject_f64 :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin" [] .f64) [bounded 127] = none := by cbv

theorem reject_unregistered_exp :
    PolicyW8A8TrigScalar.rule [] (unary "math.exp") [bounded 127] = none := by cbv

theorem reject_unknown_input :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin") [unknown] = none := by cbv

theorem reject_missing_argument :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin") [] = none := by cbv

theorem reject_extra_argument :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin") [bounded 127, bounded 127] = none := by cbv

theorem reject_uninitialized_or_regions :
    PolicyW8A8TrigScalar.rule [] (unary "math.sin")
      [{ bounded 127 with base := ⟨.scalar (.float .f32), false⟩ }] = none ∧
    PolicyW8A8TrigScalar.rule [] (.mk "math.sin" [⟨1, .float .f32⟩] [0] [] [] [.mk []] [])
      [bounded 127] = none ∧
    PolicyW8A8TrigScalar.rule [] (.mk "math.sin" [⟨1, .float .f32⟩] [0] [] [] [] [0])
      [bounded 127] = none := by cbv

theorem negation_and_safe_add :
    PolicyW8A8AttentionScalar.negationRule [] (unary "arith.negf") [bounded 142] = some [bounded 142] ∧
    PolicyW8A8AttentionScalar.addRule [] (.mk "arith.addf" [⟨2, .float .f32⟩] [0, 1] [] [] [] [])
      [bounded 146, bounded 146] = some [bounded 148] ∧
    PolicyW8A8AttentionScalar.addRule [] (.mk "arith.addf" [⟨2, .float .f32⟩] [0, 1] [] [] [] [])
      [bounded 252, bounded 0] = some [bounded 254] ∧
    PolicyW8A8AttentionScalar.addRule [] (.mk "arith.addf" [⟨2, .float .f32⟩] [0, 1] [] [] [] [])
      [bounded 253, bounded 0] = none := by cbv

theorem exact_negation_bits :
    W8A8UnarySign.negate (0 : FBits .f32) = 0x80000000 ∧
    W8A8UnarySign.negate (0x80000000 : FBits .f32) = 0 ∧
    W8A8UnarySign.negate (1 : FBits .f32) = 0x80000001 ∧
    W8A8UnarySign.negate (0x7f800000 : FBits .f32) = 0xff800000 ∧
    W8A8UnarySign.negate (0x7fc12345 : FBits .f32) = 0xffc12345 := by decide

theorem rounding_endpoints :
    W8A8RoundedReal.roundedReal 0 = (0 : FBits .f32) ∧
    W8A8RoundedReal.roundedReal 1 = (0x3f800000 : FBits .f32) ∧
    W8A8RoundedReal.roundedReal W8A8RoundedReal.halfLeastSubnormal = (0 : FBits .f32) ∧
    W8A8RoundedReal.roundedReal (-W8A8RoundedReal.halfLeastSubnormal) = (0x80000000 : FBits .f32) ∧
    W8A8RoundedReal.roundedReal W8A8RoundedReal.overflowPoint = (0x7f800000 : FBits .f32) ∧
    W8A8RoundedReal.roundedReal (-W8A8RoundedReal.overflowPoint) = (0xff800000 : FBits .f32) :=
  ⟨W8A8RoundedReal.roundedReal_zero, W8A8RoundedReal.roundedReal_one,
    W8A8RoundedReal.positive_half_subnormal_tie, W8A8RoundedReal.negative_half_subnormal_tie,
    W8A8RoundedReal.roundedReal_positive_overflow le_rfl,
    W8A8RoundedReal.roundedReal_negative_overflow le_rfl⟩

theorem transcendental_endpoints :
    W8A8RoundedReal.sine (0x80000000 : FBits .f32) = 0x80000000 ∧
    W8A8RoundedReal.cosine (0x80000000 : FBits .f32) = 0x3f800000 ∧
    W8A8RoundedReal.exponential (0 : FBits .f32) = 0x3f800000 ∧
    W8A8RoundedReal.exponential (0xff800000 : FBits .f32) = 0 ∧
    W8A8RoundedReal.exponential (0x7f800000 : FBits .f32) = 0x7f800000 ∧
    W8A8RoundedReal.sine (0x7f800000 : FBits .f32) = 0x7fc00000 ∧
    W8A8RoundedReal.cosine (0xff800000 : FBits .f32) = 0x7fc00000 :=
  ⟨W8A8RoundedReal.sine_negative_zero, W8A8RoundedReal.cosine_negative_zero,
    W8A8RoundedReal.exponential_zero, W8A8RoundedReal.exponential_negative_infinity,
    W8A8RoundedReal.exponential_positive_infinity,
    W8A8RoundedReal.sine_nonfinite (by decide), W8A8RoundedReal.cosine_nonfinite (by decide)⟩

theorem actual_ssa_chain :
    (PolicyW8A8TrigPrefix.infer 64 [(0, bounded 254)]
      [unary "math.sin", .mk "arith.negf" [⟨2, .float .f32⟩] [1] [] [] [] []]).isSome = true ∧
    PolicyW8A8TrigPrefix.bound ((PolicyW8A8TrigPrefix.infer 64 [(0, bounded 254)]
      [unary "math.sin", .mk "arith.negf" [⟨2, .float .f32⟩] [1] [] [] [] []]).getD []) 2 127 = true ∧
    PolicyW8A8TrigPrefix.infer 64 [] [unary "math.sin"] = none := by cbv

#print axioms accepted_finite_trig
#print axioms reject_fastmath
#print axioms reject_roundingmode
#print axioms reject_f64
#print axioms reject_unregistered_exp
#print axioms reject_unknown_input
#print axioms reject_missing_argument
#print axioms reject_extra_argument
#print axioms reject_uninitialized_or_regions
#print axioms negation_and_safe_add
#print axioms exact_negation_bits
#print axioms rounding_endpoints
#print axioms transcendental_endpoints
#print axioms actual_ssa_chain

end W8A8TrigControls
