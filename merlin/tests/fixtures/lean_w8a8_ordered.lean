/- Kernel controls for ordered floating accumulation and actual scalar-region recognition.
Tensor execution and whole-model coverage are separate obligations. -/
import MerlinVerify.Check.PolicyW8A8OrderedBody
import MerlinVerify.Check.PolicyW8A8IEEEContext

open MerlinVerify MerlinVerify.Rules MerlinVerify.Check
open W8A8OrderedMagnitude

set_option exponentiation.threshold 512

theorem reduction_guard_controls :
    W8A8OrderedDot.admissible 127 127 2048 = true ∧
    W8A8OrderedDot.cap 127 127 2048 = 142 ∧
    W8A8OrderedDot.admissible 127 127 (2^22) = true ∧
    W8A8OrderedDot.admissible 127 127 (2^22 + 1) = false ∧
    W8A8OrderedDot.admissible 254 254 2 = false ∧
    W8A8OrderedDot.cap 127 127 0 = 0 := by decide

/-- The proof covers every bounded input sequence, not only a numerically sampled vector. -/
theorem qkv_2048 (pairs : List W8A8OrderedDot.Pair) (count : pairs.length = 2048)
    (inputs : ∀ pair ∈ pairs, W8A8IEEEBounds.ExponentLE 127 pair.1 ∧ W8A8IEEEBounds.ExponentLE 127 pair.2) :
    W8A8IEEEBounds.ExponentLE 142 (W8A8OrderedDot.evaluate pairs 0#32) := by
  have safe : W8A8OrderedDot.admissible 127 127 pairs.length = true := by rw [count]; decide
  have bounded := W8A8OrderedDot.evaluate_bounded inputs safe
  simpa only [count, show W8A8OrderedDot.cap 127 127 2048 = 142 from by decide] using bounded

theorem ordered_rounding_controls :
    W8A8OrderedDot.evaluate [(0x4b800000, 0x3f800000), (0x3f800000, 0x3f800000),
      (0xcb800000, 0x3f800000)] 0 = 0 ∧
    W8A8OrderedDot.evaluate [(0x4b800000, 0x3f800000), (0xcb800000, 0x3f800000),
      (0x3f800000, 0x3f800000)] 0 = 0x3f800000 ∧
    W8A8OrderedDot.evaluate [(1, 0x3f800000), (1, 0x3f800000)] 0 = 2 ∧
    W8A8OrderedDot.evaluate [(0x80000000, 0x3f800000)] 0x80000000 = 0x80000000 := by decide

/-- One larger magnitude limit would admit a real rounding-to-infinity counterexample. -/
theorem unsafe_limit_overflows :
    units 0x7f7fffff + units 0x73000000 < 2^(253 + 24) ∧
    W8A8IEEE32.add 0x7f7fffff 0x73000000 = 0x7f800000 := by decide

theorem exact_body_controls :
    PolicyW8A8OrderedBody.check (W8A8OrderedBody.region false) = true ∧
    PolicyW8A8OrderedBody.check (W8A8OrderedBody.region true) = true := by
  constructor <;> simp [PolicyW8A8OrderedBody.check, W8A8OrderedBody.region, W8A8OrderedBody.body,
    alphaRegion, alphaOps, alphaOp, alphaRegions, freshIds, opHeadEq, Rho.find,
    Op.name, Op.results, Op.attrs, Op.props, Op.successors, List.lookup]

/-- The existing seven-plus-three laws and Agreement do not imply finite sin/cos.
These fallback operations are explicitly outside the constructed arithmetic slice. -/
theorem existing_context_allows_nan_trig :
    PolicyW8A8NumericalContext.ContextOk PolicyW8A8IEEEContext.candidate ∧
    W8A8IEEE32.Finite (0 : FBits .f32) ∧
    W8A8ByteFloat.isNaN (PolicyW8A8IEEEContext.candidate.M.op
      (Spec.Dequant.defaultKey "math.sin") .f32 [0]) = true ∧
    W8A8ByteFloat.isNaN (PolicyW8A8IEEEContext.candidate.M.op
      (Spec.Dequant.defaultKey "math.cos") .f32 [0]) = true := by
  exact ⟨PolicyW8A8IEEEContext.candidate_ok, by decide, by decide, by decide⟩

#print axioms qkv_2048
#print axioms ordered_rounding_controls
#print axioms unsafe_limit_overflows
#print axioms exact_body_controls
#print axioms existing_context_allows_nan_trig
