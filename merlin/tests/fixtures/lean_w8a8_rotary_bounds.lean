/- Kernel examples and negative controls for the explicit bounded candidate.
These do not supply a concrete interpretation of all primitive numerical laws. -/
import MerlinVerify.Check.PolicyW8A8RotaryDomain
import MerlinVerify.Rules.W8A8ByteBounds

open MerlinVerify MerlinVerify.Rules MerlinVerify.Check MerlinVerify.Check.Definedness

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

theorem rotary_product_finite (frequency position : FBits .f32)
    (frequencyBound : W8A8IEEEBounds.ExponentLE 250 frequency)
    (positionBound : W8A8IEEEBounds.ExponentLE 129 position) :
    W8A8IEEE32.Finite (W8A8IEEE32.mul frequency position) :=
  W8A8IEEEBounds.mul_finite frequencyBound positionBound (by decide)

theorem rotary_inputs_exist : ∃ args, PolicyW8A8RotaryDomain.Domain 64
    [.tensor [32] (.float .f32)] [none] [some 250] args :=
  PolicyW8A8RotaryDomain.check_inhabited (by decide)

theorem subnormal_frequency_domain_exists : ∃ args, PolicyW8A8RotaryDomain.Domain 64
    [.tensor [3] (.float .f32)] [none] [some 0] args :=
  PolicyW8A8RotaryDomain.check_inhabited (by decide)

example : ¬ W8A8IEEEBounds.ExponentLE 250 0x7f7fffff#32 := by
  intro bounded
  have impossible := bounded.2
  change 254 ≤ 250 at impossible
  omega
example : W8A8IEEE32.Finite (W8A8IEEE32.mul 0x7d7fffff#32 0x40e00000#32) := by decide
example : ¬ W8A8IEEE32.Finite (W8A8IEEE32.mul 0x7f7fffff#32 0x40000000#32) := by decide
example : W8A8ByteBounds.check 0#64 0x3c4c9932#32 127 = true := by decide
example : W8A8ByteBounds.check 0#64 0x3c4c9932#32 126 = false := by decide
example : W8A8ByteBounds.check 0#64 0x7f7fffff#32 254 = false := by decide
example : PolicyW8A8RotaryDomain.check 64 [.tensor [2] (.int 64)] [none] [some 250] = false := by decide
example : PolicyW8A8RotaryDomain.check 64 [.tensor [2] (.float .f32)] [none] [some 255] = false := by decide
example : PolicyW8A8RotaryDomain.check 64 [.tensor [2] (.float .f32)] [none] [] = false := by decide

#print axioms rotary_product_finite
#print axioms rotary_inputs_exist
#print axioms subnormal_frequency_domain_exists
