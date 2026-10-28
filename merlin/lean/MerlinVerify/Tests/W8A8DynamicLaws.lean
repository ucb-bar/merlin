/- Non-vacuity of the explicit dynamic-law domain. This synthetic model proves only that
the premises are jointly satisfiable; it is not an IEEE interpretation or a fidelity claim. -/
import MerlinVerify.Rules.W8A8DynamicLaws

namespace MerlinVerify.Tests.W8A8DynamicLaws

open MerlinVerify Rules

private def synthetic : FloatModel where
  op _ _ _ := 0
  cmp _ _ _ _ := false
  convert _ _ destination _ := match destination with
    | .f64 => 0x3ff0000000000000#64
    | _ => 0
  ofInt _ _ _ _ := 0
  toInt _ _ _ _ := none

private theorem source_laws : W8A8FiniteLaws.Laws synthetic where
  widen _ _ := by
    change W8A8FiniteLaws.WideScale 0x3ff0000000000000#64
    unfold W8A8FiniteLaws.WideScale
    decide
  reciprocal _ _ := by
    change W8A8FiniteLaws.Narrowable 0
    unfold W8A8FiniteLaws.Narrowable
    decide
  narrow _ _ := by change W8A8FiniteLaws.Finite 0; decide
  integer _ _ _ := by change W8A8FiniteLaws.Finite 0; decide
  multiply _ _ _ _ := by change W8A8ByteFloat.isNaN 0 = false; decide
  round _ _ := by change W8A8ByteFloat.isNaN 0 = false; decide
  add _ _ _ _ := by change W8A8ByteFloat.isNaN 0 = false; decide

private theorem dynamic_laws : Rules.W8A8DynamicLaws.Laws synthetic where
  source := source_laws
  absolute _ _ := by change W8A8FiniteLaws.Finite 0; decide
  scaleDivide _ _ := by change W8A8FiniteLaws.Finite 0; decide
  reciprocal32 _ _ := by change W8A8FiniteLaws.Finite 0; decide

theorem domain_inhabited : ∃ base, Rules.W8A8DynamicLaws.Laws base :=
  ⟨synthetic, dynamic_laws⟩

#print axioms domain_inhabited

end MerlinVerify.Tests.W8A8DynamicLaws
