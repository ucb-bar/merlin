/- Scalar conversion guards preserve initialized inputs, format directions and promise boundaries. -/
import MerlinVerify.Check.DefinedConversions

namespace MerlinVerify.Tests.DefinedConversions

open MerlinVerify MerlinVerify.Check
open MerlinVerify.Check.DefinedConversions

def op (name : String) (output : Ty) (attrs : List (String × Attr) := []) : Op :=
  .mk name [⟨1, output⟩] [0] attrs [] [] []
def clean (t : Ty) : List Definedness.Fact := [⟨.scalar t, true⟩]
def dirty (t : Ty) : List Definedness.Fact := [⟨.scalar t, false⟩]

#guard floatFormatRule (op "arith.extf" (.float .f32)) (clean (.float .f16)) == some (clean (.float .f32))
#guard floatFormatRule (op "arith.extf" (.float .f64)) (clean (.float .bf16)) == some (clean (.float .f64))
#guard floatFormatRule (op "arith.truncf" (.float .f16)) (clean (.float .f64)) == some (clean (.float .f16))
#guard floatFormatRule (op "arith.truncf" (.float .bf16)) (clean (.float .f32)) == some (clean (.float .bf16))
#guard integerRule (op "arith.sitofp" (.float .f32)) (clean (.int 1)) == some (clean (.float .f32))
#guard integerRule (op "arith.uitofp" (.float .f64)) (clean (.int 13)) == some (clean (.float .f64))
#guard floatFormatRule (op "arith.extf" (.float .f16)) (clean (.float .bf16)) == none
#guard floatFormatRule (op "arith.truncf" (.float .bf16)) (clean (.float .f16)) == none
#guard floatFormatRule (op "arith.extf" (.float .f32)) (clean (.float .f64)) == none
#guard floatFormatRule (op "arith.truncf" (.float .f64)) (clean (.float .f32)) == none
#guard floatFormatRule (op "arith.extf" (.float .f32)) (dirty (.float .f16)) == none
#guard integerRule (op "arith.sitofp" (.float .f32)) (dirty (.int 32)) == none
#guard integerRule (op "arith.sitofp" (.float .f32)) (clean .index) == none
#guard integerRule (op "arith.sitofp" (.float .f32)) (clean (.int 0)) == none
#guard integerRule (op "arith.uitofp" (.float .f32) [("nonNeg",.unit)]) (clean (.int 8)) == none
#guard floatFormatRule (op "arith.truncf" (.float .f32)
  [("roundingmode",.enum "arith" "roundingmode" "tonearest")]) (clean (.float .f64)) == none
#guard floatFormatRule (op "arith.extf" (.float .f64)
  [("fastmath",.enum "arith" "fastmath" "nnan")]) (clean (.float .f32)) == none
#guard integerRule (op "arith.sitofp" (.float .f32)
  [("fastmath",.enum "arith" "fastmath" "fast")]) (clean (.int 8)) == none
#guard floatFormatRule (op "arith.extf" (.float .f64)
  [("fastmath",.enum "arith" "fastmath" "none")]) (clean (.float .f32)) == some (clean (.float .f64))
#guard integerRule (op "arith.fptosi" (.int 32)) (clean (.float .f32)) == none
#guard integerRule (op "arith.fptoui" (.int 32)) (clean (.float .f32)) == none
#guard floatFormatRule (op "arith.extf" (.float .f32)) [⟨.tensor [3] (.float .f16),true⟩] == none
#guard (book 64).rule [] (op "arith.extf" (.float .f64) [("unknown.rounding",.unit)]) (clean (.float .f32)) == none
#guard floatFormatRule (op "arith.extf" (.float .f64)) [] == none
#guard floatFormatRule (op "arith.extf" (.int 64)) (clean (.float .f32)) == none

end MerlinVerify.Tests.DefinedConversions
