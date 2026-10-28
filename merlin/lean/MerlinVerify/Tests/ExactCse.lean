import MerlinVerify.Check.ExactCse
import MerlinVerify.Rules.Dce

namespace MerlinVerify.Tests.ExactCseRegression

open MerlinVerify MerlinVerify.Check

def ty (shape : List Nat) : Ty := .tensor (shape.map some) (.int 8) none
def empty (id : Nat) (shape : List Nat) : Op :=
  .mk "tensor.empty" [⟨id, ty shape⟩] [] [] [] [] []
def ret (ids : List Nat) : Op := .mk "func.return" [] ids [] [] [] []
def normalize (ss : List Op) := (ExactCse.base 64 []).run 0 [] [] ss []

-- A live earlier empty can supply a later identical one.
#guard (normalize [empty 1 [2], empty 2 [2], ret [1,2]]).map
  (fun ss => (ss.length, (ss.getLast!).operands)) == some (2, [1,1])

-- An unused earlier producer must not swallow the correspondence of a live one.
#guard (normalize [empty 0 [2], empty 1 [2], empty 2 [2], ret [1,2]]).map
  (fun ss => (ss.length, (ss.getLast!).operands)) == some (3, [1,1])

-- Distinct result geometry cannot merge.
#guard (normalize [empty 1 [2], empty 2 [3], ret [1,2]]).map List.length == some 3

-- Renaming remains valid through another use, rather than just at the final return.
def cast (id input : Nat) : Op :=
  .mk "tensor.cast" [⟨id, ty [2]⟩] [input] [] [] [] []
#guard (normalize [empty 1 [2], empty 2 [2], cast 3 1, cast 4 2, ret [1,2,3,4]]).map
  (fun ss => (ss.length, (ss.getLast!).operands)) == some (3, [1,1,3,3])

-- The DCE extension admits only the three modeled region-free Q/DQ operations.
def dequant (name : String) : Op := .mk name [⟨0, ty [2]⟩] [] [] [] [] []
#guard Rules.Dce.erasable (dequant "quant_ext.dequantize_per_tensor")
#guard Rules.Dce.erasable (dequant "quant_ext.dequantize_per_channel")
#guard Rules.Dce.erasable (dequant "quant_ext.quantize_per_tensor")
#guard !Rules.Dce.erasable (dequant "quant_ext.unknown")

end MerlinVerify.Tests.ExactCseRegression
