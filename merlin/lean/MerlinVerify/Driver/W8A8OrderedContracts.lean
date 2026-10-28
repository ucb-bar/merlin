/- Separate source/reference numerical demands for the same actual input domain.
An approximate numerical policy does not imply identical intermediate exponent caps. -/
import MerlinVerify.Driver.W8A8Rotary

namespace MerlinVerify.Driver.W8A8OrderedContracts

open Lean MerlinVerify

structure Contract where
  caps : List (Option Nat)
  sourceRequired : List (ValueId × Nat)
  referenceRequired : List (ValueId × Nat)

abbrev Contracts := List (String × Contract)

def decode (value : Json) : Except String Contracts := do
  let entries := (← value.getArr?).toList
  let project (side : String) : Except String Json := do
    let projected ← entries.mapM fun entry => do
      Decode.exactKeys "ordered exponent contract" entry
        ["function", "exponent_caps", "source_required_bounds", "reference_required_bounds"] []
      return Json.mkObj [("function", ← entry.getObjVal? "function"),
        ("exponent_caps", ← entry.getObjVal? "exponent_caps"),
        ("required_bounds", ← entry.getObjVal? side)]
    return toJson projected
  let source ← W8A8Rotary.decodeCaps (← project "source_required_bounds")
  let reference ← W8A8Rotary.decodeCaps (← project "reference_required_bounds")
  return source.zipWith (fun (name, left) (_, right) =>
    (name, ⟨left.caps, left.required, right.required⟩)) reference

end MerlinVerify.Driver.W8A8OrderedContracts
